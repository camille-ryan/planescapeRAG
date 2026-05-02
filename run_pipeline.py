"""End-to-end orchestrator: PDF → entity graph.

Runs the ten stages of the pipeline in sequence:
  1. OCR + combine          (chunked OpenAI batches → combined.md)
  2. Registry               (pass 1: Anthropic, single call → per-doc registry)
  3. Dedup                  (canonical-name + alias dedup, in-memory)
  4. Merge                  (per-doc registry → global registry, with LLM verifier)
  5. Sweep                  (LLM completeness sweep on doc against global → apply)
  6. Heuristic              (heuristic candidates → LLM verifier → apply to global)
  7. Canonicalize           (type → supertype mapping over the global registry, Haiku)
  8. Profiles submit        (pass 2: Anthropic batch, windowed mode against global)
  9. Profiles collect       (poll until done, validate, write entity .md files)
 10. Embed                  (OpenAI embeddings of entity bodies → FAISS-ready index for chatbot)

Per-doc state lives under output-root/doc-tag/. Global state lives under
graph-root/ (registry.json, type_supertype_mapping.json, entities/).

Each stage's outputs feed the next; --start-from lets you resume mid-pipeline.

Example:
  python -m planescapeRAG.run_pipeline \\
      --pdf "planescapeRAG/tmp/test_ingest/TSR 2603 Planes of Chaos.pdf" \\
      --profile planescapeRAG/corpus_profiles/planescape_tsr.md \\
      --doc-tag planes_of_chaos
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .chatbot.embed_entities import update_index as embed_update_index
from .pipeline.canonicalize_types import canonicalize_types
from .pipeline.completeness_sweep import apply_suggestions_to_registry, sweep_for_missing
from .pipeline.dedup_registry import dedup_registry
from .pipeline.extract_profiles import collect_profiles, submit_profiles
from .pipeline.extract_registry import extract_registry
from .pipeline.heuristic_entity_candidates import (
    extract_candidates,
    filter_against_registry,
    filter_real_candidates,
)
from .pipeline.merge_registries import _ensure_source_documents, merge_registries
from .pipeline.ocr import apply_ocr_corrections, load_corpus_profile, parse_ocr_corrections
from .pipeline.submit_batches import submit_chunked
from .pipeline.verify_candidates import apply_decisions, verify_candidates
from .pipeline.verify_fuzzy_merges import verify_merges as verify_merges_fn


STAGES = [
    "ocr",
    "registry",
    "dedup",
    "merge",
    "sweep",
    "heuristic",
    "canonicalize",
    "profiles_submit",
    "profiles_collect",
    "embed",
]


def _stage_header(idx: int, total: int, name: str) -> None:
    print(f"\n=== Stage {idx}/{total}: {name} ===", flush=True)


def _load_registry(path: Path) -> dict:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {"entities": []}


def _save_registry(path: Path, registry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


_TSR_PREFIX_RE = re.compile(r"^TSR\s*\d+\s+")


def _infer_title_from_pdf(pdf_path: Path) -> str:
    """Best-effort pretty title for a PDF. Strips the 'TSR XXXX ' product-code prefix."""
    return _TSR_PREFIX_RE.sub("", pdf_path.stem).strip()


def _register_doc_source(
    doc_tag: str,
    pdf_path: Path,
    project_root: Path,
    *,
    source_type: str = "canonical",
    title: Optional[str] = None,
) -> None:
    """Add or refresh the doc_tag entry in chatbot/doc_sources.json. The new
    schema stores an object per doc_tag with `pdf`, `type`, and `title` fields.
    Existing entries with bare-string values (legacy format) are upgraded in place."""
    sources_path = project_root / "chatbot" / "doc_sources.json"
    if sources_path.exists():
        data = json.loads(sources_path.read_text())
    else:
        data = {
            "_comment": (
                "Maps doc_tag (used in `## Source: <doc_tag>` headers) to a record "
                "describing the source. Paths are relative to the planescapeRAG/ "
                "package directory. `type` is 'canonical' or 'adventure' — the "
                "chatbot frames adventure-sourced facts as narrative content. "
                "Used by /page/{doc_tag}/{page} when PLANESCAPE_SHOW_PAGES=1."
            ),
        }

    try:
        rel = pdf_path.resolve().relative_to(project_root.resolve())
        path_str = str(rel)
    except ValueError:
        path_str = str(pdf_path.resolve())

    entry = {
        "pdf": path_str,
        "type": source_type,
        "title": title or _infer_title_from_pdf(pdf_path),
    }
    if data.get(doc_tag) != entry:
        data[doc_tag] = entry
        sources_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(
            f"  registered {doc_tag} → {path_str} "
            f"[type={source_type}, title={entry['title']!r}] in doc_sources.json",
            flush=True,
        )


def run_pipeline(
    pdf_path: Path,
    profile_path: Path,
    doc_tag: str,
    output_root: Path,
    graph_root: Path,
    start_from: str = "ocr",
    chunk_size: int = 50,
    canonicalize_model: str = "claude-haiku-4-5-20251001",
    pass2_mode: str = "windowed",
    pass2_no_skip_existing: bool = False,
    pass2_cost_threshold_usd: float = 1.00,
    auto_yes: bool = False,
    heuristic_top: int = 80,
    heuristic_min_count: int = 3,
    source_type: str = "canonical",
    source_title: Optional[str] = None,
    stop_after: Optional[str] = None,
) -> None:
    if start_from not in STAGES:
        raise ValueError(f"Unknown stage: {start_from!r}. Valid: {STAGES}")
    start_idx = STAGES.index(start_from)
    if stop_after is not None:
        if stop_after not in STAGES:
            raise ValueError(f"Unknown stop-after stage: {stop_after!r}. Valid: {STAGES}")
        stop_idx = STAGES.index(stop_after)
    else:
        stop_idx = len(STAGES) - 1

    # Per-doc paths
    work_dir = output_root / doc_tag
    work_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = pdf_path.stem.replace(" ", "_")
    # OCR'd markdown lives outside tmp/ so it survives tmp cleanups and is
    # editable as canonical source. Path is `<project>/ocr/<doc_tag>.md`.
    ocr_dir = graph_root.parent / "ocr"
    combined_md = ocr_dir / f"{doc_tag}.md"
    legacy_combined_md = work_dir / f"{pdf_stem}_combined.md"
    if not combined_md.exists() and legacy_combined_md.exists():
        # One-time migration on read: if a previous run wrote to the old
        # location (legacy tmp/<tag>/<stem>_combined.md), copy it forward.
        ocr_dir.mkdir(parents=True, exist_ok=True)
        combined_md.write_bytes(legacy_combined_md.read_bytes())
        print(f"  migrated OCR'd text: {legacy_combined_md} → {combined_md}", flush=True)
    per_doc_registry = work_dir / "registry.json"
    pass2_batch_file = work_dir / "pass2_batch_id.txt"
    sweep_path = work_dir / "sweep_suggestions.json"
    heuristic_candidates_path = work_dir / "heuristic_candidates.json"
    heuristic_decisions_path = work_dir / "heuristic_decisions.json"

    # Global paths
    global_registry = graph_root / "registry.json"
    global_supertype_map = graph_root / "type_supertype_mapping.json"
    entities_dir = graph_root / "entities"

    profile = load_corpus_profile(profile_path)

    overall_start = time.time()
    total = len(STAGES)

    # 1. OCR + combine
    if start_idx <= 0 <= stop_idx:
        _stage_header(1, total, "OCR + combine")
        t0 = time.time()
        submit_chunked(
            pdf_path=pdf_path, output_dir=work_dir, profile=profile,
            chunk_size=chunk_size, combined_md_path=combined_md,
        )
        if not combined_md.exists():
            raise FileNotFoundError(f"Expected combined markdown at {combined_md} after OCR stage")
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 2. Pass 1: entity registry
    if start_idx <= 1 <= stop_idx:
        _stage_header(2, total, "pass 1 — entity registry")
        t0 = time.time()
        extract_registry(document_path=combined_md, output_path=per_doc_registry, profile=profile)
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 3. Dedup the per-doc registry in place
    if start_idx <= 2 <= stop_idx:
        _stage_header(3, total, "dedup registry")
        t0 = time.time()
        registry = _load_registry(per_doc_registry)
        registry, stats = dedup_registry(registry)
        _save_registry(per_doc_registry, registry)
        print(f"  merged {stats['entities_merged']} entities, removed {stats['aliases_removed']} duplicate aliases ({time.time() - t0:.0f}s)", flush=True)

    # 4. Merge per-doc registry into global registry (with LLM verifier on fuzzy pairs)
    if start_idx <= 3 <= stop_idx:
        _stage_header(4, total, "merge into global registry")
        t0 = time.time()
        global_reg = _load_registry(global_registry)
        new_reg = _load_registry(per_doc_registry)
        _ensure_source_documents(new_reg, doc_tag)
        merged, stats = merge_registries(global_reg, new_reg, verify_fn=verify_merges_fn)
        _save_registry(global_registry, merged)
        print(f"  exact={stats['merged_exact']}  fuzzy_merged={stats['merged_fuzzy']}  fuzzy_split={stats['rejected_fuzzy']}  added_new={stats['added_new']}  total_now={len(merged['entities'])} ({time.time() - t0:.0f}s)", flush=True)

    # 5. LLM completeness sweep against the global registry
    if start_idx <= 4 <= stop_idx:
        _stage_header(5, total, "LLM completeness sweep")
        t0 = time.time()
        valid = sweep_for_missing(
            document_path=combined_md,
            registry_path=global_registry,
            output_path=sweep_path,
            profile=profile,
            doc_tag=doc_tag,
        )
        registry = _load_registry(global_registry)
        added = apply_suggestions_to_registry(registry, valid, doc_tag)
        _save_registry(global_registry, registry)
        print(f"  added {added} entities from sweep (total now: {len(registry['entities'])}) ({time.time() - t0:.0f}s)", flush=True)

    # 6. Heuristic candidates → LLM verifier → apply to global
    if start_idx <= 5 <= stop_idx:
        _stage_header(6, total, "heuristic candidate sweep")
        t0 = time.time()
        document = combined_md.read_text()
        document = apply_ocr_corrections(document, parse_ocr_corrections(profile))
        registry = _load_registry(global_registry)

        raw = extract_candidates(document)
        real = filter_real_candidates(raw, document)
        unmatched = filter_against_registry(real, registry)
        candidates = sorted(
            ((n, c) for n, c in unmatched.items() if c >= heuristic_min_count),
            key=lambda x: -x[1],
        )[: heuristic_top]
        heuristic_candidates_path.write_text(json.dumps(
            [{"name": n, "count": c} for n, c in candidates],
            indent=2, ensure_ascii=False,
        ))
        print(f"  heuristic surfaced {len(candidates)} candidates (top {heuristic_top}, min count {heuristic_min_count})", flush=True)

        if candidates:
            decisions = verify_candidates(
                candidates=candidates,
                document=document,
                registry=registry,
                profile=profile,
            )
            heuristic_decisions_path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False))
            stats = apply_decisions(registry, decisions, doc_tag)
            _save_registry(global_registry, registry)
            print(f"  applied: skipped={stats['skipped']} merged={stats['merged']} added={stats['added']} (total now: {len(registry['entities'])})", flush=True)
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 7. Canonicalize types over the global registry
    if start_idx <= 6 <= stop_idx:
        _stage_header(7, total, "canonicalize types (global)")
        t0 = time.time()
        canonicalize_types(
            registry_path=global_registry,
            output_path=global_supertype_map,
            profile=profile,
            model=canonicalize_model,
            doc_tag=doc_tag,
        )
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 8. Pass 2 submit (against the global registry)
    if start_idx <= 7 <= stop_idx:
        _stage_header(8, total, "pass 2 submit")
        t0 = time.time()
        batch_id = submit_profiles(
            document_path=combined_md,
            registry_path=global_registry,
            supertype_map_path=global_supertype_map,
            output_dir=entities_dir,
            doc_tag=doc_tag,
            corpus_profile=profile,
            mode=pass2_mode,
            skip_existing=not pass2_no_skip_existing,
            cost_threshold_usd=pass2_cost_threshold_usd,
            auto_yes=auto_yes,
        )
        if batch_id:
            pass2_batch_file.write_text(batch_id)
            print(f"  batch_id={batch_id} (saved to {pass2_batch_file}) ({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"  no entities to submit (all skipped) ({time.time() - t0:.0f}s)", flush=True)

    # 9. Pass 2 collect
    if start_idx <= 8 <= stop_idx:
        _stage_header(9, total, "pass 2 collect")
        t0 = time.time()
        if not pass2_batch_file.exists():
            raise FileNotFoundError(f"No batch ID at {pass2_batch_file} — was the submit stage skipped?")
        batch_id = pass2_batch_file.read_text().strip()
        if not batch_id:
            print(f"  empty batch id (nothing to collect) ({time.time() - t0:.0f}s)", flush=True)
        else:
            collect_profiles(
                batch_id=batch_id,
                document_path=combined_md,
                registry_path=global_registry,
                supertype_map_path=global_supertype_map,
                output_dir=entities_dir,
                doc_tag=doc_tag,
                corpus_profile=profile,
            )
            print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 10. Embed entities for chatbot retrieval (incremental: only new/changed)
    #     Also register the doc_tag → PDF mapping for the chatbot's page-preview endpoint.
    if start_idx <= 9 <= stop_idx:
        _stage_header(10, total, "embed entities for chatbot")
        t0 = time.time()
        project_root = graph_root.parent
        chatbot_index_dir = project_root / "chatbot" / "index"

        _register_doc_source(
            doc_tag, pdf_path, project_root,
            source_type=source_type, title=source_title,
        )

        if not entities_dir.exists():
            print(f"  skipped: no entities at {entities_dir} yet", flush=True)
        else:
            try:
                embed_update_index(
                    entities_dir=entities_dir,
                    vectors_path=chatbot_index_dir / "vectors.npy",
                    metadata_path=chatbot_index_dir / "metadata.json",
                    auto_yes=auto_yes,
                    doc_tag=doc_tag,
                )
                print(f"  done ({time.time() - t0:.0f}s)", flush=True)
            except RuntimeError as e:
                # OPENAI_API_KEY missing, etc. — chatbot is optional, don't break the run.
                print(f"  skipped: {e}", flush=True)

    print(f"\n=== Pipeline complete ({time.time() - overall_start:.0f}s total) ===", flush=True)


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="End-to-end pipeline: PDF → graph entities.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--doc-tag", type=str, required=True, help="Subdirectory name under output-root for this document's intermediate artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("planescapeRAG/tmp"))
    parser.add_argument("--graph-root", type=Path, default=Path("planescapeRAG/graph"), help="Root for global state (registry.json, type_supertype_mapping.json, entities/).")
    parser.add_argument("--start-from", type=str, default="ocr", choices=STAGES, help="Skip stages before this one (their outputs must already exist).")
    parser.add_argument("--chunk-size", type=int, default=50, help="OCR pages per chunk; sized for tier-1 enqueued-token caps.")
    parser.add_argument("--canonicalize-model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--pass2-mode", type=str, default="windowed", choices=["fulldoc", "windowed"])
    parser.add_argument("--pass2-no-skip-existing", action="store_true", help="Re-profile entities even if their .md already exists. Use when the registry has expanded and you want refreshed wikilinks.")
    parser.add_argument("--pass2-cost-threshold", type=float, default=1.00, help="USD threshold above which the pass-2 submit prompts for confirmation.")
    parser.add_argument("--yes", action="store_true", help="Skip cost-confirmation prompts (for unattended runs).")
    parser.add_argument("--heuristic-top", type=int, default=80)
    parser.add_argument("--heuristic-min-count", type=int, default=3)
    parser.add_argument(
        "--source-type", type=str, default="canonical", choices=["canonical", "adventure"],
        help="'canonical' = setting reference (boxed sets, accessories). "
             "'adventure' = adventure module — chatbot will frame these as narrative content.",
    )
    parser.add_argument(
        "--source-title", type=str, default=None,
        help="Pretty title for citations (e.g. 'The Great Modron March'). "
             "Defaults to PDF stem with the leading 'TSR XXXX ' prefix stripped.",
    )
    parser.add_argument(
        "--stop-after", type=str, default=None, choices=STAGES,
        help="Stop after this stage completes (its outputs are persisted). "
             "Use with --start-from to run a slice of the pipeline. The two-pass "
             "batch script uses --stop-after canonicalize for phase 1.",
    )
    args = parser.parse_args()

    run_pipeline(
        pdf_path=args.pdf,
        profile_path=args.profile,
        doc_tag=args.doc_tag,
        output_root=args.output_root,
        graph_root=args.graph_root,
        start_from=args.start_from,
        chunk_size=args.chunk_size,
        canonicalize_model=args.canonicalize_model,
        pass2_mode=args.pass2_mode,
        pass2_no_skip_existing=args.pass2_no_skip_existing,
        pass2_cost_threshold_usd=args.pass2_cost_threshold,
        auto_yes=args.yes,
        heuristic_top=args.heuristic_top,
        heuristic_min_count=args.heuristic_min_count,
        source_type=args.source_type,
        source_title=args.source_title,
        stop_after=args.stop_after,
    )
