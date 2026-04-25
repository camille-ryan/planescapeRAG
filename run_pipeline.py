"""End-to-end orchestrator: PDF → entity graph.

Runs the six stages of the pipeline in sequence:
  1. OCR + combine      (chunked OpenAI batches → combined.md)
  2. Registry           (pass 1: Anthropic, single call)
  3. Dedup              (canonical-name + alias dedup, in-memory)
  4. Canonicalize       (type → supertype mapping, Haiku)
  5. Profiles submit    (pass 2: Anthropic batch)
  6. Profiles collect   (poll until done, validate, write entity .md files)

Each stage uses the same intermediate paths (under output-root/doc-tag/) so a
re-run can pick up from any stage via --start-from. The pass-2 batch id is
persisted to pass2_batch_id.txt so the collect stage can resume after the
submit stage even in separate invocations.

Example:
  python -m planescapeRAG.run_pipeline \\
      --pdf "planescapeRAG/tmp/test_ingest/TSR 2603 Planes of Chaos.pdf" \\
      --profile planescapeRAG/corpus_profiles/planescape_tsr.md \\
      --doc-tag planes_of_chaos
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .canonicalize_types import canonicalize_types
from .dedup_registry import dedup_registry
from .extract_profiles import collect_profiles, submit_profiles
from .extract_registry import extract_registry
from .ocr import load_corpus_profile
from .submit_batches import submit_chunked


STAGES = ["ocr", "registry", "dedup", "canonicalize", "profiles_submit", "profiles_collect"]


def _stage_header(idx: int, total: int, name: str) -> None:
    print(f"\n=== Stage {idx}/{total}: {name} ===", flush=True)


def run_pipeline(
    pdf_path: Path,
    profile_path: Path,
    doc_tag: str,
    output_root: Path,
    graph_root: Path,
    start_from: str = "ocr",
    chunk_size: int = 50,
    canonicalize_model: str = "claude-haiku-4-5-20251001",
) -> None:
    if start_from not in STAGES:
        raise ValueError(f"Unknown stage: {start_from!r}. Valid: {STAGES}")
    start_idx = STAGES.index(start_from)

    work_dir = output_root / doc_tag
    work_dir.mkdir(parents=True, exist_ok=True)
    pdf_stem = pdf_path.stem.replace(" ", "_")
    combined_md = work_dir / f"{pdf_stem}_combined.md"
    registry_path = work_dir / "registry.json"
    supertype_map_path = work_dir / "type_supertype_mapping.json"
    pass2_batch_file = work_dir / "pass2_batch_id.txt"

    profile = load_corpus_profile(profile_path)

    overall_start = time.time()
    total = len(STAGES)

    # 1. OCR + combine
    if start_idx <= 0:
        _stage_header(1, total, "OCR + combine")
        t0 = time.time()
        submit_chunked(
            pdf_path=pdf_path,
            output_dir=work_dir,
            profile=profile,
            chunk_size=chunk_size,
        )
        if not combined_md.exists():
            raise FileNotFoundError(f"Expected combined markdown at {combined_md} after OCR stage")
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 2. Pass 1: entity registry
    if start_idx <= 1:
        _stage_header(2, total, "pass 1 — entity registry")
        t0 = time.time()
        extract_registry(
            document_path=combined_md,
            output_path=registry_path,
            profile=profile,
        )
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 3. Dedup registry (canonical-name + alias)
    if start_idx <= 2:
        _stage_header(3, total, "dedup registry")
        t0 = time.time()
        with registry_path.open() as f:
            registry = json.load(f)
        registry, stats = dedup_registry(registry)
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
        print(f"  merged {stats['entities_merged']} entities, removed {stats['aliases_removed']} duplicate aliases ({time.time() - t0:.0f}s)", flush=True)

    # 4. Canonicalize types → supertypes
    if start_idx <= 3:
        _stage_header(4, total, "canonicalize types")
        t0 = time.time()
        canonicalize_types(
            registry_path=registry_path,
            output_path=supertype_map_path,
            profile=profile,
            model=canonicalize_model,
        )
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    # 5. Pass 2 submit
    if start_idx <= 4:
        _stage_header(5, total, "pass 2 submit")
        t0 = time.time()
        batch_id = submit_profiles(
            document_path=combined_md,
            registry_path=registry_path,
            supertype_map_path=supertype_map_path,
            output_dir=graph_root,
            corpus_profile=profile,
        )
        if batch_id:
            pass2_batch_file.write_text(batch_id)
            print(f"  batch_id={batch_id} (saved to {pass2_batch_file}) ({time.time() - t0:.0f}s)", flush=True)
        else:
            print(f"  no entities to submit (all skipped) ({time.time() - t0:.0f}s)", flush=True)

    # 6. Pass 2 collect
    if start_idx <= 5:
        _stage_header(6, total, "pass 2 collect")
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
                registry_path=registry_path,
                supertype_map_path=supertype_map_path,
                output_dir=graph_root,
                corpus_profile=profile,
            )
            print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n=== Pipeline complete ({time.time() - overall_start:.0f}s total) ===", flush=True)


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="End-to-end pipeline: PDF → graph entities.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--doc-tag", type=str, required=True, help="Subdirectory name under output-root for this document's intermediate artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("planescapeRAG/tmp"))
    parser.add_argument("--graph-root", type=Path, default=Path("planescapeRAG/graph/entities"))
    parser.add_argument("--start-from", type=str, default="ocr", choices=STAGES, help="Skip stages before this one (their outputs must already exist).")
    parser.add_argument("--chunk-size", type=int, default=50, help="OCR pages per chunk; sized for tier-1 enqueued-token caps.")
    parser.add_argument("--canonicalize-model", type=str, default="claude-haiku-4-5-20251001")
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
    )
