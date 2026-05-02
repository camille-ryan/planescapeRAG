"""Two-phase batch ingest, with the OCR pipe and the registry-stages pipe
running in parallel.

Phase 1 (--phase 1)
    Producer (main thread):  serial OCR loop across all docs.
    Consumer (worker thread): serial registry stages (2-7) pulled off a queue.
    The producer enqueues each doc as soon as its OCR finishes, so the
    consumer is processing doc N's stages 2-7 while the producer is doing
    OCR for doc N+1. OpenAI and Anthropic don't compete; total wall-clock
    is dominated by OCR alone (since stages 2-7 finish faster than OCR for
    most docs).

    Stages 2-7 are kept serial behind a single worker because they all
    mutate the shared global registry — concurrent runs would race.

Phase 2 (--phase 2)
    Serial pass-2 (stages 8-10) across all docs. By this point phase 1 has
    already populated the global registry with EVERY doc's entities, so the
    pass-2 LLM has the full canonical-name list when generating wikilinks.
    Wikilinks in the campaign_setting's entity bodies (regenerated in this
    phase) will resolve to entities introduced by Hellbound, Faction War,
    etc. — the cross-doc graph quality this whole architecture is for.

Phase all (--phase all)
    Phase 1 followed by phase 2.

Skip rules:
    - OCR step (in phase 1):    skipped if combined.md exists for the doc
    - stages 2-7 (in phase 1):  skipped if doc_tag appears in graph/registry.json
    - phase 2:                  skipped if entity files reference `## Source: <doc_tag>`

Usage:
    python -m planescapeRAG.tools.batch_pipeline 1
    python -m planescapeRAG.tools.batch_pipeline 2
    python -m planescapeRAG.tools.batch_pipeline all [--cost-threshold 5.00] [--yes]
"""

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, NamedTuple, Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent     # /mnt/e/proj
_PROFILE_CANONICAL = "planescapeRAG/corpus_profiles/planescape_tsr.md"
_PROFILE_ADVENTURE = "planescapeRAG/corpus_profiles/planescape_tsr_module.md"


class Doc(NamedTuple):
    pdf: str            # repo-relative path
    tag: str
    source_type: str    # 'canonical' or 'adventure'
    title: str


DOCS: List[Doc] = [
    # Boxed Sets — canonical
    Doc("planescapeRAG/documents/Boxed Sets/TSR 2600 Planescape Campaign Setting.pdf",
        "campaign_setting", "canonical", "Planescape Campaign Setting"),
    Doc("planescapeRAG/documents/Boxed Sets/TSR 2603 Planes of Chaos.pdf",
        "planes_of_chaos", "canonical", "Planes of Chaos"),
    Doc("planescapeRAG/documents/Boxed Sets/TSR 2607 Planes of Law.pdf",
        "planes_of_law", "canonical", "Planes of Law"),
    Doc("planescapeRAG/documents/Boxed Sets/TSR 2610 Player's Primer to the Outlands.pdf",
        "players_primer_to_the_outlands", "canonical", "Player's Primer to the Outlands"),
    Doc("planescapeRAG/documents/Boxed Sets/TSR 2615 Planes of Conflict.pdf",
        "planes_of_conflict", "canonical", "Planes of Conflict"),
    Doc("planescapeRAG/documents/Boxed Sets/TSR 2621 Hellbound The Blood War.pdf",
        "hellbound", "canonical", "Hellbound: The Blood War"),

    # Accessories — canonical
    Doc("planescapeRAG/documents/Accessories/TSR 2609 In The Cage A Guide to Sigil.pdf",
        "in_the_cage", "canonical", "In the Cage: A Guide to Sigil"),
    Doc("planescapeRAG/documents/Accessories/TSR 2611 The Factol's Manifesto.pdf",
        "factols_manifesto", "canonical", "The Factol's Manifesto"),
    Doc("planescapeRAG/documents/Accessories/TSR 2620 The Planeswalker's Handbook.pdf",
        "planeswalkers_handbook", "canonical", "The Planeswalker's Handbook"),
    Doc("planescapeRAG/documents/Accessories/TSR 2623 On Hallowed Ground.pdf",
        "on_hallowed_ground", "canonical", "On Hallowed Ground"),
    Doc("planescapeRAG/documents/Accessories/TSR 2624 Uncaged Faces of Sigil.pdf",
        "uncaged_faces_of_sigil", "canonical", "Uncaged: Faces of Sigil"),
    Doc("planescapeRAG/documents/Accessories/TSR 2625 A Guide to the Astral Plane.pdf",
        "guide_to_astral", "canonical", "A Guide to the Astral Plane"),
    Doc("planescapeRAG/documents/Accessories/TSR 2630 Faces of Evil the Fiends.pdf",
        "faces_of_evil", "canonical", "Faces of Evil: The Fiends"),
    Doc("planescapeRAG/documents/Accessories/TSR 2633 Planescape A Guide to the Ethereal Plane.pdf",
        "guide_to_ethereal", "canonical", "A Guide to the Ethereal Plane"),
    Doc("planescapeRAG/documents/Accessories/TSR 2634 The Inner Planes.pdf",
        "inner_planes", "canonical", "The Inner Planes"),

    # Modules — adventure
    Doc("planescapeRAG/documents/Modules/TSR 2601 The Eternal Boundary.pdf",
        "eternal_boundary", "adventure", "The Eternal Boundary"),
    Doc("planescapeRAG/documents/Modules/TSR 2604 The Well of Worlds.pdf",
        "well_of_worlds", "adventure", "The Well of Worlds"),
    Doc("planescapeRAG/documents/Modules/TSR 2605 In the Abyss.pdf",
        "in_the_abyss", "adventure", "In the Abyss"),
    Doc("planescapeRAG/documents/Modules/TSR 2606 The Deva Spark.pdf",
        "deva_spark", "adventure", "The Deva Spark"),
    Doc("planescapeRAG/documents/Modules/TSR 2608 Fires of Dis.pdf",
        "fires_of_dis", "adventure", "Fires of Dis"),
    Doc("planescapeRAG/documents/Modules/TSR 2614 Harbinger House.pdf",
        "harbinger_house", "adventure", "Harbinger House"),
    Doc("planescapeRAG/documents/Modules/TSR 2619 Something Wild.pdf",
        "something_wild", "adventure", "Something Wild"),
    Doc("planescapeRAG/documents/Modules/TSR 2626 Doors to the Unknown.pdf",
        "doors_to_the_unknown", "adventure", "Doors to the Unknown"),
    Doc("planescapeRAG/documents/Modules/TSR 2628 The Great Modron March.pdf",
        "great_modron_march", "adventure", "The Great Modron March"),
    Doc("planescapeRAG/documents/Modules/TSR 2629 Faction War.pdf",
        "faction_war", "adventure", "Faction War"),
    Doc("planescapeRAG/documents/Modules/TSR 2631 Dead Gods.pdf",
        "dead_gods", "adventure", "Dead Gods"),
    Doc("planescapeRAG/documents/Modules/TSR 2632 Tales From the Infinite Staircase.pdf",
        "infinite_staircase", "adventure", "Tales from the Infinite Staircase"),
]


# ── Skip-detection ─────────────────────────────────────────────────────────


def has_completed_ocr(doc: Doc) -> bool:
    pdf_stem = Path(doc.pdf).stem.replace(" ", "_")
    combined = _PROJECT_ROOT / "planescapeRAG" / "tmp" / doc.tag / f"{pdf_stem}_combined.md"
    return combined.exists()


def is_in_registry(doc: Doc) -> bool:
    """The doc has gone through stages 2-4 if its tag appears anywhere in the
    global registry's source_documents fields."""
    registry = _PROJECT_ROOT / "planescapeRAG" / "graph" / "registry.json"
    if not registry.exists():
        return False
    return f'"{doc.tag}"' in registry.read_text()


def has_entity_files(doc: Doc) -> bool:
    """Pass-2 has run if any entity .md file references this doc's source tag."""
    entities_dir = _PROJECT_ROOT / "planescapeRAG" / "graph" / "entities"
    if not entities_dir.exists():
        return False
    needle = f"## Source: {doc.tag}"
    for md_file in entities_dir.rglob("*.md"):
        try:
            if needle in md_file.read_text():
                return True
        except OSError:
            continue
    return False


# ── Pipeline invocation ────────────────────────────────────────────────────


def _run_segment(
    doc: Doc,
    *,
    start_from: str,
    stop_after: Optional[str],
    cost_threshold: str,
    auto_yes: bool,
) -> int:
    profile = _PROFILE_ADVENTURE if doc.source_type == "adventure" else _PROFILE_CANONICAL
    cmd = [
        sys.executable, "-m", "planescapeRAG.run_pipeline",
        "--pdf", doc.pdf,
        "--profile", profile,
        "--doc-tag", doc.tag,
        "--source-type", doc.source_type,
        "--source-title", doc.title,
        "--pass2-cost-threshold", cost_threshold,
        "--start-from", start_from,
    ]
    if stop_after is not None:
        cmd += ["--stop-after", stop_after]
    if auto_yes:
        cmd.append("--yes")
    return subprocess.run(cmd, cwd=_PROJECT_ROOT).returncode


# ── Phase 1 ────────────────────────────────────────────────────────────────


def run_phase1(cost_threshold: str, auto_yes: bool) -> None:
    work_q: "queue.Queue[Optional[Doc]]" = queue.Queue()
    stats = {"ocr_skipped": 0, "ocr_failed": 0, "stages_skipped": 0, "stages_failed": 0, "stages_done": 0}
    stats_lock = threading.Lock()

    def consumer() -> None:
        """Pull docs whose OCR is done, run stages 2-7 against the global registry."""
        while True:
            doc = work_q.get()
            if doc is None:
                return
            if is_in_registry(doc):
                with stats_lock:
                    stats["stages_skipped"] += 1
                print(f"[stages 2-7] SKIP  {doc.tag}  (already in registry)", flush=True)
                continue
            print(f"[stages 2-7] START {doc.tag}", flush=True)
            t0 = time.time()
            rc = _run_segment(
                doc, start_from="registry", stop_after="canonicalize",
                cost_threshold=cost_threshold, auto_yes=auto_yes,
            )
            dt = int(time.time() - t0)
            with stats_lock:
                if rc == 0:
                    stats["stages_done"] += 1
                    print(f"[stages 2-7] DONE  {doc.tag}  ({dt}s)", flush=True)
                else:
                    stats["stages_failed"] += 1
                    print(f"[stages 2-7] FAIL  {doc.tag}  exit={rc}  ({dt}s)", flush=True)

    print("=" * 70)
    print("PHASE 1 — OCR (serial) || stages 2-7 (serial worker, runs in parallel with OCR)")
    print(f"docs: {len(DOCS)}    cost threshold: ${cost_threshold}    auto_yes: {auto_yes}")
    print("=" * 70)

    worker = threading.Thread(target=consumer, daemon=True)
    worker.start()

    try:
        for doc in DOCS:
            pdf_full = _PROJECT_ROOT / doc.pdf
            if not pdf_full.exists():
                print(f"[OCR]        SKIP  {doc.tag}  (PDF not found at {doc.pdf})", flush=True)
                stats["ocr_failed"] += 1
                continue
            if has_completed_ocr(doc):
                print(f"[OCR]        SKIP  {doc.tag}  (OCR already cached)", flush=True)
                stats["ocr_skipped"] += 1
                work_q.put(doc)
                continue
            print(f"[OCR]        START {doc.tag}", flush=True)
            t0 = time.time()
            rc = _run_segment(
                doc, start_from="ocr", stop_after="ocr",
                cost_threshold=cost_threshold, auto_yes=auto_yes,
            )
            dt = int(time.time() - t0)
            if rc == 0:
                print(f"[OCR]        DONE  {doc.tag}  ({dt}s) — queued for stages 2-7", flush=True)
                work_q.put(doc)
            else:
                print(f"[OCR]        FAIL  {doc.tag}  exit={rc}  ({dt}s)", flush=True)
                stats["ocr_failed"] += 1
    finally:
        work_q.put(None)
        worker.join()

    print()
    print("=" * 70)
    print("PHASE 1 complete.")
    print(f"  OCR        — done implicitly by absence: {stats['ocr_skipped']} cached, {stats['ocr_failed']} failed")
    print(f"  stages 2-7 — {stats['stages_done']} done, {stats['stages_skipped']} skipped, {stats['stages_failed']} failed")
    print("  next: python -m planescapeRAG.tools.batch_pipeline 2")
    print("=" * 70)


# ── Phase 2 ────────────────────────────────────────────────────────────────


def run_phase2(cost_threshold: str, auto_yes: bool) -> None:
    print("=" * 70)
    print("PHASE 2 — pass-2 + collect + embed (serial)")
    print(f"docs: {len(DOCS)}    cost threshold: ${cost_threshold}    auto_yes: {auto_yes}")
    print("=" * 70)

    done = skipped = failed = 0
    for doc in DOCS:
        pdf_full = _PROJECT_ROOT / doc.pdf
        print(f"\n--- {doc.tag} ({doc.source_type}) ---", flush=True)
        if has_entity_files(doc):
            print("  SKIP — entity files already exist", flush=True)
            skipped += 1
            continue
        if not pdf_full.exists():
            print(f"  FAIL — PDF not found at {doc.pdf}", flush=True)
            failed += 1
            continue
        rc = _run_segment(
            doc, start_from="profiles_submit", stop_after=None,
            cost_threshold=cost_threshold, auto_yes=auto_yes,
        )
        if rc == 0:
            done += 1
        else:
            print(f"  FAIL — exit {rc} (continuing)", flush=True)
            failed += 1

    print()
    print("=" * 70)
    print(f"PHASE 2 complete.  done={done}  skipped={skipped}  failed={failed}")
    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase batch ingest with parallel OCR + registry-stages.",
    )
    parser.add_argument("phase", choices=["1", "2", "all"])
    parser.add_argument(
        "--cost-threshold", default="5.00",
        help="Pass-2 cost threshold per doc; the pipeline prompts above this. "
             "With --yes, threshold is informational only.",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Auto-confirm cost prompts (run unattended). Required when "
             "running detached (nohup) — input() fails on closed stdin.",
    )
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        run_phase1(args.cost_threshold, args.yes)
    if args.phase in ("2", "all"):
        run_phase2(args.cost_threshold, args.yes)


if __name__ == "__main__":
    main()
