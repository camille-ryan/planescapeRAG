"""Pre-flight cost estimate for phase 2 (pass-2 submit + collect) across all
remaining docs. Builds the same per-entity requests submit_profiles would,
runs them through estimate_anthropic_batch_cost, and reports per-doc and
total projected spend — without submitting anything.

Usage:
    python -m planescapeRAG.tools.estimate_phase2

Reads the same DOCS list as batch_pipeline.py and only estimates docs that
don't already have entity files (skip-rule matches phase 2's behavior).
"""

import json
from pathlib import Path
from typing import List, Tuple

from ..config import Settings
from ..cost import estimate_anthropic_batch_cost
from ..pipeline.extract_profiles import (
    _build_request_fulldoc,
    _build_request_windowed,
    _build_registry_payload,
    _file_has_doc_section,
    _output_path,
    _parse_document_paragraphs,
)
from ..pipeline.ocr import apply_ocr_corrections, load_corpus_profile, parse_ocr_corrections
from .batch_pipeline import DOCS, _PROJECT_ROOT, _PROFILE_ADVENTURE, _PROFILE_CANONICAL, has_entity_files


GRAPH_ROOT = _PROJECT_ROOT / "planescapeRAG" / "graph"
ENTITIES_DIR = GRAPH_ROOT / "entities"
REGISTRY_PATH = GRAPH_ROOT / "registry.json"
SUPERTYPE_MAP_PATH = GRAPH_ROOT / "type_supertype_mapping.json"
OCR_DIR = _PROJECT_ROOT / "planescapeRAG" / "ocr"


def _build_pass2_requests_for_doc(
    doc_tag: str,
    corpus_profile_path: Path,
    model: str,
    mode: str = "windowed",
    cache_ttl: str = "1h",
    neighbor_paras: int = 1,
) -> Tuple[List[dict], int, int]:
    """Replicate submit_profiles' request-building loop without submitting.
    Returns (requests, n_skipped_existing, n_no_context)."""
    document_path = OCR_DIR / f"{doc_tag}.md"
    if not document_path.exists():
        return [], 0, 0

    corpus_profile = load_corpus_profile(corpus_profile_path)
    document = document_path.read_text()
    corrections = parse_ocr_corrections(corpus_profile)
    if corrections:
        document = apply_ocr_corrections(document, corrections)

    with REGISTRY_PATH.open() as f:
        registry = json.load(f)
    with SUPERTYPE_MAP_PATH.open() as f:
        supertype_map = json.load(f)

    registry_text = _build_registry_payload(registry)
    paragraphs = _parse_document_paragraphs(document) if mode == "windowed" else None

    requests: List[dict] = []
    skipped = 0
    no_context = 0
    for entity in registry["entities"]:
        supertype = supertype_map.get(entity["type"])
        if supertype is None:
            continue
        path = _output_path(ENTITIES_DIR, supertype, entity)
        if _file_has_doc_section(path, doc_tag):
            skipped += 1
            continue
        if mode == "fulldoc":
            requests.append(_build_request_fulldoc(entity, document, registry_text, model, cache_ttl))
        else:
            request = _build_request_windowed(entity, paragraphs, registry, model, cache_ttl, neighbor_paras)
            if request is None:
                no_context += 1
                continue
            requests.append(request)
    return requests, skipped, no_context


def main() -> None:
    settings = Settings()
    model = settings.extraction_model

    print(f"Estimating phase 2 cost for all pending docs (model: {model})")
    print(f"{'doc_tag':35s}  {'requests':>9}  {'est cost':>10}  status")
    print("-" * 72)

    total_cost = 0.0
    total_requests = 0
    n_pending = 0
    n_done = 0

    for doc in DOCS:
        if has_entity_files(doc):
            print(f"{doc.tag:35s}  {'-':>9}  {'-':>10}  done (skip)")
            n_done += 1
            continue

        profile_path = Path(_PROFILE_ADVENTURE if doc.source_type == "adventure" else _PROFILE_CANONICAL)
        try:
            requests, skipped, no_context = _build_pass2_requests_for_doc(
                doc.tag, profile_path, model=model,
            )
        except Exception as e:
            print(f"{doc.tag:35s}  {'-':>9}  {'-':>10}  ERROR: {e}")
            continue

        if not requests:
            print(f"{doc.tag:35s}  {'0':>9}  {'$0.00':>10}  no requests (skipped={skipped}, no_ctx={no_context})")
            continue

        estimate = estimate_anthropic_batch_cost(
            requests, model=model, is_batch=True, cache_ttl="1h",
        )
        cost = estimate.get("cost_usd") or 0.0
        total_cost += cost
        total_requests += len(requests)
        n_pending += 1
        print(f"{doc.tag:35s}  {len(requests):>9d}  ${cost:>9.2f}  pending")

    print("-" * 72)
    print(f"{'TOTAL':35s}  {total_requests:>9d}  ${total_cost:>9.2f}  ({n_pending} pending, {n_done} already done)")


if __name__ == "__main__":
    main()
