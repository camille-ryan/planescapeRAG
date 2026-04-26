"""Merge a per-document entity registry into a global multi-document registry.

When a new document is ingested, its pass-1 registry needs to be reconciled
with the global registry of entities seen across all prior documents:
- "Doomguard" already in global → merge (union aliases, OR notable, append source_documents)
- "the Doomguard" in new (case/article variant) → fuzzy-match into existing canonical
- "Hive Ward" not in global → add as new entity

Each entity carries a `source_documents` list tracking which documents it
appears in. New entities default to [new_doc_tag]; merged entities accumulate
across docs.

Usage:
  python -m planescapeRAG.merge_registries \\
      global_registry.json new_registry.json \\
      --new-doc-tag campaign_setting

If global_registry.json doesn't exist, a fresh one is created.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import regex


def _ensure_source_documents(registry: dict, default_doc_tag: Optional[str] = None) -> int:
    """Backfill missing source_documents on registry entries.

    Returns the number of entries that were backfilled. Useful when migrating
    older registries that pre-date the source_documents field.
    """
    backfilled = 0
    for entity in registry["entities"]:
        if "source_documents" not in entity:
            entity["source_documents"] = [default_doc_tag] if default_doc_tag else []
            backfilled += 1
    return backfilled


def _merge_into(existing: dict, incoming: dict) -> None:
    """Merge incoming entity into existing, in place.

    aliases: union (preserving order of first appearance).
    brief_description: longer wins.
    notable: OR.
    source_documents: union.
    type: existing wins (don't churn types after first extraction).
    """
    existing["aliases"] = list(
        dict.fromkeys((existing.get("aliases") or []) + (incoming.get("aliases") or []))
    )
    if len(incoming.get("brief_description") or "") > len(existing.get("brief_description") or ""):
        existing["brief_description"] = incoming["brief_description"]
    existing["notable"] = bool(existing.get("notable")) or bool(incoming.get("notable"))
    existing["source_documents"] = list(
        dict.fromkeys((existing.get("source_documents") or []) + (incoming.get("source_documents") or []))
    )


def _find_fuzzy_canonical(target: str, candidates: List[str]) -> Optional[str]:
    """Find a candidate canonical name that's a near-match for target.

    Edit budget: ~1 edit per 12 characters, capped at 3. Tight enough to avoid
    spurious merges (e.g. "Plane of Air" vs "Plane of War"); loose enough to
    catch case/article variants and small OCR drift.
    """
    target_norm = target.strip().lower()
    if target.lower().startswith("the "):
        target_norm = target_norm[4:]
    max_edits = max(1, min(3, len(target_norm) // 12))

    pattern = regex.compile(
        f"^({regex.escape(target_norm)}){{e<={max_edits}}}$",
        regex.IGNORECASE | regex.BESTMATCH,
    )
    for cand in candidates:
        cand_norm = cand.strip().lower()
        if cand_norm.startswith("the "):
            cand_norm = cand_norm[4:]
        if cand_norm == target_norm:
            return cand  # exact-after-normalize
        if pattern.fullmatch(cand_norm):
            return cand
    return None


def merge_registries(
    global_registry: dict,
    new_registry: dict,
    verify_fn: Optional[Callable[[List[Tuple[dict, dict]]], List[bool]]] = None,
) -> Tuple[dict, dict]:
    """Merge new_registry into global_registry. Returns (merged, stats).

    If verify_fn is provided, fuzzy-name match candidates are sent to it as a
    batched list of (existing, incoming) pairs, returning a bool per pair.
    Pairs marked False are treated as distinct entities (incoming gets added
    as new), preventing false-merge bugs like 'Manes'→'Mazes'.
    """
    by_name = {e["canonical_name"]: e for e in global_registry["entities"]}
    stats = {
        "merged_exact": 0,
        "merged_fuzzy": 0,
        "rejected_fuzzy": 0,
        "added_new": 0,
        "fuzzy_decisions": [],  # list of (incoming_name, existing_name, "merged"|"split")
    }

    # Phase 1: classify each new entity into one of three buckets.
    to_merge_exact: List[Tuple[dict, dict]] = []
    fuzzy_candidates: List[Tuple[dict, dict]] = []
    to_add_new: List[dict] = []
    for new_entity in new_registry["entities"]:
        name = new_entity["canonical_name"]
        if name in by_name:
            to_merge_exact.append((by_name[name], new_entity))
            continue
        fuzzy = _find_fuzzy_canonical(name, list(by_name.keys()))
        if fuzzy:
            fuzzy_candidates.append((by_name[fuzzy], new_entity))
            continue
        to_add_new.append(new_entity)

    # Phase 2: verify fuzzy candidates if a verifier is available.
    if verify_fn and fuzzy_candidates:
        decisions = verify_fn(fuzzy_candidates)
        if len(decisions) != len(fuzzy_candidates):
            raise RuntimeError(f"verify_fn returned {len(decisions)} decisions for {len(fuzzy_candidates)} pairs")
    else:
        decisions = [True] * len(fuzzy_candidates)

    # Phase 3: apply.
    for existing, incoming in to_merge_exact:
        _merge_into(existing, incoming)
        stats["merged_exact"] += 1

    for (existing, incoming), should_merge in zip(fuzzy_candidates, decisions):
        if should_merge:
            _merge_into(existing, incoming)
            stats["merged_fuzzy"] += 1
            stats["fuzzy_decisions"].append((incoming["canonical_name"], existing["canonical_name"], "merged"))
        else:
            by_name[incoming["canonical_name"]] = dict(incoming)
            stats["rejected_fuzzy"] += 1
            stats["fuzzy_decisions"].append((incoming["canonical_name"], existing["canonical_name"], "split"))

    for incoming in to_add_new:
        by_name[incoming["canonical_name"]] = dict(incoming)
        stats["added_new"] += 1

    global_registry["entities"] = list(by_name.values())
    return global_registry, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a per-document registry into the global multi-document registry.")
    parser.add_argument("global_registry", type=Path, help="Existing global registry (created if missing).")
    parser.add_argument("new_registry", type=Path, help="New per-doc registry to merge.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write merged registry (default: overwrite global_registry).")
    parser.add_argument("--new-doc-tag", type=str, required=True, help="Doc-tag stamped onto new registry entries' source_documents (if not already present).")
    parser.add_argument("--global-doc-tag", type=str, default=None, help="If global_registry is missing source_documents, backfill with this tag.")
    parser.add_argument("--verify-merges", action="store_true", help="Use the LLM to verify fuzzy-matched pairs before merging (recommended).")
    parser.add_argument("--verify-model", type=str, default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    if args.global_registry.exists():
        with args.global_registry.open() as f:
            global_reg = json.load(f)
    else:
        global_reg = {"entities": []}

    with args.new_registry.open() as f:
        new_reg = json.load(f)

    global_backfilled = _ensure_source_documents(global_reg, args.global_doc_tag)
    new_backfilled = _ensure_source_documents(new_reg, args.new_doc_tag)
    if global_backfilled:
        print(f"Backfilled source_documents on {global_backfilled} existing global entries.")
    if new_backfilled:
        print(f"Backfilled source_documents on {new_backfilled} new entries with tag={args.new_doc_tag!r}.")

    verify_fn = None
    if args.verify_merges:
        from .verify_fuzzy_merges import verify_merges as _verify_impl
        verify_fn = lambda pairs: _verify_impl(pairs, model=args.verify_model)

    merged, stats = merge_registries(global_reg, new_reg, verify_fn=verify_fn)

    out_path = args.output or args.global_registry
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    print(f"\nMerge stats:")
    print(f"  exact-name matches:      {stats['merged_exact']}")
    print(f"  fuzzy-name matches:      {stats['merged_fuzzy']}")
    print(f"  fuzzy rejected (split):  {stats['rejected_fuzzy']}")
    print(f"  added as new:            {stats['added_new']}")
    print(f"  total entities now:      {len(merged['entities'])}")
    if stats["fuzzy_decisions"]:
        print(f"\nFuzzy-pair decisions:")
        for new_name, existing_name, action in stats["fuzzy_decisions"][:30]:
            verb = "merged into" if action == "merged" else "kept SEPARATE from"
            print(f"  {new_name!r:<40} {verb} {existing_name!r}")
    print(f"\nWrote {out_path}")
