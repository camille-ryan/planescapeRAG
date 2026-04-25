"""Post-pass-1 dedup for entity registries.

Two passes:
1. Merge entities sharing a canonical_name (LLM occasionally emits two entries
   for the same entity from different parts of a document).
2. Remove duplicate aliases within each entity.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple


def _normalize(s: str) -> str:
    """Normalize for alias-equivalence comparison.

    Lowercases, strips whitespace, and drops a leading 'the '. Two strings that
    normalize to the same value are treated as the same alias.
    """
    s = s.strip().lower()
    if s.startswith("the "):
        s = s[4:]
    return s


def dedup_canonical_names(registry: dict) -> Tuple[dict, int]:
    """Merge entities sharing a canonical_name, in place.

    When two entries have the same canonical_name:
    - aliases are unioned (preserving order of first appearance)
    - longer brief_description wins
    - notable is OR'd
    - type from the first entry wins (could differ; we accept the first)

    Returns (registry, merge_count).
    """
    merged_count = 0
    by_name: dict = {}
    order: list = []
    for entity in registry["entities"]:
        name = entity["canonical_name"]
        if name in by_name:
            existing = by_name[name]
            existing["aliases"] = list(
                dict.fromkeys((existing.get("aliases") or []) + (entity.get("aliases") or []))
            )
            if len(entity.get("brief_description", "")) > len(existing.get("brief_description", "")):
                existing["brief_description"] = entity["brief_description"]
            existing["notable"] = bool(existing.get("notable")) or bool(entity.get("notable"))
            merged_count += 1
        else:
            by_name[name] = dict(entity)
            order.append(name)
    registry["entities"] = [by_name[n] for n in order]
    return registry, merged_count


def dedup_aliases(registry: dict) -> Tuple[dict, int]:
    """Remove duplicate aliases within each entity, in place.

    An alias is considered a duplicate if (case-insensitively, ignoring a
    leading 'the') it matches the canonical_name or another alias already kept.
    Returns (registry, total_removed).
    """
    removed = 0
    for entity in registry["entities"]:
        original = entity.get("aliases", []) or []
        canonical_norm = _normalize(entity["canonical_name"])
        seen = {canonical_norm}
        kept = []
        for alias in original:
            norm = _normalize(alias)
            if norm in seen:
                continue
            seen.add(norm)
            kept.append(alias)
        removed += len(original) - len(kept)
        entity["aliases"] = kept
    return registry, removed


def dedup_registry(registry: dict) -> Tuple[dict, dict]:
    """Run both dedup passes in order. Returns (registry, stats)."""
    registry, merged = dedup_canonical_names(registry)
    registry, alias_removed = dedup_aliases(registry)
    return registry, {"entities_merged": merged, "aliases_removed": alias_removed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedup an entity registry: merge same-canonical-name entries, then prune duplicate aliases.")
    parser.add_argument("input", type=Path, help="Registry JSON to dedup.")
    parser.add_argument("output", type=Path, help="Path to write the deduped registry (may equal input to overwrite).")
    args = parser.parse_args()

    with args.input.open() as f:
        registry = json.load(f)

    before_entities = len(registry["entities"])
    before_aliases = sum(len(e.get("aliases", []) or []) for e in registry["entities"])
    registry, stats = dedup_registry(registry)
    after_entities = len(registry["entities"])
    after_aliases = sum(len(e.get("aliases", []) or []) for e in registry["entities"])

    args.output.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
    print(f"Entities: {before_entities} → {after_entities} (merged {stats['entities_merged']})")
    print(f"Aliases:  {before_aliases} → {after_aliases} (removed {stats['aliases_removed']})")
    print(f"Wrote {args.output}")
