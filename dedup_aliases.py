import argparse
import json
from pathlib import Path


def _normalize(s: str) -> str:
    """Normalize for alias-equivalence comparison.

    Lowercases, strips whitespace, and drops a leading 'the '. Two strings that
    normalize to the same value are treated as the same alias.
    """
    s = s.strip().lower()
    if s.startswith("the "):
        s = s[4:]
    return s


def dedup_aliases(registry: dict) -> tuple[dict, int]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate aliases from an entity registry.")
    parser.add_argument("input", type=Path, help="Registry JSON to dedup.")
    parser.add_argument("output", type=Path, help="Path to write the deduped registry (may equal input to overwrite).")
    args = parser.parse_args()

    with args.input.open() as f:
        registry = json.load(f)

    before = sum(len(e.get("aliases", []) or []) for e in registry["entities"])
    registry, removed = dedup_aliases(registry)
    after = sum(len(e.get("aliases", []) or []) for e in registry["entities"])

    args.output.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
    print(f"Aliases: {before} → {after} (removed {removed} duplicates)")
    print(f"Wrote {args.output}")
