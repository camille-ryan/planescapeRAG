"""Merge accidentally-split case-variant entries for category-type entities.

The fuzzy-merge verifier sometimes treats case differences as semantic
differences for creature- or concept-type entries, e.g. splitting:

    Vargouille  ↔  vargouille
    Hollyphant  ↔  hollyphant
    Bladeling   ↔  bladeling

These should always merge — they're categories, not individuals. (The
verifier prompt has been tightened, but pre-existing splits stay until
merged manually.)

Behavior:
- Find entries that share (type, canonical_name.lower()) with type in
  {creature, concept}, where multiple distinct canonical_names exist.
- Pick the "winner": more source_documents wins; tie → uppercase-first;
  tie → alphabetical.
- Merge into the winner: union source_documents, union aliases,
  add each loser's canonical_name to the winner's aliases.
- For brief_description, keep the winner's. (Manual review if the
  descriptions diverge meaningfully.)

Always writes a backup at <registry>.bak before mutating.

Usage:
    python -m planescapeRAG.tools.merge_case_variants
    python -m planescapeRAG.tools.merge_case_variants --dry-run
    python -m planescapeRAG.tools.merge_case_variants --types creature concept character
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_REGISTRY = Path("planescapeRAG/graph/registry.json")
DEFAULT_TYPES = {"creature", "concept"}


def _pick_winner_index(entries: List[dict]) -> int:
    """Return the index of the preferred entry among case-variants."""
    def sort_key(idx: int):
        e = entries[idx]
        n_sources = len(e.get("source_documents") or [])
        starts_upper = e["canonical_name"][:1].isupper()
        return (-n_sources, not starts_upper, e["canonical_name"])
    return min(range(len(entries)), key=sort_key)


def merge_case_variants(
    registry_path: Path,
    target_types: Set[str],
    dry_run: bool = False,
) -> Tuple[int, List[Tuple[str, List[str]]]]:
    reg = json.loads(registry_path.read_text())
    entities: List[dict] = reg["entities"]

    # Group by (type, lowercased canonical_name)
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, e in enumerate(entities):
        if e.get("type") not in target_types:
            continue
        groups[(e["type"], e["canonical_name"].lower())].append(i)

    case_variant_groups = {k: v for k, v in groups.items() if len(v) > 1}
    if not case_variant_groups:
        print("No case-variant pairs found.")
        return 0, []

    if not dry_run:
        backup = registry_path.with_suffix(registry_path.suffix + ".bak.before-merge-case-variants")
        shutil.copy2(registry_path, backup)
        print(f"backup: {backup}")

    drop_indices: Set[int] = set()
    actions: List[Tuple[str, List[str]]] = []

    for (etype, key), idxs in sorted(case_variant_groups.items()):
        candidates = [entities[i] for i in idxs]
        winner_local = _pick_winner_index(candidates)
        winner_global = idxs[winner_local]
        winner = entities[winner_global]
        losers = [entities[idxs[i]] for i in range(len(idxs)) if i != winner_local]

        # Union aliases (winner's existing + each loser's canonical + each loser's aliases)
        merged_aliases = list(winner.get("aliases") or [])
        merged_alias_set = {a.lower() for a in merged_aliases}
        for loser in losers:
            for term in [loser["canonical_name"]] + (loser.get("aliases") or []):
                if term and term.lower() not in merged_alias_set:
                    merged_aliases.append(term)
                    merged_alias_set.add(term.lower())
        # Don't add the winner's own canonical to its aliases
        merged_aliases = [a for a in merged_aliases if a.lower() != winner["canonical_name"].lower()]

        # Union source_documents
        merged_sources = list(winner.get("source_documents") or [])
        merged_source_set = set(merged_sources)
        for loser in losers:
            for src in (loser.get("source_documents") or []):
                if src not in merged_source_set:
                    merged_sources.append(src)
                    merged_source_set.add(src)

        winner["aliases"] = merged_aliases
        winner["source_documents"] = merged_sources

        for i, idx in enumerate(idxs):
            if i != winner_local:
                drop_indices.add(idx)

        actions.append((
            f"{etype}/{winner['canonical_name']!r}",
            [loser["canonical_name"] for loser in losers],
        ))

    if dry_run:
        print(f"DRY RUN — {len(actions)} merges would be applied:")
    else:
        kept = [e for i, e in enumerate(entities) if i not in drop_indices]
        reg["entities"] = kept
        registry_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
        print(f"Applied {len(actions)} merges. Registry: {len(entities)} → {len(kept)} entities.")

    print()
    for survivor, dropped in actions:
        print(f"  {survivor}  ←  {dropped}")
    return len(actions), actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge case-variant entries for category types.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--types", nargs="+", default=list(DEFAULT_TYPES),
                        help="Entity types to apply the merge to. Default: creature concept.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying.")
    args = parser.parse_args()

    merge_case_variants(args.registry, set(args.types), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
