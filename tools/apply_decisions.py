"""Apply manual cleanup decisions to the global registry.

Reads a YAML file describing entity merges, type changes, and "no-op keeps"
(audit documentation only). Applies type_changes first, then merges. Always
writes a registry backup before mutating. Idempotent — running twice is safe
(the second run finds nothing to do).

Usage:
    python -m planescapeRAG.tools.apply_decisions
    python -m planescapeRAG.tools.apply_decisions --decisions tools/cleanup_decisions.yaml
    python -m planescapeRAG.tools.apply_decisions --dry-run

YAML schema:

    type_changes:                                       # optional
      - canonical: "spell key"
        type: concept                                   # new type

    merges:                                             # optional
      - winner: "The Beastlands"
        losers: [Beastlands]
      - winner: "Sigil"
        losers: ["the city"]                            # multiple losers OK

    keeps:                                              # optional, no-op
      - [The Ghetto, Ghetto]                            # documented as reviewed
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


DEFAULT_REGISTRY = Path("planescapeRAG/graph/registry.json")
DEFAULT_DECISIONS = Path("planescapeRAG/tools/cleanup_decisions.yaml")


def _find_entity(entities: List[dict], canonical: str) -> int:
    """Return index of the entity with this canonical_name, or -1.
    Tries exact match first, then case-insensitive."""
    for i, e in enumerate(entities):
        if e["canonical_name"] == canonical:
            return i
    target = canonical.lower()
    for i, e in enumerate(entities):
        if e["canonical_name"].lower() == target:
            return i
    return -1


def apply(
    registry_path: Path,
    decisions_path: Path,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    decisions = yaml.safe_load(decisions_path.read_text()) or {}

    reg = json.loads(registry_path.read_text())
    entities: List[dict] = reg["entities"]

    if not dry_run:
        backup = registry_path.with_suffix(registry_path.suffix + ".bak.before-decisions")
        shutil.copy2(registry_path, backup)
        print(f"backup: {backup}")

    # 1. Type changes first (so cross-type merges can succeed afterward).
    type_changes = decisions.get("type_changes") or []
    n_type_changes = 0
    for tc in type_changes:
        canonical = tc.get("canonical")
        new_type = tc.get("type")
        if not canonical or not new_type:
            print(f"  TYPE: SKIP (malformed): {tc!r}")
            continue
        idx = _find_entity(entities, canonical)
        if idx < 0:
            print(f"  TYPE: SKIP (not found): {canonical!r}")
            continue
        old_type = entities[idx].get("type")
        if old_type == new_type:
            print(f"  TYPE: SKIP (already {new_type!r}): {canonical!r}")
            continue
        entities[idx]["type"] = new_type
        print(f"  TYPE: {canonical!r}  {old_type!r} → {new_type!r}")
        n_type_changes += 1

    # 2. Merges.
    drop_indices: set = set()
    n_merges = 0
    for m in decisions.get("merges") or []:
        winner_name: Optional[str] = m.get("winner")
        loser_names: List[str] = m.get("losers") or []
        if not winner_name:
            print(f"  MERGE: SKIP (no winner): {m!r}")
            continue

        winner_idx = _find_entity(entities, winner_name)
        if winner_idx < 0:
            print(f"  MERGE: SKIP winner not found: {winner_name!r}")
            continue
        if winner_idx in drop_indices:
            print(f"  MERGE: SKIP (winner already dropped by earlier merge): {winner_name!r}")
            continue
        winner = entities[winner_idx]

        losers: List[Tuple[int, dict]] = []
        for ln in loser_names:
            idx = _find_entity(entities, ln)
            if idx < 0:
                print(f"  MERGE: SKIP loser not found: {ln!r}  (winner: {winner_name!r})")
                continue
            if idx == winner_idx or idx in drop_indices:
                continue
            losers.append((idx, entities[idx]))
        if not losers:
            print(f"  MERGE: no losers to apply for: {winner_name!r}")
            continue

        # Aliases: union of (winner's existing) + (each loser's canonical) + (each loser's aliases),
        # excluding the winner's canonical itself.
        merged_aliases = list(winner.get("aliases") or [])
        seen = {a.lower() for a in merged_aliases}
        seen.add(winner["canonical_name"].lower())
        for _, loser in losers:
            for term in [loser["canonical_name"]] + (loser.get("aliases") or []):
                if term and term.lower() not in seen:
                    merged_aliases.append(term)
                    seen.add(term.lower())
        winner["aliases"] = merged_aliases

        # source_documents: union, preserving winner's order.
        merged_sources = list(winner.get("source_documents") or [])
        seen_src = set(merged_sources)
        for _, loser in losers:
            for src in (loser.get("source_documents") or []):
                if src not in seen_src:
                    merged_sources.append(src)
                    seen_src.add(src)
        winner["source_documents"] = merged_sources

        for idx, _ in losers:
            drop_indices.add(idx)

        loser_display = [loser["canonical_name"] for _, loser in losers]
        print(f"  MERGE: {winner_name!r}  ←  {loser_display}")
        n_merges += 1

    # 3. Keeps — log only, no mutation.
    keeps = decisions.get("keeps") or []
    for k in keeps:
        if isinstance(k, list) and len(k) >= 2:
            print(f"  KEEP: {k}")

    # 4. Save.
    if not dry_run:
        kept_entities = [e for i, e in enumerate(entities) if i not in drop_indices]
        reg["entities"] = kept_entities
        registry_path.write_text(json.dumps(reg, indent=2, ensure_ascii=False))
        final_count = len(kept_entities)
    else:
        final_count = len(entities) - len(drop_indices)

    print()
    print("Summary:")
    print(f"  type_changes applied: {n_type_changes}")
    print(f"  merges applied:       {n_merges}")
    print(f"  keeps documented:     {len(keeps)}")
    print(f"  registry: {len(entities)} → {final_count} entities" + (" (dry-run)" if dry_run else ""))

    return n_type_changes, n_merges, len(keeps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual cleanup decisions to the registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply(args.registry, args.decisions, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
