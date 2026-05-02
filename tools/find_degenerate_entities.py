"""Flag registry entries that look like OCR artifacts or formatting noise.

Heuristics ranked from most-confident to least:

  1. Numeric-only canonical name ("8", "30", "100")
  2. Length 1-2 characters
  3. Matches known D&D stat-block tokens (HP, AC, MV, HD, THACO, ...)
  4. Matches class/alignment abbreviations (Pl, Px, Rl, NG, LE, ...)
  5. Many source documents (>=10) but notable=False
     (real minor characters appear in 1-2 docs; cross-doc coverage with low
      notability is a sign of substring matching against a common token)

Output: tools/degenerate_candidates.txt — sorted by combined confidence so the
top of the list is review-then-approve, the tail wants human judgment.

Usage:
    python -m planescapeRAG.tools.find_degenerate_entities
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

_REGISTRY = Path("planescapeRAG/graph/registry.json")
_REPORT = Path("planescapeRAG/tools/degenerate_candidates.txt")

# D&D 2e stat-block tokens — if these slipped into the registry, they're noise.
_STAT_TOKENS = {
    "HP", "HD", "AC", "MV", "AL", "XP", "INT", "WIS", "STR", "DEX", "CON", "CHA",
    "THACO", "AT", "DMG", "SZ", "ML", "F", "C", "M", "T", "D", "P",  # class abbrev
    "LG", "NG", "CG", "LN", "TN", "CN", "LE", "NE", "CE",  # alignment
    "Pl", "Px", "Rl", "Pr",  # planar status abbreviations
    "PC", "PCs", "DM", "NPC",
}


def _flag_entity(e: dict) -> Tuple[List[str], int]:
    """Returns (reason_list, confidence_score). Higher = more degenerate."""
    name = e.get("canonical_name", "")
    flags: List[str] = []
    score = 0

    if name.isdigit():
        flags.append(f"numeric-only ({name!r})")
        score += 100

    if 1 <= len(name) <= 2:
        flags.append(f"length {len(name)} ({name!r})")
        score += 80

    if name in _STAT_TOKENS:
        flags.append(f"stat token ({name!r})")
        score += 90

    # Punctuation noise
    if re.fullmatch(r"[\W_]+", name):
        flags.append("all punctuation")
        score += 100

    # Cross-doc proliferation w/ low notability
    src_docs = e.get("source_documents") or []
    if len(src_docs) >= 10 and not e.get("notable"):
        flags.append(f"in {len(src_docs)} docs, notable=False")
        score += 50

    # Page-density check: many docs but average pages/doc is 1
    src_pages = e.get("source_pages") or {}
    if isinstance(src_pages, dict) and len(src_pages) >= 6:
        avg_pages = sum(len(v) for v in src_pages.values()) / max(len(src_pages), 1)
        if avg_pages <= 1.2:
            flags.append(f"thin coverage: {len(src_pages)} docs avg {avg_pages:.1f} pages/doc")
            score += 30

    return flags, score


def main() -> None:
    reg = json.loads(_REGISTRY.read_text())
    flagged: List[Tuple[int, str, dict, List[str]]] = []
    for e in reg["entities"]:
        reasons, score = _flag_entity(e)
        if reasons:
            flagged.append((score, e.get("canonical_name", ""), e, reasons))

    flagged.sort(key=lambda x: (-x[0], x[1]))

    lines = [
        f"Total entities scanned: {len(reg['entities']):,}",
        f"Flagged candidates:     {len(flagged):,}",
        "",
        "score  canonical                         type           src_docs  notable  reasons",
        "-----  --------------------------------  -------------  --------  -------  ---------",
    ]
    for score, name, e, reasons in flagged:
        type_ = e.get("type", "?")
        nd = len(e.get("source_documents") or [])
        nb = "T" if e.get("notable") else "F"
        reason_str = "; ".join(reasons)
        lines.append(f"{score:>5d}  {name[:32]:<32s}  {type_[:13]:<13s}  {nd:>8d}  {nb:^7s}  {reason_str}")

    _REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {_REPORT}")
    print("\n".join(lines[:30]))
    if len(flagged) > 25:
        print(f"\n... ({len(flagged) - 25} more — see {_REPORT})")


if __name__ == "__main__":
    main()
