"""Scan entity files and bucket relationships into candidate predicates.

Heuristic regex against quote text. Imperfect but gives order-of-magnitude
distribution and surfaces what falls outside the proposed closed set, which is
the input we need before locking the predicate vocabulary.

Writes:
  - tools/predicate_scan_report.txt — counts + sample quotes per bucket
  - tools/predicates.yaml — proposed closed-set definitions for human review
"""

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

_ENTITIES_DIR = Path("planescapeRAG/graph/entities")
_REPORT = Path("planescapeRAG/tools/predicate_scan_report.txt")
_YAML = Path("planescapeRAG/tools/predicates.yaml")

_REL_LINE = re.compile(r'^- \[\[([^\]]+)\]\] — "(.*)" \(p\. \d+\)\s*$')
_FRONTMATTER_NAME = re.compile(r"^canonical_name:\s*(.+?)\s*$", re.MULTILINE)
_FRONTMATTER_TYPE = re.compile(r"^type:\s*(.+?)\s*$", re.MULTILINE)


# Order matters: more specific patterns first. Each quote gets one label.
PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Spatial — most distinctive verbs/prepositions
    ("layer_of", re.compile(
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|"
        r"thirteenth|fourteenth|fifteenth|sixteenth|first|topmost|bottommost|deepest|uppermost)\s+layer\b|"
        r"\blayer of\b|\bplane's \w+ layer\b",
        re.IGNORECASE)),
    ("portal_to", re.compile(
        r"\bportal\s+(to|from|between|leads|opens|connects)|\bgate\s+(to|leads)|\bdoorway\s+to|"
        r"\bportals?\s+\w+\s+(connect|link|lead)",
        re.IGNORECASE)),
    ("located_in", re.compile(
        r"\b(located|situated|lies|lie|rests?|stands?|sits?|found|nestled)\s+(in|on|at|within|atop|near)\b|"
        r"\b(in|on|atop|within)\s+the\s+(city|ward|town|village|plane|layer|realm|street|district|domain)\s+of\b|"
        r"\b(in|within)\s+(Sigil|Hive|Lower Ward|Lady's Ward|Clerk's Ward|Market Ward|Guildhall Ward)\b|"
        r"\bward\s+of\s+\[\[",
        re.IGNORECASE)),

    # Organizational
    ("leader_of", re.compile(
        r"\bfactol\s+of\b|\bfactol\s+\[\[|\bfactol\b.*?\[\[|"
        r"\b(leads|heads|rules|commands|presides over)\s+the\b|"
        r"\b(leader|head|chief|commander)\s+of\s+the\b",
        re.IGNORECASE)),
    ("founded", re.compile(
        r"\b(founded|established|created|formed)\s+(the|by)\b|\bfounder\s+of\b",
        re.IGNORECASE)),
    ("member_of", re.compile(
        r"\b(member|members)\s+of\s+the\b|\bbelongs?\s+to\s+the\b|\bjoined\s+the\b|"
        r"/(Harmonium|Mercykillers|Sensates|Bleak Cabal|Doomguard|Sign of One|Athar|"
        r"Believers of the Source|Dustmen|Fated|Free League|Fraternity of Order|Revolutionary League|"
        r"Society of Sensation|Transcendent Order|Xaositects|Anarchists|Guvners|Hardheads|Bleakers|"
        r"Indeps|Cipher|Godsmen|Sinkers|Takers|Sodkillers)/|"
        r"\b(joins?|belonging\s+to|recruited\s+by|recruited\s+into)\b",
        re.IGNORECASE)),

    # Common open predicates (NOT in closed set — captured to see if they're frequent enough to promote)
    ("worships", re.compile(r"\b(worships?|venerates?|reveres?|prays?\s+to|devoted\s+to|follower\s+of)\b", re.IGNORECASE)),
    ("allied_with", re.compile(r"\b(allied|alliance|allies)\s+(with|of)\b|\bloosely\s+allied\b", re.IGNORECASE)),
    ("opposes", re.compile(r"\b(opposes?|enemy|enemies|fights?\s+against|hostile\s+to|war\s+with|at\s+odds\s+with)\b", re.IGNORECASE)),
    ("created_by", re.compile(r"\b(created|made|forged|built|crafted)\s+by\b", re.IGNORECASE)),
    ("killed", re.compile(r"\b(killed|slew|slain|murdered|destroyed)\b", re.IGNORECASE)),
    ("ruled_by", re.compile(r"\b(ruled|governed|controlled)\s+by\b", re.IGNORECASE)),
]


def classify(quote: str) -> str:
    for label, pat in PATTERNS:
        if pat.search(quote):
            return label
    return "open"


def main() -> None:
    if not _ENTITIES_DIR.exists():
        raise SystemExit(f"missing: {_ENTITIES_DIR}")

    bucket_counts: Counter = Counter()
    bucket_samples: defaultdict = defaultdict(list)
    type_pair_counts: defaultdict = defaultdict(Counter)  # bucket -> Counter[(src_type, tgt_type_unknown)]
    total = 0

    for path in _ENTITIES_DIR.rglob("*.md"):
        text = path.read_text()
        m_name = _FRONTMATTER_NAME.search(text)
        if not m_name:
            continue
        src_name = m_name.group(1)
        m_type = _FRONTMATTER_TYPE.search(text)
        src_type = m_type.group(1) if m_type else "?"

        for line in text.splitlines():
            m = _REL_LINE.match(line)
            if not m:
                continue
            target, quote = m.group(1), m.group(2)
            bucket = classify(quote)
            bucket_counts[bucket] += 1
            type_pair_counts[bucket][src_type] += 1
            if len(bucket_samples[bucket]) < 5:
                bucket_samples[bucket].append(f"  [{src_type}] {src_name} → {target}\n    \"{quote[:200]}\"")
            total += 1

    # Report
    lines = []
    lines.append(f"Total relationships scanned: {total:,}\n")
    lines.append("Distribution by candidate predicate:")
    lines.append(f"  {'predicate':<15s}  {'count':>7s}  {'pct':>6s}  top source types")
    lines.append(f"  {'-' * 15}  {'-' * 7}  {'-' * 6}  ---")
    for bucket, n in bucket_counts.most_common():
        pct = 100 * n / total if total else 0
        top_types = ", ".join(f"{t}={c}" for t, c in type_pair_counts[bucket].most_common(4))
        lines.append(f"  {bucket:<15s}  {n:>7,d}  {pct:>5.1f}%  {top_types}")

    lines.append("\n--- Samples per bucket ---")
    for bucket, _ in bucket_counts.most_common():
        lines.append(f"\n[{bucket}]")
        for s in bucket_samples[bucket]:
            lines.append(s)

    _REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {_REPORT}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
