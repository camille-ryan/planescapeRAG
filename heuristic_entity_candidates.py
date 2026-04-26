"""Heuristic proper-noun candidate extractor.

Cheap NER-lite for finding entity-name candidates in a document, without
running an LLM. Matches proper-noun spans like 'Lady of Pain', 'Quasielemental
Plane of Radiance', 'Citadel Cavitius', and single-word names like 'Sigil' or
'Doomguard'.

Heuristic:
- A name is `CapitalWord (sep (CapitalWord | connective))* CapitalWord` where
  connectives are lowercase glue words like 'of', 'the', 'in', etc.
- Single-word matches must appear at least once in non-sentence-initial
  position (otherwise they're probably just sentence-boundary capitalization).
- Trailing connectives are trimmed; very common stopwords are skipped.

Compares the extracted candidates against an existing registry (canonical
names + aliases) to surface what's mentioned but not registered. The output
is meant as a feeder for further LLM verification or human review, not as a
final answer — it will have false positives (proper-looking strings that
aren't really entities) and may miss lowercase entities like 'tanar'ri' that
flout English capitalization conventions.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


_CONNECTIVES = {
    "of", "the", "in", "on", "for", "at", "by", "to", "and", "&",
    "de", "du", "der", "von", "van", "le", "la", "el",
}

# A "name part": capitalized word, possibly with apostrophe (O'Brien) or hyphen-internal.
# Single character "I" is excluded (English pronoun, not a proper noun).
_NAME_PART = r"[A-Z][a-zA-Z'’]+"
_CONNECTIVE_RE = "|".join(re.escape(c) for c in _CONNECTIVES)
# Allow up to 5 separator-delimited tokens; covers names like
# "Erin Darkflame Montgomery" (3) and "Quasielemental Plane of Radiance" (4).
_NAME_RE = re.compile(
    rf"\b{_NAME_PART}(?:[\s\-]+(?:{_NAME_PART}|(?:{_CONNECTIVE_RE})\b)){{0,5}}"
)

# Single-word words that are capitalized but almost never proper nouns in English.
# Filter these out post-hoc to reduce noise.
_SINGLE_WORD_BLOCKLIST = {
    # Pronouns & contractions
    "I", "I'd", "I'll", "I'm", "I've",
    # Sentence-initial pronouns/determiners frequently capitalized
    "A", "An", "The", "This", "That", "These", "Those", "It", "He", "She", "They", "We", "You",
    "It's", "It’s", "He's", "He’s", "She's", "She’s", "They're", "They’re", "We're", "We’re",
    "There", "There's", "There’s", "Here", "Here's", "Here’s",
    # Common sentence starters
    "But", "However", "And", "Or", "So", "Yet", "Yes", "No", "Now", "Then", "Thus", "Hence",
    "If", "When", "While", "Although", "Though", "Because", "Since", "As", "Whether",
    "All", "Some", "Any", "Each", "Every", "Most", "Many", "Few", "Several",
    "What", "Where", "Who", "Whose", "Which", "Why", "How",
    "More", "Less", "Other", "Another", "Such",
    "Even", "Just", "Only", "Also", "Still", "Already", "Sometimes", "Often", "Always", "Never",
    "Their", "Its", "His", "Her", "Our", "Your", "My",
    "See", "Note", "Remember", "Imagine", "Think", "Consider",
    # Honorifics that aren't names by themselves
    "Mr", "Mrs", "Ms", "Dr", "Sir",
    # Game/book abbreviations
    "PC", "PCs", "DM", "DMs", "NPC", "NPCs", "AC", "HD", "HP", "XP", "GP", "SP", "CP",
    "AD", "TSR", "Inc", "OCR",
    # Generic capitalized terms in our domain that aren't entities
    "Course", "Of", "Layer", "Page", "Pl",
}


_POSSESSIVE_RE = re.compile(r"['’]s$")


def _strip_possessive(name: str) -> str:
    return _POSSESSIVE_RE.sub("", name)


def _trim_trailing_connectives(text: str) -> str:
    parts = text.split()
    while parts and parts[-1].lower() in _CONNECTIVES:
        parts.pop()
    while parts and parts[0].lower() in _CONNECTIVES:
        parts.pop(0)
    return " ".join(parts)


def _is_sentence_initial(document: str, start: int) -> bool:
    """True if the character at `start` is the first non-whitespace after a
    sentence boundary (or the start of the doc)."""
    i = start - 1
    while i >= 0 and document[i] in " \t":
        i -= 1
    if i < 0:
        return True
    if document[i] == "\n":
        # Probably start of a paragraph; treat as sentence-initial.
        return True
    return document[i] in ".!?"


def extract_candidates(document: str) -> Dict[str, List[int]]:
    """Return {candidate_text: [positions]} for all proper-noun-like spans.
    Possessive forms are folded onto their stem ('Sigil's' counted as 'Sigil').
    """
    raw_hits: Dict[str, List[int]] = {}
    for m in _NAME_RE.finditer(document):
        text = _trim_trailing_connectives(m.group(0)).strip()
        text = _strip_possessive(text)
        if not text:
            continue
        raw_hits.setdefault(text, []).append(m.start())
    return raw_hits


def filter_real_candidates(
    candidates: Dict[str, List[int]],
    document: str,
    min_midsentence_ratio: float = 0.5,
) -> Dict[str, int]:
    """Drop candidates that are likely sentence-boundary artifacts or stopwords.

    Single-word candidates must (a) not be in the blocklist, and (b) have at
    least min_midsentence_ratio of their occurrences appearing mid-sentence.
    Multi-word candidates are kept regardless (rarely arise as accidents).

    Returns {candidate_text: occurrence_count}.
    """
    kept: Dict[str, int] = {}
    for name, positions in candidates.items():
        is_multiword = len(name.split()) > 1
        if not is_multiword:
            if name in _SINGLE_WORD_BLOCKLIST:
                continue
            mid_count = sum(1 for p in positions if not _is_sentence_initial(document, p))
            ratio = mid_count / len(positions)
            if ratio < min_midsentence_ratio:
                continue
        kept[name] = len(positions)
    return kept


def _normalize_for_registry_match(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"['’]", "", s)
    if s.startswith("the "):
        s = s[4:]
    return s


def filter_against_registry(candidates: Dict[str, int], registry: dict) -> Dict[str, int]:
    """Drop candidates whose normalized form matches a canonical_name or alias in the registry."""
    known = set()
    for entity in registry["entities"]:
        known.add(_normalize_for_registry_match(entity["canonical_name"]))
        for alias in entity.get("aliases") or []:
            known.add(_normalize_for_registry_match(alias))
    return {name: count for name, count in candidates.items() if _normalize_for_registry_match(name) not in known}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract proper-noun candidates from a document and surface ones missing from a registry.")
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Where to write the unmatched-candidates list (JSON). Default: print to stdout.")
    parser.add_argument("--min-count", type=int, default=2, help="Only keep candidates that appear at least this many times.")
    parser.add_argument("--top", type=int, default=50, help="How many to display sorted by count.")
    args = parser.parse_args()

    document = args.document.read_text()
    with args.registry.open() as f:
        registry = json.load(f)

    print(f"Document: {len(document)} chars / ~{len(document)//4} tokens")
    print(f"Registry: {len(registry['entities'])} entities")

    raw = extract_candidates(document)
    print(f"\nRaw candidates from regex: {len(raw)}")

    real = filter_real_candidates(raw, document)
    print(f"After sentence-initial / blocklist filter: {len(real)}")

    unmatched = filter_against_registry(real, registry)
    print(f"After registry filter (canonical + aliases): {len(unmatched)}")

    above_threshold = {n: c for n, c in unmatched.items() if c >= args.min_count}
    print(f"With count ≥ {args.min_count}: {len(above_threshold)}\n")

    sorted_unmatched = sorted(above_threshold.items(), key=lambda x: -x[1])
    print(f"Top {min(args.top, len(sorted_unmatched))} unmatched candidates:")
    for name, count in sorted_unmatched[: args.top]:
        print(f"  {count:>4}x  {name}")

    if args.output:
        args.output.write_text(json.dumps(
            [{"name": n, "count": c} for n, c in sorted_unmatched],
            indent=2, ensure_ascii=False,
        ))
        print(f"\nWrote {args.output}")
