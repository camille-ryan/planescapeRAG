"""Verify each canonical_name in the registry actually appears in the document.

Pass 1 occasionally produces canonical names that are slight misspellings of
the document's actual text (e.g. 'Citadel Caviitus' vs 'Citadel Cavitius').
This tool catches that class of typo by:

1. Trying an exact (case-insensitive) substring match for each canonical_name.
2. On miss, fuzzy regex search against the document with a tight edit budget
   (~1 edit per 10 characters, capped at 3).
3. If the fuzzy match finds a different spelling, suggest the document's form
   as the corrected canonical.

By default prints suggestions for human review. With --auto-apply, rewrites
the registry. With --rewrite-entities <dir>, also renames entity files and
search-replaces wikilinks across the corpus.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import regex as fuzzy_regex


def _fuzzy_find_in_document(name: str, document: str, max_edits: int) -> Optional[str]:
    """If `name` doesn't appear exactly, fuzzy-search and return the document's spelling."""
    pattern = f"({fuzzy_regex.escape(name)}){{e<={max_edits}}}"
    try:
        match = fuzzy_regex.search(pattern, document, fuzzy_regex.IGNORECASE | fuzzy_regex.BESTMATCH)
    except fuzzy_regex.error:
        return None
    return match.group(0) if match else None


def _edit_budget(name: str) -> int:
    return min(3, max(1, len(name) // 10))


def verify_canonical_names(registry: dict, document: str) -> List[Tuple[str, str]]:
    """Return [(registry_form, document_form)] pairs where the registry's
    canonical_name doesn't appear verbatim but a fuzzy match does."""
    document_lower = document.lower()
    suggestions: List[Tuple[str, str]] = []
    for entity in registry["entities"]:
        name = entity["canonical_name"]
        if name.lower() in document_lower:
            continue
        suggested = _fuzzy_find_in_document(name, document, _edit_budget(name))
        if suggested and suggested.lower() != name.lower():
            suggestions.append((name, suggested))
    return suggestions


def apply_to_registry(registry: dict, corrections: List[Tuple[str, str]]) -> int:
    name_map = {old: new for old, new in corrections}
    applied = 0
    for entity in registry["entities"]:
        if entity["canonical_name"] in name_map:
            entity["canonical_name"] = name_map[entity["canonical_name"]]
            applied += 1
    return applied


def _slug(name: str) -> str:
    s = name.lower()
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def rewrite_entities(entities_dir: Path, corrections: List[Tuple[str, str]]) -> Dict[str, int]:
    """For each correction, rename the entity file (slug change) and replace
    [[old]] / [[old|...]] / canonical_name: "old" / # old in all entity files.

    Returns counts: {files_renamed, files_modified, replacements}.
    """
    stats = {"files_renamed": 0, "files_modified": 0, "replacements": 0}
    if not corrections:
        return stats

    for old, new in corrections:
        old_slug = _slug(old)
        new_slug = _slug(new)
        for path in entities_dir.rglob(f"{old_slug}.md"):
            new_path = path.with_name(f"{new_slug}.md")
            path.rename(new_path)
            stats["files_renamed"] += 1

    name_map = {old: new for old, new in corrections}
    wikilink_re = re.compile(r"\[\[([^\]|]+)(\|[^\]]*)?\]\]")
    canonical_line_re = re.compile(r'^(canonical_name:\s*)"([^"]*)"', re.MULTILINE)
    h1_re = re.compile(r"^# (.+)$", re.MULTILINE)

    for path in entities_dir.rglob("*.md"):
        text = path.read_text()
        original = text

        def wikilink_sub(m):
            target = m.group(1).strip()
            display = m.group(2) or ""
            if target in name_map:
                return f"[[{name_map[target]}{display}]]"
            return m.group(0)
        text = wikilink_re.sub(wikilink_sub, text)

        def canonical_sub(m):
            value = m.group(2)
            if value in name_map:
                return f'{m.group(1)}"{name_map[value]}"'
            return m.group(0)
        text = canonical_line_re.sub(canonical_sub, text)

        def h1_sub(m):
            value = m.group(1).strip()
            if value in name_map:
                return f"# {name_map[value]}"
            return m.group(0)
        text = h1_re.sub(h1_sub, text)

        if text != original:
            path.write_text(text)
            stats["files_modified"] += 1
            stats["replacements"] += sum(text.count(new) - original.count(new) for new in {n for _, n in corrections})

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect canonical-name typos in a registry by fuzzy-matching against the source document.")
    parser.add_argument("registry", type=Path)
    parser.add_argument("document", type=Path)
    parser.add_argument("--auto-apply", action="store_true", help="Rewrite the registry with corrections.")
    parser.add_argument("--registry-out", type=Path, default=None, help="Where to write the corrected registry (default: overwrite input).")
    parser.add_argument("--rewrite-entities", type=Path, default=None, help="Path to entities/ directory to rename files and replace wikilinks.")
    args = parser.parse_args()

    document = args.document.read_text()
    with args.registry.open() as f:
        registry = json.load(f)

    suggestions = verify_canonical_names(registry, document)
    if not suggestions:
        print("All canonical names match the document.")
        raise SystemExit(0)

    print(f"Found {len(suggestions)} canonical names with potential typos:")
    for old, new in suggestions:
        print(f"  {old!r} → {new!r}")

    if not args.auto_apply:
        print("\nReview above. Pass --auto-apply to rewrite the registry.")
        if args.rewrite_entities:
            print("(Also pass --auto-apply to rewrite entity files.)")
        raise SystemExit(0)

    applied = apply_to_registry(registry, suggestions)
    out_path = args.registry_out or args.registry
    out_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
    print(f"\nApplied {applied} corrections to {out_path}")

    if args.rewrite_entities:
        stats = rewrite_entities(args.rewrite_entities, suggestions)
        print(f"Entity rewrite: renamed {stats['files_renamed']} files, modified {stats['files_modified']} files.")
