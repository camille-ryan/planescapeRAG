"""Remove `## Source: <doc>` sections that pass-2 produced as windowing false positives.

A false-positive section has both:
  - empty `source_pages[doc_tag]` in frontmatter (LLM correctly returned no pages)
  - no `### Relationships from <doc_tag>` block in the section body

These get written when an over-broad alias (e.g. "City of the Dead") matched a
generic phrase in a doc that doesn't actually discuss the entity. Going forward,
extract_profiles.py drops these at write time; this script handles entity files
written before that guard landed.

Usage:
    python -m planescapeRAG.tools.strip_empty_sections          # dry run
    python -m planescapeRAG.tools.strip_empty_sections --write  # apply
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ENTITIES_DIR = Path("planescapeRAG/graph/entities")
_DOC_HEADER_RE = re.compile(r"^## Source:\s*(.+?)\s*$")
_REL_HEADER_RE = re.compile(r"^### Relationships from\s+(.+?)\s*$")


def _parse_file(path: Path) -> Tuple[List[str], Dict[str, List[int]], List[str], List[Tuple[str, List[str]]], str]:
    """Returns (frontmatter_lines, source_pages, source_documents, sections, title_line).

    sections: list of (doc_tag, [body_lines]) preserving order.
    title_line: the `# <Name>` line that follows the frontmatter.
    """
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing frontmatter")
    # Well-formed files separate frontmatter from body with `\n---\n`. Some
    # legacy files have `\n---<no newline>` immediately followed by the title
    # (early writer didn't emit the blank). Handle both.
    fm_end = text.find("\n---\n", 4)
    if fm_end != -1:
        fm_lines = text[4:fm_end].splitlines()
        body = text[fm_end + 5:]
    else:
        fm_end = text.find("\n---", 4)
        if fm_end == -1:
            raise ValueError(f"{path}: malformed frontmatter")
        fm_lines = text[4:fm_end].splitlines()
        body = text[fm_end + 4:]

    source_pages: Dict[str, List[int]] = {}
    source_documents: List[str] = []
    for line in fm_lines:
        if line.startswith("source_pages: "):
            try:
                parsed = json.loads(line[len("source_pages: "):])
                if isinstance(parsed, dict):
                    source_pages = {k: list(v) for k, v in parsed.items() if isinstance(v, list)}
            except json.JSONDecodeError:
                pass
        elif line.startswith("source_documents: "):
            try:
                parsed = json.loads(line[len("source_documents: "):])
                if isinstance(parsed, list):
                    source_documents = list(parsed)
            except json.JSONDecodeError:
                pass

    body_lines = body.splitlines()
    title_line = ""
    i = 0
    while i < len(body_lines) and not body_lines[i].strip():
        i += 1
    if i < len(body_lines) and body_lines[i].startswith("# "):
        title_line = body_lines[i]
        i += 1

    sections: List[Tuple[str, List[str]]] = []
    current_tag = None
    current: List[str] = []
    for line in body_lines[i:]:
        m = _DOC_HEADER_RE.match(line)
        if m:
            if current_tag is not None:
                sections.append((current_tag, current))
            current_tag = m.group(1).strip()
            current = []
        elif current_tag is not None:
            current.append(line)
    if current_tag is not None:
        sections.append((current_tag, current))

    return fm_lines, source_pages, source_documents, sections, title_line


def _section_has_relationships(body_lines: List[str], doc_tag: str) -> bool:
    """True if the section body contains `### Relationships from <doc_tag>`."""
    for line in body_lines:
        m = _REL_HEADER_RE.match(line)
        if m and m.group(1).strip() == doc_tag:
            return True
    return False


def _rewrite(
    fm_lines: List[str],
    source_pages: Dict[str, List[int]],
    source_documents: List[str],
    sections: List[Tuple[str, List[str]]],
    title_line: str,
    drop_tags: set,
) -> str:
    """Rebuild the file with drop_tags removed from sections, source_pages, and source_documents."""
    new_source_pages = {k: v for k, v in source_pages.items() if k not in drop_tags}
    new_source_documents = [d for d in source_documents if d not in drop_tags]
    new_fm_lines: List[str] = []
    for line in fm_lines:
        if line.startswith("source_pages: "):
            new_fm_lines.append("source_pages: " + json.dumps(new_source_pages))
        elif line.startswith("source_documents: "):
            new_fm_lines.append("source_documents: " + json.dumps(new_source_documents))
        else:
            new_fm_lines.append(line)
    kept_sections = [(tag, body) for tag, body in sections if tag not in drop_tags]
    body_chunks = [title_line] if title_line else []
    for tag, body in kept_sections:
        body_chunks.extend(["", f"## Source: {tag}", ""])
        body_chunks.extend(body)
    body_text = "\n".join(body_chunks).rstrip() + "\n"
    return "---\n" + "\n".join(new_fm_lines) + "\n---\n\n" + body_text


def main() -> None:
    write = "--write" in sys.argv

    files_scanned = 0
    files_changed = 0
    files_skipped = 0
    sections_dropped_by_doc: Dict[str, int] = {}
    examples: List[str] = []

    for path in sorted(_ENTITIES_DIR.rglob("*.md")):
        files_scanned += 1
        try:
            fm_lines, source_pages, source_documents, sections, title_line = _parse_file(path)
        except ValueError:
            files_skipped += 1
            continue

        drop_tags = set()
        for tag, body in sections:
            pages = source_pages.get(tag, [])
            has_rels = _section_has_relationships(body, tag)
            if not pages and not has_rels:
                drop_tags.add(tag)
                sections_dropped_by_doc[tag] = sections_dropped_by_doc.get(tag, 0) + 1
                if len(examples) < 15:
                    examples.append(f"  {path.relative_to(_ENTITIES_DIR.parent.parent)}  →  drop {tag}")

        if drop_tags:
            files_changed += 1
            if write:
                new_text = _rewrite(fm_lines, source_pages, source_documents, sections, title_line, drop_tags)
                path.write_text(new_text)

    mode = "WRITE" if write else "DRY RUN"
    print(f"\n[{mode}] scanned={files_scanned}  skipped(unparseable)={files_skipped}  files_with_empty_sections={files_changed}")
    print(f"\nDropped sections by doc:")
    for doc, n in sorted(sections_dropped_by_doc.items(), key=lambda kv: -kv[1]):
        print(f"  {doc:40s}  {n:>5d}")
    print(f"\nExamples:")
    for ex in examples:
        print(ex)
    if not write:
        print(f"\n(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
