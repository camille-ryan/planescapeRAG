"""Backfill entity .md frontmatter from the current registry state.

Pass-2 writes each entity file ONCE — during the first ingested doc that
profiles that entity — and then sets skip_existing=True for subsequent docs.
So if aliases / supertypes / source_documents grew in the registry after
that first write (via later docs' merge / sweep / heuristic / canonicalize
stages), the frontmatter is stale.

This tool rewrites every entity .md file's frontmatter from the current
registry, preserving the body (description + per-doc sections + relationship
bullets) and per-doc source_pages from the existing frontmatter.

No LLM calls, no embeddings. Free, fast, idempotent.

Usage:
    python -m planescapeRAG.tools.refresh_entity_frontmatter
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from planescapeRAG.chatbot.embed_entities import _parse_entity_file
from planescapeRAG.pipeline.extract_profiles import _build_frontmatter, _slug


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY = _PROJECT_ROOT / "planescapeRAG" / "graph" / "registry.json"
DEFAULT_SUPERTYPE_MAP = _PROJECT_ROOT / "planescapeRAG" / "graph" / "type_supertype_mapping.json"
DEFAULT_ENTITIES_DIR = _PROJECT_ROOT / "planescapeRAG" / "graph" / "entities"


def _split_frontmatter_and_body(text: str) -> Tuple[str, str]:
    """Return (frontmatter_block_including_dashes, body). Body retains its
    original whitespace boundary. Returns ('', text) if the file has no
    frontmatter — caller should treat that as a skip."""
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end == -1:
        return "", text
    return text[: end + 5], text[end + 5:]


def refresh(
    registry_path: Path,
    supertype_map_path: Path,
    entities_dir: Path,
) -> Dict[str, int]:
    registry = json.loads(registry_path.read_text())
    supertype_map = json.loads(supertype_map_path.read_text())
    by_slug = {_slug(e["canonical_name"]): e for e in registry["entities"]}

    counts = {"updated": 0, "unchanged": 0, "no_registry_entry": 0, "no_supertype": 0, "no_frontmatter": 0}
    no_match: List[str] = []

    for md_path in sorted(entities_dir.rglob("*.md")):
        rel = md_path.relative_to(entities_dir)
        slug = md_path.stem
        original = md_path.read_text()
        fm_block, body = _split_frontmatter_and_body(original)
        if not fm_block:
            counts["no_frontmatter"] += 1
            continue

        entity = by_slug.get(slug)
        if entity is None:
            counts["no_registry_entry"] += 1
            no_match.append(str(rel))
            continue

        supertype = supertype_map.get(entity["type"])
        if supertype is None:
            counts["no_supertype"] += 1
            continue

        # Preserve per-doc source_pages from the existing frontmatter — that
        # data is computed at pass-2 collect time, not in the registry.
        existing_fm, _ = _parse_entity_file(md_path)
        existing_source_pages = existing_fm.get("source_pages")
        if not isinstance(existing_source_pages, dict):
            existing_source_pages = {}

        new_frontmatter = _build_frontmatter(entity, supertype, existing_source_pages)
        new_text = new_frontmatter + body

        if new_text == original:
            counts["unchanged"] += 1
        else:
            md_path.write_text(new_text)
            counts["updated"] += 1

    print(f"Refreshed entity frontmatter under {entities_dir}")
    print(f"  updated:           {counts['updated']}")
    print(f"  unchanged:         {counts['unchanged']}")
    print(f"  no registry entry: {counts['no_registry_entry']}")
    print(f"  no supertype map:  {counts['no_supertype']}")
    print(f"  no frontmatter:    {counts['no_frontmatter']}")
    if no_match:
        print("\nSlugs not found in registry (first 10):")
        for s in no_match[:10]:
            print(f"  {s}")
        if len(no_match) > 10:
            print(f"  ... and {len(no_match) - 10} more")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--supertype-map", type=Path, default=DEFAULT_SUPERTYPE_MAP)
    parser.add_argument("--entities-dir", type=Path, default=DEFAULT_ENTITIES_DIR)
    args = parser.parse_args()
    refresh(args.registry, args.supertype_map, args.entities_dir)


if __name__ == "__main__":
    main()
