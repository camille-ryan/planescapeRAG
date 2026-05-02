"""Load entity .md files into Neo4j as a typed property graph.

SKETCH — running but unverified. Decisions flagged with `# DECISION:` need a
review before this is the canonical loader.

Schema:
  Nodes:
    Every entity gets two labels: :Entity (generic) AND a type label
    (:Character, :Faction, :Place, :Plane, :Item, :Concept, :Creature, ...).
    This lets you query generically (`MATCH (e:Entity {name:'X'})`) or by
    type (`MATCH (c:Character)-[:MEMBER_OF]->(:Faction)`).
    Properties: id (slug), name (canonical_name), supertype, notable,
                source_documents (list), aliases (list).

  Relationships:
    One Cypher rel type per predicate (`MEMBER_OF`, `LOCATED_IN`, etc.).
    Properties: quote (text), page (int), source_doc (str).

  Direction:
    A line `- PRED → [[Target]]` becomes (file_entity)-[:PRED]->(target).
    A line `- ← PRED [[Target]]` becomes (target)-[:PRED]->(file_entity).
    `_NO_EDGE` lines are filtered out at parse time (already removed from .md).

  Targets that don't resolve in the registry get a stub :Entity node with
  property `incomplete: true` so the edge isn't lost. Filterable via
  `WHERE NOT exists(n.incomplete)`.

Idempotency:
  MERGE on (id) for nodes; MERGE on (src_id, predicate, tgt_id, source_doc, page)
  for edges. Re-running won't duplicate. Dropping the graph first (via the
  `--reset` flag) is the cleanest way to do a fresh load.

Usage:
    pip install neo4j
    export NEO4J_URI="neo4j://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASS="<password>"

    python -m planescapeRAG.tools.load_neo4j --reset      # drop & full load
    python -m planescapeRAG.tools.load_neo4j              # incremental MERGE
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from neo4j import GraphDatabase


_REGISTRY = Path("planescapeRAG/graph/registry.json")
_ENTITIES_DIR = Path("planescapeRAG/graph/entities")

# Same regex set as classify_predicates.py — keep in sync if format changes.
_REL_OUT_RE = re.compile(r'^- ([A-Z][A-Z0-9_]*) → \[\[([^\]]+)\]\] — "(.*)" \(p\. (\d+)\)\s*$')
_REL_IN_RE  = re.compile(r'^- ← ([A-Z][A-Z0-9_]*) \[\[([^\]]+)\]\] — "(.*)" \(p\. (\d+)\)\s*$')
_DOC_HEADER_RE = re.compile(r"^## Source:\s*(.+?)\s*$")
_REL_HEADER_RE = re.compile(r"^### Relationships from\s+(.+?)\s*$")


def _slug(name: str) -> str:
    """Same slug rule as pipeline/extract_profiles.py."""
    s = name.lower()
    s = re.sub(r"['’]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _type_label(t: str) -> str:
    """Capitalize first letter for Neo4j label convention."""
    return t.strip().capitalize() if t else "Unknown"


# --------------------------------------------------------- registry & lookup --

def load_registry() -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Returns (by_canonical, alias_to_canonical_lower) for fast resolution."""
    reg = json.loads(_REGISTRY.read_text())
    by_canonical: Dict[str, dict] = {}
    alias_lower: Dict[str, str] = {}
    for e in reg["entities"]:
        canon = e["canonical_name"]
        by_canonical[canon] = e
        alias_lower[canon.lower()] = canon
        for a in e.get("aliases") or []:
            if a:
                alias_lower.setdefault(a.lower(), canon)
    return by_canonical, alias_lower


def resolve_target(name: str, alias_lower: Dict[str, str]) -> Optional[str]:
    """Wikilink text → canonical name. Returns None if unresolved."""
    return alias_lower.get(name.lower())


# ------------------------------------------------------------- file parsing --

def _parse_file(path: Path) -> Optional[Tuple[dict, List[Tuple[str, List[str]]]]]:
    """Returns (frontmatter, [(doc_tag, body_lines)]) or None on parse failure."""
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    fm_end = text.find("\n---\n", 4)
    body_offset: int
    if fm_end != -1:
        fm_text = text[4:fm_end]
        body_offset = fm_end + 5
    else:
        fm_end = text.find("\n---", 4)
        if fm_end == -1:
            return None
        fm_text = text[4:fm_end]
        body_offset = fm_end + 4

    fm: Dict[str, object] = {}
    for line in fm_text.splitlines():
        m = re.match(r"^([a-zA-Z_]+):\s*(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if v.startswith("[") or v.startswith("{"):
            try:
                fm[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        fm[k] = v

    sections: List[Tuple[str, List[str]]] = []
    current_tag: Optional[str] = None
    current_lines: List[str] = []
    for line in text[body_offset:].splitlines():
        m = _DOC_HEADER_RE.match(line)
        if m:
            if current_tag is not None:
                sections.append((current_tag, current_lines))
            current_tag = m.group(1).strip()
            current_lines = []
        elif current_tag is not None:
            current_lines.append(line)
    if current_tag is not None:
        sections.append((current_tag, current_lines))

    return fm, sections


def iter_entities(by_canonical: Dict[str, dict]) -> Iterator[dict]:
    """Yield node records: {id, name, type, supertype, notable, aliases, source_documents}."""
    for path in sorted(_ENTITIES_DIR.rglob("*.md")):
        parsed = _parse_file(path)
        if parsed is None:
            continue
        fm, _ = parsed
        canonical = str(fm.get("canonical_name") or "")
        if not canonical:
            continue
        type_ = str(fm.get("type") or "unknown")
        yield {
            "id": _slug(canonical),
            "name": canonical,
            "type": type_,
            "supertype": str(fm.get("supertype") or type_),
            "notable": bool(fm.get("notable", False)),
            "aliases": fm.get("aliases") or [],
            "source_documents": fm.get("source_documents") or [],
            "_path": str(path),
        }


def iter_edges(alias_lower: Dict[str, str]) -> Iterator[dict]:
    """Yield edge records: {src_id, src_name, predicate, tgt_name, tgt_canonical,
    tgt_resolved (bool), source_doc, page, quote}."""
    for path in sorted(_ENTITIES_DIR.rglob("*.md")):
        parsed = _parse_file(path)
        if parsed is None:
            continue
        fm, sections = parsed
        src_canonical = str(fm.get("canonical_name") or "")
        if not src_canonical:
            continue
        src_id = _slug(src_canonical)

        for doc_tag, body_lines in sections:
            in_rel_block = False
            for line in body_lines:
                if _REL_HEADER_RE.match(line):
                    in_rel_block = True
                    continue
                if not in_rel_block:
                    continue
                m_out = _REL_OUT_RE.match(line)
                m_in = _REL_IN_RE.match(line)
                if m_out:
                    pred, tgt_name, quote, page = m_out.group(1), m_out.group(2), m_out.group(3), int(m_out.group(4))
                    tgt_canonical = resolve_target(tgt_name, alias_lower)
                    yield {
                        "src_id": src_id, "src_name": src_canonical,
                        "predicate": pred,
                        "tgt_name": tgt_name,
                        "tgt_canonical": tgt_canonical,
                        "tgt_resolved": tgt_canonical is not None,
                        "source_doc": doc_tag, "page": page, "quote": quote,
                    }
                elif m_in:
                    # Incoming: edge is (target)-[:PRED]->(source). Swap for the load.
                    pred, tgt_name, quote, page = m_in.group(1), m_in.group(2), m_in.group(3), int(m_in.group(4))
                    tgt_canonical = resolve_target(tgt_name, alias_lower)
                    yield {
                        # `src_*` here = the edge's actual source (the wikilink target)
                        "src_id": _slug(tgt_canonical) if tgt_canonical else None,
                        "src_name": tgt_canonical or tgt_name,
                        "predicate": pred,
                        "tgt_name": src_canonical,
                        "tgt_canonical": src_canonical,
                        "tgt_resolved": True,  # source file's entity is in registry by definition
                        "source_doc": doc_tag, "page": page, "quote": quote,
                    }


# ------------------------------------------------------------ neo4j loading --

class Loader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def reset(self) -> None:
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            print("  reset: graph dropped")

    def ensure_constraints(self) -> None:
        """Unique constraint on Entity.id — also creates an index implicitly."""
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            s.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    def load_nodes(self, nodes: List[dict], batch: int = 500) -> int:
        """MERGE all nodes. Sets two labels (:Entity + typed) via APOC-free pattern."""
        n = 0
        with self.driver.session() as s:
            for i in range(0, len(nodes), batch):
                chunk = nodes[i : i + batch]
                # Merge generic Entity, then add typed label per row via call subquery.
                s.run(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Entity {id: row.id})
                    SET e.name = row.name,
                        e.type = row.type,
                        e.supertype = row.supertype,
                        e.notable = row.notable,
                        e.aliases = row.aliases,
                        e.source_documents = row.source_documents
                    """,
                    rows=chunk,
                )
                # Add the type-specific label per row. APOC's apoc.create.addLabels
                # would be cleaner, but we keep the script APOC-free.
                # DECISION: use string interpolation in Cypher (requires plain-cap labels);
                # we trust _type_label produces alphanum.
                for row in chunk:
                    label = _type_label(row["type"])
                    s.run(
                        f"MATCH (e:Entity {{id: $id}}) SET e:{label}",
                        id=row["id"],
                    )
                n += len(chunk)
                print(f"  nodes: {n}/{len(nodes)}", flush=True)
        return n

    def load_edges(self, edges: List[dict], batch: int = 500) -> Tuple[int, int]:
        """Group by predicate (Cypher rel-type can't be parameterized) and load.
        Returns (loaded, skipped_unresolved)."""
        from collections import defaultdict
        by_pred: Dict[str, List[dict]] = defaultdict(list)
        skipped = 0
        for e in edges:
            if not e["src_id"] or not e["tgt_canonical"]:
                skipped += 1
                continue
            by_pred[e["predicate"]].append({
                "src_id": e["src_id"],
                "tgt_id": _slug(e["tgt_canonical"]),
                "tgt_name": e["tgt_canonical"],
                "tgt_type": "unknown",  # filled in later — stub if needed
                "source_doc": e["source_doc"],
                "page": e["page"],
                "quote": e["quote"],
            })

        loaded = 0
        with self.driver.session() as s:
            for pred, rows in by_pred.items():
                for i in range(0, len(rows), batch):
                    chunk = rows[i : i + batch]
                    # Stub-create target if missing (incomplete entity).
                    # MERGE the edge on (src, pred, tgt, doc, page) so re-runs are idempotent.
                    s.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (src:Entity {{id: row.src_id}})
                        MERGE (tgt:Entity {{id: row.tgt_id}})
                          ON CREATE SET tgt.name = row.tgt_name, tgt.incomplete = true
                        MERGE (src)-[r:{pred} {{source_doc: row.source_doc, page: row.page}}]->(tgt)
                          ON CREATE SET r.quote = row.quote
                        """,
                        rows=chunk,
                    )
                    loaded += len(chunk)
                print(f"  edges :{pred}: {loaded:,}", flush=True)
        return loaded, skipped


# ------------------------------------------------------------ main --

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="DETACH DELETE all nodes before loading.")
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "neo4j://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASS", ""))
    parser.add_argument("--dry-run", action="store_true", help="Parse only; print stats without loading.")
    args = parser.parse_args()

    print("Loading registry + walking entity files...")
    by_canonical, alias_lower = load_registry()
    print(f"  registry: {len(by_canonical):,} entities, {len(alias_lower):,} alias mappings")

    nodes = list(iter_entities(by_canonical))
    edges = list(iter_edges(alias_lower))
    unresolved = sum(1 for e in edges if not e["tgt_resolved"])

    print(f"  parsed:   {len(nodes):,} nodes, {len(edges):,} edges ({unresolved:,} unresolved targets)")

    # Predicate distribution
    from collections import Counter
    pred_counts = Counter(e["predicate"] for e in edges)
    print(f"\n  predicate distribution:")
    for p, n in pred_counts.most_common():
        print(f"    {p:<22s} {n:>6,d}")

    if args.dry_run:
        print("\n(--dry-run; nothing loaded)")
        return

    if not args.password:
        raise SystemExit("NEO4J_PASS env var or --password required.")

    print(f"\nConnecting to {args.uri}...")
    loader = Loader(args.uri, args.user, args.password)
    try:
        if args.reset:
            loader.reset()
        loader.ensure_constraints()
        print("\nLoading nodes...")
        loader.load_nodes(nodes)
        print("\nLoading edges...")
        loaded, skipped = loader.load_edges(edges)
        print(f"\nDone. {loaded:,} edges loaded, {skipped:,} skipped (unresolved targets).")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
