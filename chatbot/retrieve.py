"""Query → top-k similar entities → 1-hop wikilink expansion.

Loads vectors.npy + metadata.json built by embed_entities.py and serves retrieval
queries. The FAISS index (IndexFlatIP over L2-normalized vectors → cosine) is
rebuilt in memory at construction; for ~1k vectors that's instant.

The "1-hop expansion" is what makes this graphRAG vs flat RAG: for each top-k hit,
we follow [[Wikilinks]] from its body to pull related entities into the context
bundle. This recovers entities the query embedding misses but the graph knows
about (e.g. asking about a ruler retrieves the place via wikilink).
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

from ..config import Settings
from .embed_entities import EMBEDDING_MODEL, _MAIN_SECTION


# Match [[Target]] and [[Target|Display]] forms.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


@dataclass
class RetrievedEntity:
    entity_id: str
    canonical_name: str
    type: str
    similarity: float
    body: str
    is_primary: bool
    expanded_from: Optional[str] = None
    # Set on primaries to the `## Source: <section_id>` whose vector matched
    # the query. None on expansions (which come from wikilink graph traversal,
    # not vector search) and on entities with no per-source structure.
    section_id: Optional[str] = None


@dataclass
class RetrievalResult:
    query: str
    primaries: List[RetrievedEntity] = field(default_factory=list)
    expansions: List[RetrievedEntity] = field(default_factory=list)

    @property
    def all_entities(self) -> List[RetrievedEntity]:
        return self.primaries + self.expansions


class Retriever:
    """Loads index once; serves multiple queries."""

    def __init__(self, index_dir: Path, entities_dir: Path):
        self.entities_dir = entities_dir
        vectors_path = index_dir / "vectors.npy"
        metadata_path = index_dir / "metadata.json"
        if not vectors_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run "
                f"`python -m planescapeRAG.chatbot.embed_entities` first."
            )

        self.vectors = np.load(vectors_path).astype(np.float32)
        meta = json.loads(metadata_path.read_text())
        self.entities: List[dict] = meta["entities"]
        self.dim: int = meta["dim"]
        if self.vectors.shape[0] != len(self.entities):
            raise RuntimeError(
                f"Index inconsistent: {self.vectors.shape[0]} vectors vs "
                f"{len(self.entities)} metadata entries"
            )

        self.index = faiss.IndexFlatIP(self.dim)
        if self.vectors.shape[0] > 0:
            self.index.add(self.vectors)

        # Per-section schema: multiple rows share the same entity_id (one per
        # `## Source: <doc>` section). Keep the first row per entity for
        # entity-level lookups (canonical_name / type / aliases are duplicated
        # across an entity's sections, so any row works).
        self._by_id: Dict[str, dict] = {}
        for e in self.entities:
            self._by_id.setdefault(e["entity_id"], e)
        # Canonical-name → entity_id, lowercased for case-insensitive wikilink resolution.
        self._by_canonical: Dict[str, str] = {}
        for e in self.entities:
            cn = e.get("canonical_name")
            if cn:
                self._by_canonical.setdefault(cn.lower(), e["entity_id"])

        # Inverted wikilink index: target_canonical_lower → [source_entity_ids
        # whose body contains [[target]]]. Powers reverse expansion in retrieve()
        # — for "taverns in Sigil"-style queries, taverns link to wards link to
        # Sigil, but the chain is forward-only. To surface entities X that
        # mention any retrieved entity Y, we need Y → reverse → X.
        # One-time scan at startup; on the order of seconds for 4K entities.
        self._linked_by: Dict[str, List[str]] = self._build_inverted_link_index()

        settings = Settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set.")
        self.openai = OpenAI(api_key=settings.openai_api_key)

    def _build_inverted_link_index(self) -> Dict[str, List[str]]:
        """Single scan of every entity body that builds:
        - self._linked_by[target_canonical_lower] -> [source_entity_ids]  (returned)
        - self._links_from[source_entity_id]      -> [target_canonical_lower]  (set as side effect)

        Forward expansion uses _links_from to reach across an entity's FULL
        wikilink set, not just the matched section's — Sigil's guide_to_astral
        section doesn't mention wards, but we still want to walk to wards from
        Sigil so we can reverse-find the taverns that link to them.
        """
        import time
        t0 = time.time()
        inverted: Dict[str, List[str]] = {}
        forward: Dict[str, List[str]] = {}
        n_files = 0
        for entity_id in self._by_id:
            path = self.entities_dir / (entity_id + ".md")
            try:
                body = self._strip_frontmatter(path.read_text())
            except OSError:
                continue
            n_files += 1
            seen_targets: set = set()
            ordered_targets: List[str] = []
            for target in self._extract_wikilink_targets(body):
                key = target.lower()
                if self._by_canonical.get(key) == entity_id:
                    continue  # skip self-links
                if key in seen_targets:
                    continue
                seen_targets.add(key)
                ordered_targets.append(key)
                inverted.setdefault(key, []).append(entity_id)
            forward[entity_id] = ordered_targets
        self._links_from: Dict[str, List[str]] = forward
        print(
            f"  inverted wikilink index: {n_files} entity files scanned, "
            f"{len(inverted)} target canonicals indexed in {time.time() - t0:.1f}s",
            flush=True,
        )
        return inverted

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.openai.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    # Fallback per-entity budget for cases where we don't have a specific
    # matched section to load (expansions from the wikilink graph, or entities
    # whose body has no `## Source:` structure). Caps the answer prompt size.
    _MAX_BODY_CHARS = 20000

    def _strip_frontmatter(self, text: str) -> str:
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                return text[end + 5:].lstrip()
        return text

    def _load_section(self, entity_id: str, section_id: Optional[str]) -> str:
        """Load just the matched `## Source: <section_id>` block, or fall back
        to a capped full body when no section_id is available (expansions /
        early-format entities with no per-source structure).
        """
        path = self.entities_dir / (entity_id + ".md")
        text = self._strip_frontmatter(path.read_text())

        if section_id and section_id != _MAIN_SECTION:
            marker = f"## Source: {section_id}"
            start = text.find(marker)
            if start >= 0:
                next_section = text.find("\n## Source:", start + len(marker))
                end = next_section if next_section > 0 else len(text)
                # Prepend the entity heading so the answer prompt's
                # annotate_source_headers / wikilink-back logic still has the
                # canonical name visible at the top of each entity block.
                heading_end = text.find("\n", 0)
                heading = text[:heading_end] if heading_end > 0 and text.startswith("# ") else ""
                section = text[start:end].rstrip() + "\n"
                return f"{heading}\n\n{section}" if heading else section

        if len(text) <= self._MAX_BODY_CHARS:
            return text
        cut = text.rfind("\n## Source:", 0, self._MAX_BODY_CHARS)
        if cut > 0:
            return text[:cut].rstrip() + "\n"
        return text[:self._MAX_BODY_CHARS]

    def _extract_wikilink_targets(self, body: str) -> List[str]:
        seen = set()
        out: List[str] = []
        for m in _WIKILINK_RE.finditer(body):
            target = m.group(1).strip()
            key = target.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(target)
        return out

    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        max_total: int = 15,
        expand: bool = True,
    ) -> RetrievalResult:
        if self.index.ntotal == 0:
            return RetrievalResult(query=query)

        # Per-section index: each entity has up to ~25 vectors (one per source
        # doc). For top-k UNIQUE entities, fan out the FAISS search and dedup
        # by entity_id, keeping the highest-similarity section per entity.
        # Fan-out factor is chosen to comfortably cover an entity having all
        # 27 sections matched in the top-N (Sigil-tier worst case); the FAISS
        # search itself is sub-millisecond.
        k_uniq = min(k, len(self._by_id))
        if k_uniq == 0:
            return RetrievalResult(query=query)
        fan_out = min(self.index.ntotal, max(k_uniq * 30, 100))
        q_vec = self._embed_query(query)
        scores, idxs = self.index.search(q_vec, fan_out)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        primaries: List[RetrievedEntity] = []
        seen_entities = set()
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            e = self.entities[idx]
            entity_id = e["entity_id"]
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            section_id = e.get("section_id")
            primaries.append(RetrievedEntity(
                entity_id=entity_id,
                canonical_name=e.get("canonical_name", entity_id),
                type=e.get("type", ""),
                similarity=float(score),
                body=self._load_section(entity_id, section_id),
                is_primary=True,
                section_id=section_id,
            ))
            if len(primaries) >= k_uniq:
                break

        expansions: List[RetrievedEntity] = []
        if expand:
            seen_ids = {p.entity_id for p in primaries}
            total_budget = max(0, max_total - len(primaries))
            # Split the expansion budget so reverse gets a fair share —
            # otherwise wikilink-rich primaries (Sigil et al.) drain the whole
            # budget before reverse-link entities like taverns can land.
            forward_budget = total_budget // 2
            budget = forward_budget

            # Forward 1-hop: entities wikilinked from each primary's FULL body.
            # Rank targets to surface "medium hubs" (wards, factions, named
            # locations — typically 10–200 inbound links) before specific NPCs
            # AND before mega-hubs (planes, races, Sigil itself). Mega-hubs
            # have so many inbound links that reverse-from-mega-hub returns
            # essentially random entities; medium-hub reverse yields the
            # actual siblings of the primary (Hive Ward's reverse → taverns
            # in Hive Ward; Harmonium's reverse → Harmonium members).
            #
            # No type bias — it overfits ("modal primary type = place" filtered
            # out the 121 character members of Harmonium when the user asked
            # for them; queries cross the type boundary too often).

            def _hub_score(target_canonical: str) -> int:
                # Plateau at 50 — wards (50–200 inbound) and mega-hubs
                # (1000+ inbound) score equally; tie-break by mention order.
                return min(len(self._linked_by.get(target_canonical, [])), 50)

            fwd_pools: List[Tuple[RetrievedEntity, List[str]]] = []
            for primary in primaries:
                targets = self._links_from.get(primary.entity_id, [])
                candidates: List[Tuple[int, int, str]] = []
                for orig_idx, t in enumerate(targets):
                    target_id = self._by_canonical.get(t)
                    if not target_id or target_id in seen_ids:
                        continue
                    candidates.append((-_hub_score(t), orig_idx, target_id))
                candidates.sort()
                pool = [tid for _, _, tid in candidates]
                if pool:
                    fwd_pools.append((primary, pool))
            progress = True
            while budget > 0 and progress:
                progress = False
                for primary, pool in fwd_pools:
                    if budget <= 0:
                        break
                    while pool:
                        target_id = pool.pop(0)
                        if target_id in seen_ids:
                            continue
                        seen_ids.add(target_id)
                        e = self._by_id[target_id]
                        expansions.append(RetrievedEntity(
                            entity_id=target_id,
                            canonical_name=e.get("canonical_name", target_id),
                            type=e.get("type", ""),
                            similarity=0.0,
                            body=self._load_section(target_id, None),
                            is_primary=False,
                            expanded_from=primary.entity_id,
                        ))
                        budget -= 1
                        progress = True
                        break  # next primary's turn
            # Roll any leftover forward budget into reverse so total_budget
            # is fully used (rather than half-wasted when primaries have few
            # wikilinks).
            budget = total_budget - len([e for e in expansions if not e.expanded_from.startswith("reverse_link_to:")])

            # Reverse 1-hop: entities whose bodies wikilink TO any retrieved
            # entity (primary or forward expansion). For "taverns in Sigil",
            # the taverns→Hive Ward→Sigil chain is forward-only — taverns are
            # surfaced here because they link to Sigil or one of its wards.
            # Forward-expansion sources come first: they're typically hub
            # entities (wards, factions) whose reverse links surface "siblings"
            # of the primary. Reverse-from-primary yields mostly NPCs / patrons,
            # which is useful but secondary.
            if budget > 0:
                # Split reverse budget between two source classes:
                # - Forward-expansion sources (typically hub entities like
                #   wards / factions) — reverse picks here are siblings of
                #   the primaries (more taverns under each ward).
                # - Primary sources — reverse picks here are entities that
                #   directly mention the primary (members of a faction,
                #   patrons of a tavern, things connected to the entity).
                # Naive ordering would let one class drain the entire budget;
                # for "members of Harmonium" the Harmonium primary is the
                # rich source, while for "taverns in Sigil" the wards
                # (forward expansions) are.
                fwd_budget = max(1, budget // 2)
                prim_budget = budget - fwd_budget

                def _build_pool(src: RetrievedEntity) -> Optional[Tuple[RetrievedEntity, List[str]]]:
                    canon = (src.canonical_name or "").lower()
                    if not canon:
                        return None
                    pool = [eid for eid in self._linked_by.get(canon, []) if eid not in seen_ids]
                    return (src, pool) if pool else None

                fwd_pools = [r for r in (_build_pool(s) for s in expansions) if r]
                prim_pools = [r for r in (_build_pool(s) for s in primaries) if r]

                def _round_robin_draw(pools_list, draw_budget):
                    nonlocal budget
                    progress = True
                    while draw_budget > 0 and progress:
                        progress = False
                        for src, pool in pools_list:
                            if draw_budget <= 0:
                                break
                            while pool:
                                eid = pool.pop(0)
                                if eid in seen_ids:
                                    continue
                                seen_ids.add(eid)
                                e = self._by_id.get(eid)
                                if not e:
                                    continue
                                expansions.append(RetrievedEntity(
                                    entity_id=eid,
                                    canonical_name=e.get("canonical_name", eid),
                                    type=e.get("type", ""),
                                    similarity=0.0,
                                    body=self._load_section(eid, None),
                                    is_primary=False,
                                    expanded_from=f"reverse_link_to:{src.canonical_name}",
                                ))
                                draw_budget -= 1
                                budget -= 1
                                progress = True
                                break

                _round_robin_draw(fwd_pools, fwd_budget)
                _round_robin_draw(prim_pools, prim_budget)
                # Roll any leftover from one half into the other.
                if budget > 0 and fwd_pools:
                    _round_robin_draw(fwd_pools, budget)
                if budget > 0 and prim_pools:
                    _round_robin_draw(prim_pools, budget)

        return RetrievalResult(query=query, primaries=primaries, expansions=expansions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect retrieval results.")
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-total", type=int, default=15)
    parser.add_argument("--no-expand", action="store_true")
    parser.add_argument("--entities-dir", type=Path, default=Path("planescapeRAG/graph/entities"))
    parser.add_argument("--index-dir", type=Path, default=Path("planescapeRAG/chatbot/index"))
    args = parser.parse_args()

    retriever = Retriever(index_dir=args.index_dir, entities_dir=args.entities_dir)
    result = retriever.retrieve(
        args.query, k=args.k, max_total=args.max_total, expand=not args.no_expand,
    )

    print(f"\nQuery: {args.query!r}\n")
    print(f"Top-{len(result.primaries)} by cosine similarity:")
    for p in result.primaries:
        print(f"  {p.similarity:.3f}  {p.canonical_name}  ({p.type})")
    if result.expansions:
        print(f"\n1-hop expansions ({len(result.expansions)}):")
        for e in result.expansions:
            print(f"  {e.canonical_name}  ({e.type})  ← {e.expanded_from}")


if __name__ == "__main__":
    main()
