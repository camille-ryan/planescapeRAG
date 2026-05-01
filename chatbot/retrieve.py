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
from typing import Dict, List, Optional

import faiss
import numpy as np
from openai import OpenAI

from ..config import Settings
from .embed_entities import EMBEDDING_MODEL


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

        self._by_id: Dict[str, dict] = {e["entity_id"]: e for e in self.entities}
        # Canonical-name → entity_id, lowercased for case-insensitive wikilink resolution.
        self._by_canonical: Dict[str, str] = {
            e["canonical_name"].lower(): e["entity_id"]
            for e in self.entities
            if e.get("canonical_name")
        }

        settings = Settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set.")
        self.openai = OpenAI(api_key=settings.openai_api_key)

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.openai.embeddings.create(model=EMBEDDING_MODEL, input=[query])
        v = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v

    def _load_body(self, entity_id: str) -> str:
        path = self.entities_dir / (entity_id + ".md")
        text = path.read_text()
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                return text[end + 5:].lstrip()
        return text

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

        k = min(k, self.index.ntotal)
        q_vec = self._embed_query(query)
        scores, idxs = self.index.search(q_vec, k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        primaries: List[RetrievedEntity] = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            e = self.entities[idx]
            entity_id = e["entity_id"]
            primaries.append(RetrievedEntity(
                entity_id=entity_id,
                canonical_name=e.get("canonical_name", entity_id),
                type=e.get("type", ""),
                similarity=float(score),
                body=self._load_body(entity_id),
                is_primary=True,
            ))

        expansions: List[RetrievedEntity] = []
        if expand:
            seen_ids = {p.entity_id for p in primaries}
            budget = max(0, max_total - len(primaries))
            for primary in primaries:
                if budget <= 0:
                    break
                for target_canonical in self._extract_wikilink_targets(primary.body):
                    if budget <= 0:
                        break
                    target_id = self._by_canonical.get(target_canonical.lower())
                    if not target_id or target_id in seen_ids:
                        continue
                    seen_ids.add(target_id)
                    e = self._by_id[target_id]
                    expansions.append(RetrievedEntity(
                        entity_id=target_id,
                        canonical_name=e.get("canonical_name", target_id),
                        type=e.get("type", ""),
                        similarity=0.0,
                        body=self._load_body(target_id),
                        is_primary=False,
                        expanded_from=primary.entity_id,
                    ))
                    budget -= 1

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
