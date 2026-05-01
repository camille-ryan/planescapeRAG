"""Build / update the entity embedding index.

Walks graph/entities/, embeds each entity's body with OpenAI text-embedding-3-small,
and persists `vectors.npy` (float32, L2-normalized) + `metadata.json` whose rows
align by position.

Incremental: re-embeds only entities whose body sha256 changed (or are new); drops
entries whose entity file was deleted. Designed to be called as the final stage of
run_pipeline.py after profiles_collect.

The FAISS index is *not* persisted — it's rebuilt in-memory at query time from
vectors.npy. At 1044 vectors that's sub-millisecond; the FAISS file would be more
trouble than it's worth (IDMap incremental-update semantics).
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from ..config import Settings
from ..cost import log_cost, session_total


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
# USD per token (text-embedding-3-small is $0.020 / 1M tokens)
OPENAI_EMBEDDING_RATE = 0.020 / 1_000_000


def _parse_entity_file(path: Path) -> Tuple[Dict[str, object], str]:
    """Return (frontmatter_kv, body_text). Frontmatter is parsed naively but
    handles three value forms: JSON-list (`["a", "b"]`), quoted string
    (`"foo"`), and bare scalar (`42`, `true`, `bar`). Lists come back as Python
    lists — this matters for `aliases` and `source_documents`."""
    text = path.read_text()
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    frontmatter_block = text[4:end]
    body = text[end + 5:].lstrip()

    fm: Dict[str, object] = {}
    for line in frontmatter_block.split("\n"):
        m = re.match(r"^([a-zA-Z_]+):\s*(.*)$", line)
        if not m:
            continue
        key = m.group(1)
        value = m.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                fm[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass  # fall through to string handling
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        fm[key] = value
    return fm, body


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _walk_entities(entities_dir: Path) -> Dict[str, Tuple[Path, Dict[str, str], str, str]]:
    """Return {entity_id: (path, frontmatter, body, hash)} for every .md under entities_dir.

    entity_id is the path relative to entities_dir without the .md suffix,
    e.g. 'character/abacis-the-sweaty'.
    """
    out: Dict[str, Tuple[Path, Dict[str, str], str, str]] = {}
    for path in sorted(entities_dir.rglob("*.md")):
        rel = path.relative_to(entities_dir)
        entity_id = str(rel.with_suffix("")).replace("\\", "/")
        fm, body = _parse_entity_file(path)
        out[entity_id] = (path, fm, body, _body_hash(body))
    return out


def _load_state(vectors_path: Path, metadata_path: Path) -> Tuple[Optional[np.ndarray], List[dict]]:
    if not vectors_path.exists() or not metadata_path.exists():
        return None, []
    vectors = np.load(vectors_path)
    meta = json.loads(metadata_path.read_text())
    if meta.get("model") != EMBEDDING_MODEL or meta.get("dim") != EMBEDDING_DIM:
        # Model/dim mismatch — treat as fresh build.
        print(f"  warning: existing index used {meta.get('model')} @ {meta.get('dim')}d; rebuilding from scratch", flush=True)
        return None, []
    return vectors, meta.get("entities", [])


def _save_state(
    vectors: np.ndarray,
    entities: List[dict],
    vectors_path: Path,
    metadata_path: Path,
) -> None:
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    metadata_path.write_text(json.dumps(
        {"model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM, "entities": entities},
        indent=2,
        ensure_ascii=False,
    ))


def _embed_batch(client: OpenAI, texts: List[str]) -> Tuple[np.ndarray, int]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vectors, resp.usage.total_tokens


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm


def update_index(
    entities_dir: Path,
    vectors_path: Path,
    metadata_path: Path,
    *,
    batch_size: int = 100,
    auto_yes: bool = False,
    cost_threshold_usd: float = 0.50,
    force_refresh: bool = False,
    doc_tag: Optional[str] = None,
) -> Dict[str, int]:
    """Embed new/changed entities, persist vectors + metadata. Returns stats."""
    settings = Settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for embeddings.")

    on_disk = _walk_entities(entities_dir)
    old_vectors, old_entities = _load_state(vectors_path, metadata_path)
    old_by_id = {e["entity_id"]: (i, e) for i, e in enumerate(old_entities)}

    to_embed: List[Tuple[str, str]] = []   # (entity_id, body)
    keep_old: List[Tuple[str, int]] = []    # (entity_id, old_row_index)
    for entity_id, (_, _, body, h) in on_disk.items():
        prev = old_by_id.get(entity_id)
        if prev is None or prev[1].get("content_hash") != h:
            to_embed.append((entity_id, body))
        else:
            keep_old.append((entity_id, prev[0]))

    removed = [eid for eid in old_by_id if eid not in on_disk]

    print(f"  on disk: {len(on_disk)} entities", flush=True)
    print(f"  unchanged: {len(keep_old)}", flush=True)
    print(f"  new or changed: {len(to_embed)}", flush=True)
    print(f"  removed: {len(removed)}", flush=True)

    # If metadata is missing fields we now record (e.g. aliases added in a later
    # release), force a metadata refresh even when no bodies changed. The
    # rebuild reuses existing vectors — it just rewrites the JSON sidecar.
    metadata_stale = old_vectors is not None and any(
        "aliases" not in e for e in old_entities
    )

    if not to_embed and not removed and old_vectors is not None and not metadata_stale and not force_refresh:
        print("  no changes; index up to date.", flush=True)
        return {"on_disk": len(on_disk), "embedded": 0, "removed": 0, "cost_usd": 0.0}

    if not to_embed and (metadata_stale or force_refresh):
        print("  refreshing metadata (no embedding changes; aliases / new fields)", flush=True)

    if to_embed:
        total_chars = sum(len(b) for _, b in to_embed)
        est_tokens = max(1, total_chars // 4)
        est_cost = est_tokens * OPENAI_EMBEDDING_RATE
        print(f"  estimated cost: ${est_cost:.4f} (~{est_tokens} tokens)", flush=True)
        if not auto_yes and est_cost > cost_threshold_usd:
            ans = input(f"  estimate exceeds ${cost_threshold_usd:.2f} — proceed? [y/N]: ").strip().lower()
            if ans != "y":
                print("  aborted.", flush=True)
                return {"on_disk": len(on_disk), "embedded": 0, "removed": 0, "aborted": True}

    client = OpenAI(api_key=settings.openai_api_key)
    new_rows: Dict[str, np.ndarray] = {}
    tokens_used = 0
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        vecs, toks = _embed_batch(client, [b for _, b in batch])
        tokens_used += toks
        for (entity_id, _), vec in zip(batch, vecs):
            new_rows[entity_id] = vec
        print(f"  batch {i // batch_size + 1}: embedded {len(batch)} (tokens: {toks})", flush=True)

    # Assemble final state in stable order: existing entries first (preserving relative
    # order), then newly-embedded entries.
    final_entities: List[dict] = []
    final_vectors: List[np.ndarray] = []

    def _meta_entry(entity_id: str, fm: dict, h: str) -> dict:
        aliases = fm.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        return {
            "entity_id": entity_id,
            "content_hash": h,
            "canonical_name": fm.get("canonical_name", ""),
            "type": fm.get("type", ""),
            "aliases": [a for a in aliases if isinstance(a, str) and a.strip()],
        }

    for entity_id, old_row in keep_old:
        path, fm, _, h = on_disk[entity_id]
        final_entities.append(_meta_entry(entity_id, fm, h))
        final_vectors.append(old_vectors[old_row])

    for entity_id, _ in to_embed:
        path, fm, _, h = on_disk[entity_id]
        final_entities.append(_meta_entry(entity_id, fm, h))
        final_vectors.append(new_rows[entity_id])

    matrix = np.vstack(final_vectors).astype(np.float32) if final_vectors else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    matrix = _l2_normalize(matrix)

    _save_state(matrix, final_entities, vectors_path, metadata_path)

    actual_cost = tokens_used * OPENAI_EMBEDDING_RATE
    details = {"tokens": tokens_used, "embedded": len(new_rows), "removed": len(removed)}
    if doc_tag:
        details["doc_tag"] = doc_tag
    log_cost(
        operation="embed_entities",
        model=EMBEDDING_MODEL,
        cost_usd=actual_cost,
        details=details,
    )
    print(f"  done. tokens: {tokens_used}, cost: ${actual_cost:.4f}, session total: ${session_total():.2f}", flush=True)

    return {
        "on_disk": len(on_disk),
        "embedded": len(new_rows),
        "removed": len(removed),
        "tokens": tokens_used,
        "cost_usd": actual_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update entity embedding index.")
    parser.add_argument("--entities-dir", type=Path, default=Path("planescapeRAG/graph/entities"))
    parser.add_argument("--index-dir", type=Path, default=Path("planescapeRAG/chatbot/index"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--cost-threshold", type=float, default=0.50)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--force-refresh", action="store_true",
        help="Rebuild metadata.json even when bodies haven't changed. Use after "
             "changing frontmatter (e.g. running tools/refresh_entity_frontmatter).",
    )
    args = parser.parse_args()

    args.index_dir.mkdir(parents=True, exist_ok=True)
    update_index(
        entities_dir=args.entities_dir,
        vectors_path=args.index_dir / "vectors.npy",
        metadata_path=args.index_dir / "metadata.json",
        batch_size=args.batch_size,
        auto_yes=args.yes,
        cost_threshold_usd=args.cost_threshold,
        force_refresh=args.force_refresh,
    )


if __name__ == "__main__":
    main()
