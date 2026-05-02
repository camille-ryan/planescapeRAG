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

# OpenAI's text-embedding-3-small caps input at 8192 tokens. We embed at the
# `## Source: <doc_tag>` SECTION level rather than the full entity body — each
# section is a self-contained source-grounded description and almost always
# fits well under the cap. Defensive truncation (~6.5K tokens at 4 chars/token)
# is kept for the rare oversize section.
_EMBED_MAX_CHARS = 25000

# Marker for entities with no per-source structure (e.g. early-format entity
# files that predate the `## Source:` convention). The whole body becomes one
# section with this id.
_MAIN_SECTION = "main"


def _truncate_for_embedding(text: str) -> Tuple[str, bool]:
    """Hard char cap to absorb token-density variance from non-ASCII text and
    code. With per-section embedding this rarely fires — only the densest
    individual sections trip it.
    """
    if len(text) <= _EMBED_MAX_CHARS:
        return text, False
    return text[:_EMBED_MAX_CHARS], True


_SOURCE_HEADER_RE = re.compile(r"^## Source: (.+?)$", re.MULTILINE)
_RELATIONSHIPS_BLOCK_RE = re.compile(r"\n### Relationships from .+?$.*?(?=\n## |\Z)", re.DOTALL | re.MULTILINE)


def _strip_relationship_quotes(text: str) -> str:
    """Drop `### Relationships from <doc>` blocks. The verbatim quotes are
    high-noise for embedding — each is unique text but adds little semantic
    signal, since the wikilinks in the prose already carry the relational
    meaning.
    """
    return _RELATIONSHIPS_BLOCK_RE.sub("", text).rstrip() + "\n"


def _split_into_sections(body: str) -> List[Tuple[str, str]]:
    """Split an entity body into per-source sections.

    Returns [(section_id, section_text), ...] where:
    - section_id is the `## Source: <doc_tag>` value, or `_MAIN_SECTION` if
      the body doesn't follow the per-source convention
    - section_text is the prose between this header and the next, with the
      `### Relationships from X` block stripped (see `_strip_relationship_quotes`)
    """
    matches = list(_SOURCE_HEADER_RE.finditer(body))
    if not matches:
        # No per-source structure — treat the whole body as one section.
        return [(_MAIN_SECTION, _strip_relationship_quotes(body.strip()))]

    sections = []
    for i, m in enumerate(matches):
        section_id = m.group(1).strip()
        section_start = m.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[section_start:section_end]
        cleaned = _strip_relationship_quotes(raw.strip())
        if cleaned.strip():
            sections.append((section_id, cleaned))
    return sections


def _build_embedding_text(canonical_name: str, aliases: List[str], section_id: str, section_text: str) -> str:
    """Compose the per-section text sent to the embedder.

    Includes the entity's canonical name + aliases as a header so each section
    vector also carries the entity-identity signal — important when sections
    are short and heavily wikilink-driven.
    """
    parts = [f"# {canonical_name}"]
    if aliases:
        parts.append(f"Also known as: {', '.join(aliases)}")
    if section_id != _MAIN_SECTION:
        parts.append(f"From source: {section_id}")
    parts.append("")
    parts.append(section_text)
    return "\n".join(parts).strip()


def _section_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _walk_entities(entities_dir: Path) -> Dict[Tuple[str, str], Tuple[Path, Dict[str, object], str, str]]:
    """Return {(entity_id, section_id): (path, frontmatter, embed_text, hash)}
    for every per-source section in every .md under entities_dir.

    entity_id is the path relative to entities_dir without the .md suffix,
    e.g. 'character/abacis-the-sweaty'. section_id is the `## Source: <doc>`
    value or `_MAIN_SECTION` for entities with no per-source structure.

    embed_text is the section text composed for embedding (with header anchor).
    Hash is over embed_text — change-detection is per section, so adding a new
    `## Source:` section to Sigil only re-embeds that one new vector.
    """
    out: Dict[Tuple[str, str], Tuple[Path, Dict[str, object], str, str]] = {}
    for path in sorted(entities_dir.rglob("*.md")):
        rel = path.relative_to(entities_dir)
        entity_id = str(rel.with_suffix("")).replace("\\", "/")
        fm, body = _parse_entity_file(path)
        canonical = str(fm.get("canonical_name") or entity_id)
        aliases_raw = fm.get("aliases") or []
        aliases = [a for a in aliases_raw if isinstance(a, str) and a.strip()] if isinstance(aliases_raw, list) else []
        for section_id, section_text in _split_into_sections(body):
            embed_text = _build_embedding_text(canonical, aliases, section_id, section_text)
            out[(entity_id, section_id)] = (path, fm, embed_text, _section_hash(embed_text))
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
    # Defensive truncation — OpenAI rejects the entire batch if any single input
    # exceeds 8192 tokens, taking down 99 well-formed entities with one oversized one.
    safe_texts = [_truncate_for_embedding(t)[0] for t in texts]
    # Retry on TPM rate limits with capped exponential backoff. The SDK's built-in
    # retry handles 429s but gives up after a few attempts; per-section rebuilds
    # over the full corpus consistently bump the 1M TPM ceiling for ~minutes.
    import time as _time

    from openai import RateLimitError

    delay = 2.0
    for attempt in range(8):
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=safe_texts)
            vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return vectors, resp.usage.total_tokens
        except RateLimitError as e:
            if attempt == 7:
                raise
            print(f"    rate limit hit (attempt {attempt + 1}/8); sleeping {delay:.0f}s...", flush=True)
            _time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


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
    # Per-section keying. Old entries without section_id are pre-Option-C (full-body
    # embeddings) — treat them as missing so the migration re-embeds at section level.
    old_by_key = {
        (e["entity_id"], e.get("section_id", _MAIN_SECTION)): (i, e)
        for i, e in enumerate(old_entities)
        if e.get("section_id")  # require section_id for compatibility match
    }

    to_embed: List[Tuple[Tuple[str, str], str]] = []   # ((entity_id, section_id), embed_text)
    keep_old: List[Tuple[Tuple[str, str], int]] = []    # ((entity_id, section_id), old_row_index)
    for key, (_, _, embed_text, h) in on_disk.items():
        prev = old_by_key.get(key)
        if prev is None or prev[1].get("content_hash") != h:
            to_embed.append((key, embed_text))
        else:
            keep_old.append((key, prev[0]))

    removed = [k for k in old_by_key if k not in on_disk]

    n_entities = len(set(eid for eid, _ in on_disk))
    print(f"  on disk: {n_entities} entities, {len(on_disk)} sections", flush=True)
    print(f"  unchanged sections: {len(keep_old)}", flush=True)
    print(f"  new or changed sections: {len(to_embed)}", flush=True)
    print(f"  removed sections: {len(removed)}", flush=True)

    # Pre-Option-C metadata didn't store section_id. If we see entries without it,
    # the index needs a full rebuild (every entity becomes per-section).
    metadata_stale = old_vectors is not None and any(
        not e.get("section_id") for e in old_entities
    )
    if metadata_stale:
        print("  detected pre-section metadata; rebuilding index per-section.", flush=True)

    if not to_embed and not removed and old_vectors is not None and not metadata_stale and not force_refresh:
        print("  no changes; index up to date.", flush=True)
        return {"on_disk": n_entities, "sections": len(on_disk), "embedded": 0, "removed": 0, "cost_usd": 0.0}

    if not to_embed and (metadata_stale or force_refresh):
        print("  refreshing metadata (no embedding changes; new fields)", flush=True)

    if to_embed:
        # Apply truncation up-front so the cost estimate matches what's actually sent.
        truncated_keys = []
        sized_to_embed = []
        for key, embed_text in to_embed:
            text, was_cut = _truncate_for_embedding(embed_text)
            sized_to_embed.append((key, text))
            if was_cut:
                truncated_keys.append((key, len(embed_text), len(text)))
        to_embed = sized_to_embed
        total_chars = sum(len(t) for _, t in to_embed)
        est_tokens = max(1, total_chars // 4)
        est_cost = est_tokens * OPENAI_EMBEDDING_RATE
        print(f"  estimated cost: ${est_cost:.4f} (~{est_tokens} tokens across {len(to_embed)} sections)", flush=True)
        if truncated_keys:
            print(f"  truncated {len(truncated_keys)} oversized sections to fit the 8192-token cap:", flush=True)
            for (eid, sid), before, after in sorted(truncated_keys, key=lambda x: -x[1])[:10]:
                print(f"    {eid}#{sid}: {before} → {after} chars", flush=True)
            if len(truncated_keys) > 10:
                print(f"    ... and {len(truncated_keys) - 10} more", flush=True)
        if not auto_yes and est_cost > cost_threshold_usd:
            ans = input(f"  estimate exceeds ${cost_threshold_usd:.2f} — proceed? [y/N]: ").strip().lower()
            if ans != "y":
                print("  aborted.", flush=True)
                return {"on_disk": n_entities, "sections": len(on_disk), "embedded": 0, "removed": 0, "aborted": True}

    client = OpenAI(api_key=settings.openai_api_key)
    new_rows: Dict[Tuple[str, str], np.ndarray] = {}
    tokens_used = 0
    for i in range(0, len(to_embed), batch_size):
        batch = to_embed[i:i + batch_size]
        vecs, toks = _embed_batch(client, [t for _, t in batch])
        tokens_used += toks
        for (key, _), vec in zip(batch, vecs):
            new_rows[key] = vec
        print(f"  batch {i // batch_size + 1}: embedded {len(batch)} sections (tokens: {toks})", flush=True)

    # Assemble final state in stable order: existing entries first (preserving relative
    # order), then newly-embedded entries.
    final_entities: List[dict] = []
    final_vectors: List[np.ndarray] = []

    def _meta_entry(entity_id: str, section_id: str, fm: dict, h: str) -> dict:
        aliases = fm.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        return {
            "entity_id": entity_id,
            "section_id": section_id,
            "content_hash": h,
            "canonical_name": fm.get("canonical_name", ""),
            "type": fm.get("type", ""),
            "aliases": [a for a in aliases if isinstance(a, str) and a.strip()],
        }

    # When migrating from a pre-section index, skip keep_old (old vectors are full-body
    # embeddings and don't correspond to sections) — fall through to re-embed everything.
    if not metadata_stale:
        for key, old_row in keep_old:
            entity_id, section_id = key
            path, fm, _, h = on_disk[key]
            final_entities.append(_meta_entry(entity_id, section_id, fm, h))
            final_vectors.append(old_vectors[old_row])

    for key, _ in to_embed:
        entity_id, section_id = key
        path, fm, _, h = on_disk[key]
        final_entities.append(_meta_entry(entity_id, section_id, fm, h))
        final_vectors.append(new_rows[key])

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
