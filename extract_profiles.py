"""Pass 2: per-entity profile extraction.

For each entity in the registry, calls Claude with the full document + registry
in cached context, and extracts a structured profile (description, source pages,
typed relationships with verbatim quotes). Writes one markdown file per entity
under {output_dir}/{supertype}/{slug}.md.

Uses the Anthropic message batches API with 1-hour prompt caching: the document
+ registry prefix is cached once and reused across all per-entity calls.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import anthropic
import regex as fuzzy_regex
from dotenv import load_dotenv

from .config import Settings
from .ocr import apply_ocr_corrections, load_corpus_profile, parse_ocr_corrections


_SUBMIT_ENTITY_PROFILE_TOOL = {
    "name": "submit_entity_profile",
    "description": "Submit the structured profile for the current entity. Call this exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": (
                    "Multi-paragraph markdown description of the entity, written in the document's framing. "
                    "When referencing OTHER entities, use [[Canonical Name]] wikilinks where Canonical Name "
                    "exactly matches an entry in the registry. Do not self-link. Stay close to source; do not "
                    "editorialize or invent details."
                ),
            },
            "source_pages": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Page numbers (read from <!-- page N --> markers in the document) where this entity is "
                    "meaningfully discussed. Sorted and deduplicated."
                ),
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Canonical name of the related entity. MUST match a canonical_name in the registry exactly. MUST NOT equal the entity being profiled.",
                        },
                        "quote": {
                            "type": "string",
                            "description": "Verbatim sentence(s) from the document that establish this relationship. The quote must appear in the document literally (whitespace may differ).",
                        },
                        "page": {
                            "type": "integer",
                            "description": "Page where the quote appears.",
                        },
                    },
                    "required": ["target", "quote", "page"],
                },
                "description": "Directed edges from this entity to others. Only include relationships the document explicitly establishes.",
            },
        },
        "required": ["description", "source_pages", "relationships"],
    },
}


_SYSTEM_PROMPT = """You are extracting a structured profile for a single named entity from a document.

You will be given:
- The full document text (inside <document>...</document>)
- A registry of all entities in the document (inside <registry>...</registry>)
- A specific entity name to profile

Submit the profile using the `submit_entity_profile` tool, with these fields:

# description
A multi-paragraph markdown description of the entity, written in the document's framing. Use `[[Canonical Name]]` wikilinks when referencing OTHER entities, where Canonical Name must exactly match a `canonical_name` in the registry. Do not self-link. Stay close to source; do not editorialize or invent details.

# source_pages
Page numbers (read from `<!-- page N -->` markers in the document) where this entity is meaningfully discussed. Sort and deduplicate.

# relationships
Directed edges from this entity to others. For each:
- target: canonical name of the related entity (must match registry exactly; must not equal the entity being profiled).
- quote: verbatim sentence(s) from the document that establish the relationship. The quote must appear in the document literally — paraphrasing breaks downstream validation.
- page: page where the quote appears.

Only include relationships the document explicitly establishes. If the entity is mentioned only briefly, description may be short and relationships may be empty. Better to be sparse than to invent.

For entities the registry marks as notable=true, you MUST populate the relationships array with explicit edges, not just inline wikilinks in the description. Aim for 8+ relationships for highly-connected entities (factions, planes, major characters, central locations). The description and the relationships serve different purposes: description is for human reading; relationships are the structured graph edges that downstream consumers (graph-DB import, retrieval) depend on. Do not skip the relationships field by claiming the description covers it."""


_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_PAGE_MARKER_RE = re.compile(r"<!-- page (\d+) -->")


def _slug(name: str) -> str:
    """Filesystem-safe slug from a canonical entity name."""
    s = name.lower()
    s = re.sub(r"['’]", "", s)  # strip apostrophes
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


_PUNCTUATION_FOLD = str.maketrans({
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "–": "-",  # en dash
    "—": "-",  # em dash
    "…": "...",  # ellipsis
})


def _normalize_for_quote_match(text: str) -> str:
    """Fold Unicode punctuation, collapse whitespace, lowercase. For substring search of quotes.

    The OCR'd document uses curly quotes and em-dashes; the LLM tends to emit ASCII
    forms. Without folding, semantically-identical quotes fail substring match.
    """
    text = text.translate(_PUNCTUATION_FOLD)
    return re.sub(r"\s+", " ", text).strip().lower()


def _validate_quote(quote: str, document_normalized: str) -> bool:
    return _normalize_for_quote_match(quote) in document_normalized


_ELLIPSIS_SPLIT_RE = fuzzy_regex.compile(r"\.{3,}|…")


def _build_segment_pattern(segment: str, max_internal_edits: int = 5) -> str:
    """Build a fuzzy regex for one segment of a quote.

    For 4+ word segments: anchor first and last word exactly, allow up to
    max_internal_edits in the middle. For shorter segments: whole-segment
    fuzzy with edit budget proportional to length.
    """
    words = segment.split()
    if len(words) >= 4:
        first = fuzzy_regex.escape(words[0])
        last = fuzzy_regex.escape(words[-1])
        middle = fuzzy_regex.escape(" ".join(words[1:-1]))
        return rf"{first}\s+(?:{middle}){{e<={max_internal_edits}}}\s+{last}"
    max_e = max(1, len(segment) // 15)
    return f"(?:{fuzzy_regex.escape(segment)}){{e<={max_e}}}"


def _fuzzy_find_quote(emitted: str, document: str, max_internal_edits: int = 5) -> Optional[str]:
    """Find the document's text matching the emitted quote (fuzzy).

    Searches the *original* document (case-insensitively) and returns the matched
    span verbatim from the source — preserving the document's case, punctuation,
    and any markdown emphasis. Returns None on no match.

    Strategy:
    - Whitespace-collapse the emitted quote.
    - Split on ellipsis (`...` or `…`); treat each segment as a separately
      anchored fuzzy match, glued by `(?s).{0,300}?` wildcards.
    - Each segment uses first-word/last-word exact anchors with up to
      max_internal_edits edits in the middle (or whole-segment fuzzy if
      shorter than 4 words). Edits cover the model's punctuation drift
      (curly→ASCII quotes, em-dash → hyphen, dropped commas, etc.).
    """
    emitted_clean = re.sub(r"\s+", " ", emitted).strip()
    if not emitted_clean:
        return None

    segments = [s.strip() for s in _ELLIPSIS_SPLIT_RE.split(emitted_clean) if s.strip()]
    if not segments:
        return None

    sub_patterns = [_build_segment_pattern(s, max_internal_edits) for s in segments]
    pattern = r"(?s).{0,300}?".join(sub_patterns)

    try:
        match = fuzzy_regex.search(
            pattern,
            document,
            flags=fuzzy_regex.IGNORECASE | fuzzy_regex.BESTMATCH,
        )
    except fuzzy_regex.error:
        return None
    return match.group(0) if match else None


def _build_registry_payload(registry: dict) -> str:
    """Compact JSON list of entities, suitable for inclusion in the cached prefix."""
    entries = [
        {
            "canonical_name": e["canonical_name"],
            "type": e["type"],
            "aliases": e.get("aliases", []) or [],
            "brief_description": e["brief_description"],
        }
        for e in registry["entities"]
    ]
    return json.dumps(entries, ensure_ascii=False, indent=1)


def _build_request_fulldoc(
    entity: dict,
    document: str,
    registry_text: str,
    model: str,
    cache_ttl: str,
) -> dict:
    """Full-doc request builder. Caches <document>+<registry> as one big prefix."""
    cached_block = (
        "<document>\n"
        + document
        + "\n</document>\n\n<registry>\n"
        + registry_text
        + "\n</registry>"
    )
    instruction = (
        f"Now generate the profile for: {entity['canonical_name']}\n\n"
        f"(Type from registry: {entity['type']}. Aliases in registry: "
        f"{json.dumps(entity.get('aliases', []) or [])}.)"
    )
    return {
        "custom_id": _slug(entity["canonical_name"]),
        "params": {
            "model": model,
            "max_tokens": 4000,
            "system": [
                {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral", "ttl": cache_ttl}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": cached_block, "cache_control": {"type": "ephemeral", "ttl": cache_ttl}},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            "tools": [_SUBMIT_ENTITY_PROFILE_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_entity_profile"},
        },
    }


_PAGE_MARKER_BLOCK_RE = re.compile(r"^<!--\s*page\s+(\d+)\s*-->\s*\n?(.*)", re.DOTALL)


def _parse_document_paragraphs(document: str) -> List[Tuple[str, Optional[int]]]:
    """Split a combined-batch markdown document into (paragraph_text, page) tuples.

    Page markers <!-- page N --> set the current page; they don't appear as their
    own paragraphs in the output. Paragraphs are separated by blank lines.
    """
    paragraphs: List[Tuple[str, Optional[int]]] = []
    current_page: Optional[int] = None
    for raw in re.split(r"\n\s*\n", document):
        block = raw.strip()
        if not block:
            continue
        match = _PAGE_MARKER_BLOCK_RE.match(block)
        if match:
            current_page = int(match.group(1))
            remainder = match.group(2).strip()
            if remainder:
                paragraphs.append((remainder, current_page))
        else:
            paragraphs.append((block, current_page))
    return paragraphs


def _build_windowed_context(
    paragraphs: List[Tuple[str, Optional[int]]],
    entity: dict,
    neighbor_paras: int = 1,
) -> Optional[str]:
    """Find paragraphs mentioning the entity (canonical or alias) and return them
    plus N neighbor paragraphs on each side, with page markers preserved and
    `[...]` between non-contiguous spans. Returns None if no matches found.
    """
    search_terms = [entity["canonical_name"]] + (entity.get("aliases") or [])
    if not search_terms:
        return None
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in search_terms) + r")\b",
        re.IGNORECASE,
    )

    selected: Set[int] = set()
    for i, (text, _) in enumerate(paragraphs):
        if pattern.search(text):
            for j in range(max(0, i - neighbor_paras), min(len(paragraphs), i + neighbor_paras + 1)):
                selected.add(j)

    if not selected:
        return None

    chunks: List[str] = []
    last_page: Optional[int] = None
    last_index: Optional[int] = None
    for i in sorted(selected):
        text, page = paragraphs[i]
        if last_index is not None and i > last_index + 1:
            chunks.append("[...]")
            last_page = None  # force re-emit page marker after a gap
        if page is not None and page != last_page:
            chunks.append(f"<!-- page {page} -->")
            last_page = page
        chunks.append(text)
        last_index = i

    return "\n\n".join(chunks)


def _build_request_windowed(
    entity: dict,
    paragraphs: List[Tuple[str, Optional[int]]],
    registry_text: str,
    model: str,
    cache_ttl: str,
    neighbor_paras: int,
) -> Optional[dict]:
    """Windowed request builder. Caches system prompt + registry; per-entity
    context is the windowed paragraphs around mentions of the entity. Returns
    None if the entity isn't found in the document at all.
    """
    context = _build_windowed_context(paragraphs, entity, neighbor_paras=neighbor_paras)
    if context is None:
        return None
    instruction = (
        f"<context>\n{context}\n</context>\n\n"
        f"Now generate the profile for: {entity['canonical_name']}\n\n"
        f"(Type from registry: {entity['type']}. Aliases in registry: "
        f"{json.dumps(entity.get('aliases', []) or [])}.)"
    )
    return {
        "custom_id": _slug(entity["canonical_name"]),
        "params": {
            "model": model,
            "max_tokens": 4000,
            "system": [
                {"type": "text", "text": _SYSTEM_PROMPT, "cache_control": {"type": "ephemeral", "ttl": cache_ttl}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"<registry>\n{registry_text}\n</registry>", "cache_control": {"type": "ephemeral", "ttl": cache_ttl}},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            "tools": [_SUBMIT_ENTITY_PROFILE_TOOL],
            "tool_choice": {"type": "tool", "name": "submit_entity_profile"},
        },
    }


def _build_frontmatter(entity: dict, supertype: str, source_pages: List[int]) -> str:
    fields = {
        "canonical_name": entity["canonical_name"],
        "type": entity["type"],
        "supertype": supertype,
        "aliases": entity.get("aliases", []) or [],
        "notable": entity["notable"],
        "source_pages": sorted(set(source_pages)),
    }
    lines = ["---"]
    for k, v in fields.items():
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def _filter_relationships(
    relationships,
    self_canonical: str,
    canonical_names: Set[str],
    document: str,
    document_normalized: str,
) -> Tuple[List[dict], Dict[str, int], int]:
    """Drop self-relationships, unknown targets, and unverifiable quotes.

    For quotes that fail exact substring match, falls back to fuzzy regex search
    against the original document. On fuzzy match, replaces the model's emitted
    quote with the document's matched span (verbatim source text).

    Returns (kept, drop_counts, fuzzy_rescue_count).
    """
    drops = {"self": 0, "unknown_target": 0, "quote_not_in_doc": 0, "malformed": 0}
    # Model occasionally emits the whole array as a JSON-encoded string
    # instead of a parsed array. Recover by parsing.
    if isinstance(relationships, str):
        try:
            relationships = json.loads(relationships)
        except json.JSONDecodeError:
            relationships = []
    if not isinstance(relationships, list):
        return [], drops, 0

    kept = []
    rescued = 0
    for rel in relationships:
        if not isinstance(rel, dict) or "target" not in rel or "quote" not in rel:
            drops["malformed"] += 1
            continue
        if rel["target"] == self_canonical:
            drops["self"] += 1
            continue
        if rel["target"] not in canonical_names:
            drops["unknown_target"] += 1
            continue
        if _validate_quote(rel["quote"], document_normalized):
            kept.append(rel)
            continue
        found = _fuzzy_find_quote(rel["quote"], document)
        if found is not None:
            rel = dict(rel)
            rel["quote"] = re.sub(r"\s+", " ", found).strip()
            kept.append(rel)
            rescued += 1
        else:
            drops["quote_not_in_doc"] += 1
    return kept, drops, rescued


def _strip_unresolved_wikilinks(text: str, canonical_names: Set[str]) -> Tuple[str, List[str]]:
    """Resolve wikilinks against the registry.

    Handles both [[Target]] and [[Target|Display]] (Obsidian display-syntax).
    - If Target is in canonical_names, the wikilink is kept as-is (preserving
      any |Display alias for prose readability).
    - If Target is not canonical, the brackets are stripped:
      - [[Target]] → Target
      - [[Target|Display]] → Display (drop the broken target, keep the display text)
    Returns (cleaned_text, list_of_broken_targets).
    """
    broken = []

    def sub(m):
        inner = m.group(1)
        target, sep, display = inner.partition("|")
        target = target.strip()
        if target in canonical_names:
            return m.group(0)
        broken.append(target)
        return display.strip() if sep else target

    return _WIKILINK_RE.sub(sub, text), broken


def _assemble_markdown(entity: dict, supertype: str, profile: dict) -> str:
    frontmatter = _build_frontmatter(entity, supertype, profile["source_pages"])
    body = [f"# {entity['canonical_name']}", "", profile["description"]]
    if profile["relationships"]:
        body.extend(["", "## Relationships", ""])
        for rel in profile["relationships"]:
            quote = rel["quote"].strip().replace("\n", " ")
            body.append(f"- [[{rel['target']}]] — \"{quote}\" (p. {rel['page']})")
    return frontmatter + "\n\n" + "\n".join(body) + "\n"


def _output_path(output_dir: Path, supertype: str, entity: dict) -> Path:
    return output_dir / supertype / f"{_slug(entity['canonical_name'])}.md"


def submit_profiles(
    document_path: Path,
    registry_path: Path,
    supertype_map_path: Path,
    output_dir: Path,
    corpus_profile: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    cache_ttl: str = "1h",
    skip_existing: bool = True,
    mode: str = "fulldoc",
    neighbor_paras: int = 1,
) -> str:
    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set.")
    chosen_model = model or settings.extraction_model

    document = document_path.read_text()
    corrections = parse_ocr_corrections(corpus_profile) if corpus_profile else {}
    if corrections:
        document = apply_ocr_corrections(document, corrections)
        print(f"Applied {len(corrections)} OCR corrections from corpus profile.", flush=True)
    with registry_path.open() as f:
        registry = json.load(f)
    with supertype_map_path.open() as f:
        supertype_map = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    registry_text = _build_registry_payload(registry)

    if mode not in {"fulldoc", "windowed"}:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'fulldoc' or 'windowed'.")
    paragraphs = _parse_document_paragraphs(document) if mode == "windowed" else None
    if paragraphs is not None:
        print(f"Windowed mode: parsed {len(paragraphs)} paragraphs from document; neighbor_paras={neighbor_paras}", flush=True)

    # Build one request per entity, skipping any whose output file already exists.
    requests = []
    skipped = 0
    no_context = 0
    for entity in registry["entities"]:
        supertype = supertype_map.get(entity["type"])
        if supertype is None:
            print(f"WARNING: no supertype for type={entity['type']!r}; skipping {entity['canonical_name']}")
            continue
        path = _output_path(output_dir, supertype, entity)
        if skip_existing and path.exists():
            skipped += 1
            continue
        if mode == "fulldoc":
            requests.append(_build_request_fulldoc(entity, document, registry_text, chosen_model, cache_ttl))
        else:
            request = _build_request_windowed(entity, paragraphs, registry_text, chosen_model, cache_ttl, neighbor_paras)
            if request is None:
                no_context += 1
                print(f"WARNING: no document mentions found for {entity['canonical_name']!r} ({entity['type']}); skipping in windowed mode.")
                continue
            requests.append(request)

    print(f"Building {len(requests)} requests ({skipped} skipped existing, {no_context} skipped no-context).")
    if not requests:
        print("Nothing to submit.")
        return ""

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch: {batch.id}  status={batch.processing_status}  count={len(requests)}")
    return batch.id


def collect_profiles(
    batch_id: str,
    document_path: Path,
    registry_path: Path,
    supertype_map_path: Path,
    output_dir: Path,
    corpus_profile: Optional[Dict[str, str]] = None,
) -> None:
    settings = Settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    document = document_path.read_text()
    corrections = parse_ocr_corrections(corpus_profile) if corpus_profile else {}
    if corrections:
        document = apply_ocr_corrections(document, corrections)
        print(f"Applied {len(corrections)} OCR corrections from corpus profile.", flush=True)
    with registry_path.open() as f:
        registry = json.load(f)
    with supertype_map_path.open() as f:
        supertype_map = json.load(f)

    canonical_names = {e["canonical_name"] for e in registry["entities"]}
    by_slug = {_slug(e["canonical_name"]): e for e in registry["entities"]}
    document_normalized = _normalize_for_quote_match(document)

    print(f"Polling batch {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.processing_status}  succeeded={counts.succeeded}  errored={counts.errored}  processing={counts.processing}", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(60)

    written = 0
    drop_totals = {"self": 0, "unknown_target": 0, "quote_not_in_doc": 0, "malformed": 0}
    fuzzy_rescued_total = 0
    broken_link_totals = 0
    failures: List[str] = []

    for result in client.messages.batches.results(batch_id):
        slug = result.custom_id
        entity = by_slug.get(slug)
        if entity is None:
            failures.append(f"{slug}: no matching entity in registry")
            continue
        if result.result.type != "succeeded":
            err = getattr(result.result, "error", None)
            failures.append(f"{slug}: {result.result.type}  {err}")
            continue

        message = result.result.message
        tool_use = next(
            (b for b in message.content if b.type == "tool_use" and b.name == "submit_entity_profile"),
            None,
        )
        if tool_use is None:
            failures.append(f"{slug}: model did not call submit_entity_profile")
            continue

        profile = dict(tool_use.input)
        profile["relationships"], drops, rescued = _filter_relationships(
            profile.get("relationships", []),
            entity["canonical_name"],
            canonical_names,
            document,
            document_normalized,
        )
        for k in drop_totals:
            drop_totals[k] += drops[k]
        fuzzy_rescued_total += rescued

        cleaned_description, broken = _strip_unresolved_wikilinks(profile["description"], canonical_names)
        profile["description"] = cleaned_description
        broken_link_totals += len(broken)

        supertype = supertype_map[entity["type"]]
        path = _output_path(output_dir, supertype, entity)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_assemble_markdown(entity, supertype, profile))
        written += 1

    print(f"\nWrote {written} entity files under {output_dir}")
    print(f"Dropped relationships: self={drop_totals['self']}  unknown_target={drop_totals['unknown_target']}  quote_not_in_doc={drop_totals['quote_not_in_doc']}  malformed={drop_totals['malformed']}")
    print(f"Quotes rescued via fuzzy match: {fuzzy_rescued_total}")
    print(f"Stripped unresolved wikilinks in descriptions: {broken_link_totals}")
    if failures:
        print(f"\n{len(failures)} request failures:")
        for f in failures[:20]:
            print(f"  {f}")


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="Pass 2: per-entity profile extraction via Anthropic batch API.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    submit = sub.add_parser("submit", help="Build per-entity requests and submit a batch.")
    submit.add_argument("--document", type=Path, required=True)
    submit.add_argument("--registry", type=Path, required=True)
    submit.add_argument("--supertype-map", type=Path, required=True)
    submit.add_argument("--output-dir", type=Path, required=True)
    submit.add_argument("--profile", type=Path, default=None, help="Optional corpus profile (used for OCR corrections).")
    submit.add_argument("--model", type=str, default=None)
    submit.add_argument("--cache-ttl", type=str, default="1h", choices=["5m", "1h"])
    submit.add_argument("--no-skip-existing", action="store_true", help="Re-submit entities even if output file exists.")
    submit.add_argument("--mode", type=str, default="fulldoc", choices=["fulldoc", "windowed"], help="fulldoc: cache the entire document per call (needs >200K context). windowed: per-entity paragraph windowing (fits any model).")
    submit.add_argument("--neighbor-paras", type=int, default=1, help="In windowed mode, include N paragraphs before and after each match.")

    collect = sub.add_parser("collect", help="Poll a batch and write entity files from results.")
    collect.add_argument("batch_id", type=str)
    collect.add_argument("--document", type=Path, required=True)
    collect.add_argument("--registry", type=Path, required=True)
    collect.add_argument("--supertype-map", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--profile", type=Path, default=None, help="Optional corpus profile (used for OCR corrections).")

    args = parser.parse_args()
    corpus_profile = load_corpus_profile(args.profile) if args.profile else None

    if args.cmd == "submit":
        submit_profiles(
            document_path=args.document,
            registry_path=args.registry,
            supertype_map_path=args.supertype_map,
            output_dir=args.output_dir,
            corpus_profile=corpus_profile,
            model=args.model,
            cache_ttl=args.cache_ttl,
            skip_existing=not args.no_skip_existing,
            mode=args.mode,
            neighbor_paras=args.neighbor_paras,
        )
    elif args.cmd == "collect":
        collect_profiles(
            batch_id=args.batch_id,
            document_path=args.document,
            registry_path=args.registry,
            supertype_map_path=args.supertype_map,
            output_dir=args.output_dir,
            corpus_profile=corpus_profile,
        )
