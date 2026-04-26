import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import anthropic
from dotenv import load_dotenv

from .config import Settings
from .ocr import apply_ocr_corrections, load_corpus_profile, parse_ocr_corrections


_REGISTER_ENTITIES_TOOL = {
    "name": "register_entities",
    "description": "Submit the complete deduped entity registry. Call this exactly once with the full list.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_name": {
                            "type": "string",
                            "description": "The form used for wikilinks. Unique across the registry.",
                        },
                        "type": {
                            "type": "string",
                            "description": "Short lowercase common-noun category (e.g. 'faction', 'plane'). Use the document's terminology. General enough to apply to multiple entities; specific enough to be meaningful.",
                        },
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Other names the document literally uses for this entity. Only forms that appear verbatim in the source.",
                        },
                        "brief_description": {
                            "type": "string",
                            "description": "One sentence (≤25 words) describing what the entity is.",
                        },
                        "notable": {
                            "type": "boolean",
                            "description": "True if the document treats this entity as central — gives it a dedicated section, describes it in detail, returns to it across multiple contexts. False if mentioned only briefly or in passing.",
                        },
                    },
                    "required": ["canonical_name", "type", "aliases", "brief_description", "notable"],
                },
            },
        },
        "required": ["entities"],
    },
}


_SYSTEM_PROMPT_BASE = """You are extracting a structured entity registry from a document. Your output is a deduped list of every named entity that appears. Use the `register_entities` tool exactly once, with the complete list. Produce no other output.

# What to extract

Specific, named things — places, characters, factions, items, named creatures, named cosmological concepts. A useful test: would a reader plausibly look this up later as "what is X?".

# What to skip

- Generic terms ("magic", "the multiverse") unless the document treats them as named entities.
- Pronouns and unnamed roles ("the leader", "travelers").
- Quotations and aphorisms.
- Page-number footers and other marginalia.

# Choosing a type

For each entity, assign a short lowercase common-noun type that describes its category — e.g. "faction", "plane", "character", "item". Use terminology the document itself uses where possible.

A good type is general enough to apply to multiple entities and specific enough to be meaningful. Do not invent a unique type per entity.

# Deduplication

When the same entity appears under multiple names, output one entry with one canonical_name and the others in aliases. Examples:
- "Sigil" and "the Cage" → canonical: "Sigil", aliases: ["Cage"]
- "Doomguard" and "Sinkers" → canonical: "Doomguard", aliases: ["Sinkers"]

Prefer over-merging to under-merging — duplicates are harder to fix downstream than over-eager merges.

# Aliases

Include only forms that appear verbatim in the document. Do not add semantic variants ("Sinkers" goes in aliases only if the document writes "Sinkers" — not because it's a known nickname elsewhere).

# Brief descriptions

One sentence, ≤25 words, written in the document's framing. Focus on what kind of thing the entity is. Avoid editorializing.

# Notable

Mark notable=true if the document treats this entity as central — gives it a dedicated section, describes it in detail, returns to it across multiple contexts. Mark notable=false if it's mentioned only briefly or in passing. This is a semantic judgement, not a count."""


def _build_system_prompt(profile: Optional[Dict[str, str]] = None) -> str:
    if profile and profile.get("ingest"):
        return f"{_SYSTEM_PROMPT_BASE}\n\n# Corpus-specific guidance\n\n{profile['ingest']}"
    return _SYSTEM_PROMPT_BASE


def extract_registry(
    document_path: Path,
    output_path: Path,
    profile: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
) -> dict:
    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set for entity extraction.")

    document_text = document_path.read_text()
    corrections = parse_ocr_corrections(profile)
    if corrections:
        document_text = apply_ocr_corrections(document_text, corrections)
        print(f"Applied {len(corrections)} OCR corrections from corpus profile.", flush=True)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system_prompt = _build_system_prompt(profile)
    user_message = (
        f"<document>\n{document_text}\n</document>\n\n"
        "Submit the complete entity registry now via the register_entities tool."
    )
    chosen_model = model or settings.extraction_model

    print(f"Extracting registry with {chosen_model}... (document: {len(document_text)} chars)", flush=True)
    # Use streaming because high max_tokens requests can exceed the SDK's 10-minute non-streaming cutoff.
    with client.messages.stream(
        model=chosen_model,
        max_tokens=32000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[_REGISTER_ENTITIES_TOOL],
        tool_choice={"type": "tool", "name": "register_entities"},
    ) as stream:
        response = stream.get_final_message()

    tool_use = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "register_entities"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(f"Model did not call register_entities. Stop reason: {response.stop_reason}. Content: {response.content}")
    if "entities" not in tool_use.input:
        raise RuntimeError(
            f"Tool input missing 'entities' key (likely max_tokens overflow during tool call). "
            f"Stop reason: {response.stop_reason}. "
            f"Usage: input={response.usage.input_tokens} output={response.usage.output_tokens}. "
            f"Tool input keys: {list(tool_use.input.keys())}"
        )

    registry = tool_use.input
    output_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False))

    entities = registry["entities"]
    print(f"\nExtracted {len(entities)} entities → {output_path}", flush=True)

    type_counts: Dict[str, int] = {}
    for e in entities:
        type_counts[e.get("type", "?")] = type_counts.get(e.get("type", "?"), 0) + 1
    print("\nBy type:")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {n:3d}  {t}")

    notable_count = sum(1 for e in entities if e.get("notable"))
    print(f"\nNotable: {notable_count}/{len(entities)}")
    print(f"Usage: input={response.usage.input_tokens}  output={response.usage.output_tokens}  stop={response.stop_reason}")

    return registry


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="Extract an entity registry from a combined markdown document.")
    parser.add_argument("document", type=Path, help="Path to the combined markdown document.")
    parser.add_argument("output", type=Path, help="Path to write the registry JSON.")
    parser.add_argument("--profile", type=Path, default=None, help="Optional corpus profile path.")
    parser.add_argument("--model", type=str, default=None, help="Anthropic model override (default: from EXTRACTION_MODEL env / config).")
    args = parser.parse_args()

    profile = load_corpus_profile(args.profile) if args.profile else None
    extract_registry(args.document, args.output, profile=profile, model=args.model)
