"""Completeness sweep: find entities mentioned in the document but missing from the registry.

Pass 1's entity-extraction prompt biases toward entities the document
*describes* (gives a sentence or two of treatment). Entities mentioned only
in passing — like a Sigil district named once in a list of districts — get
skipped. The completeness sweep gives the model the document and the
current registry and asks specifically for the gap: what's mentioned but
not registered?

Output: a list of suggested entities (canonical_name, type, aliases,
brief_description, notable, evidence_quote). Each suggestion includes a
verbatim quote from the document, validated by substring search before
acceptance. By default the suggestions are written to a JSON file for
human review; with --apply-to-registry, valid suggestions are appended
directly to the registry.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
from dotenv import load_dotenv

from .config import Settings
from .ocr import apply_ocr_corrections, load_corpus_profile, parse_ocr_corrections


_SUGGEST_MISSING_ENTITIES_TOOL = {
    "name": "suggest_missing_entities",
    "description": "Submit the list of entities mentioned in the document that aren't in the registry. Call this exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "missing_entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_name": {
                            "type": "string",
                            "description": "Canonical name following the corpus profile's conventions.",
                        },
                        "type": {
                            "type": "string",
                            "description": "Short lowercase common-noun category (e.g. 'faction', 'place', 'character').",
                        },
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Alternative names that appear verbatim in the document.",
                        },
                        "brief_description": {
                            "type": "string",
                            "description": "One sentence (≤25 words) describing what the entity is.",
                        },
                        "notable": {
                            "type": "boolean",
                            "description": "True if the document treats this as central; false if mentioned only briefly.",
                        },
                        "evidence_quote": {
                            "type": "string",
                            "description": "Verbatim sentence(s) from the document where this entity appears. Used to validate the suggestion.",
                        },
                    },
                    "required": ["canonical_name", "type", "aliases", "brief_description", "notable", "evidence_quote"],
                },
            },
        },
        "required": ["missing_entities"],
    },
}


_SYSTEM_PROMPT_BASE = """You are auditing an entity registry against a document. Your task: find named entities that the document mentions but the registry doesn't include.

The registry was extracted from this document (or a related document) but pass 1 typically biases toward entities that get dedicated treatment in the source. Entities mentioned only in passing — districts, minor characters, named features in a list — often get skipped. Your job is to fill those gaps.

You will receive:
- The full document
- The current registry (in <registry>...</registry>) as a slim list of canonical_name + aliases

Output: every entity mentioned in the document that is NOT already covered by the registry (canonical_name OR an alias of an existing entry). Use the suggest_missing_entities tool exactly once.

# What counts as missing
- A named entity (place, character, faction, item, creature, concept) appearing in the document
- Not currently a canonical_name OR alias of any registry entry
- Specific enough that a reader could plausibly look it up later

# What to skip
- Generic terms not treated as named entities ("the air", "magic")
- Entities whose canonical_name is already in the registry as a slight variant — case differences ("the Abyss" vs "The Abyss"), leading-"the" differences, OCR-style spelling drift. These are NOT missing; they're already there.
- Pronouns and unnamed roles

# Required fields per suggestion
- canonical_name: most common or formal form in the source
- type: short lowercase common-noun category
- aliases: only forms that appear verbatim in the document
- brief_description: one sentence ≤25 words
- notable: true if the doc gives dedicated treatment, false if mentioned in passing
- evidence_quote: a verbatim sentence from the document where the entity appears (used to validate this is real, not hallucinated)

Be thorough — the goal is to fill gaps. Better to suggest an entity that turns out to be a near-variant of an existing one (we'll filter it) than to miss a real entity.

If no missing entities, return an empty list."""


def _build_system_prompt(profile: Optional[Dict[str, str]]) -> str:
    if profile and profile.get("ingest"):
        return f"{_SYSTEM_PROMPT_BASE}\n\n# Corpus-specific guidance\n\n{profile['ingest']}"
    return _SYSTEM_PROMPT_BASE


def _slim_registry_payload(registry: dict) -> str:
    entries = []
    for e in registry["entities"]:
        item = {"canonical_name": e["canonical_name"]}
        if e.get("aliases"):
            item["aliases"] = e["aliases"]
        entries.append(item)
    return json.dumps(entries, ensure_ascii=False, indent=1)


def _normalize_for_match(text: str) -> str:
    fold = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    text = text.translate(fold)
    return re.sub(r"\s+", " ", text).strip().lower()


def sweep_for_missing(
    document_path: Path,
    registry_path: Path,
    output_path: Path,
    profile: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
) -> List[dict]:
    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set.")
    chosen_model = model or settings.extraction_model

    document = document_path.read_text()
    if profile:
        corrections = parse_ocr_corrections(profile)
        if corrections:
            document = apply_ocr_corrections(document, corrections)
            print(f"Applied {len(corrections)} OCR corrections.", flush=True)

    with registry_path.open() as f:
        registry = json.load(f)
    registry_text = _slim_registry_payload(registry)

    user_message = (
        f"<document>\n{document}\n</document>\n\n"
        f"<registry>\n{registry_text}\n</registry>\n\n"
        "Find every entity mentioned in the document that's NOT already in the registry. Submit via suggest_missing_entities."
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    print(f"Running completeness sweep with {chosen_model}... (document: {len(document)} chars, registry: {len(registry['entities'])} entities)", flush=True)

    with client.messages.stream(
        model=chosen_model,
        max_tokens=64000,
        system=_build_system_prompt(profile),
        messages=[{"role": "user", "content": user_message}],
        tools=[_SUGGEST_MISSING_ENTITIES_TOOL],
        tool_choice={"type": "tool", "name": "suggest_missing_entities"},
    ) as stream:
        response = stream.get_final_message()

    tool_use = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "suggest_missing_entities"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(f"Model did not call suggest_missing_entities. Stop: {response.stop_reason}")

    suggestions = tool_use.input.get("missing_entities", [])
    if not isinstance(suggestions, list):
        raise RuntimeError(f"Model returned non-list missing_entities: {type(suggestions).__name__}")

    # Validate evidence quotes against the document
    doc_normalized = _normalize_for_match(document)
    valid: List[dict] = []
    rejected_no_evidence: List[dict] = []
    seen_canonicals = set()
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        name = s.get("canonical_name", "").strip()
        quote = s.get("evidence_quote", "")
        if not name or not quote:
            continue
        if name in seen_canonicals:
            continue
        seen_canonicals.add(name)
        if _normalize_for_match(quote) in doc_normalized:
            valid.append(s)
        else:
            rejected_no_evidence.append(s)

    output_path.write_text(json.dumps(
        {"valid_suggestions": valid, "rejected_no_evidence": rejected_no_evidence},
        indent=2, ensure_ascii=False,
    ))

    print(f"\nSuggested: {len(suggestions)} | Valid (evidence verified): {len(valid)} | Rejected (no evidence): {len(rejected_no_evidence)}")
    print(f"Wrote suggestions → {output_path}")
    if valid:
        print(f"\nSample valid suggestions (first 12):")
        for s in valid[:12]:
            notable = "*" if s.get("notable") else " "
            print(f"  {notable} [{s['type']:>10}] {s['canonical_name']}: {s['brief_description']}")
    print(f"\nUsage: input={response.usage.input_tokens} output={response.usage.output_tokens} stop={response.stop_reason}")
    from .cost import report_call
    report_call("completeness_sweep", chosen_model, response.usage)

    return valid


def apply_suggestions_to_registry(registry: dict, suggestions: List[dict], doc_tag: str) -> int:
    """Append valid suggestions to the registry as new entities. Strips evidence_quote."""
    existing_canon = {e["canonical_name"] for e in registry["entities"]}
    added = 0
    for s in suggestions:
        if s["canonical_name"] in existing_canon:
            continue
        registry["entities"].append({
            "canonical_name": s["canonical_name"],
            "type": s["type"],
            "aliases": s.get("aliases") or [],
            "brief_description": s["brief_description"],
            "notable": bool(s.get("notable", False)),
            "source_documents": [doc_tag],
        })
        existing_canon.add(s["canonical_name"])
        added += 1
    return added


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="Find entities mentioned in a document but missing from the registry.")
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Where to write suggestions JSON.")
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--apply-to-registry", action="store_true", help="Append valid suggestions directly to the registry.")
    parser.add_argument("--doc-tag", type=str, default=None, help="Provenance tag if --apply-to-registry is set.")
    args = parser.parse_args()

    profile = load_corpus_profile(args.profile) if args.profile else None
    suggestions = sweep_for_missing(
        document_path=args.document,
        registry_path=args.registry,
        output_path=args.output,
        profile=profile,
        model=args.model,
    )

    if args.apply_to_registry:
        if not args.doc_tag:
            raise SystemExit("--doc-tag required with --apply-to-registry")
        with args.registry.open() as f:
            registry = json.load(f)
        added = apply_suggestions_to_registry(registry, suggestions, args.doc_tag)
        args.registry.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
        print(f"\nApplied {added} suggestions to {args.registry}")
