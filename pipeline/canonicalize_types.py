import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
from dotenv import load_dotenv

from ..config import Settings
from .ocr import load_corpus_profile


_ASSIGN_SUPERTYPES_TOOL = {
    "name": "assign_supertypes",
    "description": "Submit the complete mapping from granular types to supertypes. Call this exactly once with every input type mapped.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "The granular type from the registry, exactly as provided.",
                        },
                        "supertype": {
                            "type": "string",
                            "description": "One of the supertypes from the closed list.",
                        },
                    },
                    "required": ["type", "supertype"],
                },
            },
        },
        "required": ["mappings"],
    },
}


def _extract_supertype_list(profile_ingest: str) -> List[Dict[str, str]]:
    """Parse the Supertypes subsection from the Ingest addendum.

    Looks for `- **name** — description` lines under a heading whose first word
    is "Supertype" or "Supertypes" (case-insensitive).
    """
    in_supertype_section = False
    supertypes: List[Dict[str, str]] = []
    bullet_re = re.compile(r'^\s*-\s*\*\*([^*]+)\*\*\s*[—–-]\s*(.*?)$')
    for line in profile_ingest.splitlines():
        if line.startswith("### "):
            in_supertype_section = line[4:].strip().lower().startswith("supertype")
            continue
        if not in_supertype_section:
            continue
        match = bullet_re.match(line)
        if match:
            supertypes.append({"name": match.group(1).strip(), "description": match.group(2).strip()})
    return supertypes


def _build_canonicalization_prompt(types: List[str], supertypes: List[Dict[str, str]]) -> str:
    supertype_lines = "\n".join(f"- **{s['name']}** — {s['description']}" for s in supertypes)
    type_lines = "\n".join(f"- {t}" for t in types)
    return f"""You are mapping document-natural entity types to a closed list of supertypes for a knowledge graph.

# Supertypes (closed list)

Choose exactly one of these for each input type. Do not invent new supertypes.

{supertype_lines}

# Input types

Map each of these granular types to one supertype from the list above:

{type_lines}

# Task

Use the `assign_supertypes` tool exactly once with the complete mapping. Every input type must appear exactly once in the output.

If a type genuinely doesn't fit any supertype, choose the closest one. The mapping is a taxonomy decision, not a factual judgement — when in doubt, prefer the broader category."""


def canonicalize_types(
    registry_path: Path,
    output_path: Path,
    profile: Dict[str, str],
    model: Optional[str] = None,
    doc_tag: Optional[str] = None,
) -> Dict[str, str]:
    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set.")
    if "ingest" not in profile:
        raise ValueError("Corpus profile has no Ingest addendum.")

    supertypes = _extract_supertype_list(profile["ingest"])
    if not supertypes:
        raise ValueError("Could not parse a Supertypes section from the corpus profile's Ingest addendum.")
    valid_supertype_names = {s["name"] for s in supertypes}

    with registry_path.open() as f:
        registry = json.load(f)
    types = sorted({e["type"] for e in registry["entities"]})

    print(f"Mapping {len(types)} unique types to {len(supertypes)} supertypes...", flush=True)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    chosen_model = model or settings.extraction_model

    response = client.messages.create(
        model=chosen_model,
        max_tokens=4000,
        messages=[{"role": "user", "content": _build_canonicalization_prompt(types, supertypes)}],
        tools=[_ASSIGN_SUPERTYPES_TOOL],
        tool_choice={"type": "tool", "name": "assign_supertypes"},
    )

    tool_use = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "assign_supertypes"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(f"Model did not call assign_supertypes. Stop: {response.stop_reason}. Content: {response.content}")

    mapping = {m["type"]: m["supertype"] for m in tool_use.input["mappings"]}

    invalid = sorted(t for t, st in mapping.items() if st not in valid_supertype_names)
    missing = sorted(t for t in types if t not in mapping)
    extra = sorted(t for t in mapping if t not in types)
    if invalid:
        print(f"WARNING: types mapped to unknown supertypes: {invalid}", flush=True)
    if missing:
        print(f"WARNING: input types not mapped: {missing}", flush=True)
    if extra:
        print(f"WARNING: mapping contains types not in registry: {extra}", flush=True)

    output_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"\nMapping → {output_path}", flush=True)

    by_supertype: Dict[str, List[str]] = {}
    for t, st in sorted(mapping.items()):
        by_supertype.setdefault(st, []).append(t)
    for st in sorted(by_supertype):
        print(f"  {st:>10}: {', '.join(by_supertype[st])}")

    print(f"\nUsage: input={response.usage.input_tokens}  output={response.usage.output_tokens}")
    from ..cost import report_call
    report_call(
        "canonicalize_types", chosen_model, response.usage,
        details={"doc_tag": doc_tag} if doc_tag else None,
    )
    return mapping


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="Map granular entity types from a registry to a closed list of supertypes from a corpus profile.")
    parser.add_argument("registry", type=Path, help="Registry JSON produced by extract_registry.py")
    parser.add_argument("output", type=Path, help="Path to write the type → supertype mapping JSON")
    parser.add_argument("profile", type=Path, help="Corpus profile markdown")
    parser.add_argument("--model", type=str, default=None, help="Anthropic model override")
    args = parser.parse_args()

    profile = load_corpus_profile(args.profile)
    canonicalize_types(args.registry, args.output, profile=profile, model=args.model)
