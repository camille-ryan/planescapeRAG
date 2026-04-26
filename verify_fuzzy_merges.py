"""LLM-assisted verification of fuzzy-matched merge candidates.

When merge_registries.py finds a fuzzy-name match between a new entity and an
existing one, the match isn't always a true alias. Examples from real runs:
  - 'Frey' vs 'Freya'  — different Norse deities (false positive)
  - 'Manes' vs 'Mazes' — different concepts (false positive)
  - 'Slaadi' vs 'slaad' — same race, plural/case variant (true positive)
  - 'Cavitius' vs 'Caviitus' — OCR variants of the same fortress (true positive)

This module batches the candidate pairs into one LLM call and returns a
yes/no decision per pair. The descriptions provide the disambiguation
signal: "Frey: Norse trickster god" vs "Freya: Norse goddess of love"
clarifies the false positive without ambiguity.

Defaults to Haiku — the task is a discrete judgment, the input is small,
and Haiku is well-suited.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

from .config import Settings


_VERIFY_MERGES_TOOL = {
    "name": "verify_merges",
    "description": "For each candidate pair, decide whether the two entities refer to the same thing. Call this exactly once with one decision per pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pair_index": {
                            "type": "integer",
                            "description": "Zero-based index matching the pair in the input prompt.",
                        },
                        "are_same_entity": {
                            "type": "boolean",
                            "description": "True if the existing and incoming entities refer to the same thing.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence explaining the decision.",
                        },
                    },
                    "required": ["pair_index", "are_same_entity"],
                },
            },
        },
        "required": ["decisions"],
    },
}


_SYSTEM_PROMPT = """You are deciding whether candidate pairs of entities refer to the same thing.

You'll receive a list of pairs. Each pair has an "existing" entity (already in the registry) and an "incoming" entity (newly extracted). They were flagged as candidates by a fuzzy-name match, which can be a false positive when:
- Two entities have similar names but different meanings ('Manes' vs 'Mazes')
- Two entities are different members of a related group ('Frey' vs 'Freya')

Use the descriptions and aliases to decide each pair.

# Decide are_same_entity = true when:
- Capitalization or article variants of the same thing ('the Abyss' vs 'The Abyss')
- Singular/plural of the same entity ('Slaadi' vs 'slaad')
- OCR-style spelling variants ('Cavitius' vs 'Caviitus' — same fortress)
- Known synonyms or alternative names for the same thing

# Decide are_same_entity = false when:
- Different members of a pantheon or related category (different gods, different planes)
- Similar names that the descriptions clarify are distinct concepts
- Anything where you're not confident — when in doubt, don't merge

For every pair, set pair_index to match the input ordering. Include a one-sentence reasoning."""


def _format_entity(label: str, entity: dict, indent: str = "  ") -> List[str]:
    lines = [f"{indent}{label}:"]
    lines.append(f"{indent}  canonical_name: {entity['canonical_name']!r}")
    lines.append(f"{indent}  type: {entity.get('type', '?')}")
    aliases = entity.get("aliases") or []
    if aliases:
        lines.append(f"{indent}  aliases: {aliases}")
    lines.append(f"{indent}  brief_description: {entity.get('brief_description', '')!r}")
    return lines


def _format_pairs(pairs: List[Tuple[dict, dict]]) -> str:
    lines = []
    for i, (existing, incoming) in enumerate(pairs):
        lines.append(f"Pair {i}:")
        lines.extend(_format_entity("existing", existing))
        lines.extend(_format_entity("incoming", incoming))
        lines.append("")
    return "\n".join(lines)


def verify_merges(
    pairs: List[Tuple[dict, dict]],
    model: str = "claude-haiku-4-5-20251001",
) -> List[bool]:
    """Decide for each (existing, incoming) pair whether they should merge.

    Returns a list of bools matching pair order. True = merge; False = treat
    as distinct (incoming gets added as a new entity instead).
    """
    if not pairs:
        return []

    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set.")

    user_message = (
        f"Here are {len(pairs)} candidate pairs. For each, decide whether the existing and "
        f"incoming entities refer to the same thing.\n\n"
        + _format_pairs(pairs)
        + f"\n\nSubmit all decisions via verify_merges. Include every pair_index from 0 to {len(pairs) - 1}."
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        tools=[_VERIFY_MERGES_TOOL],
        tool_choice={"type": "tool", "name": "verify_merges"},
    )

    tool_use = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "verify_merges"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(f"Model did not call verify_merges. Stop: {response.stop_reason}")

    raw_decisions = tool_use.input.get("decisions", [])
    if not isinstance(raw_decisions, list):
        raise RuntimeError(f"Model returned non-list decisions: {type(raw_decisions).__name__}")

    decision_map: Dict[int, bool] = {}
    reasoning_map: Dict[int, str] = {}
    for d in raw_decisions:
        if not isinstance(d, dict) or "pair_index" not in d:
            continue
        decision_map[d["pair_index"]] = bool(d.get("are_same_entity", True))
        reasoning_map[d["pair_index"]] = d.get("reasoning", "")

    # Default to True (merge) for any pair the model omitted — conservative.
    decisions = [decision_map.get(i, True) for i in range(len(pairs))]

    print(f"Verifier: {sum(decisions)} merges, {len(decisions) - sum(decisions)} rejected.")
    for i, ((existing, incoming), keep) in enumerate(zip(pairs, decisions)):
        marker = "MERGE " if keep else "SPLIT "
        print(f"  {marker} {existing['canonical_name']!r:<40} ↔ {incoming['canonical_name']!r:<30} — {reasoning_map.get(i, '')}")

    return decisions


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="Verify fuzzy-matched merge candidates via LLM.")
    parser.add_argument("pairs_json", type=Path, help='JSON: [{"existing": {...}, "incoming": {...}}, ...]')
    parser.add_argument("--output", type=Path, required=True, help="Write decisions JSON.")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    with args.pairs_json.open() as f:
        raw = json.load(f)
    pairs = [(p["existing"], p["incoming"]) for p in raw]

    decisions = verify_merges(pairs, model=args.model)
    args.output.write_text(json.dumps(
        [{"pair_index": i, "are_same_entity": d} for i, d in enumerate(decisions)],
        indent=2,
    ))
    print(f"Wrote {args.output}")
