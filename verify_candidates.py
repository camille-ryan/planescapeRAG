"""LLM-verified entity candidate processing.

Bridges between the heuristic candidate extractor and the registry. Takes a
list of unmatched candidates from `heuristic_entity_candidates`, gathers a
few sample mentions from the document for each, and asks the LLM to decide
per candidate:

  - skip       — not a real entity (generic term, fragment, stat-block field)
  - merge_into — already covered by an existing canonical entity; add as alias
  - add_new    — genuinely missing entity; emit canonical_name/type/etc.

Designed to complement the LLM completeness sweep, which we've observed
biases toward specific named figures (deities, NPCs) and misses high-
frequency cosmological groupings ('Outer Planes', 'Hive Ward', 'Toril')
that the heuristic readily surfaces. Together the two cover gaps in
opposite directions.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

from .config import Settings
from .ocr import apply_ocr_corrections, load_corpus_profile, parse_ocr_corrections


_DECIDE_CANDIDATES_TOOL = {
    "name": "decide_candidates",
    "description": "For each candidate, decide whether to skip it, merge it into an existing entity, or add it as a new entity. Call this exactly once with one decision per input candidate.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate": {
                            "type": "string",
                            "description": "The candidate text exactly as given in the input.",
                        },
                        "action": {
                            "type": "string",
                            "enum": ["skip", "merge_into", "add_new"],
                            "description": "skip: not a real entity (generic term, stat-block field, fragment). merge_into: already in the registry as a canonical_name; add this candidate as an alias. add_new: a genuinely missing entity to add.",
                        },
                        "merge_target": {
                            "type": "string",
                            "description": "When action=merge_into, the canonical_name of the existing registry entry. Must match exactly.",
                        },
                        "new_entity": {
                            "type": "object",
                            "description": "When action=add_new, the full entity fields.",
                            "properties": {
                                "canonical_name": {"type": "string", "description": "Use the form most common or formal in the source."},
                                "type": {"type": "string", "description": "Short lowercase common-noun category."},
                                "aliases": {"type": "array", "items": {"type": "string"}, "description": "Verbatim variants from the document."},
                                "brief_description": {"type": "string", "description": "One sentence ≤25 words."},
                                "notable": {"type": "boolean", "description": "True if the document treats this as central."},
                            },
                            "required": ["canonical_name", "type", "aliases", "brief_description", "notable"],
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence explaining the decision.",
                        },
                    },
                    "required": ["candidate", "action", "reasoning"],
                },
            },
        },
        "required": ["decisions"],
    },
}


_SYSTEM_PROMPT_BASE = """You are processing a list of proper-noun candidates that a heuristic extractor surfaced as potentially missing from an entity registry. Your job: for each candidate, decide whether to skip, merge into an existing registry entry, or add as a new entity.

You will receive:
- The current registry (in <registry>...</registry>) as a slim list of canonical_name + aliases
- A list of candidates, each with mention count and a few sample contexts from the document

For each candidate, choose ONE of three actions:

# action="skip" — not a real entity
The candidate is a generic term, stat-block field, abbreviation, or fragment that shouldn't be in the registry. Examples: 'Hit Dice', 'THAC0', 'Course' (when 'of course'), 'Layer' (generic), 'Intelligence' (D&D ability score, not a proper noun in this context).

# action="merge_into" — already covered
The candidate is a shorthand or alias for something already in the registry. Examples: 'Fire' is shorthand for 'Plane of Fire'; 'Sigil's' is the possessive of 'Sigil'; 'Cage' is an alias of 'Sigil'. Set merge_target to the existing canonical_name (must match exactly).

# action="add_new" — genuinely missing
The candidate is a real named entity not in the registry. Provide all entity fields (canonical_name, type, aliases, brief_description, notable). Use the corpus's conventions for the canonical name. The brief_description should reflect what the document actually says — don't invent details.

# General rules
- Always include every candidate in the output (one decision per input candidate).
- If unsure, prefer "skip" over "add_new" — adding a false-positive entity is worse than missing one (the LLM completeness sweep can catch real entities we skipped, but adding noise pollutes the registry).
- Use the sample contexts to ground your judgment. If the contexts show the candidate is being used as a generic word, skip it.
- Set reasoning to one short sentence."""


def _build_system_prompt(profile: Optional[Dict[str, str]]) -> str:
    if profile and profile.get("ingest"):
        return f"{_SYSTEM_PROMPT_BASE}\n\n# Corpus-specific guidance\n\n{profile['ingest']}"
    return _SYSTEM_PROMPT_BASE


def gather_contexts(name: str, document: str, n_samples: int = 3, chars_around: int = 180) -> List[str]:
    """Find up to n_samples mentions of `name` in the document and return surrounding context."""
    try:
        pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    except re.error:
        return []
    contexts: List[str] = []
    last_end = -1
    for m in pattern.finditer(document):
        if m.start() < last_end + 100:  # avoid overlapping/adjacent samples
            continue
        start = max(0, m.start() - chars_around)
        end = min(len(document), m.end() + chars_around)
        snippet = document[start:end].strip()
        # collapse whitespace for compactness
        snippet = re.sub(r"\s+", " ", snippet)
        contexts.append(snippet)
        last_end = m.end()
        if len(contexts) >= n_samples:
            break
    return contexts


def _slim_registry_payload(registry: dict) -> str:
    entries = []
    for e in registry["entities"]:
        item = {"canonical_name": e["canonical_name"]}
        if e.get("aliases"):
            item["aliases"] = e["aliases"]
        entries.append(item)
    return json.dumps(entries, ensure_ascii=False, indent=1)


def _format_candidates(candidates_with_contexts: List[Tuple[str, int, List[str]]]) -> str:
    lines = []
    for name, count, contexts in candidates_with_contexts:
        lines.append(f"Candidate: {name!r}  (mentions={count})")
        for i, ctx in enumerate(contexts, 1):
            lines.append(f"  context {i}: ...{ctx}...")
        lines.append("")
    return "\n".join(lines)


def verify_candidates(
    candidates: List[Tuple[str, int]],
    document: str,
    registry: dict,
    profile: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    n_samples: int = 3,
) -> List[dict]:
    """Run the LLM verifier on heuristic candidates. Returns decisions list."""
    if not candidates:
        return []

    settings = Settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set.")
    chosen_model = model or settings.extraction_model

    # Gather contexts for each candidate
    enriched: List[Tuple[str, int, List[str]]] = []
    for name, count in candidates:
        contexts = gather_contexts(name, document, n_samples=n_samples)
        if not contexts:
            continue  # candidate not findable; skip
        enriched.append((name, count, contexts))

    if not enriched:
        return []

    registry_text = _slim_registry_payload(registry)

    user_message = (
        f"<registry>\n{registry_text}\n</registry>\n\n"
        f"Decide what to do with each of the following {len(enriched)} candidates. "
        f"Output one decision per candidate via decide_candidates.\n\n"
        + _format_candidates(enriched)
    )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    print(f"Verifying {len(enriched)} candidates with {chosen_model}...", flush=True)

    with client.messages.stream(
        model=chosen_model,
        max_tokens=16000,
        system=_build_system_prompt(profile),
        messages=[{"role": "user", "content": user_message}],
        tools=[_DECIDE_CANDIDATES_TOOL],
        tool_choice={"type": "tool", "name": "decide_candidates"},
    ) as stream:
        response = stream.get_final_message()

    tool_use = next(
        (b for b in response.content if b.type == "tool_use" and b.name == "decide_candidates"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(f"Model did not call decide_candidates. Stop: {response.stop_reason}")

    decisions = tool_use.input.get("decisions", [])
    if not isinstance(decisions, list):
        raise RuntimeError(f"Model returned non-list decisions: {type(decisions).__name__}")

    print(f"Got {len(decisions)} decisions. Usage: input={response.usage.input_tokens} output={response.usage.output_tokens}")
    from .cost import report_call
    report_call("verify_candidates", chosen_model, response.usage)

    return decisions


def apply_decisions(registry: dict, decisions: List[dict], doc_tag: str) -> Dict[str, int]:
    """Apply verifier decisions to the registry, in place. Returns stats.

    Processes add_new actions before merge_into so that merges targeting
    newly-added entities resolve correctly.
    """
    stats = {"skipped": 0, "merged": 0, "added": 0, "invalid_target": 0, "invalid_payload": 0}
    canonical_index = {e["canonical_name"]: e for e in registry["entities"]}

    add_news = [d for d in decisions if isinstance(d, dict) and d.get("action") == "add_new"]
    merges = [d for d in decisions if isinstance(d, dict) and d.get("action") == "merge_into"]
    skips = [d for d in decisions if isinstance(d, dict) and d.get("action") == "skip"]
    invalid = [d for d in decisions if not isinstance(d, dict) or d.get("action") not in {"add_new", "merge_into", "skip"}]

    for d in add_news:
        new_entity = d.get("new_entity")
        if not isinstance(new_entity, dict) or not new_entity.get("canonical_name"):
            stats["invalid_payload"] += 1
            continue
        if new_entity["canonical_name"] in canonical_index:
            existing = canonical_index[new_entity["canonical_name"]]
            existing["aliases"] = list(dict.fromkeys((existing.get("aliases") or []) + (new_entity.get("aliases") or [])))
            stats["merged"] += 1
            continue
        new_entity = dict(new_entity)
        new_entity["source_documents"] = [doc_tag]
        registry["entities"].append(new_entity)
        canonical_index[new_entity["canonical_name"]] = new_entity
        stats["added"] += 1

    for d in merges:
        target_name = d.get("merge_target")
        candidate = d.get("candidate", "")
        if not target_name or target_name not in canonical_index:
            stats["invalid_target"] += 1
            continue
        target = canonical_index[target_name]
        aliases = target.get("aliases") or []
        if candidate and candidate not in aliases:
            aliases.append(candidate)
            target["aliases"] = aliases
        sources = target.get("source_documents") or []
        if doc_tag not in sources:
            sources.append(doc_tag)
            target["source_documents"] = sources
        stats["merged"] += 1

    stats["skipped"] = len(skips)
    stats["invalid_payload"] += len(invalid)
    return stats


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    parser = argparse.ArgumentParser(description="LLM-verify heuristic entity candidates against a registry.")
    parser.add_argument("--candidates", type=Path, required=True, help='Candidates JSON: [{"name": ..., "count": ...}, ...]')
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Where to write decisions JSON.")
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--top", type=int, default=80, help="Verify only the top N candidates by count.")
    parser.add_argument("--n-samples", type=int, default=3, help="Sample contexts per candidate.")
    parser.add_argument("--apply-to-registry", action="store_true", help="Apply decisions to the registry directly.")
    parser.add_argument("--doc-tag", type=str, default=None, help="Provenance tag if --apply-to-registry is set.")
    args = parser.parse_args()

    with args.candidates.open() as f:
        raw_candidates = json.load(f)
    candidates = [(c["name"], c["count"]) for c in raw_candidates[: args.top]]

    profile = load_corpus_profile(args.profile) if args.profile else None
    document = args.document.read_text()
    if profile:
        corrections = parse_ocr_corrections(profile)
        if corrections:
            document = apply_ocr_corrections(document, corrections)

    with args.registry.open() as f:
        registry = json.load(f)

    decisions = verify_candidates(
        candidates=candidates,
        document=document,
        registry=registry,
        profile=profile,
        model=args.model,
        n_samples=args.n_samples,
    )

    args.output.write_text(json.dumps(decisions, indent=2, ensure_ascii=False))
    print(f"Wrote decisions → {args.output}")

    # Summary
    from collections import Counter
    actions = Counter(d.get("action") for d in decisions if isinstance(d, dict))
    print(f"\nDecision counts:")
    for action, n in actions.most_common():
        print(f"  {n:>4}  {action}")

    if args.apply_to_registry:
        if not args.doc_tag:
            raise SystemExit("--doc-tag required with --apply-to-registry")
        stats = apply_decisions(registry, decisions, args.doc_tag)
        args.registry.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
        print(f"\nApplied to {args.registry}: {stats}")
        print(f"Registry now has {len(registry['entities'])} entities.")
