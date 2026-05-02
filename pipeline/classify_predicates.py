"""Predicate classification pass.

Reads `### Relationships from <doc>` blocks in entity .md files. For each
(target, quote, page) edge, asks the LLM to assign a predicate from a strict
enumerated vocabulary AND a direction (outgoing or incoming) — direction
encodes whether the file's entity is the predicate's natural source (outgoing)
or natural target (incoming). Writes back to the .md with a directional arrow.

Vocabulary (8 closed + 11 open + escape = 20 labels, enum-enforced via the
tool schema so the LLM cannot drift):

  Closed (state predicates with established meaning):
    LOCATED_IN, PORTAL_TO, BASED_AT,
    MEMBER_OF, LEADER_OF, RULER_OF, WORSHIPPER_OF, PROXY_OF

  Open active (subject acts on target):
    VISITS, POSSESSES, SERVES, OPPOSES, PROTECTS, KILLED, CREATED,
    ALLIED_WITH, ASSOCIATED_WITH

  Open state (subject IS something w.r.t. target):
    NATIVE_OF, IMPRISONED_AT

  Escape:
    _NO_EDGE  — quote doesn't establish a real relationship; post-processor drops the edge.

Line format on disk:
    - PREDICATE → [[Target]] — "quote" (p. N)        # outgoing — file is source
    - ← PREDICATE [[Target]] — "quote" (p. N)        # incoming — file is target

Batching: one batch request per `## Source: <doc>` section that has at least
one unclassified edge. The system prompt embeds the vocabulary + worked
examples (~5K tokens, above Haiku's 4096-token cache minimum) and is cached
with 1h TTL across all requests in the batch.

Usage:
    python -m planescapeRAG.pipeline.classify_predicates submit
    python -m planescapeRAG.pipeline.classify_predicates submit --random-sample 100
    python -m planescapeRAG.pipeline.classify_predicates collect <batch_id>
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import anthropic
import yaml
from dotenv import load_dotenv

from ..config import Settings
from ..cost import (
    compute_anthropic_cost,
    confirm_proceed,
    estimate_anthropic_batch_cost,
    log_cost,
    session_total,
)


_PREDICATES_YAML = Path("planescapeRAG/tools/predicates.yaml")
_ENTITIES_DIR = Path("planescapeRAG/graph/entities")
_BATCH_ID_FILE = Path("planescapeRAG/tmp/predicate_classify_batch_id.txt")
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# Vocabulary — kept in code so the tool schema can enforce it via JSON enum.
_CLOSED_PREDICATES = [
    "LOCATED_IN", "PORTAL_TO", "BASED_AT",
    "MEMBER_OF", "LEADER_OF", "RULER_OF",
    "WORSHIPPER_OF", "PROXY_OF",
]
_OPEN_ACTIVE_PREDICATES = [
    "VISITS", "POSSESSES", "SERVES", "OPPOSES", "PROTECTS",
    "KILLED", "CREATED", "ALLIED_WITH", "ASSOCIATED_WITH",
]
_OPEN_STATE_PREDICATES = [
    "NATIVE_OF", "IMPRISONED_AT",
]
_ESCAPE = "_NO_EDGE"
_ALL_PREDICATES = _CLOSED_PREDICATES + _OPEN_ACTIVE_PREDICATES + _OPEN_STATE_PREDICATES + [_ESCAPE]
_VALID_PREDICATES = set(_ALL_PREDICATES)
_DIRECTIONS = ["outgoing", "incoming"]
_SYMMETRIC = {"ALLIED_WITH", "ASSOCIATED_WITH"}  # direction is meaningless; canonicalize to outgoing


# Line format regexes. Three forms are recognized to support transition:
#  1. NEW_OUTGOING — `- PRED → [[X]] — "..." (p. N)`   (already classified, file is source)
#  2. NEW_INCOMING — `- ← PRED [[X]] — "..." (p. N)`   (already classified, file is target)
#  3. OLD          — `- [[X]] — "..." (p. N)`          (un-classified, from pass-2)
_REL_LINE_NEW_OUTGOING_RE = re.compile(r'^- ([A-Z][A-Z0-9_]*) → \[\[([^\]]+)\]\] — "(.*)" \(p\. (\d+)\)\s*$')
_REL_LINE_NEW_INCOMING_RE = re.compile(r'^- ← ([A-Z][A-Z0-9_]*) \[\[([^\]]+)\]\] — "(.*)" \(p\. (\d+)\)\s*$')
_REL_LINE_OLD_RE          = re.compile(r'^- \[\[([^\]]+)\]\] — "(.*)" \(p\. (\d+)\)\s*$')
_DOC_HEADER_RE = re.compile(r"^## Source:\s*(.+?)\s*$")
_REL_HEADER_RE = re.compile(r"^### Relationships from\s+(.+?)\s*$")


# ---------------------------------------------------------------- predicates --

def _load_closed_predicates_yaml() -> List[dict]:
    """Closed-vocab entries from predicates.yaml — used to build the prompt's
    detailed reference section. The actual enforcement is via the in-code list."""
    data = yaml.safe_load(_PREDICATES_YAML.read_text())
    return [p for p in data.get("predicates", []) if p.get("include")]


def build_system_prompt() -> str:
    """The cacheable system prompt. Must exceed 4096 tokens to enable caching on Haiku 4.5."""
    closed = _load_closed_predicates_yaml()
    lines: List[str] = []

    lines.extend([
        "You classify directed relationships in a Planescape RPG knowledge graph.",
        "",
        "INPUT: a SOURCE entity (the file being processed) and a list of edges.",
        "Each edge is a (target, quote, page) tuple — the quote was extracted from",
        "the source's document and mentions the target.",
        "",
        "OUTPUT: for each edge, you assign:",
        "  1. `predicate` — one of the enumerated labels below.",
        "  2. `direction` — either 'outgoing' or 'incoming'.",
        "",
        "## Direction",
        "",
        "Each predicate has a NATURAL direction: a 'subject' role and a 'target' role.",
        "  - `outgoing` means the SOURCE entity (the file's entity) plays the subject role,",
        "    and the TARGET plays the target role. Edge: SOURCE → predicate → TARGET.",
        "  - `incoming` means the SOURCE entity plays the target role, and the TARGET plays",
        "    the subject role. Edge: TARGET → predicate → SOURCE.",
        "",
        "Pick `direction` so the predicate reads naturally in its canonical orientation.",
        "DO NOT swap source/target — they are fixed by the input. The direction field",
        "is how you indicate which side of the relationship the source is on.",
        "",
        "Examples of direction:",
        "",
        "  Source = Razor Jack (character), edge target = Harmonium (faction):",
        "    quote: 'Razor Jack [Pl/F4/Harmonium/LE]'",
        "    predicate: MEMBER_OF (subject = member, target = group)",
        "    direction: outgoing  (Razor Jack IS the member; Harmonium IS the group)",
        "",
        "  Source = Harmonium (faction), edge target = Sarin (character):",
        "    quote: 'Sarin, factol of the Harmonium'",
        "    predicate: LEADER_OF (subject = leader, target = group)",
        "    direction: incoming  (Sarin IS the leader; Harmonium IS the group)",
        "",
        "  Source = Sword of Vengeance (item), edge target = Hephaestus (character):",
        "    quote: 'Hephaestus forged the Sword of Vengeance'",
        "    predicate: CREATED (subject = creator, target = creation)",
        "    direction: incoming  (Hephaestus is the creator; the Sword is the creation)",
        "",
        "  Source = Sigil (place), edge target = Lower Ward (place):",
        "    quote: 'the Lower Ward, one of Sigil's six wards'",
        "    predicate: LOCATED_IN (subject = inner, target = outer)",
        "    direction: incoming  (Lower Ward is the inner; Sigil is the outer container)",
        "",
        "## Predicate vocabulary (ENUM-ENFORCED — pick one of these exactly)",
        "",
        "### Closed predicates (high-precision, well-defined relationships)",
        "",
        "- LOCATED_IN — structural containment (inner → outer). 'A is LOCATED_IN B' means",
        "  A is physically/structurally inside B. Layers, wards, buildings inside cities,",
        "  cities inside planes. Transitive — supports `LOCATED_IN*` graph queries.",
        "",
        "- PORTAL_TO — directed gateway (A is the gateway/origin → B is the destination).",
        "  'A PORTAL_TO B' means A has a portal/gate to B.",
        "",
        "- BASED_AT — primary residence or HQ (entity → place). 'A BASED_AT B' means",
        "  A's home or headquarters is B. Use for character residence and faction HQ.",
        "  Single primary location per entity (a character with multiple homes uses VISITS",
        "  for the others). NOT for transient presence.",
        "",
        "- MEMBER_OF — formal organizational membership (member → group). The faction tag",
        "  in stat blocks ('[Pl/F4/Harmonium/LN]') is the strongest signal.",
        "",
        "- LEADER_OF — leads an organization (leader → group). Factols, heads, founders",
        "  who currently lead. Distinct from RULER_OF (which is for places, not orgs).",
        "",
        "- RULER_OF — sovereign over a place or realm (ruler → domain). Bel rules",
        "  Avernus; Asmodeus rules Nessus. Reserve for clear sovereignty — the Lady of",
        "  Pain is NOT RULER_OF Sigil (use _NO_EDGE; she presides but doesn't rule).",
        "",
        "- WORSHIPPER_OF — devotion to a deity or power (devotee → power). For",
        "  organizations whose stance is REJECTION (Athar reject the powers), use OPPOSES",
        "  instead — denial is not worship.",
        "",
        "- PROXY_OF — sanctioned divine agent (proxy → power). A proxy is a chosen",
        "  servant who acts on the power's behalf, distinct from an ordinary worshipper.",
        "",
        "### Open active predicates (SOURCE acts on TARGET)",
        "",
        "- VISITS — transient presence (visitor → place or entity visited). Use for",
        "  characters who travel to / appear in places without it being their home.",
        "  Replaces all of: VISITED, FREQUENTS, INHABITS, TRAVELS_THROUGH.",
        "",
        "- POSSESSES — owns / has / wields / is equipped with (owner → item).",
        "  Replaces all of: OWNS, EQUIPPED_WITH, WIELDS, HAS, CARRIES.",
        "  Whether it's a weapon, an artifact, mundane gear — all POSSESSES.",
        "",
        "- SERVES — works for / commanded by (servant → master). Use when the",
        "  relationship is 'works for' but not formal MEMBER_OF and not PROXY_OF.",
        "  Note: for the inverse ('master commands subordinates'), use SERVES with",
        "  direction=incoming, not a separate COMMANDS predicate.",
        "",
        "- OPPOSES — declared antagonism (opposer → opposed). Faction rivalry, personal",
        "  feud, declared hostility, attacks, betrayals, war. The Blood War is OPPOSES",
        "  in both directions between baatezu and tanar'ri (emit two edges if both",
        "  perspectives appear). Replaces: ATTACKED, BETRAYED, HOSTILE_TO, FIGHTS.",
        "",
        "- PROTECTS — guards / defends / watches over (protector → protected).",
        "  Replaces: GUARDS, DEFENDS, WATCHES_OVER.",
        "",
        "- KILLED — slew / destroyed (killer → killed). Past-tense — the event happened.",
        "  Distinct from OPPOSES: opposing is ongoing, killed is permanent.",
        "",
        "- CREATED — made / forged / built / founded (creator → creation). Replaces:",
        "  MADE_BY, FORGED, BUILT, FOUNDED, ESTABLISHED. For 'A was created by B',",
        "  use CREATED with direction=incoming.",
        "",
        "- ALLIED_WITH — declared alliance, symmetric. Use direction=outgoing only",
        "  (the relation is bidirectional; no need to emit both directions).",
        "",
        "- ASSOCIATED_WITH — LAST RESORT generic relation, symmetric. Use ONLY when",
        "  no other predicate fits AND the quote does establish some real connection.",
        "  If the quote is just incidental mention with no real relationship, use",
        "  _NO_EDGE instead. Use direction=outgoing only.",
        "",
        "### Open state predicates (SOURCE IS something — direction always outgoing)",
        "",
        "- NATIVE_OF — origin/birthplace (subject is native of target). For characters,",
        "  the world or plane they came from. For creatures, their plane of origin.",
        "",
        "- IMPRISONED_AT — held captive (subject is imprisoned at the target location).",
        "",
        "### Escape",
        "",
        "- _NO_EDGE — emit when the quote does NOT establish a real directed relationship",
        "  between the source and target. Common cases:",
        "    * Quote is a stat block / type description (no relational verb)",
        "    * Quote describes other entities, target is mentioned in passing",
        "    * Source is mentioned briefly, no semantic link to target in the quote",
        "    * Type-only descriptions ('X is a tiefling', 'X is a hamatula')",
        "  The post-processor will drop these edges from the graph entirely.",
        "",
        "## Worked examples",
        "",
        "Format below: `[Source (type)] target = X | quote | predicate | direction`.",
        "",
        "[Razor Jack (character)] target=Harmonium | 'Razor Jack [Pl/F4/Harmonium/LE]'",
        "  → MEMBER_OF, outgoing",
        "",
        "[Harmonium (faction)] target=Razor Jack | 'A dashing tiefling named Razor Jack",
        "  [Pl/F4/Harmonium/LE]' → MEMBER_OF, incoming  (RJ is the member, Harmonium the group)",
        "",
        "[Sarin (character)] target=Harmonium | 'Sarin, factol of the Harmonium'",
        "  → LEADER_OF, outgoing",
        "",
        "[Harmonium (faction)] target=Sarin | 'Sarin, factol of the Harmonium'",
        "  → LEADER_OF, incoming",
        "",
        "[Bel (character)] target=Avernus | 'Bel rules Avernus from his iron tower'",
        "  → RULER_OF, outgoing",
        "",
        "[Vault (place)] target=Colothys | 'the Vault rests atop one of Colothys' peaks'",
        "  → LOCATED_IN, outgoing  (Vault is in Colothys)",
        "",
        "[Sigil (place)] target=Lower Ward | 'the Lower Ward, one of Sigil's six wards'",
        "  → LOCATED_IN, incoming  (Lower Ward is in Sigil)",
        "",
        "[Naratyr (place)] target=Acheron | 'a portal in Naratyr to Acheron'",
        "  → PORTAL_TO, outgoing",
        "",
        "[Harmonium (faction)] target=City Barracks | 'the Harmonium operates from the",
        "  City Barracks' → BASED_AT, outgoing",
        "",
        "[Athar (faction)] target=powers | 'the Athar deny the divinity of the powers'",
        "  → OPPOSES, outgoing  (denial = opposition; NOT WORSHIPPER_OF)",
        "",
        "[Hvix'mnac (character)] target=TorNav'roc | 'Hvix'mnac arrives in TorNav'roc'",
        "  → VISITS, outgoing  (arrival, not residence)",
        "",
        "[Razor Jack (character)] target=vorpal sword | 'Razor Jack wields a vorpal sword'",
        "  → POSSESSES, outgoing",
        "",
        "[Yisifel (character)] target=Throne Archon | 'a former throne archon who",
        "  voluntarily accepted demotion' → _NO_EDGE  (type/role description, not a relationship)",
        "",
        "[Halitsu (character)] target=spinagon | 'Halitsu, a least baatezu",
        "  (Planar/♀ spinagon/HD 3+3/LE)' → _NO_EDGE  (stat block — type, not relationship)",
        "",
        "[Hephaestus (character)] target=Sword of Vengeance | 'Hephaestus forged the Sword'",
        "  → CREATED, outgoing",
        "",
        "[Sword of Vengeance (item)] target=Hephaestus | 'Hephaestus forged the Sword'",
        "  → CREATED, incoming",
        "",
        "[Vault (place)] target=Harmonium | 'the three lawgiving factions created the Vault'",
        "  → CREATED, incoming",
        "",
        "[Cyrus (character)] target=Carceri | 'Cyrus is imprisoned at the Vault on Carceri'",
        "  → IMPRISONED_AT, outgoing",
        "",
        "[8 (character)] target=Hamatula | 'One of the hamatula defends Nihmron while he",
        "  casts spells' → _NO_EDGE  (quote describes Nihmron, not the source '8'; '8' isn't",
        "  even mentioned. Pass-2 noise — drop.)",
        "",
        "[The Dustmen (faction)] target=Naratyr | 'most customers are Dustmen' (in",
        "  Naratyr's tavern) → ASSOCIATED_WITH, outgoing  (real link but no specific predicate)",
        "",
        "## Critical constraints",
        "",
        "- TARGET is fixed by the input. Never invent or rename targets.",
        "- QUOTE is fixed. Don't echo it back; reference by index.",
        "- Output exactly N classifications for N input edges, in the same order.",
        "- `predicate` MUST be one of the 20 enum values listed above. Anything else is invalid.",
        "- `direction` MUST be 'outgoing' or 'incoming'.",
        "- For symmetric (ALLIED_WITH, ASSOCIATED_WITH) and pure-state (NATIVE_OF,",
        "  IMPRISONED_AT) predicates, always use direction=outgoing.",
        "- When in doubt between a real predicate and _NO_EDGE: prefer _NO_EDGE if the",
        "  quote doesn't actually establish the relationship. We'd rather lose a weak",
        "  edge than pollute the graph with noise.",
        "- When in doubt between two real predicates: pick the more specific one. Use",
        "  ASSOCIATED_WITH only if NOTHING else fits.",
        "",
        "Call the `classify_edges` tool.",
        "",
        "## Reference: closed predicate definitions (verbatim from predicates.yaml)",
        "",
        "(For your reference. The summary above is what you should use day-to-day.)",
        "",
        "```yaml",
        _PREDICATES_YAML.read_text(),
        "```",
    ])

    return "\n".join(lines)


_CLASSIFY_TOOL = {
    "name": "classify_edges",
    "description": "Submit predicate classifications for the input edges. One entry per input edge, in the same order.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "0-based index of the input edge being classified.",
                        },
                        "predicate": {
                            "type": "string",
                            "enum": _ALL_PREDICATES,
                            "description": "Closed/open predicate label, or _NO_EDGE to drop the edge.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": _DIRECTIONS,
                            "description": (
                                "'outgoing' = source is the predicate's natural subject. "
                                "'incoming' = source is the predicate's natural target. "
                                "For symmetric and pure-state predicates, always 'outgoing'."
                            ),
                        },
                    },
                    "required": ["index", "predicate", "direction"],
                },
            },
        },
        "required": ["classifications"],
    },
}


# ---------------------------------------------------------------- file walker --

def _parse_file(path: Path) -> Optional[Tuple[List[str], dict, List[Tuple[str, List[str], int, int]]]]:
    """Return (full_lines, frontmatter_dict, sections).
    sections: list of (doc_tag, body_lines, body_start_line_index, body_end_line_index)
    """
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    fm_end = text.find("\n---\n", 4)
    body_start_offset: int
    if fm_end != -1:
        fm_text = text[4:fm_end]
        body_start_offset = fm_end + 5
    else:
        fm_end = text.find("\n---", 4)
        if fm_end == -1:
            return None
        fm_text = text[4:fm_end]
        body_start_offset = fm_end + 4

    fm: Dict[str, object] = {}
    for line in fm_text.splitlines():
        m = re.match(r"^([a-zA-Z_]+):\s*(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if v.startswith("[") or v.startswith("{"):
            try:
                fm[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        fm[k] = v

    full_lines = text.splitlines()
    body_line_start = text[:body_start_offset].count("\n")

    sections: List[Tuple[str, List[str], int, int]] = []
    current_tag: Optional[str] = None
    current_lines: List[str] = []
    current_start = body_line_start
    for i in range(body_line_start, len(full_lines)):
        line = full_lines[i]
        m = _DOC_HEADER_RE.match(line)
        if m:
            if current_tag is not None:
                sections.append((current_tag, current_lines, current_start, i))
            current_tag = m.group(1).strip()
            current_lines = []
            current_start = i + 1
        elif current_tag is not None:
            current_lines.append(line)
    if current_tag is not None:
        sections.append((current_tag, current_lines, current_start, len(full_lines)))

    return full_lines, fm, sections


def _parse_section_edges(section_lines: List[str]) -> Tuple[List[dict], bool, List[int]]:
    """Returns (edges, all_classified, line_indices_within_section).

    edges: list of {predicate, direction, target, quote, page}. predicate=None
           and direction=None mean unclassified.
    """
    edges: List[dict] = []
    line_indices: List[int] = []
    in_rel_block = False
    all_classified = True
    for i, line in enumerate(section_lines):
        if _REL_HEADER_RE.match(line):
            in_rel_block = True
            continue
        if not in_rel_block:
            continue
        m = _REL_LINE_NEW_OUTGOING_RE.match(line)
        if m:
            edges.append({"predicate": m.group(1), "direction": "outgoing",
                          "target": m.group(2), "quote": m.group(3), "page": int(m.group(4))})
            line_indices.append(i)
            continue
        m = _REL_LINE_NEW_INCOMING_RE.match(line)
        if m:
            edges.append({"predicate": m.group(1), "direction": "incoming",
                          "target": m.group(2), "quote": m.group(3), "page": int(m.group(4))})
            line_indices.append(i)
            continue
        m = _REL_LINE_OLD_RE.match(line)
        if m:
            edges.append({"predicate": None, "direction": None,
                          "target": m.group(1), "quote": m.group(2), "page": int(m.group(3))})
            line_indices.append(i)
            all_classified = False
    return edges, all_classified, line_indices


def _walk_unclassified_sections() -> Iterator[Tuple[Path, str, str, str, List[dict], List[int], List[str], int, int]]:
    """Yield one tuple per section that has at least one unclassified edge."""
    for path in sorted(_ENTITIES_DIR.rglob("*.md")):
        parsed = _parse_file(path)
        if parsed is None:
            continue
        full_lines, fm, sections = parsed
        canonical = str(fm.get("canonical_name") or path.stem)
        rel = path.relative_to(_ENTITIES_DIR)
        entity_id = str(rel.with_suffix("")).replace("\\", "/")
        for doc_tag, sec_lines, start, end in sections:
            edges, all_classified, line_indices = _parse_section_edges(sec_lines)
            if not edges or all_classified:
                continue
            yield (path, entity_id, canonical, doc_tag, edges, line_indices, sec_lines, start, end)


# ---------------------------------------------------------------- request build --

def _slug_from_section(entity_id: str, doc_tag: str) -> str:
    safe = entity_id.replace("/", "__") + "__" + re.sub(r"[^a-zA-Z0-9_]+", "-", doc_tag)
    return safe[:64]


def _build_request(
    entity_id: str,
    canonical: str,
    entity_type: str,
    doc_tag: str,
    edges: List[dict],
    model: str,
) -> dict:
    edge_payload = [
        {"index": i, "target": e["target"], "quote": e["quote"], "page": e["page"]}
        for i, e in enumerate(edges)
    ]
    user_text = (
        f"SOURCE entity: {canonical}\n"
        f"SOURCE type: {entity_type}\n"
        f"SOURCE document: {doc_tag}\n\n"
        f"Classify each edge below. Return one entry per edge, same order.\n\n"
        f"<edges>\n{json.dumps(edge_payload, ensure_ascii=False, indent=2)}\n</edges>"
    )
    return {
        "custom_id": _slug_from_section(entity_id, doc_tag),
        "params": {
            "model": model,
            "max_tokens": 2000,
            "system": [
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ],
            "tools": [_CLASSIFY_TOOL],
            "tool_choice": {"type": "tool", "name": "classify_edges"},
        },
    }


_SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------- write back --

def _format_rel_line(predicate: str, direction: str, target: str, quote: str, page: int) -> str:
    if direction == "incoming":
        return f'- ← {predicate} [[{target}]] — "{quote}" (p. {page})'
    return f'- {predicate} → [[{target}]] — "{quote}" (p. {page})'


def _rewrite_section_lines(
    section_lines: List[str],
    line_indices: List[int],
    edges: List[dict],
    classifications: Dict[int, Tuple[str, str]],
) -> Tuple[List[str], int]:
    """Returns (new_lines, n_dropped). Lines marked _NO_EDGE are removed entirely."""
    drop_line_set: set = set()
    new_lines = list(section_lines)
    for k, edge in enumerate(edges):
        cls = classifications.get(k)
        if cls is None:
            continue  # leave existing line untouched
        predicate, direction = cls
        if predicate not in _VALID_PREDICATES:
            continue  # invalid predicate — leave line, will retry next run
        if predicate == _ESCAPE:
            drop_line_set.add(line_indices[k])
            continue
        # Force outgoing direction for symmetric and pure-state predicates.
        if predicate in _SYMMETRIC or predicate in _OPEN_STATE_PREDICATES:
            direction = "outgoing"
        if direction not in _DIRECTIONS:
            direction = "outgoing"
        new_lines[line_indices[k]] = _format_rel_line(
            predicate, direction, edge["target"], edge["quote"], edge["page"]
        )
    if drop_line_set:
        new_lines = [ln for i, ln in enumerate(new_lines) if i not in drop_line_set]
    return new_lines, len(drop_line_set)


# ---------------------------------------------------------------- submit --

def submit_classification(
    model: str,
    auto_yes: bool,
    cost_threshold: float,
    dry_run: bool = False,
    limit: Optional[int] = None,
    random_sample: Optional[int] = None,
    random_seed: int = 42,
) -> str:
    load_dotenv()
    settings = Settings()

    requests: List[dict] = []
    section_count = 0
    edge_count = 0

    registry = json.loads(Path("planescapeRAG/graph/registry.json").read_text())
    by_canonical = {e["canonical_name"]: e for e in registry["entities"]}

    candidates = list(_walk_unclassified_sections())
    if random_sample is not None and random_sample < len(candidates):
        import random as _random
        rng = _random.Random(random_seed)
        candidates = rng.sample(candidates, random_sample)

    for path, entity_id, canonical, doc_tag, edges, _, _, _, _ in candidates:
        unclassified = [e for e in edges if e["predicate"] is None]
        edge_count += len(unclassified)
        section_count += 1
        entity_type = (by_canonical.get(canonical) or {}).get("type", "unknown")
        requests.append(_build_request(entity_id, canonical, entity_type, doc_tag, edges, model))
        if limit is not None and len(requests) >= limit:
            break

    if not requests:
        print("Nothing to classify — every section's edges already carry predicates.")
        return ""

    print(f"Built {len(requests)} requests across {section_count} sections, {edge_count} unclassified edges.")
    estimate = estimate_anthropic_batch_cost(
        requests, model=model, cache_ttl="1h", output_tokens_per_request=300,
    )
    print(f"\nCost estimate (Haiku batch, with caching):")
    print(f"  uncached input tokens: {estimate.get('uncached_input_tokens', 0):>10,}")
    print(f"  cache write tokens:    {estimate.get('cache_write_tokens', 0):>10,}")
    print(f"  cache read tokens:     {estimate.get('cache_read_tokens', 0):>10,}")
    print(f"  output (estimate):     {estimate.get('output_tokens_estimate', 0):>10,}")
    print(f"  estimated cost:        ${estimate.get('cost_usd') or 0:.2f}")

    if dry_run:
        print("\n(--dry-run set; not submitting.)")
        return ""

    if not confirm_proceed(estimate, "predicate classification submit", threshold_usd=cost_threshold, auto_yes=auto_yes):
        print("Aborted.")
        return ""

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Pre-warm the prompt cache. Verified empirically (tools/test_cache_prewarm.py)
    # that a single non-batch call writes a cache the subsequent batch can read,
    # avoiding the per-worker cache-write overhead (~$8 savings on 10K-req batches).
    print("Pre-warming prompt cache...")
    warm = client.messages.create(
        model=model, max_tokens=10,
        system=[{"type": "text", "text": _SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        messages=[{"role": "user", "content": "ack"}],
    )
    cw = getattr(warm.usage, "cache_creation_input_tokens", 0) or 0
    print(f"  warm-up cache_write={cw} tokens (one-time, non-batch rate)")

    batch = client.messages.batches.create(requests=requests)
    print(f"\nSubmitted batch: {batch.id}  status={batch.processing_status}  count={len(requests)}")

    _BATCH_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BATCH_ID_FILE.write_text(batch.id)

    log_cost(
        "predicate_classify_submit_estimate",
        model,
        cost_usd=None,
        is_batch=True,
        details={
            "batch_id": batch.id,
            "n_requests": len(requests),
            "n_sections": section_count,
            "n_edges": edge_count,
            "estimate_usd": estimate.get("cost_usd"),
            **{k: v for k, v in estimate.items() if k not in ("cost_usd", "n_requests")},
        },
    )
    return batch.id


# ---------------------------------------------------------------- collect --

def collect_classification(batch_id: str) -> None:
    load_dotenv()
    settings = Settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    print(f"Polling batch {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"  status={batch.processing_status}  succeeded={c.succeeded}  errored={c.errored}  processing={c.processing}", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(60)

    work_index: Dict[str, Tuple[Path, str, str, List[dict], List[int]]] = {}
    for path, entity_id, canonical, doc_tag, edges, line_indices, _, _, _ in _walk_unclassified_sections():
        cid = _slug_from_section(entity_id, doc_tag)
        work_index[cid] = (path, entity_id, doc_tag, edges, line_indices)

    classifications_by_cid: Dict[str, Dict[int, Tuple[str, str]]] = {}
    failures: List[str] = []
    invalid_predicate_counts: Dict[str, int] = {}
    total_cost = 0.0
    total_input = total_output = total_cw = total_cr = 0

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type != "succeeded":
            failures.append(f"{cid}: {result.result.type}")
            continue
        message = result.result.message
        usage = message.usage
        cost = compute_anthropic_cost(usage, message.model, is_batch=True, cache_ttl="1h") or 0.0
        total_cost += cost
        total_input += getattr(usage, "input_tokens", 0) or 0
        total_output += getattr(usage, "output_tokens", 0) or 0
        total_cw += getattr(usage, "cache_creation_input_tokens", 0) or 0
        total_cr += getattr(usage, "cache_read_input_tokens", 0) or 0

        tool_use = next(
            (b for b in message.content if b.type == "tool_use" and b.name == "classify_edges"),
            None,
        )
        if tool_use is None:
            failures.append(f"{cid}: no tool_use returned")
            continue
        items = tool_use.input.get("classifications", [])
        if not isinstance(items, list):
            failures.append(f"{cid}: classifications not a list")
            continue
        cls: Dict[int, Tuple[str, str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            pred = item.get("predicate")
            direction = item.get("direction", "outgoing")
            if not isinstance(idx, int) or not isinstance(pred, str):
                continue
            pred_normalized = pred.strip().upper()
            if pred_normalized not in _VALID_PREDICATES:
                invalid_predicate_counts[pred_normalized] = invalid_predicate_counts.get(pred_normalized, 0) + 1
                continue
            if direction not in _DIRECTIONS:
                direction = "outgoing"
            cls[idx] = (pred_normalized, direction)
        classifications_by_cid[cid] = cls

    files_touched: Dict[Path, List[Tuple[str, Dict[int, Tuple[str, str]]]]] = {}
    for cid, cls in classifications_by_cid.items():
        if cid not in work_index:
            failures.append(f"{cid}: no matching section in current entity files (file changed?)")
            continue
        path, _, doc_tag, _, _ = work_index[cid]
        files_touched.setdefault(path, []).append((doc_tag, cls))

    rewritten = 0
    edges_rewritten = 0
    edges_dropped = 0
    for path, doc_classifications in files_touched.items():
        parsed = _parse_file(path)
        if parsed is None:
            failures.append(f"{path}: file unparseable on rewrite")
            continue
        full_lines, _, sections = parsed
        new_full = list(full_lines)
        # Apply each section's rewrite. Recompute sections after each splice to handle line count changes from _NO_EDGE drops.
        for doc_tag, cls in doc_classifications:
            # Re-parse from the current new_full so spans are accurate.
            sections_now = _parse_file_lines(new_full)
            for sec_doc, sec_lines, start, end in sections_now:
                if sec_doc != doc_tag:
                    continue
                edges, _, line_indices = _parse_section_edges(sec_lines)
                rewritten_lines, n_drops = _rewrite_section_lines(sec_lines, line_indices, edges, cls)
                edges_rewritten += sum(1 for _ in cls)
                edges_dropped += n_drops
                new_full[start:end] = rewritten_lines
                break
        path.write_text("\n".join(new_full) + "\n")
        rewritten += 1

    log_cost(
        "predicate_classify_collect",
        "claude-haiku-4-5-20251001",
        cost_usd=total_cost,
        is_batch=True,
        details={
            "batch_id": batch_id,
            "n_results": len(classifications_by_cid),
            "n_failures": len(failures),
            "files_rewritten": rewritten,
            "edges_rewritten": edges_rewritten,
            "edges_dropped_no_edge": edges_dropped,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_write_tokens": total_cw,
            "cache_read_tokens": total_cr,
            "invalid_predicates": invalid_predicate_counts,
        },
    )

    print(f"\nFiles rewritten: {rewritten}")
    print(f"Edges rewritten: {edges_rewritten}")
    print(f"Edges dropped (_NO_EDGE): {edges_dropped}")
    print(f"Failures: {len(failures)}")
    if invalid_predicate_counts:
        print(f"Invalid predicates returned (should be 0 with enum enforcement):")
        for p, n in invalid_predicate_counts.items():
            print(f"  {p!r}: {n}")
    for f in failures[:10]:
        print(f"  {f}")
    print(f"\nTotal batch cost: ${total_cost:.4f}")
    print(f"Tokens: input={total_input:,} output={total_output:,} cache_write={total_cw:,} cache_read={total_cr:,}")
    print(f"Session total so far: ${session_total():.2f}")


def _parse_file_lines(full_lines: List[str]) -> List[Tuple[str, List[str], int, int]]:
    """Re-derive section spans from a list of body lines. Used after rewrites so
    subsequent section edits use accurate line indices."""
    # Find body start (after closing ---)
    body_start = 0
    saw_open = False
    for i, line in enumerate(full_lines):
        if line.strip() == "---":
            if not saw_open:
                saw_open = True
            else:
                body_start = i + 1
                break
        elif i > 0 and not saw_open:
            break
    sections: List[Tuple[str, List[str], int, int]] = []
    current_tag = None
    current_lines: List[str] = []
    current_start = body_start
    for i in range(body_start, len(full_lines)):
        line = full_lines[i]
        m = _DOC_HEADER_RE.match(line)
        if m:
            if current_tag is not None:
                sections.append((current_tag, current_lines, current_start, i))
            current_tag = m.group(1).strip()
            current_lines = []
            current_start = i + 1
        elif current_tag is not None:
            current_lines.append(line)
    if current_tag is not None:
        sections.append((current_tag, current_lines, current_start, len(full_lines)))
    return sections


# ---------------------------------------------------------------- main --

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify entity-relationship predicates.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="Build batch and submit.")
    p_submit.add_argument("--model", default=_DEFAULT_MODEL)
    p_submit.add_argument("--yes", action="store_true")
    p_submit.add_argument("--cost-threshold", type=float, default=5.0)
    p_submit.add_argument("--dry-run", action="store_true", help="Print estimate without submitting.")
    p_submit.add_argument("--limit", type=int, default=None, help="Cap the request count (smoke testing).")
    p_submit.add_argument("--random-sample", type=int, default=None, help="Randomly sample N sections from the unclassified pool (smoke testing).")
    p_submit.add_argument("--random-seed", type=int, default=42)

    p_collect = sub.add_parser("collect", help="Poll and apply classifications from a batch.")
    p_collect.add_argument("batch_id", nargs="?", help="Defaults to id stored in tmp/predicate_classify_batch_id.txt")

    p_preview = sub.add_parser("preview", help="Show the system prompt and one example request without submitting.")

    args = parser.parse_args()

    if args.cmd == "submit":
        submit_classification(
            args.model, args.yes, args.cost_threshold,
            dry_run=args.dry_run, limit=args.limit,
            random_sample=args.random_sample, random_seed=args.random_seed,
        )
    elif args.cmd == "collect":
        bid = args.batch_id or (_BATCH_ID_FILE.read_text().strip() if _BATCH_ID_FILE.exists() else None)
        if not bid:
            raise SystemExit("No batch_id provided and none stored.")
        collect_classification(bid)
    elif args.cmd == "preview":
        sys_tokens_est = len(_SYSTEM_PROMPT) // 4
        print(f"System prompt: {len(_SYSTEM_PROMPT)} chars (~{sys_tokens_est} tokens)")
        registry = json.loads(Path("planescapeRAG/graph/registry.json").read_text())
        by_canonical = {e["canonical_name"]: e for e in registry["entities"]}
        for path, entity_id, canonical, doc_tag, edges, _, _, _, _ in _walk_unclassified_sections():
            entity_type = (by_canonical.get(canonical) or {}).get("type", "unknown")
            req = _build_request(entity_id, canonical, entity_type, doc_tag, edges, _DEFAULT_MODEL)
            print(f"\n--- example request: {req['custom_id']} ---")
            print(f"file: {path}")
            print(f"section: {doc_tag}")
            print(f"edges: {len(edges)}")
            print("user message preview:")
            print(req["params"]["messages"][0]["content"][0]["text"][:1200])
            break


if __name__ == "__main__":
    main()
