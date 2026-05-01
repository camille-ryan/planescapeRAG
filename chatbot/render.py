"""Shared HTML rendering helpers for the FastAPI/HTMX UI.

- Linkify [[wikilinks]] to entity URLs (HTMX-augmented).
- Linkify (doc_tag, p. N) citations to source-page URLs (gated).
- Build the pyvis interactive graph for a retrieval result.
- Convert the resulting markdown-with-inline-HTML to safe HTML.
"""

import re
from typing import Dict, List

import markdown as md_lib

from .retrieve import RetrievalResult


_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_CITATION_RE = re.compile(r"\(([a-zA-Z][\w]*),\s*p\.\s*(\d+)\)")
_LEADING_H1_RE = re.compile(r"^# .*\n+")


def linkify_wikilinks(text: str, name_to_id: Dict[str, str]) -> str:
    """Replace `[[Name]]` with an HTMX-augmented anchor that loads the entity
    fragment into #entity-panel without a full page reload. Names that don't
    resolve in the corpus render as plain text (brackets stripped)."""
    def repl(m: re.Match) -> str:
        name = m.group(1).strip()
        eid = name_to_id.get(name.lower())
        if not eid:
            return name
        return (
            f'<a href="/entity/{eid}" '
            f'hx-get="/entity/{eid}" '
            f'hx-target="#entity-panel" '
            f'hx-swap="innerHTML" '
            f'hx-push-url="true" '
            f'class="wikilink">{name}</a>'
        )
    return _WIKILINK_RE.sub(repl, text)


def linkify_citations(text: str, api_base: str, show_pages: bool) -> str:
    """Replace `(doc_tag, p. N)` with an `<a target=_blank>` to the page image.
    No-op when show_pages is False (the link target would 404)."""
    if not show_pages:
        return text
    def repl(m: re.Match) -> str:
        tag, page = m.group(1), m.group(2)
        return (
            f'<a href="{api_base}/page/{tag}/{page}" '
            f'target="_blank" class="citation">({tag}, p. {page})</a>'
        )
    return _CITATION_RE.sub(repl, text)


def render_body_to_html(
    text: str,
    name_to_id: Dict[str, str],
    api_base: str,
    show_pages: bool,
    *,
    strip_leading_h1: bool = False,
) -> str:
    """Linkify wikilinks + citations, then convert markdown to HTML.

    Order matters: linkify runs on the raw markdown text so the regex sees
    `[[Name]]` and `(tag, p.N)` cleanly. The resulting `<a>` tags pass through
    python-markdown as inline HTML.
    """
    if strip_leading_h1:
        text = _LEADING_H1_RE.sub("", text, count=1)
    text = linkify_wikilinks(text, name_to_id)
    text = linkify_citations(text, api_base, show_pages)
    md = md_lib.Markdown(extensions=["extra", "sane_lists"])
    return md.convert(text)


def build_pyvis_html(retrieval: RetrievalResult) -> str:
    """Render the retrieved subgraph as an interactive pyvis network HTML."""
    from pyvis.network import Network

    retrieved = retrieval.all_entities

    net = Network(
        height="520px", width="100%",
        bgcolor="#ffffff", font_color="#222222",
        directed=True, notebook=False,
    )

    name_to_id = {e.canonical_name.lower(): e.entity_id for e in retrieved}
    ids_in_set = {e.entity_id for e in retrieved}

    for e in retrieved:
        if e.is_primary:
            color = "#1f77b4"
            size = 20 + 30 * max(0.0, e.similarity)
            tooltip = (
                f"{e.canonical_name} ({e.type})\\n"
                f"primary · sim={e.similarity:.3f}"
            )
        else:
            color = "#bbbbbb"
            size = 14
            tooltip = (
                f"{e.canonical_name} ({e.type})\\n"
                f"expansion ← {e.expanded_from}"
            )
        net.add_node(
            e.entity_id,
            label=e.canonical_name,
            title=tooltip,
            color=color,
            size=size,
        )

    for e in retrieved:
        for m in _WIKILINK_RE.finditer(e.body or ""):
            target_id = name_to_id.get(m.group(1).strip().lower())
            if target_id and target_id != e.entity_id and target_id in ids_in_set:
                net.add_edge(e.entity_id, target_id)

    net.set_options("""
    {
      "nodes": {
        "font": {"size": 16, "face": "Arial", "color": "#222222", "strokeWidth": 4, "strokeColor": "#ffffff"},
        "borderWidth": 1
      },
      "edges": {
        "smooth": {"type": "continuous"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "color": {"color": "#888888"}
      },
      "physics": {
        "barnesHut": {"gravitationalConstant": -8000, "springLength": 140, "springConstant": 0.04},
        "stabilization": {"iterations": 200}
      },
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)

    return net.generate_html(notebook=False)


def entity_summary_dicts(retrieval: RetrievalResult) -> List[dict]:
    """Flatten primaries + expansions into the dicts the templates expect."""
    rows: List[dict] = []
    for e in retrieval.primaries:
        rows.append({
            "entity_id": e.entity_id,
            "canonical_name": e.canonical_name,
            "type": e.type,
            "is_primary": True,
            "similarity": e.similarity,
            "badge": f"primary · sim={e.similarity:.3f}",
        })
    for e in retrieval.expansions:
        rows.append({
            "entity_id": e.entity_id,
            "canonical_name": e.canonical_name,
            "type": e.type,
            "is_primary": False,
            "similarity": 0.0,
            "badge": f"1-hop ← {e.expanded_from}",
        })
    return rows
