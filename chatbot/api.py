"""FastAPI server: serves the HTMX-driven HTML UI and the JSON API.

HTML endpoints (HTMX-friendly):
  GET  /                       — main page
  POST /chat                   — answer fragment for a question (form-encoded)
  GET  /entity/{entity_id:path}— entity-detail fragment for any entity in the corpus
  GET  /graph/{retrieval_id}   — pyvis interactive graph for a stored retrieval

JSON / utility endpoints:
  GET  /health                 — liveness + indexed-entity count
  GET  /capabilities           — flags the UI uses to render conditionally
  POST /query                  — JSON Q&A (the "API as product" surface)
  GET  /page/{doc_tag}/{page}  — source-page PNG (gated by PLANESCAPE_SHOW_PAGES=1)

Run:
  uvicorn planescapeRAG.chatbot.api:app --host 0.0.0.0 --port 8000

The retriever and a small LRU of recent retrievals (for the graph endpoint)
live as module-level state.
"""

import json
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .answer import DEFAULT_MODEL, SUPPORTED_MODELS, answer_question
from .embed_entities import _parse_entity_file
from .render import (
    build_pyvis_html,
    entity_summary_dicts,
    render_body_to_html,
)
from .retrieve import RetrievalResult, Retriever


_PROJECT_ROOT = Path(__file__).resolve().parent.parent           # planescapeRAG/
_PACKAGE_DIR = Path(__file__).resolve().parent                    # chatbot/
DEFAULT_ENTITIES_DIR = _PROJECT_ROOT / "graph" / "entities"
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "chatbot" / "index"
DOC_SOURCES_PATH = _PACKAGE_DIR / "doc_sources.json"
PAGE_CACHE_DIR = _PROJECT_ROOT / "tmp" / "page_cache"
PAGE_RENDER_DPI = 150


def _show_pages_enabled() -> bool:
    return os.environ.get("PLANESCAPE_SHOW_PAGES", "").strip() in {"1", "true", "yes"}


_TSR_PREFIX_RE = re.compile(r"^TSR\s*\d+\s+")


def _pretty_title(pdf_stem: str) -> str:
    """Strip the 'TSR XXXX ' product-code prefix from a PDF stem."""
    return _TSR_PREFIX_RE.sub("", pdf_stem).strip()


def _load_doc_sources() -> Dict[str, dict]:
    """Map doc_tag → {pdf: Path, type: 'canonical'|'adventure', title: str}.

    Tolerates both schemas:
    - Legacy: bare string value (the PDF path) — treated as canonical with title
      inferred from the PDF stem (TSR-product-code prefix stripped).
    - New:    object value with `pdf`, `type`, `title` keys.
    """
    if not DOC_SOURCES_PATH.exists():
        return {}
    raw = json.loads(DOC_SOURCES_PATH.read_text())
    out: Dict[str, dict] = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if isinstance(v, str):
            pdf = (_PROJECT_ROOT / v).resolve()
            out[k] = {"pdf": pdf, "type": "canonical", "title": _pretty_title(pdf.stem)}
        elif isinstance(v, dict) and "pdf" in v:
            pdf = (_PROJECT_ROOT / v["pdf"]).resolve()
            out[k] = {
                "pdf": pdf,
                "type": v.get("type", "canonical"),
                "title": v.get("title") or _pretty_title(pdf.stem),
                # Optional: list of {name, start, end} describing the physical books
                # bound inside a boxed set, for book-level citation display.
                "books": v.get("books") or [],
            }
    return out


app = FastAPI(
    title="Planescape graphRAG",
    description=(
        "Retrieval-augmented Q&A over the Planescape entity graph. "
        "Embeddings via OpenAI text-embedding-3-small; answers grounded in "
        "source bullets via Claude."
    ),
    version="0.2.0",
)

# Serve CSS / static assets from chatbot/static/.
app.mount("/static", StaticFiles(directory=_PACKAGE_DIR / "static"), name="static")

# Jinja2 templates from chatbot/templates/.
templates = Jinja2Templates(directory=_PACKAGE_DIR / "templates")

_retriever: Optional[Retriever] = None
_doc_sources: Dict[str, dict] = _load_doc_sources()

# Recent retrievals keyed by short UUID; the answer fragment references its
# retrieval here so /graph/{id} can render the matching subgraph and browser
# back/forward restores the right graph.
_RETRIEVALS_CACHE_MAX = 50
_retrievals_cache: "OrderedDict[str, RetrievalResult]" = OrderedDict()


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            index_dir=DEFAULT_INDEX_DIR,
            entities_dir=DEFAULT_ENTITIES_DIR,
        )
    return _retriever


def _store_retrieval(retrieval: RetrievalResult) -> str:
    rid = uuid.uuid4().hex[:10]
    _retrievals_cache[rid] = retrieval
    while len(_retrievals_cache) > _RETRIEVALS_CACHE_MAX:
        _retrievals_cache.popitem(last=False)
    return rid


def _global_name_to_id() -> Dict[str, str]:
    """Lowercase canonical_name → entity_id, over the entire corpus.
    Used to linkify wikilinks to ANY entity, not just the retrieved set."""
    r = get_retriever()
    return {e["canonical_name"].lower(): e["entity_id"] for e in r.entities if e.get("canonical_name")}


# ─── JSON request/response models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question.")
    k: int = Field(5, ge=1, le=50)
    max_total: int = Field(15, ge=1, le=50)
    expand: bool = Field(True)
    model: str = Field(DEFAULT_MODEL)


class EntityRef(BaseModel):
    entity_id: str
    canonical_name: str
    type: str
    similarity: float
    is_primary: bool
    expanded_from: Optional[str] = None
    body: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    cost_usd: Optional[float]
    primaries: List[EntityRef]
    expansions: List[EntityRef]


class CapabilitiesResponse(BaseModel):
    show_pages: bool
    available_doc_tags: List[str]


# ─── HTML endpoints (HTMX-driven UI) ───────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/chat", response_class=HTMLResponse)
def chat(
    request: Request,
    question: str = Form(...),
    k: int = Form(5),
    max_total: int = Form(15),
    expand: bool = Form(False),  # checkbox: present='true', absent=missing
    model: str = Form(DEFAULT_MODEL),
):
    if model not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unsupported model {model!r}")

    r = get_retriever()
    retrieval = r.retrieve(question, k=k, max_total=max_total, expand=expand)
    result = answer_question(question, retrieval, model=model, doc_metadata=_doc_sources)

    name_to_id = _global_name_to_id()
    api_base = str(request.base_url).rstrip("/")
    show_pages = _show_pages_enabled()

    answer_html = render_body_to_html(
        result.answer, name_to_id, api_base, show_pages,
        doc_metadata=_doc_sources,
    )
    retrieval_id = _store_retrieval(retrieval)

    return templates.TemplateResponse(
        request,
        "fragments/answer.html",
        {
            "question": question,
            "answer_html": answer_html,
            "entities": entity_summary_dicts(retrieval),
            "cost_usd": result.cost_usd,
            "model": result.model,
            "retrieval_id": retrieval_id,
            "show_graph": True,
        },
    )


@app.get("/entity/{entity_id:path}", response_class=HTMLResponse)
def entity_html(request: Request, entity_id: str):
    # If the user landed here via direct browser navigation (refresh, bookmark,
    # paste), this URL only resolves to a content fragment, not a full page.
    # Send them back to / so the chat shell renders. The HX-Request header is
    # set by HTMX on its own AJAX requests, so we only return the fragment then.
    if request.headers.get("HX-Request") != "true":
        return RedirectResponse(url="/", status_code=303)

    r = get_retriever()
    if entity_id not in r._by_id:
        raise HTTPException(404, f"Entity {entity_id!r} not found in the corpus.")

    file_path = r.entities_dir / (entity_id + ".md")
    if not file_path.exists():
        raise HTTPException(500, f"Entity metadata exists but file is missing: {file_path}")
    fm, body = _parse_entity_file(file_path)
    indexed = r._by_id[entity_id]

    api_base = str(request.base_url).rstrip("/")
    body_html = render_body_to_html(
        body,
        _global_name_to_id(),
        api_base,
        _show_pages_enabled(),
        strip_leading_h1=True,  # body starts with `# Title`; we already show it in the header.
        doc_metadata=_doc_sources,
    )

    entity = {
        "entity_id": entity_id,
        "canonical_name": indexed.get("canonical_name") or fm.get("canonical_name") or entity_id,
        "type": indexed.get("type") or fm.get("type", ""),
        "supertype": fm.get("supertype", ""),
        "aliases": fm.get("aliases", "").strip("[]") if fm.get("aliases") else "",
    }

    return templates.TemplateResponse(
        request,
        "fragments/entity.html",
        {"entity": entity, "body_html": body_html},
    )


@app.get("/graph/{retrieval_id}", response_class=HTMLResponse)
def graph_for_retrieval(retrieval_id: str):
    retrieval = _retrievals_cache.get(retrieval_id)
    if retrieval is None:
        return HTMLResponse(
            "<p style='padding:1em;color:#999;text-align:center;font-family:sans-serif;'>"
            "Graph state expired. Ask the question again to regenerate.</p>"
        )
    return HTMLResponse(build_pyvis_html(retrieval))


# ─── JSON / utility endpoints ──────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    try:
        r = get_retriever()
        return {"status": "ok", "indexed_entities": len(r.entities)}
    except FileNotFoundError as e:
        return {"status": "no_index", "detail": str(e)}


@app.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        show_pages=_show_pages_enabled(),
        available_doc_tags=sorted(_doc_sources.keys()),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """JSON-only Q&A endpoint — the 'API as product' surface for downstream consumers."""
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unsupported model {req.model!r}. Allowed: {SUPPORTED_MODELS}")
    try:
        r = get_retriever()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    retrieval = r.retrieve(req.question, k=req.k, max_total=req.max_total, expand=req.expand)
    result = answer_question(req.question, retrieval, model=req.model, doc_metadata=_doc_sources)

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        model=result.model,
        cost_usd=result.cost_usd,
        primaries=[
            EntityRef(
                entity_id=p.entity_id,
                canonical_name=p.canonical_name,
                type=p.type,
                similarity=p.similarity,
                is_primary=True,
                body=p.body,
            )
            for p in retrieval.primaries
        ],
        expansions=[
            EntityRef(
                entity_id=e.entity_id,
                canonical_name=e.canonical_name,
                type=e.type,
                similarity=0.0,
                is_primary=False,
                expanded_from=e.expanded_from,
                body=e.body,
            )
            for e in retrieval.expansions
        ],
    )


@app.get("/page/{doc_tag}/{page}")
def get_page(doc_tag: str, page: int) -> FileResponse:
    if not _show_pages_enabled():
        raise HTTPException(404, "Source-page rendering is disabled.")
    if page < 1:
        raise HTTPException(400, "page must be >= 1")
    meta = _doc_sources.get(doc_tag)
    if meta is None:
        raise HTTPException(404, f"Unknown doc_tag {doc_tag!r}")
    pdf_path = meta["pdf"]
    if not pdf_path.exists():
        raise HTTPException(500, f"PDF for {doc_tag!r} is missing on disk: {pdf_path}")

    cache_dir = PAGE_CACHE_DIR / doc_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{page:04d}.png"

    if not cache_file.exists():
        from pdf2image import convert_from_path
        try:
            images = convert_from_path(
                str(pdf_path), dpi=PAGE_RENDER_DPI,
                first_page=page, last_page=page,
            )
        except Exception as e:
            raise HTTPException(500, f"PDF render failed: {e}")
        if not images:
            raise HTTPException(404, f"page {page} not found in {doc_tag}")
        images[0].save(cache_file, "PNG", optimize=True)

    return FileResponse(
        cache_file,
        media_type="image/png",
        filename=f"{doc_tag}_p{page}.png",
        content_disposition_type="inline",
    )
