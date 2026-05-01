# planescapeRAG

End-to-end pipeline that turns a stack of scanned RPG sourcebooks into a queryable, cited entity graph with a retrieval-augmented chat UI.

The current build has ingested three TSR Planescape books (Campaign Setting, Planes of Chaos, Planes of Law — ~600 PDF pages total) into **1,063 entities** across seven types (characters, places, planes, creatures, factions, items, concepts), each with structured relationships, verbatim source quotes, and page-level citations.

## What it does

1. **OCR** scanned PDFs into clean markdown using GPT-4o vision via the OpenAI Batch API.
2. **Extract** entities and their relationships from that markdown using Claude with tool use (structured output).
3. **Reconcile** entities across multiple sourcebooks into one global graph — same entity from different books gets one file with per-source sections.
4. **Embed + retrieve** entity bodies with FAISS, then expand 1-hop along the wikilink graph for context.
5. **Answer** natural-language questions with Claude, grounded in retrieved bullets, with inline `(source, p. N)` citations.

The chat UI ([chatbot/](chatbot/)) is a FastAPI + HTMX app that renders cited answers, an interactive pyvis graph of what was retrieved, and clickable wikilinks that load any entity's full dossier into a side panel.

## Architecture

The pipeline is ten stages, orchestrated by [run_pipeline.py](run_pipeline.py). Each stage's outputs are checkpointed; `--start-from <stage>` resumes mid-run.

| # | Stage | What it does | Tech |
|---|---|---|---|
| 1 | **OCR + combine** | Chunk PDF → batch JSONL → S3 presigned URLs → GPT-4o vision → combined markdown with `<!-- page N -->` markers. Tier-1 token caps respected; partial runs resume. | `pdf2image`, `boto3`, OpenAI Batch API |
| 2 | **Pass 1: registry** | Single Claude call extracts every entity into a typed registry (canonical name, type, aliases, brief description, notable flag). Tool use enforces schema. | Anthropic SDK + tool use |
| 3 | **Dedup** | Merges canonical-name and alias duplicates within the per-doc registry (case-insensitive, leading-"the" insensitive). | in-memory |
| 4 | **Merge** | Folds the per-doc registry into the global one. Exact matches merge silently; fuzzy matches go to an LLM verifier (`Frey` vs `Freya` are different deities; `the Lady of Pain` and `Lady of Pain` are not). | Anthropic SDK |
| 5 | **Completeness sweep** | Gives the model the full doc + slim registry, asks "what's mentioned but not registered?" Suggestions are substring-validated against the source before being added. | Anthropic SDK |
| 6 | **Heuristic candidates** | Regex-finds proper-noun spans the LLM missed (high-frequency cosmological terms — *Outer Planes*, *Hive Ward*, *Toril* — that the LLM systematically biased away from). LLM verifier decides skip / merge_into / add_new. | `regex` + Anthropic SDK |
| 7 | **Canonicalize types** | Maps freeform types ("demipower", "proxy", "petitioner") to a small supertype set ("character", "creature"…) over the global registry. | Anthropic SDK |
| 8 | **Pass 2: profile submit** | For each entity, builds a windowed local-registry slice and submits a batch request asking Claude to write a structured profile (description with `[[wikilinks]]`, source pages, typed relationships with verbatim quotes). | Anthropic Batches |
| 9 | **Pass 2: profile collect** | Polls the batch; on completion validates each profile (quote substring + Unicode-fold + fuzzy regex match against the source), then upserts a `## Source: <doc_tag>` section into the entity's markdown file. | Anthropic Batches + custom validator |
| 10 | **Embed for chatbot** | Embeds each entity body with `text-embedding-3-small`, writes `vectors.npy` + `metadata.json` for the FAISS retriever. Incremental: only embeds new/changed entities. | OpenAI embeddings, NumPy |

The output is a directory of markdown files (one per entity, sharded by supertype) where each file looks like:

```markdown
---
canonical_name: "Abriymoch"
type: "place"
source_documents: ["campaign_setting", "planes_of_law"]
source_pages: {"campaign_setting": [54, 292], "planes_of_law": [123, 293]}
---

# Abriymoch

## Source: campaign_setting
Abriymoch is a city located on [[Phlegethos]], the fourth layer of [[Baator]]…

### Relationships from campaign_setting
- [[Phlegethos]] — "The only city is Abriymoch, built in the caldera…" (p. 54)
- [[Baator]] — "LAYER 4: PHLEGETHOS** - Abriymoch" (p. 292)

## Source: planes_of_law
…
```

The wikilinks make the markdown corpus a graph today; the design is also clean for migration to a graph DB (Neo4j) later.

## Chat UI ([chatbot/](chatbot/))

```
question
  → embed (text-embedding-3-small)
  → top-k FAISS over entity vectors
  → 1-hop expansion along [[wikilinks]]
  → Claude (Haiku/Sonnet/Opus) with strict citation rules
  → markdown answer rendered with clickable wikilinks + (source, p. N) citations
```

- **FastAPI** serves both the HTML UI and a JSON `/query` API. Endpoints: see [chatbot/api.py](chatbot/api.py).
- **HTMX** drives the front-end — no SPA framework. Submitting a question swaps a fragment into `#answer-panel`; clicking a wikilink loads that entity's dossier into `#entity-panel`. `hx-push-url` makes the browser back/forward buttons work.
- **pyvis** renders an interactive force-directed graph of the retrieved subgraph at `/graph/{retrieval_id}`.
- **Source-page viewer** (gated by `PLANESCAPE_SHOW_PAGES=1`): `(doc_tag, p. N)` citations link to a rendered PDF page for verification — closes the loop from generated answer back to the original scan.

The "1-hop graph expansion" is what makes this graphRAG and not flat RAG: a query for *"who rules Phlegethos?"* embeds well to **Phlegethos**, but the answer lives on **Gazra** — which is pulled in via the wikilink and provided to Claude alongside.

## Notable engineering decisions

- **Two-pass extraction.** Pass 1 only commits to a registry of names + types. Pass 2 writes the full per-entity profile, which means relationships can use `[[wikilinks]]` that *will resolve* — the registry is the contract between passes.
- **Per-source sections, not synthesized profiles.** When the same entity appears in three sourcebooks, the entity file gets three `## Source: …` sections rather than one merged description. Avoids "synthesize across sources" hallucination, makes ingest fully incremental (adding a new book only profiles entities that book mentions), and keeps every quote attributable.
- **Local registry over global registry in pass 2.** Initial design sent the full global registry (~75K tokens) cached as a prefix. Anthropic's batch workers all hit the cache before any could populate it, paying cache-*write* rate on every request. Switched to a per-entity windowed slice (~0.5–3K tokens) and removed `cache_control` entirely. Side benefit: the architecture now scales to arbitrarily large registries.
- **Corpus profiles as a portability seam.** All Planescape-specific quirks (decorative-font OCR rules, pull-quote conventions, OCR mis-transcriptions) live in [corpus_profiles/planescape_tsr.md](corpus_profiles/planescape_tsr.md). The pipeline core is corpus-agnostic; pointing it at a different scanned corpus is a new profile, not a code change.
- **OCR corrections are applied in-memory at ingest time, not baked into the OCR output.** A profile section like `### OCR corrections` lists `Caviitus → Cavitius` rules. Provenance of the original OCR is preserved on disk; corrections apply whenever a downstream stage loads the doc. New correction = profile edit, not re-OCR.
- **Layered quote validation.** Pass 2 emits `"verbatim quote" (p. N)` relationships. Strict substring matching dropped ~330 quotes per run on the Campaign Setting. Layered fix: Unicode-punctuation fold (curly quotes/dashes) + fuzzy regex match anchored on first/last words allowing a few internal edits + handling Obsidian-style `[[Target|Display]]` wikilinks. Recovered ~210 of the 330.
- **Cost tracking is first-class.** [cost.py](cost.py) maintains a JSONL ledger at `tmp/cost_log.jsonl` with per-call model/usage/USD entries, pre-flight batch estimates that prompt for confirmation above a threshold, and de-duping on re-collected batches so the ledger stays honest.

For the longer war stories — non-determinism in vision OCR, decorative typography, the cache-write surprise, the $200 lesson — see [Learnings.md](Learnings.md).

## Layout

```
planescapeRAG/
├── run_pipeline.py          ← orchestrator (10 stages, --start-from to resume)
├── ocr.py                   ← OCR prompt builder, profile loader, batch JSONL
├── submit_batches.py        ← chunked OCR batch submission with polling
├── combine_batch.py         ← OCR results JSONL → combined markdown
├── extract_registry.py      ← pass 1: doc → typed entity registry
├── dedup_registry.py        ← in-doc canonical/alias dedup
├── merge_registries.py      ← per-doc → global, with fuzzy LLM verifier
├── verify_fuzzy_merges.py   ← LLM same-entity-or-not? for fuzzy name pairs
├── completeness_sweep.py    ← LLM "what's missing from the registry?" pass
├── heuristic_entity_candidates.py  ← regex-found proper-noun candidates
├── verify_candidates.py     ← LLM verifier for heuristic candidates
├── canonicalize_types.py    ← freeform type → supertype mapping
├── extract_profiles.py      ← pass 2: per-entity windowed profile extraction
├── cost.py                  ← pricing tables, ledger, pre-flight estimates
├── config.py                ← env-driven Settings dataclass
├── corpus_profiles/
│   └── planescape_tsr.md    ← TSR Planescape OCR/ingest addenda + corrections
├── graph/
│   ├── registry.json                    ← global entity registry
│   ├── type_supertype_mapping.json      ← freeform-type → supertype map
│   └── entities/<supertype>/<slug>.md   ← 1,063 entity files
└── chatbot/
    ├── api.py               ← FastAPI: HTML UI + JSON /query
    ├── retrieve.py          ← FAISS top-k + 1-hop wikilink expansion
    ├── answer.py            ← Claude with strict citation system prompt
    ├── embed_entities.py    ← incremental embedding builder
    ├── render.py            ← markdown → HTML, wikilink/citation linkifier, pyvis
    ├── templates/           ← Jinja2 (HTMX fragments)
    └── static/style.css
```

## Setup

### Conda environment

```bash
conda env create -f ../environment.yml      # first time
conda activate graphRAG
python -m pip install -r requirements.txt
```

`direnv` users can `direnv allow` from the workspace root to auto-activate.

### System dependency (PDF rendering)

```bash
sudo apt install poppler-utils    # Ubuntu/Debian
brew install poppler              # macOS
```

### Environment variables (`planescapeRAG/.env`)

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
S3_OCR_BUCKET=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
OPENAI_OCR_MODEL=gpt-4o     # optional, default
```

OCR uploads each page image to S3 and gives OpenAI a 24h presigned URL; the bucket needs to allow object reads via presigned URL.

## Usage

### Ingest a sourcebook end-to-end

```bash
python -m planescapeRAG.run_pipeline \
    --pdf "documents/Boxed Sets/TSR 2600 Planescape Campaign Setting.pdf" \
    --profile corpus_profiles/planescape_tsr.md \
    --doc-tag campaign_setting
```

Resume from any stage:

```bash
python -m planescapeRAG.run_pipeline ... --start-from profiles_submit
```

### Run the chatbot

```bash
cd chatbot
make run
# → http://localhost:8000   (UI)
# → http://localhost:8000/docs  (Swagger for the JSON API)
```

Re-embed after ingesting a new doc:

```bash
make embed
```

## Status

- **Done:** OCR pipeline, two-pass entity extraction, cross-doc merge with LLM verifier, completeness sweep, heuristic candidate fill-in, type canonicalization, pass-2 profile extraction with quote validation, embedding + FAISS retrieval, FastAPI/HTMX chat UI with cited answers, 1-hop graph expansion, interactive pyvis graph, source-page viewer, cost tracking with pre-flight estimates.
- **Ingested:** 3 of ~12 Planescape sourcebooks → 1,063 entities.
- **Next:** ingest the remaining accessories; migrate the markdown-as-graph to Neo4j (the design is migration-clean — entity files map cleanly to nodes, `### Relationships` bullets to typed edges).
