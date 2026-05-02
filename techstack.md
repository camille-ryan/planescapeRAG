# Tech Stack

Technologies used by planescapeRAG, grouped by purpose.

## Language & runtime

| Technology | Purpose |
|---|---|
| **Python 3.11** | Sole implementation language. Pinned in the parent workspace's [../environment.yml](../environment.yml). |
| **Conda** | Provides the `graphRAG` env (defined in [../environment.yml](../environment.yml)) on top of which project deps are pip-installed. |
| **pip** | Installs project-specific Python packages from [requirements.txt](requirements.txt). |
| **direnv** | Auto-activates the conda env on entry to the workspace (see [.envrc](.envrc)). |

## LLM / AI providers

| Technology | Purpose |
|---|---|
| **OpenAI Python SDK** (`openai`) | Two roles: (1) GPT-4o vision OCR via the **Batch API** for scanned PDF pages — page images are uploaded to S3 and handed to OpenAI as presigned URLs inside batch JSONL for cost-efficient bulk OCR; (2) `text-embedding-3-small` to embed entity files for retrieval. |
| **Anthropic Python SDK** (`anthropic`) | Used throughout the pipeline: building and updating the entity registry ([extract_registry.py](extract_registry.py), [extract_profiles.py](extract_profiles.py), [merge_registries.py](merge_registries.py)), verification passes ([verify_candidates.py](verify_candidates.py), [verify_fuzzy_merges.py](verify_fuzzy_merges.py)), type canonicalization ([canonicalize_types.py](canonicalize_types.py)), the completeness sweep ([completeness_sweep.py](completeness_sweep.py)), and the final cited answer in the chat UI ([chatbot/answer.py](chatbot/answer.py)). Supported answer-time models: Haiku 4.5 (default), Sonnet 4.6, Opus 4.7. |

## Vector search & retrieval

| Technology | Purpose |
|---|---|
| **FAISS** (`faiss-cpu`) | In-memory `IndexFlatIP` over L2-normalized embedding vectors (≈ cosine similarity). Rebuilt at startup from `vectors.npy` in [chatbot/retrieve.py](chatbot/retrieve.py). |
| **NumPy** (`numpy`) | Stores the embedding matrix on disk as `vectors.npy` and handles vector math feeding FAISS. |

## Web service (chat UI)

| Technology | Purpose |
|---|---|
| **FastAPI** | HTTP framework serving both the HTMX-driven HTML UI and the JSON `/query` API. Entrypoint: [chatbot/api.py](chatbot/api.py). |
| **Uvicorn** | ASGI server that runs the FastAPI app (`make run` in [chatbot/Makefile](chatbot/Makefile)). |
| **Pydantic** | Request/response models for the JSON API (`QueryRequest`, `QueryResponse`, `CapabilitiesResponse`). |
| **Jinja2** (`jinja2`) | Server-side templating for the HTML page and HTMX fragments in [chatbot/templates/](chatbot/templates/). |
| **python-multipart** | FastAPI dependency for parsing `application/x-www-form-urlencoded` POSTs from the HTMX form. |
| **HTMX** (CDN, v1.9.10) | Front-end interactivity without a JS framework: form posts, fragment swaps into `#answer-panel` / `#entity-panel`, and `hx-push-url` for back/forward navigation. |
| **Markdown** (`markdown`) | Renders entity bodies and Claude's markdown answers to HTML before injection into the page. |
| **pyvis** | Builds the interactive HTML graph view of a retrieval result (`/graph/{retrieval_id}`). |

## OCR / document pipeline

| Technology | Purpose |
|---|---|
| **pdf2image** | Renders PDF pages to PIL images for OCR submission and for the on-demand source-page viewer (`/page/{doc_tag}/{page}`). PNGs are cached under `tmp/page_cache/`. |
| **Poppler** (`poppler-utils`, system package) | Native dependency of `pdf2image`; performs the actual PDF rasterization. |
| **boto3** | Uploads page images to S3 and mints presigned URLs for the OpenAI Batch OCR requests. |
| **regex** | More-capable replacement for stdlib `re`, used in cleanup/extraction passes over OCR'd markdown. |

## Configuration & secrets

| Technology | Purpose |
|---|---|
| **python-dotenv** | Loads `.env` (OpenAI key, S3 bucket, AWS creds, optional model overrides) into a `Settings` dataclass in [config.py](config.py). |
| **AWS S3** | Storage for OCR page images consumed via presigned URLs by the OpenAI Batch API. |

## Tooling

| Technology | Purpose |
|---|---|
| **Git** | Version control — planescapeRAG is its own repo. |
| **GNU Make** | Convenience targets for the chatbot (`install`, `embed`, `run`) in [chatbot/Makefile](chatbot/Makefile). |
