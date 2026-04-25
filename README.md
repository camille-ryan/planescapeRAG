# planescapeRAG

Pipeline for OCR'ing TSR Planescape sourcebooks (and similar scanned PDFs) into clean markdown, with corpus-specific guidance baked into the OCR prompt.

## Status

**Built:** OCR pipeline — chunked OpenAI batch submission, corpus-profile-aware prompting, and per-page → combined-markdown merging.

**Pending:** GraphRAG construction over the combined markdown (entity extraction, graph storage, retrieval). Not yet implemented.

## Setup

### Conda environment

```bash
conda env create -f ../environment.yml      # first time
conda activate graphRAG
python -m pip install -r requirements.txt
```

`direnv` users can `direnv allow` from `/mnt/e/proj` to auto-activate.

### System dependencies (PDF rendering)

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler
```

### Environment variables

Create `planescapeRAG/.env` with:

```bash
OPENAI_API_KEY=...
S3_OCR_BUCKET=...           # bucket where page images are uploaded for batch OCR
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
OPENAI_OCR_MODEL=gpt-4o     # optional; defaults to gpt-4o
```

The OCR pipeline uploads each page image to S3 and gives OpenAI a presigned URL (24h expiry). The S3 bucket needs to permit object reads via presigned URL.

## Usage

### Submit a PDF for OCR

`submit_batches.py` chunks the PDF into batches sized to fit the org's enqueued-token cap, submits each, polls until completion, and writes a final combined markdown.

```bash
python -m planescapeRAG.submit_batches \
    --pdf path/to/document.pdf \
    --output-dir tmp/output_dir \
    --profile corpus_profiles/planescape_tsr.md \
    --chunk-size 50
```

Outputs (in `--output-dir`):
- `<stem>_chunk<NN>_p<start>-<end>.jsonl` — per-chunk batch input
- `<stem>_chunk<NN>_p<start>-<end>_results.jsonl` — per-chunk batch output
- `<stem>_all_results.jsonl` — concatenated batch outputs
- `<stem>_combined.md` — final combined markdown with `<!-- page N -->` markers

Re-running skips chunks whose results JSONL already exists, so partial runs resume cleanly.

### Combine an existing batch result

```bash
python -m planescapeRAG.combine_batch \
    path/to/results.jsonl \
    path/to/output.md
```

## Corpus profiles

A corpus profile is a markdown file with level-2 sections that contribute addenda to specific pipeline-stage prompts:

```markdown
## OCR addendum

Per-page rules the OCR model can apply (typography quirks, what to discard,
heading conventions, image/map handling).

## Ingest addendum

Reserved for the future entity-extraction stage. Cross-page rules, entity
type list, etc.
```

The first word of each `##` heading (lowercased) is the section key. `build_ocr_prompt(profile)` concatenates the base prompt with the OCR addendum and sends it for every page request. See `corpus_profiles/planescape_tsr.md` for a worked example.

## Layout

```
planescapeRAG/
  ocr.py              # OCR prompt builder, profile loader, batch JSONL builder
  submit_batches.py   # chunked batch submission with polling and resume
  combine_batch.py    # batch results JSONL → combined markdown
  config.py           # Settings dataclass (env-driven)
  corpus_profiles/
    planescape_tsr.md # worked profile for TSR Planescape sourcebooks
```
