# TODO

## Pipeline / quality

- **Haiku-vs-Sonnet quality comparison on local-registry pass 2.** Earlier Haiku test failed because the (then-incomplete) global registry made Haiku's "thoroughness" produce many unknown-target relationships. With local registry constraining the option space, Haiku is theoretically viable. ~$0.10 to run on 6 representative entities; estimate already computed (`~$0.10`). If quality is acceptable, default new docs to Haiku for ~3× cost reduction.
- **Cross-document profile synthesis.** Current per-doc-section format preserves provenance but doesn't synthesize across sources. A future LLM step could read all `## Source: X` sections and produce a unified description while keeping the per-doc evidence as appendices.
- **Pre-pass-1 canonical-name verification at ingest time.** `verify_canonical_names.py` exists but only runs on demand. Wire into the orchestrator so canonical-name typos surface during ingest, not after pass 2 has propagated them everywhere.
- **Better cache investigation.** Anthropic batches charged cache_write ~N times instead of 1× write + (N−1)× reads on the failed PoL batch. Possibly a known parallel-race issue with batches; possibly a bug in our request shape. Worth a small synchronous test to isolate.
- **Heuristic candidate extractor improvements** ([heuristic_entity_candidates.py](heuristic_entity_candidates.py)): currently word-boundary regex misses lowercase-only D&D names like `tanar'ri`, `baatezu`, `slaad` because they don't follow capitalization conventions. A separate "known-creature-name patterns" pass could pick these up.

## Frontend

- **Typo reporting UI** — let users flag canonical names or quoted text in the rendered graph that look wrong. Capture `(entity, span, suggested correction)`. Pipe into the corpus profile's `### OCR corrections` section as new `old → new` bullets, then re-run the affected pipeline stages. The mechanism for applying corrections is already built ([ocr.py](ocr.py) `apply_ocr_corrections`); the gap is the reporting + persistence flow.

## Retrieval / chat layer

- **Build the actual chatbot.** The graph is built; nothing consumes it yet. Embed entity files, build a query → top-k retrieval → 1-hop wikilink expansion → Claude answer pipeline.
- **Citation rendering in answers.** Each entity file has `[[Target]] — "verbatim quote" (p. N)` bullets that already give us source citations. Surface them in chatbot answers ("according to Campaign Setting page 49, ...").
- **Eval / quality measurement.** No way today to tell whether retrieval is good. Build a small set of (question → expected entity) pairs and measure recall@k.

## Migration / scale

- **Graph DB importer.** When ready to migrate from markdown-as-graph to a real graph DB, a one-shot importer can walk `graph/entities/`, parse each file's frontmatter and Relationships bullets, and emit Cypher (Neo4j) or equivalent. Keep an eye on canonical-name consistency (wikilinks must match canonical_name exactly) so the importer is trivial.
- **Confluence-scale architecture sketch.** Local-registry approach already adopted is a prerequisite. Beyond that: page-level change detection (webhooks or `lastUpdated` polling), per-page entity diffing, permissions-aware retrieval. None of this is built; the existing code is the foundation.
- **Make the OCR step optional.** For Confluence/HTML/structured sources, skip the OCR pipeline entirely. Generic interface: any source ingestion produces a `combined.md` that downstream stages consume.

## Hygiene

- **README rewrite.** Currently describes only the OCR foundation. The README should reflect the full pipeline — pass 1, dedup, merge, sweep, heuristic, canonicalize, pass 2 (windowed + local registry), per-doc-section format, cost tracking, orchestrator usage.
- **Orchestrator guard against re-collect double-runs.** Cost-log dedup catches it after the fact; the orchestrator could detect "submit produced no new batch in this run" and skip the collect stage entirely.
- **Dependency on `regex` package.** Pinned in requirements but not version-locked. Could add a minimum version once we know what features we depend on.
