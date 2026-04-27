# TODO

Prioritized roughly by what unblocks the most project value per unit of effort. Tiers are aspirational ordering, not a strict gate.

## Tier 1 — unblocks the actual purpose

- **Build the chatbot / retrieval layer.** The graph is built; nothing consumes it. Concrete steps: embed each entity file's body (one-time, ~$0.20 with OpenAI text-embedding-3-small or equivalent), build query → top-k similarity → 1-hop wikilink expansion → Claude answer-with-citations. Each entity file already has `[[Target]] — "verbatim quote" (p. N)` bullets that give us source citations for free. Per-query dev cost is negligible if Haiku is the answer model. This is the "demoable artifact" — everything else is supporting work.

## Tier 2 — meaningfully improves what's there

- **Eval / quality measurement.** A smoke-test set of ~20 (question → expected_entity) pairs to measure retrieval recall@k. Doesn't matter if retrieval is good if you can't measure it. Free, small effort, naturally paired with the chatbot build.
- **README rewrite.** Currently describes only the OCR foundation. Should reflect the full pipeline — pass 1, dedup, merge, sweep, heuristic, canonicalize, pass 2 (windowed + local registry), per-doc-section format, cost tracking, orchestrator usage. Free, ~30 min, reduces re-onboarding cost the next time you pick this up.
- **Pre-pass-1 canonical-name verification at ingest time.** [verify_canonical_names.py](verify_canonical_names.py) exists but only runs on demand. Wire into the orchestrator so canonical-name typos surface during ingest, not after pass 2 has propagated them everywhere. Cheap (~$0.05 per ingest).
- **Code cleanup before adding new features.** Audit existing modules for dead code or scripts no longer used. Bundle related scripts into folders (e.g., `pipeline/` for orchestrator stages, `tools/` for one-off helpers like `verify_canonical_names`, `dedup_registry`). Reduces friction for the chatbot work.

## Tier 3 — quality polishes

- **Cross-document profile synthesis.** Per-doc-section format preserves provenance but doesn't synthesize across sources. A future LLM step could read all `## Source: X` sections and produce a unified description while keeping the per-doc evidence as appendices. Defer until current per-doc output proves insufficient (it might be fine for retrieval). Costs ~$10–15/doc to redo.
- **Heuristic candidate extractor improvements** ([heuristic_entity_candidates.py](heuristic_entity_candidates.py)) — currently word-boundary regex misses lowercase-only D&D names like `tanar'ri`, `baatezu`, `slaad`. A separate "known-creature-name patterns" pass could pick these up. Modest quality gain, cheap.
- **Better cache investigation.** Anthropic batches charged cache_write ~N times instead of 1× write + (N−1)× reads on the original PoL batch. Sidestepped by local registry; remains an open question. Mostly intellectual curiosity at this point.

## Tier 4 — out of scope until prerequisites exist

- **Frontend typo reporting UI** — let users flag canonical names or quoted text in the rendered graph that look wrong. Capture `(entity, span, suggested correction)`. Pipe into the corpus profile's `### OCR corrections` section as new `old → new` bullets. The mechanism for applying corrections is already built ([ocr.py](ocr.py) `apply_ocr_corrections`); the gap is the reporting + persistence flow. **Needs a frontend to exist first.**
- **Graph DB importer.** When ready to migrate from markdown-as-graph to a real graph DB, a one-shot importer can walk `graph/entities/`, parse each file's frontmatter and Relationships bullets, and emit Cypher (Neo4j) or equivalent. Keep canonical-name consistency so the importer is trivial. **Wait until there's a reason to migrate.**
- **Confluence-scale architecture sketch.** Local-registry approach already adopted is a prerequisite. Beyond that: page-level change detection, per-page entity diffing, permissions-aware retrieval. **Speculative; revisit if/when there's an actual second corpus.**
- **Make the OCR step optional.** For Confluence/HTML/structured sources, skip the OCR pipeline entirely. Generic interface: any source ingestion produces a `combined.md` that downstream stages consume. **Coupled with Confluence-scale concern.**

## Tier 5 — hygiene

- **Orchestrator guard against re-collect double-runs.** Cost-log dedup catches it after the fact ([extract_profiles.py](extract_profiles.py) `_batch_already_billed`); the orchestrator could detect "submit produced no new batch in this run" and skip the collect stage entirely. Belt-and-suspenders.
- **Pin `regex` package version** in [requirements.txt](requirements.txt). Only worth doing if we hit a compatibility issue.
