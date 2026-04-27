# Learnings

Challenges encountered building the planescapeRAG pipeline and how we resolved them. Roughly chronological — earlier items shaped later architectural decisions.

## OCR is non-deterministic on the same source

The same PDF page produced different OCR text on different runs. `Citadel Cavitius` came out as `Caviitus` once and `Cavitius` another time, even from a single run on a single PDF. This is a property of LLM-based vision OCR with `temperature > 0` (or model-internal nondeterminism even at 0).

**Resolution:** corpus profile gained a `### OCR corrections` section listing known mis-transcriptions as `old → new` bullets. `parse_ocr_corrections` extracts these from the profile; `apply_ocr_corrections` runs them as plain string replacement on the document at processing time. This keeps the on-disk OCR output unchanged (provenance preserved) and corrections apply in-memory whenever a downstream stage loads the document. Adding a new correction is an edit to the profile, not a re-OCR.

## Decorative typography breaks character-level OCR

TSR Planescape books use a decorative font for pull quotes that mis-OCRs `t` as `+` (e.g., `AR+IS+S` for `ARTISTS`) and `O` as `@` (e.g., `L@CK` for `LOCK`).

**Resolution:** corpus-profile addendum gives the OCR model a semantic instruction ("transcribe pull quotes as the intended words, not letter-for-letter"). Avoided per-character substitution rules because we kept discovering new mappings; semantic instruction is more general.

## Header levels drift between OCR runs

Per-page OCR has no document-wide context, so it assigned `#`, `##`, `###` based on visual size. Same heading level appeared as `#` on one page and `###` on another. Headers also leaked into pull quotes that were rendered as `### `.

**Resolution:** per-page rules (corpus profile) plus an explicit instruction that pull quotes are blockquotes (`> ...`), never headings. Cross-page consistency was never fully achieved — papered over by ingest-time normalization.

## Pass 1 entity extraction misses minor entities

Pass 1's prompt biases toward entities the document *describes* (gives a paragraph or section of treatment). Entities mentioned only in passing — e.g., `Hive Ward` listed once as a Sigil district — get skipped. Pass 2 then strips wikilinks to those entities because they're not in the registry.

**Resolution:** two-mechanism completeness pass after pass 1.

1. **LLM completeness sweep** ([completeness_sweep.py](completeness_sweep.py)) — gives the model the full document plus the slim registry, asks "what's mentioned but not registered?" Returns suggestions with verbatim evidence quotes that we substring-validate.
2. **Heuristic candidate extractor + LLM verifier** ([heuristic_entity_candidates.py](heuristic_entity_candidates.py), [verify_candidates.py](verify_candidates.py)) — regex-finds proper-noun spans, filters against the registry, hands the remaining candidates to the LLM for skip / merge_into / add_new decisions.

The two mechanisms turned out to be **complementary** rather than redundant. The LLM sweep biased toward specific named figures (deities, NPCs); the heuristic surfaced high-frequency cosmological groupings (Outer Planes, Hive Ward, Toril) the LLM systematically missed. Together they fill different classes of registry gap.

## Quote validation in pass 2 was brittle

Pass 2 emits `[[entity]] — "verbatim quote" (p. N)` relationships. Strict substring matching against the document dropped 330 quotes per run on the Campaign Setting — many false negatives.

**Resolution:** layered validation, each layer recovering a different class of false negative.

1. **Unicode punctuation fold** — OCR'd text uses curly `'`/`"`/`—`; LLM output uses ASCII `'`/`"`/`-`. Folding both sides recovered ~160 quotes on a 330-failure run.
2. **Fuzzy regex matching with substitution** (using the `regex` package's `{e<=N}` syntax) — anchors on first/last words exactly, allows up to 5 internal edits. When fuzzy-match succeeds, the model's emitted quote is *replaced* with the document's matched span, upgrading the stored evidence. Recovered ~50 more quotes per run.
3. **Pipe-syntax wikilinks** — the model emitted `[[Target|Display]]` (Obsidian-style display-name syntax) which the original validator captured as one token. Fixed by partitioning on `|` and validating just the target.

## Slug collisions

Multiple registry entries can share a filesystem slug after lowercasing and stripping non-alphanumerics — e.g., `Manes` (creature) and `Mazes` (extradimensional prison), or `Manes` (capitalized) and `manes` (lowercase). Pass 2 batch submission failed with "duplicate custom_id" before any LLM call.

**Resolution:** two-phase fix.

1. `dedup_registry.py` extended to merge canonical-name duplicates (case + leading-"the" insensitive, in-registry).
2. `merge_registries.py` (cross-doc merging) gained an LLM verifier on fuzzy-name pairs (`verify_fuzzy_merges.py`). The verifier sees both entities' descriptions and decides "same entity" or "different." Catches false positives that pure edit-distance matching can't, like `Frey` vs `Freya` (different Norse deities).

## max_tokens overflow corrupted tool calls silently

When an Anthropic call hits `max_tokens` mid-tool-call, the partial JSON is returned but the tool input may be empty or malformed. Symptom on pass 1 was `Tool input missing 'entities' key: {}` — confusing because the schema said `entities` is required.

**Resolution:** bumped pass-1 `max_tokens` to 32K, switched to streaming (Anthropic's SDK refuses non-streaming calls that may exceed 10 minutes), and improved error messages to surface `stop_reason` and usage so future overflows diagnose at a glance.

## Models occasionally violate their own schema

Haiku in particular sometimes:

- Emits `relationships` as a JSON-encoded *string* instead of an array (double-encoded).
- Omits required fields like `source_pages`.

**Resolution:** defensive parsing in `_filter_relationships` (json.loads if string, isinstance guards per element) and `setdefault` for required fields in `collect_profiles`. Added a `malformed` drop counter to surface frequency.

## Single-doc pass 2 overwrites multi-doc context

Initial pass-2 design assumed one entity = one description = one source. Running pass 2 on a second document either skipped the entity (if `skip_existing=True`) or overwrote the first doc's profile (if `--no-skip-existing`). Either way, *the entity file reflected one document's perspective*, not the corpus's.

**Resolution:** per-doc-section markdown format. Each entity file has multiple `## Source: <doc_tag>` sections, one per ingested doc. `_upsert_entity_file` parses existing sections, replaces or appends the doc's section, and writes back. Page citations stay simple (`(p. N)`) because they're scoped to the enclosing section. `source_pages` in frontmatter became a per-doc dict `{"campaign_setting": [...], "planes_of_chaos": [...]}`.

This let us avoid the more complex "synthesize a unified profile from all docs" approach and is incremental-friendly: adding a new doc only requires profiling entities that doc mentions, then appending sections to existing files (no re-running across all docs).

## Anthropic batch caching didn't behave as we expected

Set `cache_control` on a 75K-token cached prefix (system + global registry) for a 309-request batch. Expected: 1 cache write + 308 cache reads. Observed: ~22.97M cache_write tokens + ~0.4M cache_read tokens — i.e., almost every request paid the cache-write rate.

**Hypothesis** (not confirmed in docs): batch workers process requests in parallel. With 309 workers starting near-simultaneously, each finds an empty cache and writes. By the time any could read, they've all already paid the write. The "warmup synchronous call before submitting the batch" pattern is the documented workaround.

**Resolution:** sidestepped the question by adopting a **local registry** approach instead — for each entity, build a per-request registry slice containing only entities mentioned in that entity's windowed paragraphs. Per-request payload dropped from ~75K to ~0.5–3K tokens. With payloads that small, caching savings are immaterial; we removed `cache_control` from windowed-mode requests entirely.

Side benefit: this is the right architecture for any registry that gets large. At Confluence scale (10K+ entities), sending the global registry per request isn't possible; local registry is the only path forward.

## Cost surprises (the $200 lesson)

Hit Anthropic's spending limit unexpectedly because I (the assistant) had been running expensive operations without surfacing cumulative cost. The estimator didn't exist, and I didn't track running totals out loud.

**Resolution:** [cost.py](cost.py) — pricing tables, per-call cost reporting (`report_call`), pre-flight estimation for batches (`estimate_anthropic_batch_cost`), and a JSONL ledger at `tmp/cost_log.jsonl` so cumulative spend is visible on every run. Pre-flight cost estimates print before submission and prompt for confirmation when above a configurable threshold. `--yes` skips the prompt for unattended runs.

Secondary lesson: the *first* estimator was wildly wrong (predicted $7, actual $72) because it modeled cache-write-once-read-many while reality was cache-write-every-time. Updated the estimator and the underlying architecture (local registry) to be honest about batch cache behavior.

## Cost log double-counting on re-collect

When the orchestrator re-runs `--start-from profiles_submit` against a doc whose batch was already collected, the submit step finds nothing new (`skip_existing` matches all entities) but the collect step reads the old `pass2_batch_id.txt` and re-iterates the batch results, logging the cost again as if it were new spend.

**Resolution:** `_batch_already_billed` checks `cost_log.jsonl` for an existing `pass2_collect` entry with the same `batch_id` and skips logging if found. The entity files are still rewritten (idempotent and cheap), but the ledger doesn't double-count.

## What I'd do differently next time

- **Build cost tracking before any LLM-heavy work.** It would have saved the $200 surprise.
- **Test new request shapes (caching, schema, etc.) on tiny batches before running at scale.** The cache-write surprise would have shown up immediately on a 5-request test.
- **Default to local registry from day one.** Sending the full registry per request only worked because Planescape Campaign Setting is small. The architecture didn't scale and we paid for that in iteration cost.
- **Track running cumulative cost out loud during the session.** Not just per-call but explicit "we've now spent $X this session" callouts at major decision points.
