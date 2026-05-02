"""Empirical test: does a non-batch warm-up call let a subsequent batch read
from the same prompt cache, or do batch workers maintain independent caches?

Procedure:
  1. Send ONE non-batch messages.create() with the predicate-classifier system
     prompt + cache_control. Verify it pays a cache write.
  2. Immediately submit a 5-request batch with the same system prompt.
  3. Poll the batch to completion, sum cache_write vs cache_read across results.
  4. If batch cache_write == 0: pre-warming works. Recommend it.
     If batch cache_write > 0: pre-warming doesn't carry through; per-worker caches.

Cost: ~$0.02. Quick.
"""

import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from planescapeRAG.config import Settings
from planescapeRAG.pipeline.classify_predicates import (
    _SYSTEM_PROMPT,
    _CLASSIFY_TOOL,
    _build_request,
    _walk_unclassified_sections,
    _DEFAULT_MODEL,
)


def main() -> None:
    load_dotenv()
    settings = Settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Step 1 — single non-batch warm-up
    print("=== Step 1: warm-up (non-batch) ===")
    warm_msg = client.messages.create(
        model=_DEFAULT_MODEL,
        max_tokens=100,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ],
        messages=[
            {"role": "user", "content": "Reply with the single word: ready"},
        ],
    )
    u = warm_msg.usage
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    print(f"  warm-up: input={inp}  output={out}  cache_write={cw}  cache_read={cr}")
    if cw == 0:
        print("  WARNING: warm-up didn't write cache. Prompt may already be cached from prior session.")

    # Step 2 — small batch with same system prompt
    print("\n=== Step 2: 5-request batch, same prompt ===")
    registry = json.loads(Path("planescapeRAG/graph/registry.json").read_text())
    by_canonical = {e["canonical_name"]: e for e in registry["entities"]}

    requests = []
    for path, entity_id, canonical, doc_tag, edges, _, _, _, _ in _walk_unclassified_sections():
        entity_type = (by_canonical.get(canonical) or {}).get("type", "unknown")
        requests.append(_build_request(entity_id, canonical, entity_type, doc_tag, edges, _DEFAULT_MODEL))
        if len(requests) >= 5:
            break

    if len(requests) < 5:
        print(f"  only {len(requests)} unclassified sections available — stop early")
        return

    batch = client.messages.batches.create(requests=requests)
    print(f"  submitted: {batch.id}")

    print("\n=== Step 3: polling batch ===")
    import time
    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        print(f"  status={b.processing_status}  succeeded={c.succeeded}  errored={c.errored}")
        if b.processing_status == "ended":
            break
        time.sleep(15)

    print("\n=== Step 4: per-result usage ===")
    total_cw = total_cr = total_inp = total_out = 0
    for i, result in enumerate(client.messages.batches.results(batch.id)):
        if result.result.type != "succeeded":
            print(f"  req {i}: failed — {result.result.type}")
            continue
        u = result.result.message.usage
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        inp = getattr(u, "input_tokens", 0) or 0
        out = getattr(u, "output_tokens", 0) or 0
        print(f"  req {i}: input={inp}  output={out}  cache_write={cw}  cache_read={cr}")
        total_cw += cw
        total_cr += cr
        total_inp += inp
        total_out += out

    print(f"\n=== Verdict ===")
    print(f"Batch totals: cache_write={total_cw}  cache_read={total_cr}")
    if total_cw == 0:
        print("✓ PRE-WARM WORKS: batch read from the warmed cache. Recommend pre-warming the full run.")
    else:
        write_ratio = total_cw / (total_cw + total_cr) if (total_cw + total_cr) else 0
        print(f"✗ PRE-WARM DID NOT HELP: batch wrote cache anyway ({write_ratio:.0%} write ratio).")
        print("  Workers maintain independent caches. Submit normally and accept the overhead.")


if __name__ == "__main__":
    main()
