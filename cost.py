"""Cost tracking for Anthropic API calls.

Pricing tables (USD per million tokens) plus helpers to:
- Compute the cost of an Anthropic Usage object (handles cache write/read split).
- Estimate the cost of a pre-built batch request list before submitting.
- Log each call to a project-wide JSONL ledger so cumulative spend is visible.

Layer 1 (post-call accounting): every LLM-calling module logs cost after the
response. Cumulative session total prints alongside.

Layer 2 (pre-flight estimate): submit_profiles computes expected cost from
the request bodies before calling the API and prompts for confirmation when
the estimate exceeds a threshold.

NOTE: prices are hand-maintained and may drift from Anthropic's posted rates.
Treat estimates as ~10% accurate, not authoritative billing.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# USD per million tokens. cache_write rates vary by TTL.
PRICING: Dict[str, Dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write_5m": 1.25,
        "cache_write_1h": 2.00,
        "cache_read": 0.10,
    },
    "claude-opus-4-7": {
        "input": 15.00,
        "output": 75.00,
        "cache_write_5m": 18.75,
        "cache_write_1h": 30.00,
        "cache_read": 1.50,
    },
}


_DEFAULT_LOG_PATH = Path("planescapeRAG/tmp/cost_log.jsonl")


def _log_path() -> Path:
    override = os.environ.get("PLANESCAPERAG_COST_LOG")
    return Path(override) if override else _DEFAULT_LOG_PATH


def _rate(model: str, key: str) -> Optional[float]:
    p = PRICING.get(model)
    if not p:
        return None
    return p.get(key)


def compute_anthropic_cost(
    usage,
    model: str,
    is_batch: bool = False,
    cache_ttl: str = "1h",
) -> Optional[float]:
    """USD cost for a single Anthropic response Usage object.

    Returns None for unknown models (price table miss).
    """
    if not PRICING.get(model):
        return None

    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0

    cache_write_key = "cache_write_1h" if cache_ttl == "1h" else "cache_write_5m"

    cost = 0.0
    cost += input_tokens * (_rate(model, "input") or 0) / 1_000_000
    cost += output_tokens * (_rate(model, "output") or 0) / 1_000_000
    cost += cache_write_tokens * (_rate(model, cache_write_key) or 0) / 1_000_000
    cost += cache_read_tokens * (_rate(model, "cache_read") or 0) / 1_000_000
    if is_batch:
        cost *= 0.5
    return cost


def log_cost(
    operation: str,
    model: str,
    cost_usd: Optional[float],
    *,
    is_batch: bool = False,
    details: Optional[dict] = None,
) -> None:
    """Append one entry to the project-wide cost log."""
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "operation": operation,
        "model": model,
        "is_batch": is_batch,
        "cost_usd": cost_usd if cost_usd is not None else None,
        "details": details or {},
    }
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def session_total() -> float:
    """Cumulative USD spent according to the local log. Returns 0 if no log."""
    path = _log_path()
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cost = entry.get("cost_usd")
                if isinstance(cost, (int, float)):
                    total += cost
            except json.JSONDecodeError:
                continue
    return total


def report_call(
    operation: str,
    model: str,
    usage,
    *,
    is_batch: bool = False,
    cache_ttl: str = "1h",
    details: Optional[dict] = None,
) -> Optional[float]:
    """Compute, log, and print the cost of a single response. Returns the cost (or None)."""
    cost = compute_anthropic_cost(usage, model, is_batch=is_batch, cache_ttl=cache_ttl)
    log_cost(operation, model, cost, is_batch=is_batch, details=details)
    if cost is not None:
        total = session_total()
        print(f"  cost: ${cost:.4f}  |  session total: ${total:.2f}", flush=True)
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight estimation for batch submission
# ─────────────────────────────────────────────────────────────────────────────

# Conservative chars-per-token. English-prose-heavy text averages ~4 chars per
# token; we use 3 to overestimate slightly (better surprise low than high).
_CHARS_PER_TOKEN = 3


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_anthropic_batch_cost(
    requests: List[dict],
    *,
    model: str,
    is_batch: bool = True,
    cache_ttl: str = "1h",
    output_tokens_per_request: int = 1000,
) -> Dict[str, float]:
    """Estimate USD cost for a list of Anthropic batch requests.

    Walks each request's system + messages structure, sums tokens, and applies
    cache-write-once / cache-read-many heuristic: the first request that
    presents a given cacheable block writes it; subsequent requests with the
    same block read it (much cheaper).

    Returns a dict with the breakdown so the caller can show users where the
    cost goes, plus a `cost_usd` total.
    """
    if not PRICING.get(model):
        return {"cost_usd": None, "note": f"No pricing table for {model!r}"}

    cache_write_key = "cache_write_1h" if cache_ttl == "1h" else "cache_write_5m"
    p = PRICING[model]

    cached_seen: set = set()
    cache_write_tokens = 0
    cache_read_tokens = 0
    uncached_input_tokens = 0

    for req in requests:
        params = req.get("params", {})

        # System blocks
        for block in params.get("system", []) or []:
            text = block.get("text", "") if isinstance(block, dict) else ""
            tokens = _estimate_tokens(text)
            if isinstance(block, dict) and block.get("cache_control"):
                key = ("system", text[:64], len(text))
                if key in cached_seen:
                    cache_read_tokens += tokens
                else:
                    cache_write_tokens += tokens
                    cached_seen.add(key)
            else:
                uncached_input_tokens += tokens

        # User-message blocks
        for msg in params.get("messages", []) or []:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    text = block.get("text", "") if isinstance(block, dict) else ""
                    tokens = _estimate_tokens(text)
                    if isinstance(block, dict) and block.get("cache_control"):
                        key = ("user", text[:64], len(text))
                        if key in cached_seen:
                            cache_read_tokens += tokens
                        else:
                            cache_write_tokens += tokens
                            cached_seen.add(key)
                    else:
                        uncached_input_tokens += tokens
            elif isinstance(content, str):
                uncached_input_tokens += _estimate_tokens(content)

    output_tokens = output_tokens_per_request * len(requests)

    cost = 0.0
    cost += uncached_input_tokens * p["input"] / 1_000_000
    cost += cache_write_tokens * p[cache_write_key] / 1_000_000
    cost += cache_read_tokens * p["cache_read"] / 1_000_000
    cost += output_tokens * p["output"] / 1_000_000
    if is_batch:
        cost *= 0.5

    return {
        "uncached_input_tokens": uncached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens_estimate": output_tokens,
        "n_requests": len(requests),
        "cost_usd": cost,
    }


def confirm_proceed(
    estimate: Dict[str, float],
    operation: str,
    *,
    threshold_usd: float = 1.00,
    auto_yes: bool = False,
) -> bool:
    """Pretty-print the cost estimate and prompt for confirmation if above threshold.

    Returns True if user approves (or estimate is below threshold, or auto_yes).
    """
    cost = estimate.get("cost_usd")
    n = estimate.get("n_requests", 0)
    print(f"\n--- {operation} cost estimate ---", flush=True)
    print(f"  requests:                {n}", flush=True)
    print(f"  uncached input tokens:   {estimate.get('uncached_input_tokens', 0):,}", flush=True)
    print(f"  cache-write tokens:      {estimate.get('cache_write_tokens', 0):,}", flush=True)
    print(f"  cache-read tokens:       {estimate.get('cache_read_tokens', 0):,}", flush=True)
    print(f"  estimated output tokens: {estimate.get('output_tokens_estimate', 0):,}", flush=True)
    if cost is None:
        print(f"  cost:                    UNKNOWN ({estimate.get('note', '')})", flush=True)
        return auto_yes
    print(f"  estimated cost:          ${cost:.2f}", flush=True)
    total = session_total()
    print(f"  session total so far:    ${total:.2f}  →  after: ${total + cost:.2f}", flush=True)

    if auto_yes or cost < threshold_usd:
        return True
    answer = input(f"\n  Proceed? [y/N]: ").strip().lower()
    return answer in ("y", "yes")
