"""Cost summary for pass-2 profiling.

Walks `cost_log.jsonl` and prints three views:
  1. Per-doc breakdown — historical vs new-framework cost for each doc
  2. Totals — actual spent vs counterfactual ("if new framework from start")
  3. Per-model breakdown — Sonnet vs Haiku request and cost split

The "new framework" marker is the presence of a `by_model` dict in the
`pass2_collect` entry's `details`. Two-tier routing started landing those after
the cleanup waves; older entries (single-model Sonnet runs from the original
$18-estimate experiments) lack the field.

Usage:
    python -m planescapeRAG.tools.cost_summary
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


_LOG = Path("planescapeRAG/cost_log.jsonl")


def _read_log() -> List[dict]:
    if not _LOG.exists():
        return []
    return [json.loads(l) for l in _LOG.read_text().splitlines() if l.strip()]


def main() -> None:
    entries = _read_log()
    pass2 = [e for e in entries if e.get("operation") == "pass2_collect"]

    # Group per doc, keep historical (no by_model) and new-framework (has by_model)
    # entries separately. If a doc was profiled twice, the latest in each category wins.
    per_doc: Dict[str, Dict[str, Optional[dict]]] = defaultdict(lambda: {"old": None, "new": None})
    for e in pass2:
        det = e.get("details", {})
        tag = det.get("doc_tag", "unknown")
        bucket = "new" if det.get("by_model") else "old"
        prev = per_doc[tag][bucket]
        if prev is None or e.get("ts", "") > prev.get("ts", ""):
            per_doc[tag][bucket] = e

    # Per-doc breakdown.
    print("=" * 100)
    print(f"{'doc_tag':38s}  {'old $':>10s}  {'new $':>10s}  {'savings $':>10s}  {'Sonnet/Haiku':>20s}")
    print("-" * 100)
    total_old = 0.0
    total_new = 0.0
    actual_spent = 0.0
    new_only_total = 0.0
    for tag in sorted(per_doc):
        old_e = per_doc[tag]["old"]
        new_e = per_doc[tag]["new"]
        old_cost = (old_e or {}).get("cost_usd") or 0
        new_cost = (new_e or {}).get("cost_usd") or 0
        actual_spent += old_cost + new_cost
        if new_e is not None:
            new_only_total += new_cost
            total_new += new_cost
            total_old += old_cost if old_e is not None else 0
        else:
            new_only_total += old_cost  # no new run yet — count old as the current cost
        # Sonnet/Haiku split (from the new entry)
        sh = ""
        if new_e is not None:
            bm = new_e["details"].get("by_model", {})
            s = bm.get("claude-sonnet-4-6", {}) or {}
            h = bm.get("claude-haiku-4-5-20251001", {}) or {}
            sn = s.get("n", 0) if isinstance(s, dict) else 0
            hn = h.get("n", 0) if isinstance(h, dict) else 0
            sh = f"{sn}/{hn}"
        savings = (old_cost - new_cost) if (old_e is not None and new_e is not None) else 0
        savings_str = f"${savings:>8.2f}" if savings else ""
        print(f"{tag:38s}  ${old_cost:>8.2f}  ${new_cost:>8.2f}  {savings_str:>10s}  {sh:>20s}")

    print("-" * 100)
    print(f"\n{'Actual spent (sum of every pass2_collect):':50s}  ${actual_spent:>8.2f}")
    print(f"{'Counterfactual (new framework from start):':50s}  ${new_only_total:>8.2f}")
    if actual_spent > 0:
        diff = actual_spent - new_only_total
        print(f"{'Difference (re-billing for re-profiled docs):':50s}  ${diff:>8.2f}")

    # Where re-billing came from — docs profiled both old and new
    rebilled = [(tag, per_doc[tag]) for tag in per_doc if per_doc[tag]["old"] and per_doc[tag]["new"]]
    if rebilled:
        print(f"\n--- Re-profiled docs (cost paid twice) ---")
        for tag, both in rebilled:
            old_c = both["old"]["cost_usd"] or 0
            new_c = both["new"]["cost_usd"] or 0
            print(f"  {tag:38s}  old=${old_c:.2f}  new=${new_c:.2f}  savings_per_run=${old_c - new_c:.2f}")

    # Aggregate per-model totals from new entries.
    by_model_totals: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0, "cost": 0.0, "input": 0, "output": 0})
    for tag, buckets in per_doc.items():
        new_e = buckets["new"]
        if not new_e:
            continue
        bm = new_e["details"].get("by_model", {}) or {}
        for model, stats in bm.items():
            if not isinstance(stats, dict):
                continue
            by_model_totals[model]["n"] += stats.get("n", 0)
            by_model_totals[model]["cost"] += stats.get("cost", 0)
            by_model_totals[model]["input"] += stats.get("input", 0)
            by_model_totals[model]["output"] += stats.get("output", 0)

    if by_model_totals:
        print("\n--- Per-model totals (new framework runs only) ---")
        total_n = sum(t["n"] for t in by_model_totals.values())
        for model, t in sorted(by_model_totals.items()):
            pct = 100 * t["n"] / total_n if total_n else 0
            cpr = t["cost"] / t["n"] if t["n"] else 0
            print(f"  {model:35s}  {t['n']:>5d} reqs ({pct:>4.1f}%)  ${t['cost']:>7.2f}  ${cpr:.4f}/req  in={t['input']:,}  out={t['output']:,}")
        print(f"  {'TOTAL':35s}  {total_n:>5d} reqs           ${sum(t['cost'] for t in by_model_totals.values()):>7.2f}")


if __name__ == "__main__":
    main()
