#!/usr/bin/env python3
"""Re-score an existing KEREN V0.6 result JSONL without running a model.

This is intended for early result files produced before the unified evaluator.
It fixes the benchmark's naive forbidden-substring handling, but it cannot
recover Qwen thinking boundaries if the old evaluator decoded them with
skip_special_tokens=True. Use run_model_eval_v06.py for the authoritative rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.run_model_eval_v06 import load_jsonl, score

DEFAULT_BENCH = ROOT / "evaluation" / "benchmark_v0.6_holdout.jsonl"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    args = p.parse_args()

    cases = {row["id"]: row for row in load_jsonl(args.benchmark)}
    rows = load_jsonl(args.results)
    passed = 0
    category_stats: dict[str, list[int]] = {}
    changed = []

    for row in rows:
        rid = row["id"]
        if rid not in cases:
            raise SystemExit(f"Result id {rid!r} is not in benchmark")
        visible = row.get("final_output")
        if visible is None:
            visible = row.get("output", "")
        ok, detail = score(cases[rid], visible, [])
        old = bool(row.get("passed", False))
        if old != ok:
            changed.append((rid, old, ok, detail))
        passed += int(ok)
        cat = cases[rid].get("category", "unknown")
        category_stats.setdefault(cat, [0, 0])
        category_stats[cat][0] += int(ok)
        category_stats[cat][1] += 1

    total = len(rows)
    print(f"RESCORED: {passed}/{total} = {100.0 * passed / total:.1f}%")
    print("CATEGORY SCORES:")
    for cat in sorted(category_stats):
        pcat, tcat = category_stats[cat]
        print(f"  {cat}: {pcat}/{tcat} = {100.0 * pcat / tcat:.1f}%")
    print(f"CHANGED DECISIONS: {len(changed)}")
    for rid, old, new, detail in changed:
        print(f"  {rid}: {'PASS' if old else 'FAIL'} -> {'PASS' if new else 'FAIL'} | {detail}")
    print("NOTE: old outputs may still mix hidden thinking with the final answer; this no-GPU rescore is diagnostic, not authoritative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
