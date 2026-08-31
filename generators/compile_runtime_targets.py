#!/usr/bin/env python3
"""Compile rich KEREN traces into compact student targets."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "datasets" / "gold" / "gold_v0.1_seed.jsonl"
DEFAULT_OUTPUT = ROOT / "datasets" / "candidates" / "runtime_targets_v0.1.jsonl"

def compact(record: dict) -> dict:
    decision = record.get("decision") or {}
    if decision.get("state") in {"clarify", "await_confirmation", "blocked", "recover", "failed"}:
        return {"state": decision.get("state"), "reason": decision.get("reason_code"), "next": decision.get("next_action")}
    intent = record.get("intent") or {}; node = record.get("node") or {}; tool = record.get("tool") or {}
    return {"i": intent.get("name"), "n": node.get("selected"), "t": tool.get("selected"), "a": record.get("arguments") or {}}

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT); p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for raw in src:
            if not raw.strip(): continue
            r = json.loads(raw)
            row = {"id": r["id"], "input": r["user_input"]["text"], "target": compact(r)}
            dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"); count += 1
    print(f"Compiled {count} compact targets -> {args.output}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
