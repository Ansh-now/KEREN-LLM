#!/usr/bin/env python3
"""Normalize mechanical KEREN Gold V0.1 metadata inconsistencies.

This script intentionally performs only deterministic, low-risk repairs:
- map legacy/non-taxonomy categories to current canonical categories
- make task.output_mode match final_output.mode when the final mode is valid

It does NOT rewrite user text, decisions, safety labels, observations, verification,
or code answers. Those require human-quality review.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "datasets" / "gold"
TAXONOMY = ROOT / "taxonomy"

CATEGORY_MAP = {
    "intent_boundaries": "model_tool_awareness",
    "failure_recovery": "agentic_execution",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    output_modes = set(load_json(TAXONOMY / "output_modes.json")["modes"].keys())
    changed_files = 0
    changed_records = 0
    category_fixes = 0
    mode_fixes = 0

    for path in sorted(GOLD_DIR.glob("gold_v0.1_*.jsonl")):
        records = []
        file_changed = False

        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            record = json.loads(raw)
            changed = False

            task = record.get("task")
            if isinstance(task, dict):
                category = task.get("category")
                mapped = CATEGORY_MAP.get(category)
                if mapped:
                    task["category"] = mapped
                    category_fixes += 1
                    changed = True

                final_output = record.get("final_output")
                final_mode = final_output.get("mode") if isinstance(final_output, dict) else None
                task_mode = task.get("output_mode")
                if (
                    isinstance(final_mode, str)
                    and final_mode in output_modes
                    and isinstance(task_mode, str)
                    and task_mode != final_mode
                ):
                    task["output_mode"] = final_mode
                    mode_fixes += 1
                    changed = True

            if changed:
                changed_records += 1
                file_changed = True
            records.append(record)

        if file_changed:
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records),
                encoding="utf-8",
            )
            changed_files += 1

    print(
        f"Normalized Gold: files={changed_files}, records={changed_records}, "
        f"category_fixes={category_fixes}, output_mode_fixes={mode_fixes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
