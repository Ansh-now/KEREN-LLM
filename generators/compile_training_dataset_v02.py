#!/usr/bin/env python3
"""Compile validated KEREN Gold V0.1 + V0.2 traces into QLoRA-ready JSONL.

The Gold files remain the source of truth. Output rows match training/train_qlora.py:
    {"id": "...", "input": "...", "target": {...}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "datasets" / "gold"
DEFAULT_OUTPUT = ROOT / "datasets" / "compiled" / "keren_train_v0.2.jsonl"

TERMINAL_OR_GATED_STATES = {
    "clarify", "await_confirmation", "confirm", "refuse",
    "blocked", "recover", "failed",
}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        out = {k: _clean(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        out = [_clean(v) for v in value]
        return [v for v in out if v not in (None, "", [], {})]
    return value


def _intent_name(record: dict) -> str | None:
    intent = record.get("intent")
    if isinstance(intent, str):
        return intent
    if isinstance(intent, dict):
        for key in ("name", "intent", "type", "id"):
            value = intent.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _selected(obj: Any) -> str | None:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("selected", "name", "id", "tool", "node"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def build_input(record: dict) -> str:
    user = record.get("user_input") or {}
    task = record.get("task") or {}
    payload = {
        "text": user.get("text", ""),
        "language": user.get("language"),
        "input_type": user.get("input_type"),
        "category": task.get("category"),
        "context": record.get("context") or {},
    }
    return json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":"))


def build_target(record: dict) -> dict:
    decision = record.get("decision") or {}
    final_output = record.get("final_output") or {}
    state = decision.get("state") or "continue"
    mode = final_output.get("mode") or final_output.get("type") or (record.get("task") or {}).get("output_mode") or "answer"
    text = final_output.get("text")

    target: dict[str, Any] = {"state": state, "mode": mode}
    reason = decision.get("reason_code")
    if reason:
        target["reason"] = reason

    if mode in {"answer", "status", "debug", "code"} or state in TERMINAL_OR_GATED_STATES:
        if isinstance(text, str) and text:
            target["text"] = text
        next_action = decision.get("next_action")
        if next_action:
            target["next"] = next_action
        return _clean(target)

    target.update({
        "intent": _intent_name(record),
        "node": _selected(record.get("node")),
        "tool": _selected(record.get("tool")),
        "arguments": record.get("arguments") or {},
    })
    if isinstance(text, str) and text:
        target["text"] = text
    return _clean(target)


def iter_gold() -> list[Path]:
    files = sorted(GOLD_DIR.glob("gold_v*.jsonl"))
    if not files:
        raise SystemExit(f"No Gold files found in {GOLD_DIR}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-records", type=int, default=550)
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[str] = set()
    for path in iter_gold():
        with path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                record = json.loads(raw)
                rid = str(record.get("id", ""))
                if not rid:
                    raise SystemExit(f"Missing id in {path.name}:{line_no}")
                if rid in seen:
                    raise SystemExit(f"Duplicate id while compiling: {rid}")
                seen.add(rid)
                rows.append({"id": rid, "input": build_input(record), "target": build_target(record)})

    if len(rows) != args.expected_records:
        raise SystemExit(f"Expected {args.expected_records} Gold records, found {len(rows)}; refusing to write incomplete V0.2 dataset")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    modes: dict[str, int] = {}
    for row in rows:
        mode = str(row["target"].get("mode", "unknown"))
        modes[mode] = modes.get(mode, 0) + 1
    print(f"Compiled {len(rows)} Gold records -> {args.output}")
    print("Modes: " + ", ".join(f"{k}={v}" for k, v in sorted(modes.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
