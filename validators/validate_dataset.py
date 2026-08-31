#!/usr/bin/env python3
"""Validate KEREN Dataset V0.1 JSONL records.

Checks:
- JSON syntax and schema
- duplicate IDs across one or many files
- taxonomy values (category, intent, output mode, reason code)
- basic decision/training-label coherence
- confirmation and verification safety invariants
- code-output quality heuristic

Examples:
    python validators/validate_dataset.py
    python validators/validate_dataset.py datasets/gold/gold_v0.1_batch10.jsonl
    python validators/validate_dataset.py --all-gold
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schemas" / "keren_dataset_v0.1.schema.json"
DEFAULT_DATASET = ROOT / "datasets" / "gold" / "gold_v0.1_seed.jsonl"
GOLD_DIR = ROOT / "datasets" / "gold"
TAXONOMY_DIR = ROOT / "taxonomy"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def taxonomy_values(path: Path, key: str) -> set[str]:
    data = load_json(path)
    value = data[key]
    if isinstance(value, dict):
        return set(value.keys())
    return set(value)


def iter_dataset_files(dataset: Path | None, all_gold: bool) -> Iterable[Path]:
    if all_gold:
        files = sorted(GOLD_DIR.glob("gold_v0.1_*.jsonl"))
        if not files:
            raise SystemExit(f"No Gold JSONL files found in {GOLD_DIR}")
        return files
    return [dataset or DEFAULT_DATASET]


def get_nested(record: dict, *keys):
    cur = record
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def extract_reason_codes(record: dict) -> list[str]:
    found: list[str] = []
    candidates = [
        get_nested(record, "decision", "reason_code"),
        get_nested(record, "decision", "reason_codes"),
        get_nested(record, "safety", "reason_code"),
        get_nested(record, "safety", "reason_codes"),
        record.get("reason_code"),
        record.get("reason_codes"),
    ]
    for value in candidates:
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, list):
            found.extend(x for x in value if isinstance(x, str))
    return found


def looks_like_complete_code(text: str) -> bool:
    """Conservative heuristic: reject prose-only promises in code mode.

    This intentionally does not try to compile arbitrary languages. It only
    verifies that a code-mode target contains substantial code-like structure.
    """
    if not text or len(text.strip()) < 40:
        return False

    lower = text.lower()
    promise_only = [
        "implement karunga",
        "create karunga",
        "code likhunga",
        "i will implement",
        "i'll implement",
        "i will create",
        "here's how i would",
    ]
    if any(p in lower for p in promise_only):
        # A promise is acceptable only if real code is also present.
        pass

    fenced = re.search(r"```(?:python|py|dart|kotlin|java|cpp|c\+\+|c|arduino|json|bash|sh|powershell|yaml|yml)?\s*\n.+?```", text, re.S | re.I)
    if fenced:
        return True

    code_signals = [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bimport\s+[\w.]+",
        r"\bfrom\s+[\w.]+\s+import\b",
        r"\bfun\s+\w+\s*\(",
        r"\bvoid\s+\w+\s*\(",
        r"\bint\s+\w+\s*\(",
        r"\bFuture<[^>]+>\s+\w+\s*\(",
        r"\bWidget\s+build\s*\(",
        r"#include\s*[<\"]",
        r"\bvoid\s+(setup|loop)\s*\(",
        r"@app\.(get|post|put|delete|patch)\s*\(",
        r"\basync\s+def\s+\w+\s*\(",
    ]
    hits = sum(bool(re.search(pattern, text, re.I)) for pattern in code_signals)
    structural = text.count("{") + text.count("}") + text.count(";") + text.count("\n")
    return hits >= 1 and structural >= 4


def semantic_errors(
    record: dict,
    categories: set[str],
    intents: set[str],
    output_modes: set[str],
    reason_codes: set[str],
) -> list[str]:
    errors: list[str] = []

    category = get_nested(record, "task", "category")
    if isinstance(category, str) and category not in categories:
        errors.append(f"unknown task.category {category!r}")

    intent_obj = record.get("intent")
    intent_name = None
    if isinstance(intent_obj, str):
        intent_name = intent_obj
    elif isinstance(intent_obj, dict):
        for key in ("name", "intent", "type", "id"):
            if isinstance(intent_obj.get(key), str):
                intent_name = intent_obj[key]
                break
    if intent_name and intent_name not in intents:
        errors.append(f"unknown intent {intent_name!r}")

    task_mode = get_nested(record, "task", "output_mode")
    final_mode = get_nested(record, "final_output", "mode")
    if isinstance(task_mode, str) and task_mode not in output_modes:
        errors.append(f"unknown task.output_mode {task_mode!r}")
    if isinstance(final_mode, str) and final_mode not in output_modes:
        errors.append(f"unknown final_output.mode {final_mode!r}")
    if isinstance(task_mode, str) and isinstance(final_mode, str) and task_mode != final_mode:
        errors.append(f"output mode mismatch task={task_mode!r} final={final_mode!r}")

    for code in extract_reason_codes(record):
        if code not in reason_codes:
            errors.append(f"unknown reason code {code!r}")

    labels = record.get("training_labels") or {}
    decision = record.get("decision") or {}
    state = decision.get("state") if isinstance(decision, dict) else None

    if labels.get("should_clarify") is True and state not in {"clarify", "blocked"}:
        errors.append(f"should_clarify=true but decision.state={state!r}")
    if labels.get("should_confirm") is True and state not in {"await_confirmation", "confirm", "blocked"}:
        errors.append(f"should_confirm=true but decision.state={state!r}")
    if labels.get("should_refuse") is True and state not in {"refuse", "blocked"}:
        errors.append(f"should_refuse=true but decision.state={state!r}")
    if labels.get("should_recover") is True and state not in {"recover", "continue", "blocked"}:
        errors.append(f"should_recover=true but decision.state={state!r}")

    safety = record.get("safety") or {}
    confirmation_required = False
    if isinstance(safety, dict):
        confirmation_required = bool(
            safety.get("confirmation_required")
            or safety.get("requires_confirmation")
            or safety.get("needs_confirmation")
        )
    if confirmation_required and labels.get("should_confirm") is False:
        errors.append("safety requires confirmation but should_confirm=false")

    verification = record.get("verification")
    if isinstance(verification, dict):
        verified = verification.get("verified")
        status = verification.get("status")
        failed = verified is False or status in {"failed", "failure", "not_verified", "unverified"}
        if failed and state == "complete":
            errors.append("decision.state=complete despite failed/unverified verification")

    if task_mode == "code" or final_mode == "code":
        text = get_nested(record, "final_output", "text")
        if not isinstance(text, str) or not looks_like_complete_code(text):
            errors.append("code output does not contain substantial code-like content")

    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", type=Path, default=None)
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--all-gold", action="store_true", help="Validate all datasets/gold/gold_v0.1_*.jsonl files together")
    p.add_argument("--schema-only", action="store_true", help="Skip semantic/taxonomy checks")
    args = p.parse_args()

    if args.all_gold and args.dataset is not None:
        p.error("dataset path and --all-gold are mutually exclusive")

    validator = Draft202012Validator(load_json(args.schema))
    categories = taxonomy_values(TAXONOMY_DIR / "categories.json", "categories")
    intents = taxonomy_values(TAXONOMY_DIR / "intents.json", "intents")
    output_modes = taxonomy_values(TAXONOMY_DIR / "output_modes.json", "modes")
    reason_codes = taxonomy_values(TAXONOMY_DIR / "reason_codes.json", "reason_codes")

    errors = 0
    records = 0
    ids: dict[str, str] = {}
    files = list(iter_dataset_files(args.dataset, args.all_gold))

    for dataset_file in files:
        with dataset_file.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                if not raw.strip():
                    continue
                records += 1
                source = f"{dataset_file.name}:{line_no}"
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(f"FAIL {source}: invalid JSON: {exc}")
                    errors += 1
                    continue

                rid = record.get("id")
                if isinstance(rid, str):
                    if rid in ids:
                        print(f"FAIL {source}: duplicate id {rid}; first seen at {ids[rid]}")
                        errors += 1
                    else:
                        ids[rid] = source

                for err in sorted(validator.iter_errors(record), key=lambda e: list(e.path)):
                    loc = ".".join(str(x) for x in err.path) or "<root>"
                    print(f"FAIL {rid or source} schema.{loc}: {err.message}")
                    errors += 1

                if not args.schema_only:
                    for message in semantic_errors(record, categories, intents, output_modes, reason_codes):
                        print(f"FAIL {rid or source} semantic: {message}")
                        errors += 1

    scope = "all Gold files" if args.all_gold else files[0].name
    print(f"Validated {records} records from {scope}; errors={errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
