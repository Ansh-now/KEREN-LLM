#!/usr/bin/env python3
"""Validate KEREN JSONL records against Dataset V0.1 schema."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schemas" / "keren_dataset_v0.1.schema.json"
DEFAULT_DATASET = ROOT / "datasets" / "gold" / "gold_v0.1_seed.jsonl"

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = p.parse_args()
    validator = Draft202012Validator(load_json(args.schema))
    errors = 0; records = 0; ids = set()
    with args.dataset.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            if not raw.strip(): continue
            records += 1
            try: record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"FAIL line {line_no}: invalid JSON: {exc}"); errors += 1; continue
            rid = record.get("id")
            if rid in ids:
                print(f"FAIL line {line_no}: duplicate id {rid}"); errors += 1
            ids.add(rid)
            for err in sorted(validator.iter_errors(record), key=lambda e: list(e.path)):
                loc = ".".join(str(x) for x in err.path) or "<root>"
                print(f"FAIL {rid or line_no} {loc}: {err.message}"); errors += 1
    print(f"Validated {records} records; errors={errors}")
    return 1 if errors else 0

if __name__ == "__main__": raise SystemExit(main())
