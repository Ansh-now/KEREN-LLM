#!/usr/bin/env python3
"""Validate the KEREN V0.1 compiled dataset and GPU environment before QLoRA."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--expected-records", type=int, default=400)
    args = p.parse_args()

    ids: set[str] = set()
    modes = Counter()
    count = 0
    with args.dataset.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            for key in ("id", "input", "target"):
                if key not in row:
                    raise SystemExit(f"FAIL line {line_no}: missing {key}")
            rid = str(row["id"])
            if rid in ids:
                raise SystemExit(f"FAIL line {line_no}: duplicate id {rid}")
            ids.add(rid)
            count += 1
            target = row["target"]
            if isinstance(target, dict):
                if "mode" in target:
                    modes[str(target["mode"])] += 1
                elif "state" in target:
                    modes["state"] += 1
                else:
                    modes["structured"] += 1
            else:
                modes["text"] += 1

    print(f"Dataset records: {count}")
    if count != args.expected_records:
        raise SystemExit(f"FAIL expected {args.expected_records} records, got {count}")
    print(f"Unique ids: {len(ids)}")
    print("Target shapes:", dict(sorted(modes.items())))
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM GiB: {props.total_memory / 1024**3:.2f}")
    else:
        print("NOTE: dataset preflight passed, but training requires a CUDA GPU runtime.")

    print("PREFLIGHT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
