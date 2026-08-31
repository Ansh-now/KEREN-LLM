#!/usr/bin/env python3
"""Check that a KEREN LoRA output directory contains the expected PEFT artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("adapter", type=Path)
    args = p.parse_args()

    required = [
        "adapter_config.json",
        "keren_training_manifest.json",
    ]
    missing = [name for name in required if not (args.adapter / name).exists()]
    weight_candidates = [
        args.adapter / "adapter_model.safetensors",
        args.adapter / "adapter_model.bin",
    ]
    if not any(path.exists() for path in weight_candidates):
        missing.append("adapter_model.safetensors|adapter_model.bin")
    if missing:
        raise SystemExit("FAIL missing: " + ", ".join(missing))

    with (args.adapter / "keren_training_manifest.json").open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("ADAPTER CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
