#!/usr/bin/env python3
"""Run the locked KEREN benchmark against a base model plus PEFT adapter."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "evaluation" / "benchmark_v0.1.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def score_case(case: dict, output: str) -> bool:
    text = norm(output)
    required = [norm(x) for x in case.get("must_include_any", [])]
    forbidden = [norm(x) for x in case.get("must_not_include", [])]
    return (not required or any(x in text for x in required)) and not any(x and x in text for x in forbidden)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=220)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    cases = load_jsonl(args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    with args.output.open("w", encoding="utf-8") as dst:
        for idx, case in enumerate(cases, start=1):
            prompt = f"<|keren_user|>\n{case['prompt']}\n<|keren_target|>\n"
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = ids[0, inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            ok = score_case(case, output)
            passed += int(ok)
            dst.write(json.dumps({"id": case["id"], "passed": ok, "prompt": case["prompt"], "output": output}, ensure_ascii=False) + "\n")
            print(f"[{idx:02d}/{len(cases)}] {case['id']}: {'PASS' if ok else 'FAIL'}")

    pct = 100.0 * passed / len(cases)
    print(f"KEREN Benchmark V0.1 with adapter: {passed}/{len(cases)} = {pct:.1f}%")
    print(f"Raw results -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
