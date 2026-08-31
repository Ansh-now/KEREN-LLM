#!/usr/bin/env python3
"""Run the locked KEREN V0.1 unseen benchmark against a local HF causal LM.

This is intentionally a lightweight lexical/contract benchmark. It records raw
model output so manual review can supplement the automatic score.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "evaluation" / "benchmark_v0.1.jsonl"
DEFAULT_OUT = ROOT / "evaluation" / "results" / "benchmark_v0.1_results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def score_case(case: dict, output: str) -> tuple[bool, dict]:
    text = norm(output)
    required = [norm(x) for x in case.get("must_include_any", [])]
    forbidden = [norm(x) for x in case.get("must_not_include", [])]
    include_ok = True if not required else any(x in text for x in required)
    forbidden_hits = [x for x in forbidden if x and x in text]
    passed = include_ok and not forbidden_hits
    return passed, {
        "include_ok": include_ok,
        "forbidden_hits": forbidden_hits,
        "required_any": case.get("must_include_any", []),
    }


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    rendered = f"<|keren_user|>\n{prompt}\n<|keren_target|>\n"
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local model directory")
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-new-tokens", type=int, default=220)
    args = p.parse_args()

    cases = load_jsonl(args.benchmark)
    if not cases:
        raise SystemExit("Benchmark is empty")
    ids = [x["id"] for x in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate benchmark IDs")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading model: {args.model}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed_count = 0
    category_stats: dict[str, list[int]] = {}
    with args.output.open("w", encoding="utf-8") as dst:
        for index, case in enumerate(cases, start=1):
            output = generate(model, tokenizer, case["prompt"], args.max_new_tokens)
            passed, detail = score_case(case, output)
            passed_count += int(passed)
            cat = case["category"]
            category_stats.setdefault(cat, [0, 0])
            category_stats[cat][0] += int(passed)
            category_stats[cat][1] += 1
            result = {
                "id": case["id"],
                "category": cat,
                "mode": case["mode"],
                "passed": passed,
                "score_detail": detail,
                "prompt": case["prompt"],
                "output": output,
            }
            dst.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{index:02d}/{len(cases)}] {case['id']}: {'PASS' if passed else 'FAIL'}")

    pct = 100.0 * passed_count / len(cases)
    print(f"\nKEREN Benchmark V0.1: {passed_count}/{len(cases)} = {pct:.1f}%")
    for cat in sorted(category_stats):
        got, total = category_stats[cat]
        print(f"  {cat}: {got}/{total}")
    print(f"Raw results -> {args.output}")
    print("NOTE: automatic score is lexical; inspect raw outputs before making quality claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
