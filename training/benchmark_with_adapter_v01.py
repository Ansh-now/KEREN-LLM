#!/usr/bin/env python3
"""Run the locked KEREN benchmark against a base model plus PEFT adapter."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
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


def quality_flags(prompt: str, output: str) -> list[str]:
    flags: list[str] = []
    p, o = norm(prompt), norm(output)
    if not o:
        return ["empty_output"]
    if o == p or (len(p) >= 24 and p in o):
        flags.append("prompt_echo")
    if "<|" in output or "|>" in output:
        flags.append("special_token_artifact")
    if re.search(r"\b(?:assistant|system)\s*:", output, re.IGNORECASE):
        flags.append("role_leak")

    prompt_has_options = bool(re.search(r"(?:\boption\b|\ba\)|\bb\)|\bc\)|\bd\))", p))
    output_has_mcq = sum(bool(re.search(rf"(?m)^\s*{letter}[\).:]", output, re.IGNORECASE)) for letter in "ABCD") >= 2
    if output_has_mcq and not prompt_has_options:
        flags.append("fabricated_mcq")

    if "sirf code" in p or "code only" in p or "sirf function" in p:
        after_block = re.sub(r"```[\s\S]*?```", "", output).strip()
        if after_block and len(after_block.split()) > 3:
            flags.append("extra_text_after_code")

    words = re.findall(r"\w+", o)
    if len(words) >= 24:
        ngrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
        if ngrams:
            counts = Counter(ngrams)
            repeated = sum(c - 1 for c in counts.values() if c > 1)
            if repeated / len(ngrams) >= 0.18:
                flags.append("repetition_loop")
    return flags


def score_case(case: dict, output: str) -> tuple[bool, dict]:
    text = norm(output)
    required = [norm(x) for x in case.get("must_include_any", [])]
    forbidden = [norm(x) for x in case.get("must_not_include", [])]
    include_ok = not required or any(x in text for x in required)
    forbidden_hits = [x for x in forbidden if x and x in text]
    flags = quality_flags(case["prompt"], output)
    passed = include_ok and not forbidden_hits and not flags
    return passed, {
        "include_ok": include_ok,
        "forbidden_hits": forbidden_hits,
        "required_any": case.get("must_include_any", []),
        "quality_flags": flags,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--ids", nargs="*", help="Optional benchmark IDs to run")
    args = p.parse_args()

    tokenizer_source = args.adapter if (args.adapter / "tokenizer.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    cases = load_jsonl(args.benchmark)
    if args.ids:
        wanted = set(args.ids)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            raise SystemExit(f"Unknown benchmark IDs: {', '.join(sorted(missing))}")
    if not cases:
        raise SystemExit("Benchmark is empty")

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
                    max_new_tokens=min(args.max_new_tokens, 120),
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = ids[0, inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            ok, detail = score_case(case, output)
            passed += int(ok)
            dst.write(json.dumps({
                "id": case["id"],
                "passed": ok,
                "score_detail": detail,
                "prompt": case["prompt"],
                "output": output,
            }, ensure_ascii=False) + "\n")
            flags = detail["quality_flags"]
            suffix = f" [{','.join(flags)}]" if flags else ""
            print(f"[{idx:02d}/{len(cases)}] {case['id']}: {'PASS' if ok else 'FAIL'}{suffix}")

    pct = 100.0 * passed / len(cases)
    print(f"KEREN Benchmark V0.1 with adapter: {passed}/{len(cases)} = {pct:.1f}%")
    print(f"Raw results -> {args.output}")
    print("NOTE: score includes lexical and generation-quality guards; manual semantic review remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
