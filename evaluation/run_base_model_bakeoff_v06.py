#!/usr/bin/env python3
"""Run the frozen KEREN V0.6 holdout against an unadapted model.

Adapter-free comparison only. Uses each tokenizer's native chat template and the
same shared KEREN policy for every candidate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.keren_policy import KEREN_POLICY

DEFAULT_BENCH = ROOT / "evaluation" / "benchmark_v0.6_holdout.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def quality_flags(prompt: str, output: str) -> list[str]:
    flags: list[str] = []
    o = norm(output)
    p = norm(prompt)
    if not o:
        return ["empty"]
    if "<|" in output or "|>" in output:
        flags.append("pseudo_special_token")
    if re.search(r"\b(?:assistant|system|user)\s*:", output, re.I):
        flags.append("role_leak")
    labels = sum(bool(re.search(rf"(?m)^\s*{x}[\).:]", output, re.I)) for x in "ABCD")
    if labels >= 2:
        flags.append("fabricated_mcq")
    if "code only" in p or "sirf expression" in p or "sirf function" in p:
        non_code = re.sub(r"```[\s\S]*?```", "", output).strip()
        if "```" in output and len(non_code.split()) > 3:
            flags.append("extra_text_after_code")
    words = re.findall(r"\w+", o)
    if len(words) >= 28:
        grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        c = Counter(grams)
        if grams and sum(v-1 for v in c.values() if v > 1) / len(grams) >= 0.18:
            flags.append("repetition")
    return flags


def score(case: dict, output: str) -> tuple[bool, dict]:
    text = norm(output)
    required_all = [norm(x) for x in case.get("must_include_all", [])]
    required_any = [norm(x) for x in case.get("must_include_any", [])]
    forbidden = [norm(x) for x in case.get("must_not_include", [])]
    all_ok = all(x in text for x in required_all)
    any_ok = not required_any or any(x in text for x in required_any)
    forbidden_hits = [x for x in forbidden if x and x in text]
    flags = quality_flags(case["prompt"], output)

    persona_ok = True
    if case.get("persona") == "female_hinglish":
        masculine = ["kar raha hoon", "karunga", "samajh gaya", "dekh raha hoon", "bataunga"]
        persona_ok = not any(x in text for x in masculine)

    passed = all_ok and any_ok and not forbidden_hits and not flags and persona_ok
    return passed, {
        "all_ok": all_ok,
        "any_ok": any_ok,
        "forbidden_hits": forbidden_hits,
        "quality_flags": flags,
        "persona_ok": persona_ok,
    }


def format_prompt(tokenizer, user_prompt: str) -> str:
    # Fold policy into the user turn so models whose templates reject a system
    # role (including some lightweight instruction models) are still compared
    # with identical semantic context.
    content = f"{KEREN_POLICY}\n\nTASK:\n{user_prompt}"
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{content}\n\nANSWER:"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--ids", nargs="*")
    args = p.parse_args()

    cases = load_jsonl(args.benchmark)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("Benchmark has duplicate IDs")
    if args.ids:
        wanted = set(args.ids)
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        raise SystemExit("No benchmark cases selected")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    total_tokens = 0
    total_seconds = 0.0
    category_stats: dict[str, list[int]] = {}

    with args.output.open("w", encoding="utf-8") as dst:
        for i, case in enumerate(cases, 1):
            rendered = format_prompt(tokenizer, case["prompt"])
            inputs = tokenizer(rendered, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.inference_mode():
                ids_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.08,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.perf_counter() - t0
            new_ids = ids_out[0, inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            n_new = int(new_ids.numel())
            total_tokens += n_new
            total_seconds += elapsed
            ok, detail = score(case, output)
            passed += int(ok)
            cat = case.get("category", "unknown")
            category_stats.setdefault(cat, [0, 0])
            category_stats[cat][0] += int(ok)
            category_stats[cat][1] += 1
            row = {
                "id": case["id"],
                "category": cat,
                "passed": ok,
                "score_detail": detail,
                "latency_seconds": round(elapsed, 4),
                "new_tokens": n_new,
                "tokens_per_second": round(n_new / elapsed, 3) if elapsed > 0 else None,
                "prompt": case["prompt"],
                "output": output,
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i:02d}/{len(cases)}] {case['id']}: {'PASS' if ok else 'FAIL'} | {elapsed:.2f}s | {n_new} tok")

    pct = 100.0 * passed / len(cases)
    speed = total_tokens / total_seconds if total_seconds > 0 else 0.0
    print(f"MODEL: {args.model}")
    print(f"SCORE: {passed}/{len(cases)} = {pct:.1f}%")
    print(f"GENERATION: {total_tokens} tokens / {total_seconds:.2f}s = {speed:.2f} tok/s")
    print("CATEGORY SCORES:")
    for cat in sorted(category_stats):
        pcat, tcat = category_stats[cat]
        print(f"  {cat}: {pcat}/{tcat} = {100.0*pcat/tcat:.1f}%")
    print(f"RAW: {args.output}")
    print("Manual semantic review is required before selecting a base model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
