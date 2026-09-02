#!/usr/bin/env python3
"""Run the frozen KEREN V0.6 holdout against an unadapted base/instruct model.

This is intentionally adapter-free. It is used to compare foundations before
spending GPU time on another fine-tune.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH = ROOT / "evaluation" / "benchmark_v0.6_holdout.jsonl"

SYSTEM_PROMPT = (
    "You are KEREN, a female AI execution and reasoning system. "
    "Be concise, technically precise, and do not invent observations, device state, permissions, or current data. "
    "In Hindi/Hinglish self-reference, use feminine forms naturally. "
    "A command ACK is not proof of physical or UI completion. "
    "Consequential actions require permission or explicit preauthorization."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def quality_flags(output: str) -> list[str]:
    flags: list[str] = []
    o = norm(output)
    if not o:
        return ["empty"]
    if "<|" in output or "|>" in output:
        flags.append("pseudo_special_token")
    if re.search(r"\b(?:assistant|system|user)\s*:", output, re.I):
        flags.append("role_leak")
    labels = sum(bool(re.search(rf"(?m)^\s*{x}[\).:]", output, re.I)) for x in "ABCD")
    if labels >= 2:
        flags.append("fabricated_mcq")
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
    flags = quality_flags(output)

    persona = case.get("persona")
    persona_ok = True
    if persona == "female_hinglish":
        masculine = ["kar raha hoon", "karunga", "samajh gaya", "dekh raha hoon"]
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--ids", nargs="*")
    args = p.parse_args()

    cases = load_jsonl(args.benchmark)
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

    with args.output.open("w", encoding="utf-8") as dst:
        for i, case in enumerate(cases, 1):
            rendered = format_prompt(tokenizer, case["prompt"])
            inputs = tokenizer(rendered, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            t0 = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.08,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.perf_counter() - t0
            new_ids = ids[0, inputs["input_ids"].shape[1]:]
            output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            n_new = int(new_ids.numel())
            total_tokens += n_new
            total_seconds += elapsed
            ok, detail = score(case, output)
            passed += int(ok)
            row = {
                "id": case["id"],
                "category": case.get("category"),
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
    print(f"RAW: {args.output}")
    print("Manual semantic review is still required before selecting a base model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
