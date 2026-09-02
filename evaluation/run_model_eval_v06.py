#!/usr/bin/env python3
"""Unified KEREN V0.6 evaluator for base models and PEFT adapters.

Key fixes over the early bakeoff scripts:
- base and adapter runs use the same prompt, generation budget, scorer, and output schema
- Qwen-style hidden thinking is separated from the user-visible final answer
- lexical forbidden phrases are not counted when they appear in a nearby explicit negation
- raw output, thinking, final answer, truncation, and category scores are all preserved

The frozen holdout remains read-only. Manual semantic review is still required.
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
NEGATIONS = (
    "nahi", "nahin", "mat", "not", "never", "don't", "dont", "do not",
    "cannot", "can't", "cant", "without", "avoid", "shouldn't", "shouldnt",
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def strip_known_special_tokens(text: str, tokenizer) -> str:
    out = text
    tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    tokens.update({"<|im_end|>", "<|im_start|>", "<|endoftext|>"})
    for tok in sorted(tokens, key=len, reverse=True):
        if tok:
            out = out.replace(tok, "")
    return out.strip()


def split_thinking(raw_output: str, tokenizer, hit_limit: bool) -> tuple[str, str, list[str]]:
    """Return (thinking, final_answer, flags) without scoring hidden reasoning."""
    flags: list[str] = []
    text = raw_output.strip()

    if "</think>" in text:
        before, final = text.rsplit("</think>", 1)
        thinking = before.split("<think>", 1)[-1] if "<think>" in before else before
        return thinking.strip(), strip_known_special_tokens(final, tokenizer), flags

    if "<think>" in text:
        thinking = text.split("<think>", 1)[1]
        if hit_limit:
            flags.append("thinking_truncated")
            return thinking.strip(), "", flags
        return thinking.strip(), "", ["missing_final_answer"]

    return "", strip_known_special_tokens(text, tokenizer), flags


def phrase_is_negated(text: str, phrase: str) -> bool:
    """Conservative local negation check for benchmark lexical exclusions."""
    t = norm(text)
    p = norm(phrase)
    if not p:
        return False
    for m in re.finditer(re.escape(p), t):
        left = t[max(0, m.start() - 48):m.start()]
        right = t[m.end():min(len(t), m.end() + 48)]
        window = f"{left} {right}"
        if any(neg in window for neg in NEGATIONS):
            continue
        return False
    return p in t


def quality_flags(prompt: str, output: str) -> list[str]:
    flags: list[str] = []
    o = norm(output)
    p = norm(prompt)
    if not o:
        return ["empty_final_answer"]
    if "<|keren_" in o:
        flags.append("pseudo_token")
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
        counts = Counter(grams)
        if grams and sum(v - 1 for v in counts.values() if v > 1) / len(grams) >= 0.18:
            flags.append("repetition_loop")
    return flags


def score(case: dict, final_output: str, generation_flags: list[str]) -> tuple[bool, dict]:
    text = norm(final_output)
    required_all = [norm(x) for x in case.get("must_include_all", [])]
    required_any = [norm(x) for x in case.get("must_include_any", [])]
    forbidden = [norm(x) for x in case.get("must_not_include", [])]

    all_ok = all(x in text for x in required_all)
    any_ok = not required_any or any(x in text for x in required_any)
    forbidden_hits = [x for x in forbidden if x and x in text and not phrase_is_negated(final_output, x)]
    flags = list(generation_flags) + quality_flags(case["prompt"], final_output)

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


def render_prompt(tokenizer, user_prompt: str) -> str:
    content = f"{KEREN_POLICY}\n\nTASK:\n{user_prompt}"
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{content}\n\nANSWER:"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Base model ID/path used to instantiate weights and tokenizer")
    p.add_argument("--adapter", type=Path, help="Optional PEFT adapter directory")
    p.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=384)
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
        device_map = {"": 0}
    else:
        dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device_map,
    )
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Adapter evaluation requires peft: pip install -U peft") from exc
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    total_tokens = 0
    total_seconds = 0.0
    category_stats: dict[str, list[int]] = {}

    with args.output.open("w", encoding="utf-8") as dst:
        for i, case in enumerate(cases, 1):
            rendered = render_prompt(tokenizer, case["prompt"])
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
            n_new = int(new_ids.numel())
            hit_limit = n_new >= args.max_new_tokens
            raw_output = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
            thinking, final_output, generation_flags = split_thinking(raw_output, tokenizer, hit_limit)
            ok, detail = score(case, final_output, generation_flags)

            passed += int(ok)
            total_tokens += n_new
            total_seconds += elapsed
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
                "hit_generation_limit": hit_limit,
                "tokens_per_second": round(n_new / elapsed, 3) if elapsed > 0 else None,
                "prompt": case["prompt"],
                "raw_output": raw_output,
                "thinking": thinking,
                "final_output": final_output,
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            dst.flush()
            print(f"[{i:02d}/{len(cases)}] {case['id']} {'PASS' if ok else 'FAIL'} {n_new}tok {elapsed:.1f}s")

    pct = 100.0 * passed / len(cases)
    speed = total_tokens / total_seconds if total_seconds > 0 else 0.0
    print("\n============================")
    print(f"MODEL: {args.model}")
    print(f"ADAPTER: {args.adapter or 'NONE'}")
    print(f"SCORE: {passed}/{len(cases)} = {pct:.1f}%")
    print(f"GENERATION: {total_tokens} tokens / {total_seconds:.1f}s = {speed:.2f} tok/s")
    print("CATEGORY SCORES:")
    for cat in sorted(category_stats):
        pcat, tcat = category_stats[cat]
        print(f"  {cat}: {pcat}/{tcat} = {100.0 * pcat / tcat:.1f}%")
    print(f"RAW: {args.output}")
    print("Manual semantic review is required before accepting an adapter.")
    print("============================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
