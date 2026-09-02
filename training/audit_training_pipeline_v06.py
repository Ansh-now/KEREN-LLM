#!/usr/bin/env python3
"""KEREN V0.6 pre-training audit.

CPU-only checks. This does not train anything.
It inspects:
- custom marker tokenization
- target masking boundaries
- exact/near duplicate prompts
- conflicting targets for identical prompts
- suspicious architecture/persona terms
- masculine/feminine Hindi self-reference signals
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

USER_MARKER = "<|keren_user|>\n"
TARGET_MARKER = "\n<|keren_target|>\n"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            for key in ("id", "input", "target"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing {key}")
            rows.append(row)
    return rows


def norm(text: object) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    words = re.findall(r"[\w.+#-]+", norm(text))
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_near_duplicates(rows: list[dict], threshold: float) -> list[tuple[str, str, float]]:
    grams = [(str(r["id"]), ngrams(str(r["input"]))) for r in rows]
    hits: list[tuple[str, str, float]] = []
    for i in range(len(grams)):
        id_a, a = grams[i]
        for j in range(i + 1, len(grams)):
            id_b, b = grams[j]
            s = jaccard(a, b)
            if s >= threshold:
                hits.append((id_a, id_b, s))
    hits.sort(key=lambda x: x[2], reverse=True)
    return hits


def show_marker_audit(tokenizer) -> None:
    print("\n=== MARKER TOKENIZATION ===")
    for label, text in (("USER_MARKER", USER_MARKER), ("TARGET_MARKER", TARGET_MARKER)):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        pieces = tokenizer.convert_ids_to_tokens(ids)
        print(f"{label}: {text!r}")
        print(f"  token_ids={ids}")
        print(f"  pieces={pieces}")
        print(f"  token_count={len(ids)}")
    vocab = tokenizer.get_vocab()
    for token in ("<|keren_user|>", "<|keren_target|>"):
        print(f"registered_exact_token[{token}]={token in vocab}")


def show_mask_audit(rows: list[dict], tokenizer, max_length: int) -> None:
    print("\n=== TARGET MASK AUDIT ===")
    for row in rows[:3]:
        prompt = f"{USER_MARKER}{row['input']}{TARGET_MARKER}"
        answer = row["target"] if isinstance(row["target"], str) else json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + answer_ids)[:max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        supervised = sum(x != -100 for x in labels)
        print(f"{row['id']}: total={len(input_ids)} prompt={prompt_len} supervised={supervised}")
        print("  boundary_before=", tokenizer.decode(input_ids[max(0, prompt_len-12):prompt_len]))
        print("  boundary_target=", tokenizer.decode(input_ids[prompt_len:prompt_len+20]))
        assert all(x == -100 for x in labels[:prompt_len])
        assert all(x != -100 for x in labels[prompt_len:])


def term_hits(rows: Iterable[dict], terms: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {t: [] for t in terms}
    for row in rows:
        text = norm(str(row["input"]) + "\n" + norm(row["target"]))
        for term in terms:
            if term.casefold() in text:
                out[term].append(str(row["id"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("datasets/compiled/keren_train_v0.5.jsonl"))
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--near-dup-threshold", type=float, default=0.82)
    p.add_argument("--near-dup-limit", type=int, default=50)
    args = p.parse_args()

    rows = load_jsonl(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Records: {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    show_marker_audit(tokenizer)
    show_mask_audit(rows, tokenizer, args.max_length)

    print("\n=== EXACT DUPLICATES / CONFLICTS ===")
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_prompt[norm(row["input"])].append(row)
    duplicate_groups = [g for g in by_prompt.values() if len(g) > 1]
    conflicts = []
    for group in duplicate_groups:
        targets = {norm(r["target"]) for r in group}
        if len(targets) > 1:
            conflicts.append(group)
    print(f"exact_duplicate_prompt_groups={len(duplicate_groups)}")
    print(f"conflicting_exact_prompt_groups={len(conflicts)}")
    for group in conflicts[:20]:
        print("CONFLICT", [r["id"] for r in group])
        for r in group:
            print("  TARGET:", norm(r["target"])[:240])

    print("\n=== NEAR DUPLICATES ===")
    near = find_near_duplicates(rows, args.near_dup_threshold)
    print(f"near_duplicate_pairs>={args.near_dup_threshold:.2f}: {len(near)}")
    for a, b, score in near[:args.near_dup_limit]:
        print(f"{a} <-> {b}: {score:.3f}")

    print("\n=== SUSPICIOUS FACT/PERSONA SIGNALS ===")
    terms = [
        "xtensa lx6", "xtensa lx7", "risc-v", "8051", "arm",
        "kar raha hoon", "kar rahi hoon", "samajh gaya", "samajh gayi",
        "main keren hoon", "female", "woman", "girl",
        "<|keren_", "<|keran_",
    ]
    hits = term_hits(rows, terms)
    for term, ids in hits.items():
        print(f"{term!r}: {len(ids)} {ids[:30]}")

    masculine = sum(len(hits[t]) for t in ("kar raha hoon", "samajh gaya"))
    feminine = sum(len(hits[t]) for t in ("kar rahi hoon", "samajh gayi"))
    print(f"persona_signal_masculine={masculine}")
    print(f"persona_signal_feminine={feminine}")

    print("\n=== RECOMMENDATION GATES ===")
    marker_registered = all(t in tokenizer.get_vocab() for t in ("<|keren_user|>", "<|keren_target|>"))
    print("custom_markers_registered=", marker_registered)
    print("BLOCK_TRAINING if conflicting_exact_prompt_groups > 0")
    print("BLOCK_TRAINING if custom markers are emitted in targets or not intentionally registered")
    print("BLOCK_TRAINING until persona policy is explicit and holdout benchmark is frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
