#!/usr/bin/env python3
"""KEREN V0.6+ QLoRA trainer.

Design goals:
- use each model's native chat template
- never hand-invent pseudo-special KEREN markers
- never reverse-engineer assistant token boundaries
- represent every example as conversational prompt + completion
- delegate completion-only loss masking to Hugging Face TRL
- keep --validate-only CPU-safe and model-weight-free
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig

from keren_policy import KEREN_POLICY

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all rows and native chat-template rendering on CPU, then exit before model loading/training.",
    )
    return p.parse_args()


def render_target(target: object) -> str:
    if isinstance(target, str):
        return target
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def user_content(raw_input: str) -> str:
    return f"{KEREN_POLICY}\n\nTASK:\n{raw_input}"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            for key in ("id", "input", "target"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing {key!r}")
            rid = str(row["id"])
            if rid in ids:
                raise ValueError(f"Duplicate training id: {rid}")
            ids.add(rid)
            answer = render_target(row["target"])
            if not str(row["input"]).strip():
                raise ValueError(f"Empty input for {rid}")
            if not answer.strip():
                raise ValueError(f"Empty target for {rid}")
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def build_prompt_completion_dataset(rows: list[dict]) -> Dataset:
    records: list[dict] = []
    for row in rows:
        records.append({
            "id": str(row["id"]),
            "prompt": [{"role": "user", "content": user_content(str(row["input"]))}],
            "completion": [{"role": "assistant", "content": render_target(row["target"])}],
        })
    return Dataset.from_list(records)


def validate_template(rows: list[dict], tokenizer, max_length: int) -> None:
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("BLOCK_TRAINING: selected tokenizer has no native chat_template")

    lengths: list[int] = []
    over_limit = 0
    for row in rows:
        messages = [
            {"role": "user", "content": user_content(str(row["input"]))},
            {"role": "assistant", "content": render_target(row["target"])},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(rendered, str) or not rendered.strip():
            raise ValueError(f"Native chat template rendered empty/non-text output for {row['id']}")
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError(f"Unexpected batched tokenization for {row['id']}")
            ids = ids[0]
        n = len(ids)
        if n <= 2:
            raise ValueError(f"Suspicious native-template token length={n} for {row['id']}")
        lengths.append(n)
        if n > max_length:
            over_limit += 1

    print(f"Loaded rows: {len(rows)}")
    print(f"Base model: {tokenizer.name_or_path}")
    print(f"Native-template token lengths: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f}")
    print(f"Rows above max_length={max_length}: {over_limit}")
    print("FORMAT CHECK PASS: conversational prompt/completion + native chat template")
    print("LOSS POLICY: TRL completion_only_loss=True; no manual token-boundary masking")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    validate_template(rows, tokenizer, args.max_length)
    train_dataset = build_prompt_completion_dataset(rows)

    if args.validate_only:
        print("VALIDATION ONLY PASS: all rows rendered; no model weights loaded and no training started")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for QLoRA training")

    try:
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Missing training dependency. Install with: pip install -U trl peft bitsandbytes accelerate") from exc

    torch.cuda.manual_seed_all(args.seed)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    sft_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        model_init_kwargs={"quantization_config": quant, "device_map": "auto", "trust_remote_code": True},
    )
    trainer = SFTTrainer(
        model=args.model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    manifest = {
        "format_version": "v0.6-trl-conversational-prompt-completion",
        "base_model": args.model,
        "dataset": str(args.dataset),
        "records": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "female_persona_policy": True,
        "pseudo_keren_markers": False,
        "completion_only_loss": True,
    }
    with (args.output / "keren_training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Training complete -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
