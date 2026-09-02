#!/usr/bin/env python3
"""KEREN V0.6+ QLoRA trainer using the model's native chat template.

Key differences from the legacy trainer:
- no handwritten <|keren_...|> pseudo-special markers
- shared explicit KEREN female/execution policy
- prompt tokens masked from loss; only assistant answer tokens supervised
- hard preflight checks for template availability and mask boundary correctness
- CPU-safe --validate-only mode that exits before any model/GPU training setup
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

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
        help="Validate dataset/chat-template/assistant-only masking on CPU and exit before model loading/training.",
    )
    return p.parse_args()


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
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def render_target(target: object) -> str:
    return target if isinstance(target, str) else json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def user_content(raw_input: str) -> str:
    return f"{KEREN_POLICY}\n\nTASK:\n{raw_input}"


def build_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("BLOCK_TRAINING: tokenizer has no native chat_template")

    def encode(row: dict) -> dict:
        user = user_content(str(row["input"]))
        answer = render_target(row["target"])
        prompt_messages = [{"role": "user", "content": user}]
        full_messages = prompt_messages + [{"role": "assistant", "content": answer}]

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
        if isinstance(full_ids, torch.Tensor):
            full_ids = full_ids.tolist()
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat-template boundary mismatch for {row['id']}: "
                "prompt is not an exact prefix of full conversation"
            )

        input_ids = full_ids[:max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        if not any(x != -100 for x in labels):
            raise ValueError(f"No supervised assistant tokens remain for {row['id']}")
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    ds = Dataset.from_list(rows)
    return ds.map(encode, remove_columns=ds.column_names, desc="Tokenizing KEREN V0.6")


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("BLOCK_TRAINING: selected model tokenizer has no native chat_template")

    train_dataset = build_dataset(rows, tokenizer, args.max_length)
    lengths = [len(x["input_ids"]) for x in train_dataset]
    supervised = [sum(1 for t in x["labels"] if t != -100) for x in train_dataset]
    print(f"Loaded rows: {len(rows)}")
    print(f"Base model: {args.model}")
    print(f"Token lengths: min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f}")
    print(f"Supervised answer tokens: min={min(supervised)} max={max(supervised)} avg={sum(supervised)/len(supervised):.1f}")
    print("FORMAT CHECK PASS: native chat template + assistant-only loss")

    if args.validate_only:
        print("VALIDATION ONLY PASS: no model weights loaded and no training started")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for QLoRA training")
    torch.cuda.manual_seed_all(args.seed)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    model.print_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
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
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=train_dataset, data_collator=collator)
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    manifest = {
        "format_version": "v0.6-native-chat-template",
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
    }
    with (args.output / "keren_training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Training complete -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
