#!/usr/bin/env python3
"""KEREN Student V0.1 QLoRA training entrypoint.

Designed for a CUDA GPU runtime (Kaggle/Colab). The base model remains frozen;
only a LoRA adapter is trained. The script consumes the compiled dataset shape:
{"id": ..., "input": ..., "target": ...}.
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

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"
USER_MARKER = "<|keren_user|>\n"
TARGET_MARKER = "\n<|keren_target|>\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=Path("outputs/keren-student-v0.1"))
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", default=None)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            for key in ("id", "input", "target"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing key {key!r}")
            record_id = str(row["id"])
            if record_id in ids:
                raise ValueError(f"Duplicate training id: {record_id}")
            ids.add(record_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def render_target(target: object) -> str:
    if isinstance(target, str):
        return target
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def build_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    def encode(row: dict) -> dict:
        prompt = f"{USER_MARKER}{row['input']}{TARGET_MARKER}"
        answer = render_target(row["target"])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        input_ids = (prompt_ids + answer_ids)[:max_length]
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    ds = Dataset.from_list(rows)
    return ds.map(encode, remove_columns=ds.column_names, desc="Tokenizing KEREN Gold")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Use a Kaggle/Colab GPU runtime for QLoRA training.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_rows(args.dataset)
    print(f"Loaded training rows: {len(rows)}")
    print(f"Base model: {args.model}")
    print(f"Output adapter: {args.output}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_dataset = build_dataset(rows, tokenizer, args.max_length)
    lengths = [len(x["input_ids"]) for x in train_dataset]
    print(
        "Token lengths: "
        f"min={min(lengths)} max={max(lengths)} avg={sum(lengths)/len(lengths):.1f} "
        f"limit={args.max_length}"
    )

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

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("Starting KEREN Student V0.1 QLoRA training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))

    summary = {
        "base_model": args.model,
        "dataset": str(args.dataset),
        "records": len(rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "adapter_output": str(args.output),
    }
    with (args.output / "keren_training_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training complete.")
    print(f"Adapter saved -> {args.output}")
    print("Base weights were not overwritten; deploy this directory as a PEFT/LoRA adapter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
