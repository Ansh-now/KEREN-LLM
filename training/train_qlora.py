#!/usr/bin/env python3
"""KEREN Student V0.1 QLoRA training entrypoint.
Designed for a CUDA cloud GPU; do not expect useful training on the low-RAM CPU development machine.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Base"

def load_rows(path: Path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            target=json.dumps(r["target"], ensure_ascii=False, separators=(",", ":"))
            rows.append({"text": f"<|keren_user|>\n{r['input']}\n<|keren_target|>\n{target}"})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--output", default="outputs/keren-v0.1-lora"); p.add_argument("--epochs", type=float, default=3.0); args=p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU required for this initial QLoRA recipe.")
    tokenizer=AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model=AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="auto", trust_remote_code=True)
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    train_args=TrainingArguments(output_dir=args.output,num_train_epochs=args.epochs,per_device_train_batch_size=2,gradient_accumulation_steps=8,learning_rate=2e-4,logging_steps=5,save_strategy="epoch",report_to="none",fp16=not torch.cuda.is_bf16_supported(),bf16=torch.cuda.is_bf16_supported(),gradient_checkpointing=True)
    trainer=SFTTrainer(model=model,args=train_args,train_dataset=Dataset.from_list(load_rows(args.dataset)),peft_config=lora,processing_class=tokenizer)
    trainer.train(); trainer.save_model(args.output); tokenizer.save_pretrained(args.output)

if __name__=="__main__": main()
