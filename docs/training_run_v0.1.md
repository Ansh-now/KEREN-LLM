# KEREN Student V0.1 GPU Training Runbook

This runbook trains a QLoRA adapter on the compiled 400-record Gold V0.1 dataset. The base model stays frozen.

## Prerequisites

- CUDA GPU runtime (Kaggle or Colab)
- repository checkout containing `datasets/compiled/keren_train_v0.1.jsonl`
- Python training dependencies from `requirements-training.txt`

## Install

```bash
pip install -r requirements-training.txt
```

## Preflight

```bash
python - <<'PY'
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16:', torch.cuda.is_bf16_supported())
PY

wc -l datasets/compiled/keren_train_v0.1.jsonl
```

Expected dataset line count: `400`.

## Train

```bash
python training/train_qlora.py \
  --dataset datasets/compiled/keren_train_v0.1.jsonl \
  --model Qwen/Qwen3-0.6B-Base \
  --output outputs/keren-student-v0.1 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --learning-rate 2e-4 \
  --max-length 2048 \
  --save-steps 25 \
  --logging-steps 5 \
  --seed 42
```

The effective batch size is 16 examples per optimizer step. Checkpoints are saved during the run and only the latest three are retained.

## Resume after interruption

Find a checkpoint under `outputs/keren-student-v0.1/checkpoint-*`, then run:

```bash
python training/train_qlora.py \
  --dataset datasets/compiled/keren_train_v0.1.jsonl \
  --model Qwen/Qwen3-0.6B-Base \
  --output outputs/keren-student-v0.1 \
  --resume-from-checkpoint outputs/keren-student-v0.1/checkpoint-N
```

Use the same hyperparameters as the original run when resuming.

## Expected output

The final directory contains the PEFT/LoRA adapter, tokenizer files, checkpoints (subject to retention), and `keren_training_manifest.json`. Do not commit base-model weights or training outputs to GitHub.

## Evaluation rule

Do not alter or train on `evaluation/benchmark_v0.1.jsonl`. Compare the trained adapter against the frozen pre-training baseline using the exact same locked 30-case benchmark. Automatic lexical scores must be supplemented by raw-output inspection before making quality claims.
