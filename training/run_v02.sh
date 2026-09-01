#!/usr/bin/env bash
set -euo pipefail

# KEREN Student V0.2: train a fresh adapter from the same frozen base model
# using the complete validated 550-record Gold mix. On multi-GPU Kaggle
# runtimes, invoke with CUDA_VISIBLE_DEVICES=0 to avoid device_map/Trainer
# placement conflicts observed during V0.1.

python training/preflight_v01.py \
  --dataset datasets/compiled/keren_train_v0.2.jsonl \
  --expected-records 550

python training/train_qlora.py \
  --dataset datasets/compiled/keren_train_v0.2.jsonl \
  --model Qwen/Qwen3-0.6B-Base \
  --output outputs/keren-student-v0.2 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --learning-rate 2e-4 \
  --max-length 2048 \
  --save-steps 25 \
  --logging-steps 5 \
  --warmup-ratio 0.05 \
  --seed 42

python training/check_adapter_v01.py outputs/keren-student-v0.2
