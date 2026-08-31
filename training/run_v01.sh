#!/usr/bin/env bash
set -euo pipefail

python training/preflight_v01.py \
  --dataset datasets/compiled/keren_train_v0.1.jsonl \
  --expected-records 400

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
  --warmup-ratio 0.05 \
  --seed 42

python training/check_adapter_v01.py outputs/keren-student-v0.1
