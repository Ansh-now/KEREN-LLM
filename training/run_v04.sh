#!/usr/bin/env bash
set -euo pipefail

python generators/generate_v04_precision_repair.py
python generators/compile_training_dataset_v02.py \
  --output datasets/compiled/keren_train_v0.4.jsonl \
  --expected-records 820
python training/preflight_v01.py \
  --dataset datasets/compiled/keren_train_v0.4.jsonl \
  --expected-records 820
python training/train_qlora.py \
  --dataset datasets/compiled/keren_train_v0.4.jsonl \
  --model Qwen/Qwen3-0.6B-Base \
  --output outputs/keren-student-v0.4 \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 8 \
  --learning-rate 1.5e-4 \
  --max-length 2048 \
  --save-steps 25 \
  --logging-steps 5 \
  --warmup-ratio 0.05 \
  --seed 42
python training/check_adapter_v01.py outputs/keren-student-v0.4
