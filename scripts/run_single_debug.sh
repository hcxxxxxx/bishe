#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=./src:${PYTHONPATH:-}
python -m fgemo_tts.train.train \
  --train_manifest data/manifests/train.jsonl \
  --val_manifest data/manifests/val.jsonl \
  --output_dir exp/debug_full \
  --ablation full \
  --batch_size 4 \
  --num_workers 2 \
  --max_steps 200 \
  --eval_every 50 \
  --save_every 100
