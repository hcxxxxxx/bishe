#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=./src:${PYTHONPATH:-}

TRAIN_MANIFEST=${1:-data/manifests/train.jsonl}
VAL_MANIFEST=${2:-data/manifests/val.jsonl}
OUT_ROOT=${3:-exp/ablation}

for ABL in none rule_only full; do
  torchrun --nproc_per_node=8 --master_port=29511 \
    -m fgemo_tts.train.train \
    --train_manifest "${TRAIN_MANIFEST}" \
    --val_manifest "${VAL_MANIFEST}" \
    --output_dir "${OUT_ROOT}/${ABL}" \
    --ablation "${ABL}" \
    --batch_size 12 \
    --num_workers 8 \
    --max_steps 20000 \
    --eval_every 1000 \
    --save_every 1000

done
