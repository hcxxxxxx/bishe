#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=./src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

COSYVOICE_ROOT=${1:-../CosyVoice}
COSYVOICE_MODEL_DIR=${2:-../CosyVoice/pretrained_models/CosyVoice2-0.5B}
DATA_ROOT=${3:-data/cosyvoice_esd}
EXP_ROOT=${4:-exp/cosyvoice_esd}
MODELS=${5:-llm,flow}

for ABL in none rule_only full; do
  python3 -m fgemo_tts.train.train \
    --cosyvoice_root "${COSYVOICE_ROOT}" \
    --cosyvoice_model_dir "${COSYVOICE_MODEL_DIR}" \
    --ablation "${ABL}" \
    --models "${MODELS}" \
    --data_root "${DATA_ROOT}" \
    --exp_root "${EXP_ROOT}" \
    --nproc_per_node 8 \
    --master_port 29511

done
