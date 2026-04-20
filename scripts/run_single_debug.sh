#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=./src:${PYTHONPATH:-}

COSYVOICE_ROOT=${1:-../CosyVoice}
COSYVOICE_MODEL_DIR=${2:-../CosyVoice/pretrained_models/CosyVoice2-0.5B}
DATA_ROOT=${3:-data/cosyvoice_esd}
EXP_ROOT=${4:-exp/cosyvoice_esd_debug}

python3 -m fgemo_tts.train.train \
  --cosyvoice_root "${COSYVOICE_ROOT}" \
  --cosyvoice_model_dir "${COSYVOICE_MODEL_DIR}" \
  --ablation full \
  --models llm \
  --data_root "${DATA_ROOT}" \
  --exp_root "${EXP_ROOT}" \
  --nproc_per_node 1 \
  --cuda_visible_devices 0 \
  --master_port 29611
