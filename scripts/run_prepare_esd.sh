#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=./src:${PYTHONPATH:-}

ESD_ROOT=${1:-../dataset_esd_sorted}
COSYVOICE_ROOT=${2:-../CosyVoice}
OUT_ROOT=${3:-data/cosyvoice_esd}

python3 scripts/prepare_cosyvoice_esd_data.py \
  --esd_root "${ESD_ROOT}" \
  --cosyvoice_root "${COSYVOICE_ROOT}" \
  --out_root "${OUT_ROOT}"

echo "CosyVoice ESD data prepared: ${OUT_ROOT}"
