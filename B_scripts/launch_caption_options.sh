#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1
GPU_ID=${2:-0}

CUDA_VISIBLE_DEVICES=${GPU_ID} \
python src/inference_caption_options.py \
  --exp_config "${CONFIG}" \
  --device 0
