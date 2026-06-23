#!/usr/bin/env bash
set -euo pipefail
CATEGORY=${1:-bottle}
IMAGE_SIZE=${2:-256}
trtexec \
  --onnx=artifacts/${CATEGORY}/feature_extractor.onnx \
  --saveEngine=artifacts/${CATEGORY}/feature_extractor_fp16.engine \
  --fp16 \
  --minShapes=input:1x3x${IMAGE_SIZE}x${IMAGE_SIZE} \
  --optShapes=input:8x3x${IMAGE_SIZE}x${IMAGE_SIZE} \
  --maxShapes=input:16x3x${IMAGE_SIZE}x${IMAGE_SIZE}
