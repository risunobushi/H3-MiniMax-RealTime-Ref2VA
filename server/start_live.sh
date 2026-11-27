#!/usr/bin/env bash
# Start the live mirror server on the pod. Options come from the environment (see README.md):
#   ASPECTS=16:9,4:3,3:4,1:1 STEPS=3 bash start_live.sh
set -euo pipefail
cd /workspace/Minimax-H3-Turbo
source env_live.sh
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
# the largest setting that keeps up with a 24 fps camera on each card (3 steps, 39-frame windows)
case "$GPU" in
  *B300*) DEF_SIZE=288x512 DEF_REF=224 ;;
  *)      DEF_SIZE=256x448 DEF_REF=192 ;;
esac
SIZE=${SIZE:-$DEF_SIZE}
GARMENTS=${GARMENTS:-/workspace/garments}
FIRST=$(find "$GARMENTS" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \) | sort | head -1)
[ -n "$FIRST" ] || { echo "no garment images in $GARMENTS"; exit 1; }
echo "GPU: $GPU -> ${SIZE}, ref ${REF:-$DEF_REF}, start garment $(basename "$FIRST")"
exec python live_server.py --port "${PORT:-8000}" --w "${SIZE%x*}" --h "${SIZE#*x}" --ref "${REF:-$DEF_REF}" \
  --window 39 --steps "${STEPS:-3}" --compile --no-early-cond \
  --image "$FIRST" --garments-dir "$GARMENTS" \
  --lora "${LORA:-minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors}" --video-shift "${SHIFT:-12}" \
  ${ASPECTS:+--aspects "$ASPECTS"} ${TOKEN:+--token "$TOKEN"}
