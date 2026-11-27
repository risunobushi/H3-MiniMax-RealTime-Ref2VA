#!/usr/bin/env bash
# Fresh pod setup for the live mirror server. Detects Hopper (cu128 torch + FA3 hub kernel) vs Blackwell (cu130 torch,
# CUDA-13 ptxas for Triton, SDPA). Inputs (garment.jpg) and scripts are scp'd separately to /workspace/inputs and /workspace/stage.
set -euo pipefail
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p /workspace/hf /workspace/inputs /workspace/outputs /workspace/stage
CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
echo "== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l), compute cap major $CC_MAJOR"

echo "== apt"
apt-get update -qq > /dev/null && apt-get install -y -qq ffmpeg git > /dev/null
echo "== pip"
pip install -q -U pip
pip install -q --ignore-installed cryptography
if [ "$CC_MAJOR" -ge 10 ]; then
  pip install -q -U torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
  EXTRA='nvidia-cuda-nvcc==13.0.*'
else
  pip install -q -U torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
  EXTRA=''
fi
pip install -q -U "diffusers==0.40.0" "transformers>=4.57" accelerate peft av "kernels==0.16.0" \
  huggingface_hub hf_xet sentencepiece imageio imageio-ffmpeg bitsandbytes einops \
  fastapi "uvicorn[standard]" websockets python-multipart $EXTRA
pip uninstall -y -q torchao 2>/dev/null || true

echo "== turbo repo + LoRA"
cd /workspace
[ -d Minimax-H3-Turbo ] || git clone -q https://github.com/ModelTC/Minimax-H3-Turbo.git
cd Minimax-H3-Turbo
if [ -n "${H3_WEIGHTS_DIR:-}" ]; then
  # weights already on a mounted volume (the serverless global volume: <dir>/MiniMax-H3, <dir>/lora): no download
  ln -sf "$H3_WEIGHTS_DIR/lora/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors" .
  echo "weights from $H3_WEIGHTS_DIR:"; ls "$H3_WEIGHTS_DIR/MiniMax-H3"
else
# v0.1 (544p, shift 12) is the measured baseline; v1.0 768p (shift 6) is the newer release, see experiments/run_lora_v1.sh
hf download lightx2v/Minimax-h3-Turbo minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors   minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors --local-dir . > /dev/null

echo "== base model (ref2va partition + shared components)"
hf download MiniMaxAI/MiniMax-H3 \
  --include "tokenizer/*" "processor/*" "vae/*" "audio_vae/*" "scheduler/*" "audio_scheduler/*" "*.json" "*.txt" \
  --exclude "transformer/*" "transformer_ref/*" "text_encoder/*" > /dev/null
hf download MiniMaxAI/MiniMax-H3 --include "text_encoder/*" > /dev/null
hf download MiniMaxAI/MiniMax-H3 --include "transformer_ref/*" > /dev/null
SNAP=$(ls -d /workspace/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/*)
echo "snapshot: $SNAP"; du -shL "$SNAP"/transformer_ref "$SNAP"/text_encoder "$SNAP"/vae
fi

if [ "$CC_MAJOR" -lt 10 ]; then
  echo "== kernel warm (FA3 hub kernel into cache)"
  python - <<'PY'
from kernels import get_kernel
get_kernel("kernels-community/flash-attn3", revision="7cb368cf8278b583132eb72cbf312d54586df2e2")
print("FA3 kernel cached")
PY
fi
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
python - <<'PY'
import torch, diffusers, transformers, peft, bitsandbytes, kernels, fastapi, uvicorn, torchvision
print("torch", torch.__version__, "gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0),
      "diffusers", diffusers.__version__, "transformers", transformers.__version__, "bnb", bitsandbytes.__version__, "tv", torchvision.__version__)
x = torch.randn(64, 64, device="cuda").prod()  # NVRTC/JIT smoke
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference  # noqa: F401
print("MiniMax-H3 pipeline: OK")
PY
echo "SETUP DONE"
