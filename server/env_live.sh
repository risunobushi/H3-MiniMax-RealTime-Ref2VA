export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=0 PIP_BREAK_SYSTEM_PACKAGES=1
# weights on a mounted volume (H3_WEIGHTS_DIR, e.g. the serverless global volume) win over the HF cache snapshot
if [ -n "${H3_WEIGHTS_DIR:-}" ]; then export H3_MODEL_PATH=$H3_WEIGHTS_DIR/MiniMax-H3
else export H3_MODEL_PATH=$(ls -d /workspace/hf/hub/models--MiniMaxAI--MiniMax-H3/snapshots/* 2>/dev/null | head -1); fi
CC_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$CC_MAJOR" -ge 10 ]; then
  export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
  export TRITON_PTXAS_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/ptxas
  export H3_ATTN=            # Blackwell: SDPA (FA4 hub not usable for H3 yet)
else
  export H3_ATTN=_flash_3_hub  # Hopper: FA3 hub kernel (hub must be reachable; check the log says "attention backend: _flash_3_hub")
fi
# CPU thread pools: without this OMP/MKL size themselves to every core the host shows (344 on US-WA-2), not the pod's
# quota. live_server.py also defaults to 8 when unset. For the "old behaviour" baseline: H3_THREADS=$(nproc).
export H3_THREADS=${H3_THREADS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$H3_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS:-$H3_THREADS}
