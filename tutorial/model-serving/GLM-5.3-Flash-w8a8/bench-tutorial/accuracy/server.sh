#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_PATH=${MODEL_PATH:-/data/weight/GLM-5.3-Flash-w8a8-v2}
RUN_ID=${ACCURACY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${ACCURACY_RUN_DIR:-$HERE/results/$RUN_ID}
mkdir -p "$RUN_DIR"
cd /workspace/vllm-ascend-dly-git
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-400}
export VLLM_USE_MODELSCOPE=False
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}
exec vllm serve "$MODEL_PATH" --host 0.0.0.0 --port 8077 \
  --max-model-len 133120 --data-parallel-size 1 --tensor-parallel-size 16 \
  --enable-expert-parallel --seed 1024 --served-model-name glm \
  --safetensors-load-strategy prefetch --max-num-seqs 32 \
  --max-num-batched-tokens 8192 --trust-remote-code --quantization ascend \
  --limit-mm-per-prompt '{"image":1,"video":0}' --gpu-memory-utilization 0.85 \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128]}' \
  --api-server-count 1 >> "$RUN_DIR/server.log" 2>&1
