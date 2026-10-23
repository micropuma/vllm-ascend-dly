#!/bin/sh
# GLM-5.3-Flash baseline service: TP16, DP1, expert parallel.
# 并发参数旋钮 max-num-seqs / max-num-batched-tokens 由 config.env 的
# MAX_NUM_SEQS / MAX_NUM_BATCHED_TOKENS 提供：改动只需编辑 config.env 后重启本服务。
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
[ -f "$ROOT/config.env" ] && . "$ROOT/config.env"
cd /workspace/vllm-ascend-dly-git
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-400}
export VLLM_USE_MODELSCOPE=False
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3000}
MODEL_PATH=${MODEL_PATH:-/data/weight/GLM-5.3-Flash-w8a8-v2}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
LOG=/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/serve_mtp.log
: > "$LOG"
printf '%s\n' "### serve MTP start $(date -u +%FT%TZ) pid=$$" \
  "### topology dp=1 tp=16" \
  "### max-num-seqs=$MAX_NUM_SEQS max-num-batched-tokens=$MAX_NUM_BATCHED_TOKENS" >> "$LOG"
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 8077 --max-model-len 133120 \
  --data-parallel-size 1 --tensor-parallel-size 16 --enable-expert-parallel \
  --seed 1024 --served-model-name glm --safetensors-load-strategy prefetch \
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --trust-remote-code \
  --quantization ascend --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"num_speculative_tokens":3,"method":"deepseek_mtp","enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,96,128]}' \
  --api-server-count 1 >> "$LOG" 2>&1
