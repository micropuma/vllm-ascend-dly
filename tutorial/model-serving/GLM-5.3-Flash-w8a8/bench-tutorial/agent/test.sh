#!/bin/bash
# test.sh —— AIPerf + tracelab agentic 轨迹回放（GLM-5.3-Flash-w8a8 @ 910C）
#
# 容器内执行：
#   docker exec vllm-ascend-dly-v028 bash \
#     /workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/agent/test.sh
#
# 依赖：上一级 config.env（SERVER_URL / MODEL / TOKENIZER / AIPERF 全在，source 即可）。
# 数据集：/workspace/vllm-ascend-dly-git/tutorial/model-serving/datasets/ 下的
#   syfi_coding_trace.jsonl.gz（原始全量，100.9MB）
#   tracelab_filtered.jsonl.gz（过滤峰值 input_tokens_total >120K 的 session 后：5916 session / 102529 轮）
#   过滤原因：vllm max-model-len=133120，长 session 尾部 input 涨到 132K~183K 必撞天花板报 error_isl。
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
source "$ROOT/config.env"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

DATASET_DIR="/workspace/vllm-ascend-dly-git/tutorial/model-serving/datasets"
TRACELAB_FILTERED="$DATASET_DIR/tracelab_filtered.jsonl.gz"
OUT="$ROOT/out/agent-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

# 关键：--no-fixed-schedule 缺省 --request-count = max(concurrency*2, 10)，必须显式给大，否则只跑 10 个请求。
echo "===== A) tracelab smoke（filtered，25 sessions / 并发 4，全速闭环 + think-time 封顶 5s）====="
"$AIPERF" profile \
  --url "$SERVER_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --endpoint-type "$ENDPOINT_TYPE" --streaming \
  --custom-dataset-type tracelab --input-file "$TRACELAB_FILTERED" \
  --num-dataset-entries 25 --concurrency 4 --request-count 400 \
  --no-fixed-schedule --inter-turn-delay-cap-seconds 5 \
  --output-artifact-dir "$OUT/tracelab_smoke" --profile-export-prefix tracelab_smoke \
  --export-level records 2>&1 | tee "$OUT/tracelab_smoke.log"

echo "===== B) tracelab 全量（filtered，200 sessions / 并发 16，全速闭环 + think-time 封顶 5s）====="
"$AIPERF" profile \
  --url "$SERVER_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --endpoint-type "$ENDPOINT_TYPE" --streaming \
  --custom-dataset-type tracelab --input-file "$TRACELAB_FILTERED" \
  --num-dataset-entries 200 --concurrency 16 --request-count 3000 \
  --no-fixed-schedule --inter-turn-delay-cap-seconds 5 \
  --output-artifact-dir "$OUT/tracelab_full" --profile-export-prefix tracelab_full \
  --export-level records 2>&1 | tee "$OUT/tracelab_full.log"

echo "完成 -> $OUT"
