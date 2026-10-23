#!/bin/bash
# run_open_loop.sh —— 开环按到达率 → 可持续 λ（过三门）
# 方法：--request-rate λ --arrival-pattern poisson（不设 --concurrency = 无并发上限）
# RATES 分输入档：从 out/<RUN_ID>/rates.txt 读取（每行 "isl rate1 rate2 ..."，
#   由 `analyze.py --run-id <id> --emit-rates` 生成，比例 0.45/0.70/0.85/1.20 × μ_max(isl)）。
# 用法：RUN_ID=<id> bash run_open_loop.sh
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
source "$ROOT/config.env"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RATES_FILE="$ROOT/out/$RUN_ID/rates.txt"
if [ ! -f "$RATES_FILE" ]; then
  echo "ERROR: $RATES_FILE 不存在。请先跑闭环并用 analyze.py --emit-rates 反推分档 RATES。" >&2
  exit 1
fi

OUT_ROOT="$ROOT/out/$RUN_ID/rps"
mkdir -p "$OUT_ROOT"

{
  printf 'date_utc=%s\nload_mode=open_loop_no_max_concurrency\nserver_url=%s\nmodel=%s\ninput_lens=%s\noutput_len=%s\nrates_file=%s\nrequest_count=%s\nextra_inputs=%s\narrival_pattern=%s\n' \
    "$(date -u +%FT%TZ)" "$SERVER_URL" "$MODEL" "$INPUT_LENS" "$OUTPUT_LEN" \
    "$RATES_FILE" "$OPEN_REQUEST_COUNT" "$EXTRA_INPUTS" "$ARRIVAL_PATTERN"
} > "$OUT_ROOT/metadata.txt"

status=0
for isl in $INPUT_LENS; do
  # 从 rates.txt 取该 isl 的速率列表（第 1 列匹配，输出第 2..N 列）
  rates=$(awk -v isl="$isl" '$1 == isl { for (i=2; i<=NF; i++) printf "%s%s", $i, (i<NF ? " " : ""); print "" }' "$RATES_FILE")
  if [ -z "$rates" ]; then
    echo "WARN: rates.txt 无 isl=$isl 的速率，跳过" >&2
    continue
  fi
  for rate in $rates; do
    safe=${rate//./_}
    run="$OUT_ROOT/isl${isl}/rate-${safe}"
    mkdir -p "$run"
    cmd=("$AIPERF" profile \
      --url "$SERVER_URL" --model "$MODEL" --endpoint-type "$ENDPOINT_TYPE" \
      --tokenizer "$TOKENIZER" --streaming \
      --isl "$isl" --osl "$OUTPUT_LEN" --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --extra-inputs "$EXTRA_INPUTS" \
      --request-rate "$rate" --arrival-pattern "$ARRIVAL_PATTERN" \
      --request-count "$OPEN_REQUEST_COUNT" --random-seed "$SEED" \
      --output-artifact-dir "$run" --profile-export-prefix "rate-${safe}" \
      --export-level records)
    echo "===== open_loop isl=$isl rate=$rate $(date -u +%FT%TZ) ====="
    { printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'; "${cmd[@]}"; } 2>&1 | tee "$run/stdout.log" || status=1
  done
done
exit "$status"
