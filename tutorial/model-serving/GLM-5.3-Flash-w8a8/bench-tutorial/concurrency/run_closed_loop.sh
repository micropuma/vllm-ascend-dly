#!/bin/bash
# run_closed_loop.sh —— 闭环扫并发 → μ(L) 曲线 + μ_max + SLO 可用并发
# 方法：--concurrency N（闭环，AIPerf 完成一个立即发下一个，维持 N in-flight）
# 产出：out/<run-id>/concurrency/<isl>/c<N>/smoke.json（AIPerf summary）+ metadata
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
source "$ROOT/config.env"

# per-run 覆盖（A 轴实验用不同并发/输入档；不覆盖则用 config.env 默认）
INPUT_LENS=${RUN_INPUT_LENS:-$INPUT_LENS}
CONCURRENCIES=${RUN_CONCURRENCIES:-$CONCURRENCIES}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT="$ROOT/out/$RUN_ID/concurrency"
mkdir -p "$OUT_ROOT"

{
  printf 'date_utc=%s\nload_mode=closed_loop\nserver_url=%s\nmodel=%s\ninput_lens=%s\noutput_len=%s\nconcurrencies=%s\nrequest_count=%s\nextra_inputs=%s\narrival_pattern=%s\n' \
    "$(date -u +%FT%TZ)" "$SERVER_URL" "$MODEL" "$INPUT_LENS" "$OUTPUT_LEN" \
    "$CONCURRENCIES" "$CLOSED_REQUEST_COUNT" "$EXTRA_INPUTS" "$ARRIVAL_PATTERN"
} > "$OUT_ROOT/metadata.txt"

status=0
for isl in $INPUT_LENS; do
  for c in $CONCURRENCIES; do
    run="$OUT_ROOT/isl${isl}/c${c}"
    mkdir -p "$run"
    cmd=("$AIPERF" profile \
      --url "$SERVER_URL" --model "$MODEL" --endpoint-type "$ENDPOINT_TYPE" \
      --tokenizer "$TOKENIZER" --streaming \
      --isl "$isl" --osl "$OUTPUT_LEN" --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --extra-inputs "$EXTRA_INPUTS" \
      --concurrency "$c" --request-count "$CLOSED_REQUEST_COUNT" --random-seed "$SEED" \
      --arrival-pattern "$ARRIVAL_PATTERN" \
      --output-artifact-dir "$run" --profile-export-prefix "c${c}" \
      --export-level records)
    echo "===== closed_loop isl=$isl c=$c $(date -u +%FT%TZ) ====="
    { printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'; "${cmd[@]}"; } 2>&1 | tee "$run/stdout.log" || status=1
  done
done
exit "$status"
