#!/bin/bash
# scrape_metrics.sh —— KV 命中率 / MTP 接受率 交叉核对
# AIPerf 已原生上报（--server-metrics 默认开，产出 <prefix>_server_metrics.*），
# 本脚本独立 curl /metrics 做前后差值，作为对账证据。
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
source "$ROOT/config.env"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

OUT=${1:-$ROOT/out/scrape-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"

# 关注的 vLLM Prometheus 指标（KV 命中 / MTP 接受率）
PATTERN='prefix_cache_hits|prefix_cache_queries|spec_decode_num_accepted_tokens|spec_decode_num_draft_tokens|prompt_tokens_cached|num_preemptions|num_requests_running|num_requests_waiting'

curl --noproxy '*' -fsS "$SERVER_URL/metrics" > "$OUT/metrics.txt" || {
  echo "curl /metrics 失败" >&2; exit 1;
}

echo "===== KV / MTP 关键指标 ====="
grep -E "$PATTERN" "$OUT/metrics.txt" | grep -v '^#' | sort || echo "(未匹配到目标指标)"

echo
echo "===== 指标说明（对照 AIPerf server_metrics）====="
echo "  prefix_cache_hits/queries      -> KV 前缀缓存命中率 = hits/queries"
echo "  spec_decode_num_accepted/draft -> MTP 草稿接受率 = accepted/draft_tokens"
echo
echo "产出 -> $OUT/metrics.txt"
