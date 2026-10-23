#!/bin/bash
# run_smoke.sh —— 门禁：/v1/models + 真实请求 + AIPerf smoke（验证指标字段）
# 阶段 3 验收：装好 AIPerf 后 smoke 验证指标字段（TTFT/ITL/output_token_throughput/
#   output_token_throughput_per_user/request_throughput/KV 命中/MTP 接受率）。
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)
source "$ROOT/config.env"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

OUT="$ROOT/out/smoke-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

echo "===== 1) /v1/models ====="
curl --noproxy '*' -fsS "$SERVER_URL/v1/models" > "$OUT/models.json"
echo "models.json: $(head -c 200 "$OUT/models.json")"

echo "===== 2) 真实请求（8K/512）====="
curl --noproxy '*' -fsS "$SERVER_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$(printf '1 %.0s' $(seq 1 2000))\"}],\"max_completion_tokens\":32,\"temperature\":0}" \
  > "$OUT/real_request.json"
echo "real_request.json: $(head -c 300 "$OUT/real_request.json")"

echo "===== 3) AIPerf smoke（8K/512，并发 1，少量请求）====="
"$AIPERF" profile \
  --url "$SERVER_URL" --model "$MODEL" --endpoint-type "$ENDPOINT_TYPE" \
  --tokenizer "$TOKENIZER" --streaming \
  --isl 8192 --osl 512 --random-range-ratio "$RANDOM_RANGE_RATIO" \
  --extra-inputs "$EXTRA_INPUTS" \
  --concurrency 1 --request-count 16 --random-seed "$SEED" \
  --arrival-pattern "$ARRIVAL_PATTERN" \
  --output-artifact-dir "$OUT/aiperf" --profile-export-prefix smoke \
  --export-level records 2>&1 | tee "$OUT/aiperf_stdout.log"

echo "===== 4) 产物清单 ====="
find "$OUT/aiperf" -type f | sort
echo
echo "===== 5) summary JSON 关键字段 ====="
python3 - "$OUT/aiperf/smoke.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
def walk(o, prefix=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, prefix + k + ".")
    elif isinstance(o, list):
        for i, v in enumerate(o[:2]):
            walk(v, prefix + f"[{i}].")
    else:
        print(f"{prefix.rstrip('.')} = {o!r}")
walk(d)
PY

echo "SMOKE 完成 -> $OUT"
