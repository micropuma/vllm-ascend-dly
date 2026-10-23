#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV=${AISBENCH_VENV:-/opt/aisbench-glm53-client-venv}
PYTHON="$VENV/bin/python"
AIS_BENCH="$VENV/bin/ais_bench"
RUN_ID=${ACCURACY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${ACCURACY_RUN_DIR:-$HERE/results/$RUN_ID}
NUM_PROMPTS=${AISBENCH_NUM_PROMPTS:-}
if [ -e "$RUN_DIR" ] && [ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]; then
  echo "refusing to overwrite existing run: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

started_service=0
stop_owned_service() {
  if [ "$started_service" -eq 1 ]; then
    ACCURACY_RUN_ID="$RUN_ID" ACCURACY_RUN_DIR="$RUN_DIR" "$HERE/service.sh" stop || true
  fi
}
trap stop_owned_service EXIT
if ! curl --noproxy '*' -fsS -m 3 http://127.0.0.1:8077/v1/models > "$RUN_DIR/models.json" 2>/dev/null; then
  ACCURACY_RUN_ID="$RUN_ID" ACCURACY_RUN_DIR="$RUN_DIR" "$HERE/service.sh" start
  started_service=1
  curl --noproxy '*' -fsS -m 5 http://127.0.0.1:8077/v1/models > "$RUN_DIR/models.json"
fi
grep -q '"glm"' "$RUN_DIR/models.json" || { echo "served model glm not present" >&2; exit 4; }

"$PYTHON" "$HERE/prepare_data.py" > "$RUN_DIR/prepare.log" 2>&1
cp "$HERE/configs/models/glm53_vllm_api.py" "$RUN_DIR/"
cp "$HERE/configs/datasets/ceval_local.py" "$RUN_DIR/"
cp "$HERE/configs/datasets/gsm8k_local.py" "$RUN_DIR/"

cat > "$RUN_DIR/manifest.json" <<EOF
{
  "run_id": "$RUN_ID",
  "started_utc": "$(date -u +%FT%TZ)",
  "model": "GLM-5.3-Flash-w8a8-v2",
  "served_model": "glm",
  "server_url": "http://127.0.0.1:8077",
  "aisbench_version": "3.1.20260630",
  "aisbench_python": "$($PYTHON --version 2>&1)",
  "config_dir": "$HERE/configs",
  "dataset_split": {"ceval": "all subjects / val", "gsm8k": "main / test"},
  "num_prompts_limit": "${NUM_PROMPTS:-all}"
}
EOF

for dataset in ceval_local gsm8k_local; do
  case "$dataset" in
    ceval_local) name=ceval ;;
    gsm8k_local) name=gsm8k ;;
  esac
  out="$RUN_DIR/$name"
  mkdir -p "$out"
  set +e
  extra_args=()
  if [ -n "$NUM_PROMPTS" ]; then extra_args+=(--num-prompts "$NUM_PROMPTS"); fi
  "$AIS_BENCH" --config-dir "$HERE/configs" \
    --models glm53_vllm_api --datasets "$dataset" \
    --work-dir "$out" --dump-eval-details "${extra_args[@]}" \
    2>&1 | tee "$out/ais_bench.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$out/exit_code"
  if [ "$rc" -ne 0 ]; then
    echo "AISBench failed: $dataset (exit=$rc)" >&2
    exit "$rc"
  fi
done

"$PYTHON" "$HERE/summarize_results.py" "$RUN_DIR"
printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)" >> "$RUN_DIR/service-state.txt"
