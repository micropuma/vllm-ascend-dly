#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
source "$ROOT/config.env"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1

readonly INPUT_LEN=8192
readonly PERF_OUTPUT_LEN=512
readonly CLIENT_CONCURRENCY=8
readonly REQUEST_COUNT=256
readonly WARMUP_REQUEST_COUNT=16
readonly RANDOM_SEED=1024
readonly REPEAT_COUNT=1
readonly RUN_ID=${PERF_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
readonly OUT_DIR="$HERE/results/$RUN_ID"
export PERF_SOURCE_REPO=${PERF_SOURCE_REPO:?set actual server source checkout}
export PERF_SERVER_RUN_DIR=${PERF_SERVER_RUN_DIR:?set frozen server startup artifact directory}
[[ -f "$PERF_SERVER_RUN_DIR/command.txt" && -f "$PERF_SERVER_RUN_DIR/import-paths.txt" ]] || { echo "server identity artifacts missing" >&2; exit 2; }
grep -F "$PERF_SOURCE_REPO/vllm_ascend/" "$PERF_SERVER_RUN_DIR/import-paths.txt" >/dev/null || { echo "server import path does not match source checkout" >&2; exit 2; }

[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid PERF_RUN_ID: use letters, digits, dot, underscore, or dash" >&2; exit 2; }
[[ "$MAX_NUM_SEQS" == "32" ]] || { echo "expected MAX_NUM_SEQS=32, got $MAX_NUM_SEQS" >&2; exit 2; }
[[ "$MAX_NUM_BATCHED_TOKENS" == "8192" ]] || { echo "expected MAX_NUM_BATCHED_TOKENS=8192, got $MAX_NUM_BATCHED_TOKENS" >&2; exit 2; }
[[ "$INPUT_LENS" == *"8192"* ]] || { echo "shared testbench contract no longer contains ISL 8192" >&2; exit 2; }
[[ "$OUTPUT_LEN" == "$PERF_OUTPUT_LEN" ]] || { echo "expected OUTPUT_LEN=512, got $OUTPUT_LEN" >&2; exit 2; }
[[ "$MODEL" == "glm" ]] || { echo "expected MODEL=glm, got $MODEL" >&2; exit 2; }
[[ "$EXTRA_INPUTS" == "ignore_eos:true" ]] || { echo "unexpected EXTRA_INPUTS=$EXTRA_INPUTS" >&2; exit 2; }
[[ "$ENDPOINT_TYPE" == "chat" && "$STREAMING" == "true" ]] || { echo "expected streaming chat endpoint" >&2; exit 2; }
[[ "$RANDOM_RANGE_RATIO" == "0" && "$SEED" == "$RANDOM_SEED" ]] || { echo "input/seed contract changed" >&2; exit 2; }
[[ "$ARRIVAL_PATTERN" == "poisson" ]] || { echo "expected ARRIVAL_PATTERN=poisson, got $ARRIVAL_PATTERN" >&2; exit 2; }
[[ "$SERVER_URL" == "http://127.0.0.1:8077" ]] || { echo "unexpected SERVER_URL=$SERVER_URL" >&2; exit 2; }
[[ "$AIPERF" == "/workspace/aiperf-venv/bin/aiperf" ]] || { echo "unexpected AIPERF path: $AIPERF" >&2; exit 2; }
[[ -x "$AIPERF" ]] || { echo "AIPerf not found: $AIPERF" >&2; exit 3; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite run directory: $OUT_DIR" >&2; exit 4; }
mkdir -p "$OUT_DIR"

curl --noproxy '*' -fsS "$SERVER_URL/v1/models" > "$OUT_DIR/models.json"
grep -q '"glm"' "$OUT_DIR/models.json" || { echo "served model glm not present" >&2; exit 5; }

"$AIPERF" --version > "$OUT_DIR/aiperf-version.txt" 2>&1

python3 - "$OUT_DIR/manifest.json" "$RUN_ID" "$OUT_DIR" "$ROOT" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, run_id, out_dir, root = sys.argv[1:]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def git(*args):
    result = subprocess.run(["git", "-C", os.environ["PERF_SOURCE_REPO"], *args], text=True, capture_output=True, check=True)
    return result.stdout.strip()

manifest = {
    "run_id": run_id,
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "load_mode": "closed_loop",
    "model": "glm",
    "model_path": "/data/weight/GLM-5.3-Flash-w8a8-v2",
    "tokenizer_path": "/data/weight/GLM-5.3-Flash-w8a8-v2",
    "server_url": "http://127.0.0.1:8077",
    "endpoint_type": "chat",
    "streaming": True,
    "input_length_tokens": 8192,
    "output_length_tokens": 512,
    "random_range_ratio": 0,
    "extra_inputs": "ignore_eos:true",
    "seed": 1024,
    "arrival_pattern": "poisson",
    "client_concurrency": 8,
    "request_count_per_repeat": 256,
    "warmup_request_count_per_repeat": 16,
    "measured_batches_per_server_start": 1,
    "independent_server_start_repeats": 3,
    "server_max_num_seqs": 32,
    "server_max_num_batched_tokens": 8192,
    "aiperf_version": (Path(out_dir) / "aiperf-version.txt").read_text().strip(),
    "served_models_sha256": sha(Path(out_dir) / "models.json"),
    "git_head": git("rev-parse", "HEAD"),
    "git_status": git("status", "--short"),
    "source_repo": os.environ["PERF_SOURCE_REPO"],
    "server_run_dir": os.environ["PERF_SERVER_RUN_DIR"],
    "actual_server_command_sha256": sha(Path(os.environ["PERF_SERVER_RUN_DIR"]) / "command.txt"),
    "actual_server_imports_sha256": sha(Path(os.environ["PERF_SERVER_RUN_DIR"]) / "import-paths.txt"),
    "actual_server_patch_sha256": sha(Path(os.environ["PERF_SERVER_RUN_DIR"]) / "source.patch"),
    "config_env_sha256": sha(Path(root) / "config.env"),
    "server_script_sha256": sha(Path(root) / "baseline/server.sh"),
    "testbench_closed_loop_script_sha256": sha(Path(root) / "concurrency/run_closed_loop.sh"),
    "prefill8k_script_sha256": sha(Path(root) / "baseline/perf/run_prefill8k.sh"),
}
Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
PY

status=0
for ((repeat = 1; repeat <= REPEAT_COUNT; repeat++)); do
  repeat_id=$(printf 'r%02d' "$repeat")
  repeat_dir="$OUT_DIR/$repeat_id"
  mkdir -p "$repeat_dir"
  cmd=("$AIPERF" profile
    --url "$SERVER_URL" --model "$MODEL" --endpoint-type "$ENDPOINT_TYPE"
    --tokenizer "$TOKENIZER" --streaming
    --isl "$INPUT_LEN" --osl "$PERF_OUTPUT_LEN" --random-range-ratio "$RANDOM_RANGE_RATIO"
    --extra-inputs "$EXTRA_INPUTS"
    --concurrency "$CLIENT_CONCURRENCY" --request-count "$REQUEST_COUNT"
    --warmup-request-count "$WARMUP_REQUEST_COUNT" --warmup-concurrency "$CLIENT_CONCURRENCY"
    --random-seed "$RANDOM_SEED" --arrival-pattern "$ARRIVAL_PATTERN"
    --output-artifact-dir "$repeat_dir" --profile-export-prefix "prefill8k-c8-$repeat_id"
    --export-level records)
  echo "===== prefill8k repeat=$repeat_id $(date -u +%FT%TZ) ====="
  { printf 'command='; printf '%q ' "${cmd[@]}"; printf '\n'; "${cmd[@]}"; } 2>&1 | tee "$repeat_dir/stdout.log" || status=1
  if [[ "$status" -ne 0 ]]; then
    echo "AIPerf failed in $repeat_id; preserving partial artifacts in $OUT_DIR" >&2
    exit "$status"
  fi
done

python3 - "$OUT_DIR/manifest.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
echo "prefill8k performance batch saved to $OUT_DIR"
