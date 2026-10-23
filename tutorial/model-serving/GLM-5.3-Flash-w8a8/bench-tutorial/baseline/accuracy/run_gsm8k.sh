#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TUTORIAL_ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
ACCURACY_ROOT="$TUTORIAL_ROOT/accuracy"
VENV=${AISBENCH_VENV:-/opt/aisbench-glm53-client-venv}
PYTHON="$VENV/bin/python"
AIS_BENCH="$VENV/bin/ais_bench"
RUN_ID=${ACCURACY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR=${ACCURACY_RUN_DIR:-$HERE/results/$RUN_ID}
NUM_PROMPTS=${AISBENCH_NUM_PROMPTS:-}

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to overwrite existing run: $RUN_DIR" >&2
  exit 2
fi
[[ -x "$PYTHON" ]] || { echo "AISBench Python not found: $PYTHON" >&2; exit 3; }
[[ -x "$AIS_BENCH" ]] || { echo "AISBench executable not found: $AIS_BENCH" >&2; exit 3; }
mkdir -p "$RUN_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy=127.0.0.1,localhost,::1 NO_PROXY=127.0.0.1,localhost,::1
if ! curl --noproxy '*' -fsS -m 3 http://127.0.0.1:8077/v1/models > "$RUN_DIR/models.json"; then
  echo "No healthy service at http://127.0.0.1:8077; start the frozen baseline service first." >&2
  exit 4
fi
grep -q '"glm"' "$RUN_DIR/models.json" || { echo "served model glm not present" >&2; exit 4; }

"$PYTHON" "$ACCURACY_ROOT/prepare_data.py" > "$RUN_DIR/prepare.log" 2>&1
cp "$ACCURACY_ROOT/configs/models/glm53_vllm_api.py" "$RUN_DIR/"
cp "$ACCURACY_ROOT/configs/datasets/gsm8k_local.py" "$RUN_DIR/"

"$PYTHON" - "$RUN_DIR/manifest.json" "$RUN_ID" "$RUN_DIR" "$ACCURACY_ROOT" "$NUM_PROMPTS" <<'PY'
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path, run_id, run_dir, accuracy_root, num_prompts = sys.argv[1:]
repo = Path(accuracy_root).parents[4]
def aisbench_version():
    try:
        return importlib.metadata.version("ais_bench")
    except importlib.metadata.PackageNotFoundError:
        import ais_bench
        return getattr(ais_bench, "__version__", "unknown")
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = {
    "run_id": run_id,
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "dataset": "GSM8K main/test",
    "expected_examples": 1319,
    "aisbench_version": aisbench_version(),
    "python": sys.version,
    "server_url": "http://127.0.0.1:8077",
    "models_response_sha256": sha(Path(run_dir) / "models.json"),
    "prepared_test_sha256": sha(Path(accuracy_root) / "data-prepared/gsm8k/test.jsonl"),
    "model_config_sha256": sha(Path(accuracy_root) / "configs/models/glm53_vllm_api.py"),
    "dataset_config_sha256": sha(Path(accuracy_root) / "configs/datasets/gsm8k_local.py"),
    "git_head": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
    "git_status": subprocess.run(["git", "-C", str(repo), "status", "--short"], text=True, capture_output=True).stdout,
    "num_prompts_limit": num_prompts or "all",
}
Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
PY

out="$RUN_DIR/gsm8k"
mkdir -p "$out"
extra_args=()
if [[ -n "$NUM_PROMPTS" ]]; then extra_args+=(--num-prompts "$NUM_PROMPTS"); fi
set +e
"$AIS_BENCH" --config-dir "$ACCURACY_ROOT/configs" \
  --models glm53_vllm_api --datasets gsm8k_local \
  --work-dir "$out" --dump-eval-details "${extra_args[@]}" \
  2>&1 | tee "$out/ais_bench.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$out/exit_code"
if [[ "$rc" -ne 0 ]]; then
  echo "AISBench GSM8K failed (exit=$rc)" >&2
  exit "$rc"
fi
"$PYTHON" "$ACCURACY_ROOT/summarize_results.py" "$RUN_DIR"
"$PYTHON" - "$RUN_DIR/manifest.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
printf 'GSM8K run saved to %s\n' "$RUN_DIR"
