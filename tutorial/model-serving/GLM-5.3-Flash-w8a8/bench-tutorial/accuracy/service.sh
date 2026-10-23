#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUN_ID=${ACCURACY_RUN_ID:-}
if [ -z "$RUN_ID" ]; then
  echo "ACCURACY_RUN_ID is required" >&2
  exit 2
fi
RUN_DIR=${ACCURACY_RUN_DIR:-$HERE/results/$RUN_ID}
PID_FILE="$RUN_DIR/server.pid"
mkdir -p "$RUN_DIR"

api_model_id() {
  curl --noproxy '*' -fsS -m 5 http://127.0.0.1:8077/v1/models 2>/dev/null \
    | python3 -c 'import json,sys; print("\n".join(x.get("id", "") for x in json.load(sys.stdin).get("data", [])))'
}

case "${1:-}" in
  start)
    if curl --noproxy '*' -fsS -m 3 http://127.0.0.1:8077/v1/models >/dev/null 2>&1; then
      echo "port 8077 already serves a model; refusing to replace it" >&2
      exit 3
    fi
    nohup env ACCURACY_RUN_ID="$RUN_ID" ACCURACY_RUN_DIR="$RUN_DIR" \
      "$HERE/server.sh" > "$RUN_DIR/server-launch.log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PID_FILE"
    printf '%s\n' "started_pid=$pid" > "$RUN_DIR/service-state.txt"
    printf 'Started server PID %s; waiting for /v1/models\n' "$pid"
    for _ in $(seq 1 3600); do
      if models=$(api_model_id); then
        if printf '%s\n' "$models" | grep -Fxq glm; then
          printf 'ready pid=%s model=glm utc=%s\n' "$pid" "$(date -u +%FT%TZ)" >> "$RUN_DIR/service-state.txt"
          exit 0
        fi
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "server process exited before readiness; see $RUN_DIR/server.log" >&2
        exit 4
      fi
      sleep 1
    done
    echo "server readiness timeout; process left running for diagnosis" >&2
    exit 5
    ;;
  status)
    if models=$(api_model_id) && printf '%s\n' "$models" | grep -Fxq glm; then
      echo "ready model=glm"
      exit 0
    fi
    echo "not ready"
    exit 1
    ;;
  stop)
    if [ ! -s "$PID_FILE" ]; then
      echo "no owned server PID recorded; leaving all processes untouched"
      exit 0
    fi
    pid=$(cat "$PID_FILE")
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmdline" in
      *"vllm serve /data/weight/GLM-5.3-Flash-w8a8-v2"*"--served-model-name glm"*) ;;
      *)
        echo "PID $pid no longer matches this GLM service; refusing to stop it" >&2
        exit 6
        ;;
    esac
    kill -TERM "$pid"
    for _ in $(seq 1 300); do
      if ! kill -0 "$pid" 2>/dev/null; then
        printf 'stopped_pid=%s utc=%s\n' "$pid" "$(date -u +%FT%TZ)" >> "$RUN_DIR/service-state.txt"
        exit 0
      fi
      sleep 1
    done
    echo "server did not exit within 300s; left it running for diagnosis" >&2
    exit 7
    ;;
  *)
    echo "usage: $0 {start|status|stop}" >&2
    exit 2
    ;;
esac
