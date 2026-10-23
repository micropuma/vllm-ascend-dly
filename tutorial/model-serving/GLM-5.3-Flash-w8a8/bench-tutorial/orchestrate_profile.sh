#!/usr/bin/env bash
# orchestrate_profile.sh —— 阶段2：profile 4 场景诊断 + 分析，结束后恢复基线服务。
# 场景顺序：prefill-8k → prefill-32k → decode-8k-c8 → decode-32k-c4
# 每个场景：停当前服务 → 起 profiler 服务（torch profiler, max_iterations=20）→
#           run_profile.sh（start_profile→bench→stop_profile）→ analyse.sh。
# 任一场景失败即中止（保留现场），最后恢复基线（config 32/8192 + 服务）。
set -uo pipefail

CTR_ROOT=/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial
PP1="$CTR_ROOT/baseline/server-exp/profile-plan-1"
URL=http://127.0.0.1:8077/v1/models
LAUNCH_LOG=/tmp/server_launch.log

cd "$CTR_ROOT" || exit 1
log() { echo "[$(date -u +%FT%TZ)] $*"; }

svc_pid() { pgrep -f 'vllm serve' | head -1; }

# ---- 等 /v1/models 返回期望码 ----
wait_http() { # $1=expect(200|000) $2=timeout_s
  local expect="$1" tmo="$2" i=0 code
  while [ "$i" -lt "$tmo" ]; do
    code=$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null)
    [ -z "$code" ] && code=000
    if [ "$expect" = "200" ] && [ "$code" = "200" ]; then log "http ready (200) @ ${i}s"; return 0; fi
    if [ "$expect" = "000" ] && [ "$code" = "000" ]; then log "http down (000) @ ${i}s"; return 0; fi
    sleep 5; i=$((i+5))
  done
  log "TIMEOUT waiting http=$expect (last=$code) @ ${i}s"
  return 1
}

# ---- 等 vllm 进程树清空 ----
wait_procs_gone() { # $1=timeout_s
  local tmo="$1" i=0 n
  while [ "$i" -lt "$tmo" ]; do
    n=$(ps -ef | grep -E 'vllm serve|VLLM::EngineCore|VLLM::Worker' | grep -v grep | wc -l)
    [ "$n" -eq 0 ] && { log "vllm procs gone @ ${i}s"; return 0; }
    sleep 5; i=$((i+5))
  done
  log "TIMEOUT: $n vllm procs still alive @ ${i}s"
  return 1
}

# ---- 停当前服务（profiler 或 baseline 通用）----
stop_service() {
  local pid
  pid=$(svc_pid)
  log "stop_service: current pid=${pid:-none}"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    log "sent SIGTERM to $pid"
  else
    log "no live vllm serve pid; will clean stragglers"
  fi
  wait_procs_gone 180 || {
    log "procs lingered; force-killing"
    pkill -9 -f 'VLLM::EngineCore|VLLM::Worker' 2>/dev/null || true
    pkill -9 -f 'vllm serve' 2>/dev/null || true
    sleep 5
  }
  wait_http 000 60 || log "warn: port still not free after kill"
}

# ---- 起 profiler 服务（scenario env 经 set -a 导出，避免未 export 变量丢失）----
start_profiler() { # $1=scenario
  local scenario="$1"
  ( cd "$PP1" && set -a && . "scenarios/$scenario.env" && set +a && \
      nohup setsid bash start_server_profile.sh > /dev/null 2>&1 & )
  log "launched profiler service for $scenario (out/$scenario/server.log)"
  wait_http 200 3600 || { log "FATAL: profiler service not ready in 60min"; return 1; }
  return 0
}

# ---- 跑一个 profile 场景 ----
run_scenario() { # $1=scenario
  local scenario="$1"
  log "===== PROFILE $scenario ====="
  stop_service || { log "FATAL: stop_service failed for $scenario"; return 1; }
  start_profiler "$scenario" || { log "FATAL: start_profiler failed for $scenario"; return 1; }
  log "running profile workload for $scenario"
  ( cd "$PP1" && set -a && . "scenarios/$scenario.env" && set +a && bash run_profile.sh ) \
    >> "$PP1/out/$scenario/run_profile.log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    log "FATAL: run_profile.sh rc=$rc for $scenario (see out/$scenario/run_profile.log)"
    return 1
  fi
  log "profile captured for $scenario; run.json:"
  cat "$PP1/out/$scenario/run/run.json" 2>/dev/null
  log "analysing $scenario (rank0-only, non-eager) ..."
  ( cd "$PP1" && bash analyse_rank0.sh "out/$scenario" ) >> "$PP1/out/$scenario/analyse.log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    log "FATAL: analyse_rank0.sh rc=$rc for $scenario (see out/$scenario/analyse.log)"
    return 1
  fi
  log "===== PROFILE $scenario COMPLETE (analysis: out/$scenario/analysis/) ====="
  return 0
}

# ================= 主流程 =================
log "orchestrate_profile START"
for sc in prefill-8k prefill-32k decode-8k-c8 decode-32k-c4; do
  run_scenario "$sc" || { log "ABORT: scenario $sc failed; leaving current state"; exit 1; }
done

# 恢复基线服务
log "restore baseline service (ns=32 bt=8192)"
stop_service
( cd "$CTR_ROOT/baseline" && nohup setsid bash server.sh > "$LAUNCH_LOG" 2>&1 & )
log "launched baseline server.sh"
wait_http 200 3600 || { log "FATAL: baseline restore not ready in 60min"; exit 1; }
log "baseline restored (config 32/8192)"

{
  echo "done_at=$(date -u +%FT%TZ)"
  echo "models_http=$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "$URL")"
  echo "scenarios:"
  for sc in prefill-8k prefill-32k decode-8k-c8 decode-32k-c4; do
    echo "  $sc: $( [ -f "$PP1/out/$sc/run/run.json" ] && cat "$PP1/out/$sc/run/run.json" || echo 'NO run.json' )"
  done
} > "$CTR_ROOT/PROFILE_DONE"
log "PROFILE_DONE written"
log "orchestrate_profile END"
