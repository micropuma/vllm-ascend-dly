#!/usr/bin/env bash
# orchestrate_phase1.sh —— 阶段1：E2→E3→E5→E6 闭环扫描，结束后恢复基线（config 32/8192 + 服务）。
#
# 跑在容器内（vllm-ascend-dly-v028），由 docker exec -d 启动后自身再 nohup setsid 脱壳，
# 以便长跑（~16h）不被 SSH/会话中断。
#
# 实验矩阵（对齐 task-10-18.md §1）：
#   B 轴（num-seqs 固定 32，扫 batched-tokens）：E2=16384, E3=32768  —— 闭环 8K/32K × c1/8/16/32
#   A 轴（batched-tokens 固定 8192，扫 num-seqs）：E5=64, E6=96       —— 闭环 8K × c32/48/64、c64/96
#   E1（基线 32/8192）已在跑，本脚本先等它收尾。
set -uo pipefail

CTR_ROOT=/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial
SERVE_LOG=/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/serve_mtp.log
LAUNCH_LOG=/tmp/server_launch.log
URL=http://127.0.0.1:8077/v1/models
WARM_SECONDS=900

cd "$CTR_ROOT" || exit 1

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# ---- 读写 config.env 并发旋钮 ----
set_knobs() { # $1=ns $2=bt
  sed -i -E "s/^MAX_NUM_SEQS=.*/MAX_NUM_SEQS=\"$1\"/; s/^MAX_NUM_BATCHED_TOKENS=.*/MAX_NUM_BATCHED_TOKENS=\"$2\"/" "$CTR_ROOT/config.env"
  log "knobs -> $(grep -E '^(MAX_NUM_SEQS|MAX_NUM_BATCHED_TOKENS)=' "$CTR_ROOT/config.env" | tr '\n' ' ')"
}

main_pid() {
  awk '/^### serve MTP start/{gsub(/pid=/,"",$NF); print $NF}' "$SERVE_LOG" 2>/dev/null | tail -1
}

# ---- 等 /v1/models 返回期望码（200=就绪 / 000=已停）----
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

# ---- 重启服务：kill 旧进程树 → 起新 → 等 200 ----
restart_service() {
  local pid
  pid=$(main_pid)
  log "restart: current main pid=${pid:-none}"
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    log "sent SIGTERM to $pid"
  else
    log "no live main pid (kill -0 failed); will clean stragglers"
  fi
  wait_procs_gone 180 || {
    log "procs lingered; force-killing"
    pkill -9 -f 'VLLM::EngineCore|VLLM::Worker' 2>/dev/null || true
    pkill -9 -f 'vllm serve' 2>/dev/null || true
    sleep 5
  }
  wait_http 000 60 || log "warn: port still not free after kill"
  ( cd "$CTR_ROOT/baseline" && nohup setsid bash server.sh > "$LAUNCH_LOG" 2>&1 & )
  log "launched server.sh (knobs from config.env)"
  wait_http 200 3600 || { log "FATAL: server not ready in 60min"; return 1; }
  log "server ready; log header: $(head -1 "$SERVE_LOG")"
  return 0
}

# ---- 计算期望 summary JSON 数 ----
expected_count() { # $1=inlens $2=conc -> stdout 整数
  local inlens="$1" conc="$2" ni nc
  ni=$(echo "$inlens" | wc -w)
  nc=$(echo "$conc" | wc -w)
  echo $((ni * nc))
}

# ---- 数已产出的 summary JSON ----
summary_count() { # $1=run_id $2=inlens $3=conc
  local run_id="$1" inlens="$2" conc="$3" isl c n=0
  for isl in $inlens; do
    for c in $conc; do
      [ -f "$CTR_ROOT/out/$run_id/concurrency/isl${isl}/c${c}/c${c}.json" ] && n=$((n+1))
    done
  done
  echo "$n"
}

# ---- 等闭环跑完（全 JSON 出，或 launcher 退出）----
wait_closed_loop() { # $1=run_id $2=inlens $3=conc
  local run_id="$1" inlens="$2" conc="$3" exp cur alive
  exp=$(expected_count "$inlens" "$conc")
  log "wait_closed_loop $run_id: expect $exp summaries"
  while true; do
    cur=$(summary_count "$run_id" "$inlens" "$conc")
    alive=$(pgrep -f "run_closed_loop[.]sh" | wc -l)
    log "  $run_id: summaries=$cur/$exp launcher_alive=$alive"
    if [ "$cur" -ge "$exp" ]; then
      log "  $run_id DONE (all $exp summaries present)"
      return 0
    fi
    if [ "$alive" -eq 0 ]; then
      log "  $run_id launcher exited but only $cur/$exp summaries -> FAILED"
      return 1
    fi
    sleep 60
  done
}

# ---- 跑一个实验 ----
run_experiment() { # $1=run_id $2=ns $3=bt $4=inlens $5=conc
  local run_id="$1" ns="$2" bt="$3" inlens="$4" conc="$5"
  log "===== EXPERIMENT $run_id (ns=$ns bt=$bt) ====="
  set_knobs "$ns" "$bt"
  restart_service || { log "FATAL: restart failed for $run_id"; return 1; }
  log "warm: sleeping ${WARM_SECONDS}s"
  sleep "$WARM_SECONDS"
  log "smoke gate..."
  mkdir -p "$CTR_ROOT/out/$run_id"
  ( cd "$CTR_ROOT" && bash baseline/run_smoke.sh ) >> "$CTR_ROOT/out/$run_id/smoke.log" 2>&1 || log "warn: smoke rc=$?"
  log "launching closed-loop $run_id"
  ( cd "$CTR_ROOT" && RUN_ID="$run_id" RUN_INPUT_LENS="$inlens" RUN_CONCURRENCIES="$conc" \
      nohup setsid bash concurrency/run_closed_loop.sh >> "out/$run_id/run.log" 2>&1 & )
  wait_closed_loop "$run_id" "$inlens" "$conc" || { log "FATAL: closed-loop failed for $run_id"; return 1; }
  log "===== EXPERIMENT $run_id COMPLETE ====="
  return 0
}

# ================= 主流程 =================
log "orchestrate_phase1 START"

# 0) 等 E1（基线，已在跑）
log "step0: wait for E1 (e1-ns32-bt8192) to finish"
wait_closed_loop "e1-ns32-bt8192" "8192 32768" "1 8 16 32" || log "warn: E1 not all summaries (may still be running or failed); continue anyway"

# 1) B 轴
run_experiment "e2-ns32-bt16384" 32 16384 "8192 32768" "1 8 16 32" || log "FATAL E2"
run_experiment "e3-ns32-bt32768" 32 32768 "8192 32768" "1 8 16 32" || log "FATAL E3"

# 2) A 轴
run_experiment "e5-ns64-bt8192" 64 8192 "8192" "32 48 64" || log "FATAL E5"
run_experiment "e6-ns96-bt8192" 96 8192 "8192" "64 96" || log "FATAL E6"

# 3) 恢复基线（config 32/8192 + 服务）
log "restore baseline (ns=32 bt=8192) + restart service"
set_knobs 32 8192
restart_service || log "FATAL: baseline restore restart failed"

# 4) 标记完成
{
  echo "done_at=$(date -u +%FT%TZ)"
  echo "knobs=$(grep -E '^(MAX_NUM_SEQS|MAX_NUM_BATCHED_TOKENS)=' "$CTR_ROOT/config.env" | tr '\n' ' ')"
  echo "models_http=$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "$URL")"
  echo "runs:"
  for r in e1-ns32-bt8192 e2-ns32-bt16384 e3-ns32-bt32768 e5-ns64-bt8192 e6-ns96-bt8192; do
    echo "  $r: $(find "$CTR_ROOT/out/$r" -name 'c[0-9]*.json' ! -name '*_server_metrics*' 2>/dev/null | wc -l) summaries"
  done
} > "$CTR_ROOT/PHASE1_DONE"
log "PHASE1_DONE written"
log "orchestrate_phase1 END"
