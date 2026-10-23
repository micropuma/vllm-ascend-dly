#!/bin/bash
# monitor_closed_loop.sh —— 监视闭环进度，完成时写 CLOSED_LOOP_DONE 标记
# 完成判据：8 个 summary JSON（isl8192/isl32768 × c1/c8/c16/c32）全出，或 launcher 退出。
# 注意：AIPerf 还产出 <prefix>_server_metrics.json，不能用通配 c*.json 计数（会误判）。
cd "$(dirname "$0")"
ROOT=out/run-20261017/concurrency
MARK=out/run-20261017/CLOSED_LOOP_DONE
LOG=out/run-20261017/monitor.log
rm -f "$MARK"
while true; do
  n=0
  for isl in 8192 32768; do
    for c in 1 8 16 32; do
      if [ -f "$ROOT/isl${isl}/c${c}/c${c}.json" ]; then n=$((n+1)); fi
    done
  done
  alive=$(pgrep -f 'run_closed_loop[.]sh' 2>/dev/null | wc -l)
  echo "[$(date -u +%FT%TZ)] summaries=$n/8 launcher_alive=$alive" >> "$LOG"
  if [ "$n" -ge 8 ] || [ "$alive" -eq 0 ]; then
    {
      echo "done_at=$(date -u +%FT%TZ)"
      echo "summaries=$n"
      echo "launcher_alive=$alive"
      echo "---- summary files ----"
      find "$ROOT" -name 'c[0-9]*.json' ! -name '*_server_metrics*' 2>/dev/null | sort
    } > "$MARK"
    echo "CLOSED LOOP FINISHED (summaries=$n launcher_alive=$alive)" >> "$LOG"
    exit 0
  fi
  sleep 60
done
