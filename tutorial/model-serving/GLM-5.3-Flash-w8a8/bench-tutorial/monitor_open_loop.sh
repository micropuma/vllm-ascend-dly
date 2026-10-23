#!/bin/bash
# monitor_open_loop.sh —— 监视开环进度，完成时写 OPEN_LOOP_DONE 标记
# 完成判据：8 个 summary JSON（isl8192/isl32768 × 4 档 rate）全出，或 launcher 退出。
# 注意排除 <prefix>_server_metrics.json（用 ! -name '*_server_metrics*'）。
cd "$(dirname "$0")"
ROOT=out/run-20261017/rps
MARK=out/run-20261017/OPEN_LOOP_DONE
LOG=out/run-20261017/open_monitor.log
rm -f "$MARK"
while true; do
  n=$(find "$ROOT" -name 'rate-*.json' ! -name '*_server_metrics*' 2>/dev/null | wc -l)
  alive=$(pgrep -f 'run_open_loop[.]sh' 2>/dev/null | wc -l)
  echo "[$(date -u +%FT%TZ)] summaries=$n/8 launcher_alive=$alive" >> "$LOG"
  if [ "$n" -ge 8 ] || [ "$alive" -eq 0 ]; then
    {
      echo "done_at=$(date -u +%FT%TZ)"
      echo "summaries=$n"
      echo "launcher_alive=$alive"
      echo "---- summary files ----"
      find "$ROOT" -name 'rate-*.json' ! -name '*_server_metrics*' 2>/dev/null | sort
    } > "$MARK"
    echo "OPEN LOOP FINISHED (summaries=$n launcher_alive=$alive)" >> "$LOG"
    exit 0
  fi
  sleep 300
done
