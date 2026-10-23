#!/bin/bash
# finalize_env.sh —— 补齐 env.txt 的算子数 + npu-smi 段（collect_env.sh 的 torch 段曾中断）
cd "$(dirname "$0")"
{
  echo
  echo "----- enable_custom_op 算子数 -----"
  python3 /tmp/op_count4.py 2>&1 | grep -v -E 'INFO|WARNING|platform\.py|__init__\.py'
  echo
  echo "----- npu-smi (8x910C, server 运行中) -----"
  npu-smi info 2>/dev/null | head -40
} >> out/run-20261017/env.txt
echo "env.txt finalized -> out/run-20261017/env.txt"
