#!/bin/bash
# collect_env.sh —— 环境采集：镜像 / git commit / 算子 .so / npu-smi
# 产出：out/<run-id>/env.txt（阶段 2 三对齐证据留档）
set -euo pipefail
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=${1:-$HERE/out/env.txt}
mkdir -p "$(dirname "$OUT")"

{
  echo "===== collect_env $(date -u +%FT%TZ) ====="
  echo
  echo "----- 镜像 digest -----"
  # 容器运行于本地镜像（build.md 注明：quay 远端 digest 为 bded7da4f3...，本容器用本地 tag）
  echo "image_id=$(docker inspect --format '{{.Image}}' "$(hostname)" 2>/dev/null || echo 'n/a(容器内不可查，见 build.md)')"
  echo "expected_remote_digest=sha256:bded7da4f300800ec4715dbde44531c4735cf0950fa313977248ef0dd74866b8"
  echo
  echo "----- git commit -----"
  cd /workspace/vllm-ascend-dly-git 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "git unavailable"
  echo "expected=3caf228dd8667a525809e92b887a809a88ce17ff"
  echo
  echo "----- vllm-ascend 算子 .so -----"
  VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")
  echo "vllm_dir=$VLLM_DIR"
  ls -la /workspace/vllm-ascend-dly-git/*.so 2>/dev/null || find / -name 'vllm_ascend_C*.so' 2>/dev/null | head -5
  echo
  echo "----- enable_custom_op 算子数 -----"
  python3 - <<'PY'
try:
    import torch
    print("npu_available:", torch.npu.is_available())
    print("device_count:", torch.npu.device_count())
    try:
        import vllm_ascend
        ops = torch._C._jit_get_all_schemas()
        ascend_ops = [str(o) for o in ops if "_C_ascend" in str(o)]
        print("_C_ascend ops:", len(ascend_ops))
    except Exception as e:
        print("vllm_ascend import err:", e)
except Exception as e:
    print("torch err:", e)
PY
  echo
  echo "----- npu-smi -----"
  npu-smi info 2>/dev/null | grep -E "NPU|Chip|Process" | head -40 || echo "npu-smi unavailable"
} > "$OUT" 2>&1

echo "环境采集完成 -> $OUT"
