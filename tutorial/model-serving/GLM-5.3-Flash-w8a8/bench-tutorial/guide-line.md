# 910C 系统性能测试

参考 vLLM-ascend 在 950DT 上的系统性能测试，进行对照分析。主要包含如下测试项：

## 测试项清单

| 场景 | 目录 | 测什么 | 本轮状态 |
|---|---|---|---|
| **基线服务** | `baseline/` | MTP3 基线（TP16+DP1+EP、max-num-seqs 32、chunked prefill 8192） | ✅ 本轮 |
| **闭环（concurrency）** | `concurrency/` | 扫并发 1/8/16/32 → μ(L) 曲线 + μ_max + SLO 可用并发 | ✅ 本轮 |
| **开环（rps）** | `rps/` | 按到达率 λ（0.45μ/0.85μ/1.2μ）→ 可持续 λ（过三门） | ✅ 本轮 |
| 长上下文 | `long-context/` | 长上下文性能 | ⏳ 预留（本轮范围外） |
| Agent 轨迹 | `agent/` | Agent 多轮轨迹回放 | ⏳ 预留（本轮范围外） |

## 方法论核心（对齐 950）

> **闭环测产能 μ，开环测可持续到达率 λ，两步靠 μ_max 反推布点 + Little's Law 对账串起来。**

### 闭环做到什么地步
1. 扫并发到曲线压平（相邻档吞吐增幅 <5% → μ_max 收敛；否则续扫 64/96）。
2. 得到 μ(L) 曲线，取 μ_max = 最高点，同时定出 SLO 可用并发（TTFT P95 ≤ SLO 的最高并发档）。
3. 定位硬约束（断崖劣化点 → 对到 max-num-seqs / max-num-batched-tokens 等参数）。

### 开环做到什么地步
1. 布点骑跨饱和线（至少一个 ρ>1 的点）。
2. 每个可用区点过三门：① ρ=λ/μ_max < 0.85 ② 达成率 >95% ③ TTFT P95 ≤ SLO。
3. 可持续 λ = 全过三门的最高 λ。
4. Little's Law 对账：L = 完成RPS × (TTFT_P50 + 512×ITL_P50) 回查闭环曲线，偏差 <5% 判自洽。
5. 统计精度：每点 n≥200（±7%），低档 n 过小须标注。

### 最终交付四样东西
μ_max（产能天花板）· 可持续 λ（容量输入）· 推荐工作点（部署配置）· 瓶颈定位（调优方向）。

## 怎么跑

```bash
# 0. 环境对齐（镜像 digest / git commit / 算子 .so / npu-smi）
bash collect_env.sh out/<run-id>/env.txt

# 1. 基线服务（MTP3）+ warm ≥15min
bash baseline/server.sh            # 后台启动
bash baseline/run_smoke.sh         # 门禁：/v1/models + 真实请求 + AIPerf smoke

# 2. 闭环扫并发
RUN_ID=<run-id> bash concurrency/run_closed_loop.sh

# 3. 用 μ_max 反推开环到达率
RATES=$(python3 analyze.py --run-id <run-id> --emit-rates)
RUN_ID=<run-id> RATES="$RATES" bash rps/run_open_loop.sh

# 4. 汇总对账
python3 analyze.py --run-id <run-id> --output out/<run-id>/summary.md

# 5. 清理（释放 NPU）
bash <repo>/tutorial/inference/vllm-ascend/GLM5.x/kill-process.sh
```

## SLO（v0，依据本机 xieyu-workload-reproduction-v1 + 950 同档实测）

| 指标 | 工作 SLO |
|---|---|
| TTFT P95 @ 8K | ≤ 3.0 s |
| TTFT P95 @ 16K | ≤ 5.0 s |
| TTFT P95 @ 32K | ≤ 8.0 s |
| ITL P99（全档） | ≤ 1.0 s（关注项） |
| 达成率 | ≥ 95% |

## 参考文件

- 方法论：`tutorial/inference/vllm-ascend/GLM5.x/inference-exp/douliyang/baseline/950-result/950-method.md`、`950-analysis-03.md`
- 对标对象：`950-result/950-ref.md`
- 本轮计划：`910-result/task-10-17.md`
