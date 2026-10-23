# agent 轨迹回放 —— AIPerf 原生 loader（tracelab）

用 AIPerf `--custom-dataset-type tracelab` 原生回放 **TraceLab（uw-syfi）agentic 编码轨迹**，
压 GLM-5.3-Flash-w8a8（端口 8077，served-model `glm`），测多轮 agentic 负载下的
TTFT / TPOT / ITL / prefix-cache 命中 / MTP 接受率。

> 替代 950DT 上 thoughtworks 自写 driver 的那条链路：这里不再自己拼多轮请求、
> 自己算前缀，直接用 AIPerf 的 tracelab loader 把「session → turn → subagent → KV 前缀链」
> 一次重建出来，口径与 NVIDIA 官方教程一致。

## 数据集

| 文件 | 大小 | 内容 |
|---|---|---|
| `syfi_coding_trace.jsonl.gz` | 100.9 MB | 全量：665,453 行 = **8,058 session**（92.7% 多轮，轮数 p50=16 / p90=146 / max 21351） |
| `tracelab_slice.jsonl.gz` | 42 KB | smoke 切片：25 个短 session / 245 轮（2~20 轮/session） |

来源：`github.com/uw-syfi/TraceLab` release v0.0.2。一行 = 一轮（按 `session_id` 分组），
字段含 `input_tokens_total / prefix_tokens / newly_append_tokens / timing_events`。
gzip 直读，无需解压。

## 4 层轨迹结构（loader 重建出来的）

AIPerf 的 `tracelab` loader（`aiperf/dataset/loader/tracelab_trace.py` + `_tracelab_convert.py`）把
原始 JSONL 重建为 4 层：

1. **session（root trace）** —— 一次 agentic 编码会话（`session_id` 分组）。
2. **turn（round）** —— 每轮一次 LLM 请求；`t` 为该 session 内相对秒（首轮 ≈0）。
3. **subagent（子代理）** —— 语料里子代理是独立 `session_id`，loader 用「时序包含」
   （`build_join_index`：子 session 的整段 span 落在父 session 的 spawn 工具调用窗口内）
   恢复父子嵌套，挂成 `SPAWN/SPAWN_JOIN`。
4. **KV 前缀链（hash_ids）** —— TraceLab 无 content hash，loader 用引擎上报的
   `prefix_tokens / newly_append_tokens` 切分，按 `block_size=64` 合成 block id，
   复现逐轮的 prefix 复用（**这就是 prefix-cache 命中率的来源**）。

## 调度语义（关键）

- **`--fixed-schedule`**：按录制时间戳回放，`--fixed-schedule-end-offset 300000` 只回放
  每 session 前 300s 内的轮次（loader 按 `req.t` 过滤）。
- **`--no-fixed-schedule`**：闭环全速回放，`--inter-turn-delay-cap-seconds 5` 把轮间
  思考时间封顶到 5s。
- **`--concurrency N`** = 并发 session 数（lane），两种模式都受它约束（每 session 首轮抢
  一个 session slot，`credit/issuer.py`）；**`--num-dataset-entries N`** = 回放多少 session
  （sequential 采样，取文件前 N 个）。
- 两者缺一不可：`--concurrency` 管并行度，`--num-dataset-entries` 管总量。缺了会把
  8058 个 session 首轮在 t≈0 同时开火（服务 `max-num-seqs=32` 直接打爆）。

## 与 950DT 对照

| 维度 | 950DT（thoughtworks 自写 driver） | 本方案（AIPerf 原生 loader） |
|---|---|---|
| 轨迹来源 | 自拼多轮 | TraceLab 官方语料，gzip 直读 |
| 前缀/缓存 | 手工算前缀 | loader 合成 hash_ids 复现 prefix 复用 |
| 子代理 | 无 | timing containment 恢复嵌套 |
| 调度 | 自实现 | `--fixed-schedule` / `--no-fixed-schedule` + concurrency |
| 指标 | 自采 | TTFT/TPOT/ITL/prefix-cache 命中/MTP 接受率，`--export-level records` |

## 怎么跑

```bash
# 容器内（服务已在 8077 跑）：
docker exec vllm-ascend-dly-v028 bash \
  /workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/agent/test.sh
```

产出在 `bench-tutorial/out/agent-<UTC>/`：
- `tracelab_smoke/`（summary JSON + records，验证链路）
- `tracelab_full/`（fixed-schedule 正式回放）

> 铁律：压测结束**不 kill 远端 vllm 服务**（端口 8077），只等 aiperf 客户端自己退出。

## weka 备选（需容器内 HF 下载，走代理）

```bash
"$AIPERF" profile --url "$SERVER_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --endpoint-type chat --streaming \
  --public-dataset semianalysis-cc-traces-weka-with-subagents --fixed-schedule \
  --output-artifact-dir "$OUT/weka" --profile-export-prefix weka --export-level records
```
