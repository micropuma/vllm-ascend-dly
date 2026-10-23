# Fixed GLM-5.3-Flash Prefill 8K Performance Point

This is the performance benchmark entry for the KDA and MoE prefill work. It is a fixed AIPerf closed-loop point based on the existing `bench-tutorial/config.env` contract. It does not call `baseline/run_smoke.sh`.

## Frozen workload

- AIPerf 0.13.0 from `/workspace/aiperf-venv/bin/aiperf`.
- Chat completions, streaming, `glm`, existing server at `http://127.0.0.1:8077`.
- Fixed input/output lengths: 8192 / 512 tokens, `random-range-ratio=0`.
- `ignore_eos:true`, seed 1024, Poisson arrival pattern as in the shared benchmark contract.
- Client concurrency: **8**. No explicit request-rate cap; AIPerf maintains eight in-flight requests and sends replacements as requests complete (closed loop).
- One measured batch of 256 requests per service startup, preceded by 16 warmup requests. Repeat with three independent service startups per experiment arm, using a unique run ID each time (768 measured requests and 48 warmup requests per arm).
- Server settings remain the shared baseline: TP16/DP1/EP, `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=8192`, MTP3 and existing graph/quantization settings. Client concurrency 8 is distinct from the server's maximum sequence capacity of 32.

Concurrency 8 matches the intended loaded-serving point and allows cross-request scheduling/batching to affect the measured TTFT. Compare B0 and every candidate at this same concurrency. This single point is not a concurrency sweep or a maximum-throughput claim.

## Run

Run inside `vllm-ascend-dly-v028`, with the intended B0 or candidate service already active. The script checks `/v1/models`; it does not start, stop, or restart the service. Each run ID is immutable and writes under `results/<run-id>/`.

```bash
cd /workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/baseline/perf
export PERF_SOURCE_REPO=/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/baseline/round1-20261022/workspaces/B0
export PERF_SERVER_RUN_DIR=/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/baseline/round1-20261022/B0/environment/perf-r1
PERF_RUN_ID=B0-prefill8k-20261022-r1 ./run_prefill8k.sh
```

Run the script once per service startup. Restart the service between repetitions, use a new run ID for each startup, and complete three paired, balanced startup repetitions per experiment arm. Run only one arm on the NPU at a time. Do not compare B0 and a candidate when server settings, model/tokenizer, AIPerf version, data-generation seed, cache policy, or source identity differ. Raw AIPerf artifacts, complete commands, service identity, and manifests are retained per startup.

Set both identity paths to the service actually running. The script verifies the recorded plugin import path and hashes the real startup command, imports and source patch. The shared baseline server script alone does not identify an isolated experiment arm.

## Matching profile workload

The profile plan has a dedicated `prefill-8k-c8-256` scenario at `baseline/profile/profile-plan-1/scenarios/prefill-8k-c8-256.env`: input/output 8192/512, 256 prompts, max concurrency 8. Use it for B0 and every candidate. The server profiler is configured for its first 20 model iterations, so it captures a bounded slice of the full workload. Check that the capture includes the concurrent prefill iterations; the remaining client requests validate workload completion but are not all expected inside the trace. The profile client's endpoint/tokenization differs from AIPerf, so record actual token counts and use profile for attribution, not as the performance score.

This fixed point is the primary local regression signal for 8K prefill. It does not by itself establish the maximum serving throughput or a production SLA.
