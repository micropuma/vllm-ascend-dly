# GLM-5.3-Flash GSM8K Accuracy Regression

This directory defines the accuracy gate for the KDA and MoE performance work. It is separate from `baseline/run_smoke.sh`: accuracy must pass before performance A/B, MSTX annotation, or profiling.

## Three comparison levels

1. **Task score:** run all 1,319 examples from GSM8K `main/test` through the existing AISBench suite and compare paired per-example outcomes. The initial non-inferiority margin is -1.0 percentage point; freeze it in the experiment contract before evaluating candidates.
2. **Generated output:** on a fixed 15-example fixture, compare raw response text and generated token IDs. Text equality does not imply token-ID equality. Record answer extraction, termination reason, first differing token, and output length.
3. **Logits:** compare full-vocabulary, pre-sampler raw logits at each next-token position under the same teacher-forced prefix. Use the normalized gold answer continuation, not each arm's divergent free-generation context.

Bitwise equality is reported as a strict diagnostic. It is a hard gate only when B0 self-repeat is bitwise stable and the candidate follows the same floating-point operation order. Otherwise, freeze tolerance metrics from B0 self-repeat before examining candidate results. NaN/Inf and mismatched tokenization or positions fail the comparison.

## Run order

1. Start from the same service configuration intended for the testbench. Record source/image/model/tokenizer identities and confirm the server is healthy.
2. Create a new immutable run directory with `run_gsm8k.sh`. Run B0 twice before any candidate. Do not reuse or overwrite a run ID.
3. Verify the 15-row fixture and implement the model-runner teacher-forcing logits exporter. OpenAI API `logprobs` is not a substitute for full-vocabulary logits.
4. Compare B0 with its repeat. If scores, outputs, or logits are unstable beyond the predeclared limits, stop and investigate the harness/environment.
5. Run each candidate with the same fixture, service contract, and artifacts. Pass the correctness gate before performance A/B. Run MSTX/profile only after correctness and unprofiled performance checks pass.

## Files and commands

- `run_gsm8k.sh`: invokes the repository's existing `bench-tutorial/accuracy` preparation and AISBench GSM8K evaluator, writing a unique result directory under this directory's `results/`.
- `make_fixture.py`: creates the immutable first-15 fixture from the exact prepared `test.jsonl` order. IDs are row-based and the input file hash is recorded. The actual AISBench-rendered prompt and tokenizer IDs must be added by the logits-capture adapter before comparing logits.
- `compare_outputs.py`: compares normalized per-example JSONL records with `example_id`, `prediction`, `extracted_answer`, `gold_answer`, and boolean `correct`; optional `token_ids` and `finish_reason` add generated-sequence checks. It reports paired score/non-inferiority and exact McNemar results, per-sample answer/text/token agreement, and first token divergence. The AISBench detail-to-normalized-JSONL adapter remains a required integration task because output schemas depend on the installed AISBench version.
- `compare_logits.py`: compares two NumPy `.npz` captures containing `logits` (`[positions, vocab]`), `prompt_token_ids`, `token_ids` (gold continuation IDs), and optional `valid_vocab_size` scalar.

Example (inside the test container):

```bash
cd /workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/baseline/accuracy
ACCURACY_RUN_ID=B0-20261022-r1 ./run_gsm8k.sh
python3 make_fixture.py --input ../../accuracy/data-prepared/gsm8k/test.jsonl --output results/B0-20261022-r1/fixture-15.jsonl
python3 compare_outputs.py results/B0-20261022-r1/predictions.jsonl results/B0-20261022-r2/predictions.jsonl --details results/B0-20261022-r2/prediction-diff.jsonl
python3 compare_logits.py results/B0-20261022-r1/logits.npz results/B0-20261022-r2/logits.npz --output results/B0-20261022-r2/logits-vs-B0.json
```

The run script and utilities do not implement the model-runner logits hook or claim that AISBench outputs include token IDs. Those are explicit adapter work before level 3 can pass. Preserve raw service/evaluator output and all manifests; never edit a B0 result in place.
