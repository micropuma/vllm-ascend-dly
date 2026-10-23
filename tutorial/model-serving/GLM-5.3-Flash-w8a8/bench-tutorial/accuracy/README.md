# GLM-5.3-Flash W8A8 Accuracy Baseline

This directory contains the API-serving accuracy baseline. The server command follows `../baseline/server.sh`; the baseline directory is not modified.

## Environment

- vLLM service: `vllm-ascend-dly-v028`, using the existing fixed image and `/workspace/vllm-ascend-dly-git` bind mount.
- AISBench client: `/opt/aisbench-glm53-client-venv`, Python 3.12, AISBench `3.1.20260630`.
- The client virtualenv uses `--system-site-packages` so it can reuse the image's preinstalled PyTorch stack. Every AISBench/extra package is installed into the virtualenv; do not run a dependency-upgrading install against the base interpreter.
- The client does not execute an offline model and sends requests only to `127.0.0.1:8077`.

## Datasets and scoring

- C-Eval: every available subject, `val` split, 1,346 labeled examples. AISBench built-in C-Eval prompt and answer-letter accuracy evaluator.
- GSM8K: `main/test`, 1,319 labeled examples. AISBench built-in GSM8K CoT prompt, final-number postprocessor, and exact numeric accuracy evaluator.
- `prepare_data.py` converts source parquet files into the CSV/JSONL layout expected by the installed AISBench loaders. It never edits the source dataset directory. The conversion manifests record source and generated-file hashes and row counts.
- Results are per dataset. Do not average C-Eval and GSM8K into a single score.

## Run

Run inside the service container. `run_accuracy.sh` starts the service only if port 8077 is not already serving model `glm`; if it started the service, it stops only the recorded matching process after evaluation. A pre-existing service is left running.

```bash
cd /workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/accuracy
AISBENCH_NUM_PROMPTS=1 ./run_accuracy.sh   # small smoke: 1 prompt per configured subject/task
./run_accuracy.sh                         # full labeled splits
```

Set a distinct `ACCURACY_RUN_ID` to name a run. Existing run directories are never overwritten. Results, logs, generated predictions, evaluation details, and manifest are stored in `results/<run-id>/`.

To manage a service started separately by this suite:

```bash
ACCURACY_RUN_ID=<run-id> ./service.sh status
ACCURACY_RUN_ID=<run-id> ./service.sh stop
```

The stop command validates the saved PID command line and refuses to signal an unrelated process.

## Reproducibility record

Before comparing model revisions, record the vLLM-Ascend Git HEAD and dirty status, image digest, model weight identity, server log/config, AISBench venv package lock, dataset conversion manifest, and full command. The current source tree has pre-existing worktree changes; do not reset/clean it as part of this accuracy run.
