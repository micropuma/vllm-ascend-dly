from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_noncot_chat_prompt import ceval_datasets as _builtin_ceval

ceval_local_datasets = _builtin_ceval
dataset_root = "/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/accuracy/data-prepared/ceval"
for dataset in ceval_local_datasets:
    dataset["path"] = dataset_root
