from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_noncot_chat_prompt import ceval_datasets as _builtin_ceval

ceval_smoke_datasets = [_builtin_ceval[6]]
ceval_smoke_datasets[0]["path"] = "/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/accuracy/data-prepared/ceval"
