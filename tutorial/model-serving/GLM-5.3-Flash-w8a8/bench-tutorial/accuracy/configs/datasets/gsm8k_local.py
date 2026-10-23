from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as _builtin_gsm8k

gsm8k_local_datasets = _builtin_gsm8k
gsm8k_local_datasets[0]["path"] = "/workspace/vllm-ascend-dly-git/tutorial/model-serving/GLM-5.3-Flash-w8a8/bench-tutorial/accuracy/data-prepared/gsm8k"
