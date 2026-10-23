from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="glm53-w8a8-vllm-api",
        path="",
        model="glm",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="127.0.0.1",
        host_port=8077,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=True,
        generation_kwargs=dict(temperature=0.0, top_p=1.0, ignore_eos=False),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
