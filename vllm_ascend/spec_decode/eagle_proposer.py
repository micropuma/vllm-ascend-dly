# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendEagleProposer(EagleProposer, AscendSpecDecodeBaseProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        # MTP 和 Eagle1&2&3都复用同一个Proposer，EagleProposer的init中会根据vllm_config.speculative_config来判断是MTP还是Eagle1&2&3
        # 关键点：pass_hidden_states_to_model 设定为true，表示在推理过程中会将隐藏状态传递给模型，以便模型可以利用这些隐藏状态进行更高效的推理。
        AscendSpecDecodeBaseProposer.__init__(self, vllm_config, device, True, runner=runner)
