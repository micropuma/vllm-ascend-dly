# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Two-NPU coverage of real AllToAll metadata, permutations and combine.

Run directly to avoid loading model-serving pytest fixtures:
    python tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_alltoall_preprocess.py
"""

import socket
import traceback
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV
from vllm_ascend.quantization.quant_type import QuantType


class _WorldGroupDispatcher(TokenDispatcherWithAll2AllV):
    """Use a real HCCL world group without constructing a serving engine."""

    @property
    def ep_group(self):
        return dist.group.WORLD

    @property
    def ep_size(self):
        return dist.get_world_size()

    @property
    def ep_rank(self):
        return dist.get_rank()


def _check_metadata(dispatcher, all_ids, rank):
    histograms = torch.stack([torch.bincount(ids.flatten(), minlength=dispatcher.num_experts) for ids in all_ids])
    local_experts = dispatcher.num_local_experts
    table = histograms[:, rank * local_experts : (rank + 1) * local_experts]
    ids = all_ids[rank].to(device="npu", dtype=torch.int32)
    counts, input_cpu, output_cpu, device_table, done, num_out = dispatcher._preprocess(ids)
    done.synchronize()
    np.testing.assert_array_equal(input_cpu.numpy(), histograms[rank].reshape(2, local_experts).sum(1).numpy())
    np.testing.assert_array_equal(output_cpu.numpy(), table.sum(1).numpy())
    torch.testing.assert_close(counts.cpu(), table.sum(0))
    torch.testing.assert_close(device_table.cpu(), table.to(device_table.dtype))
    assert num_out == ids.numel()

    if all(source_ids.shape[0] > 0 for source_ids in all_ids):
        result = dispatcher._dispatch_preprocess(torch.ones((ids.shape[0], 64), device="npu"), ids)
        np.testing.assert_array_equal(result[3], input_cpu.numpy())
        np.testing.assert_array_equal(result[4], output_cpu.numpy())
        if local_experts > 1:
            expected = torch.tensor(
                [expert for row in table.tolist() for expert, count in enumerate(row) for _ in range(count)],
                dtype=torch.int32,
            )
            torch.testing.assert_close(result[5].cpu(), expected)
        else:
            assert result[5] is None


def _check_roundtrip(dispatcher, all_ids, rank, quant_type):
    ids_cpu = all_ids[rank]
    num_tokens, top_k = ids_cpu.shape
    # Exact binary fractions and nonzero inputs keep quantization error bounded
    # while making swapped tokens, experts, or TopK weights observable.
    hidden_cpu = (torch.arange(num_tokens * 64).reshape(num_tokens, 64) % 61 + 1).to(torch.bfloat16) / 64
    hidden_cpu += rank
    weights_cpu = torch.arange(1, top_k + 1, dtype=torch.float32).expand(num_tokens, -1).clone()
    weights_cpu /= weights_cpu.sum(-1, keepdim=True)
    dispatched = dispatcher.token_dispatch(
        MoETokenDispatchInput(
            hidden_states=hidden_cpu.npu(),
            topk_ids=ids_cpu.to(device="npu", dtype=torch.int32),
            topk_weights=weights_cpu.npu(),
            routing=MoeRouterInput(
                expert_map=None, global_redundant_expert_num=0, mc2_mask=None, apply_router_weight_on_input=False
            ),
            quant=MoEQuantParams(quant_type=quant_type),
        )
    )
    histograms = torch.stack([torch.bincount(ids.flatten(), minlength=dispatcher.num_experts) for ids in all_ids])
    local_experts = dispatcher.num_local_experts
    counts = histograms[:, rank * local_experts : (rank + 1) * local_experts].sum(0)
    torch.testing.assert_close(dispatched.group_list.cpu(), counts)
    assert dispatched.hidden_states.shape[0] == int(counts.sum())
    expanded_experts = torch.repeat_interleave(torch.arange(local_experts) + rank * local_experts, counts)

    expert_input = dispatched.hidden_states.to(torch.bfloat16)
    if quant_type != QuantType.NONE:
        assert dispatched.dynamic_scale is not None
        expert_input = (dispatched.hidden_states.float() * dispatched.dynamic_scale.reshape(-1, 1)).to(torch.bfloat16)
    # A per-expert scalar is an independent, easily checked stand-in for FFN.
    # It verifies both permutations, both collectives and the final weighting.
    factors = (expanded_experts % 4 + 1).to(device="npu", dtype=torch.bfloat16)
    expert_output = expert_input * factors.unsqueeze(-1)
    actual = dispatcher.token_combine(expert_output, dispatched.combine_metadata).cpu()
    expected = (
        hidden_cpu.float().unsqueeze(1)
        * (ids_cpu % 4 + 1).float().unsqueeze(-1)
        * weights_cpu.unsqueeze(-1)
    ).sum(1).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


@torch.inference_mode()
def _worker(rank, port, result_queue):
    try:
        torch.npu.set_device(rank)
        dist.init_process_group(
            "hccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
        )
        cases = [
            (4, [torch.tensor([[0, 1], [0, 2], [1, 0]]), torch.tensor([[0, 3], [1, 2]])]),
            (4, [torch.tensor([[0, 1], [1, 0], [0, 1]]), torch.tensor([[1, 0], [0, 1]])]),
            (2, [torch.tensor([[0, 1], [1, 0], [0, 1]]), torch.tensor([[1, 0], [0, 1]])]),
        ]
        # Actual GLM expert count and TopK, with both tiny and larger traffic.
        for num_tokens in (1, 64):
            cases.append(
                (
                    288,
                    [
                        (torch.arange((num_tokens + source_rank) * 8).reshape(-1, 8) * 17 + source_rank) % 288
                        for source_rank in range(2)
                    ],
                )
            )
        for num_experts, all_ids in cases:
            dispatcher = _WorldGroupDispatcher(
                num_experts=num_experts, num_local_experts=num_experts // 2, top_k=all_ids[rank].shape[1]
            )
            _check_metadata(dispatcher, all_ids, rank)
            for quant_type in (QuantType.NONE, QuantType.W8A8):
                _check_roundtrip(dispatcher, all_ids, rank, quant_type)

        # Metadata must also be safe when a source rank, or the entire step,
        # has no tokens. No claim is made about empty-input FFN kernel support.
        dispatcher = _WorldGroupDispatcher(num_experts=4, num_local_experts=2, top_k=2)
        empty = torch.empty((0, 2), dtype=torch.int64)
        _check_metadata(dispatcher, [empty, torch.tensor([[0, 3], [1, 2]])], rank)
        _check_metadata(dispatcher, [empty, empty], rank)
        torch.npu.synchronize()
        result_queue.put((rank, None))
    except Exception:
        result_queue.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_alltoall_preprocess_and_roundtrip():
    if torch.npu.device_count() < 2:
        raise RuntimeError("This test requires two Ascend NPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [context.Process(target=_worker, args=(rank, port, result_queue)) for rank in range(2)]
    try:
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=240) for _ in processes]
        for process in processes:
            process.join(timeout=30)
        errors = [f"rank {rank}: {error}" for rank, error in results if error is not None]
        assert not errors, "\n".join(errors)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


if __name__ == "__main__":
    test_alltoall_preprocess_and_roundtrip()
    print("PASS: two-NPU metadata and BF16/W8A8 dispatch/combine roundtrips")
