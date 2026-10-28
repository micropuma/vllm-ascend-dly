# SPDX-License-Identifier: Apache-2.0
"""Single-kernel indexer KPool slot mapping for GLM-5.3-Flash."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _kpool_slot_mapping_kernel(
    slot_ptr,
    pos_ptr,
    out_ptr,
    num_tokens,
    lbs: tl.constexpr,
    lbs_shift: tl.constexpr,
    lbs_div_cr: tl.constexpr,
    cr: tl.constexpr,
    cr_shift: tl.constexpr,
):
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    mask = offsets < num_tokens
    # Slot and position values fit in int32; keeping arithmetic narrow avoids
    # software-emulated int64 division on Ascend AIV.
    slot = tl.load(slot_ptr + offsets, mask=mask, other=-1).to(tl.int32)
    pos = tl.load(pos_ptr + offsets, mask=mask, other=0).to(tl.int32)
    valid = (slot >= 0) & ((pos + 1) % cr == 0)
    safe = tl.maximum(slot, 0)
    if lbs_shift >= 0:
        block = safe >> lbs_shift
        in_block = safe & (lbs - 1)
    else:
        block = safe // lbs
        in_block = safe % lbs
    if cr_shift >= 0:
        compressed_offset = in_block >> cr_shift
    else:
        compressed_offset = in_block // cr
    value = tl.where(valid, block * lbs_div_cr + compressed_offset, -1)
    tl.store(out_ptr + offsets, value.to(tl.int64), mask=mask)


def _pow2_shift(value: int) -> int:
    if value > 0 and value & (value - 1) == 0:
        return value.bit_length() - 1
    return -1


def kpool_slot_mapping_triton(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    logical_block_size: int,
    compress_ratio: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute compressed slots, optionally writing an existing buffer."""
    if compress_ratio <= 1 or logical_block_size <= 0 or logical_block_size % compress_ratio:
        raise ValueError(
            f"logical_block_size={logical_block_size} must be divisible by "
            f"compress_ratio={compress_ratio}."
        )
    num_tokens = slot_mapping.shape[0]
    if positions.shape[0] != num_tokens:
        raise ValueError("slot_mapping and positions must have the same length")
    if out is None:
        out = torch.empty_like(slot_mapping, dtype=torch.int64)
    elif out.shape[0] < num_tokens or out.dtype != torch.int64:
        raise ValueError("out must be an int64 buffer covering all tokens")
    if num_tokens == 0:
        return out[:0]
    _kpool_slot_mapping_kernel[(triton.cdiv(num_tokens, 1024),)](
        slot_mapping,
        positions,
        out,
        num_tokens,
        lbs=logical_block_size,
        lbs_shift=_pow2_shift(logical_block_size),
        lbs_div_cr=logical_block_size // compress_ratio,
        cr=compress_ratio,
        cr_shift=_pow2_shift(compress_ratio),
        num_warps=4,
    )
    return out[:num_tokens]
