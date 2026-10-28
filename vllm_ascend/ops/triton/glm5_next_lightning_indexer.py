# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fast path for the narrow GLM5 Next KPool lightning indexer.

The heavy part is scoring every compressed pool against the token's
head-weighted query: a paged-cache gather plus a 128-dim matvec. Doing that in
torch lowers to an aclnnIndex/SearchSorted small-op flood, so it stays in one
triton kernel that writes the raw pool scores to a scratch buffer. The top-k
selection itself is a single fused aclnn ``topk`` call, followed by a few
element-wise ops for pool->token expansion and causal tail append.

There is no hard length limit: the kernel tiles pools dynamically, and the
wrapper chunks the token dimension so the fp32 scores scratch stays under
``TRITON_SCORES_CHUNK_BYTES`` even for long-context prefill (e.g. 128K input).

Note: greedy/sort-based in-kernel top-k is deliberately avoided — reductions
over wide vectors scalarize on the Ascend triton backend, which is slower than
the torch fallback by two orders of magnitude.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2
from vllm_ascend import envs
from vllm_ascend.models.glm5next.indexer_contract import indexer_pool_id_enabled

# Fix the pool tile width to avoid recompiling as sequence lengths grow.
TRITON_POOL_CHUNK_SIZE = 2048
# Sub-tile of pools whose K rows are loaded as one coalesced 2D block.
# 128 pools x 128 dims x fp32 = 64KB, keeping the UB footprint small.
TRITON_POOL_SUB_TILE_SIZE = 128
# Chunk tokens to limit the FP32 score buffer to this budget where possible.
TRITON_SCORES_CHUNK_BYTES = 256 * 1024 * 1024

# CANN's fused TopKV2 high_performance kernel faults (aicore VEC 507035) on -inf
# cells produced by a ragged mask at a non-aligned width. The fp32-lowest finite
# value orders identically -- no real score can reach it -- so the wrapper
# rewrites the mask sentinel to it on the way into top-k. Producers keep using
# -inf; the guard sits at the consumer that faults.
NEG_INF_SENTINEL = torch.finfo(torch.float32).min


# Keep batch-varying inputs unspecialized to avoid recompiling per step.
# REQ_POW2 stays constexpr for tl.arange; warm up its power-of-two variants.
@triton.jit(do_not_specialize=["token_offset", "max_pool_seq_len", "num_reqs", "num_cache_blocks"])
def _glm5_next_lightning_indexer_score_kernel(
    qbar_ptr,
    indexer_cache_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    scores_ptr,
    token_offset,
    max_pool_seq_len,
    num_reqs,
    num_cache_blocks,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    BLOCK_POOL: tl.constexpr,
    SUB_POOL: tl.constexpr,
    OPTIMIZED_INF_FILL: tl.constexpr,
):
    # qbar/scores rows are chunk-local; positions/query-boundary lookups use
    # the batch-global token index.
    local_token_idx = tl.program_id(0)
    token_idx = local_token_idx + token_offset
    chunk = tl.program_id(1)

    req_offsets = tl.arange(0, REQ_POW2)
    query_ends = tl.load(cum_query_lens_ptr + req_offsets, mask=req_offsets < num_reqs, other=2147483647)
    req_id = tl.sum(tl.where(token_idx >= query_ends, 1, 0))
    # Full ACL graphs keep padded rows beyond the last request; keep their
    # pointer arithmetic in bounds even though their outputs are unused.
    req_id = tl.minimum(req_id, num_reqs - 1)

    pos = tl.load(positions_ptr + token_idx).to(tl.int32)
    request_pool_len = tl.load(indexer_seq_lens_ptr + req_id).to(tl.int32)
    causal_pool_len = (pos + 1) // INDEX_KPOOL
    visible_pool_len = tl.minimum(causal_pool_len, request_pool_len)

    dim_offsets = tl.arange(0, HEAD_DIM)
    qbar = tl.load(qbar_ptr + local_token_idx * HEAD_DIM + dim_offsets)

    chunk_start = chunk * BLOCK_POOL
    # Dynamic trip count: requests shorter than the static max pool length
    # skip their out-of-range sub-tiles even inside captured graphs.
    chunk_visible = tl.maximum(tl.minimum(visible_pool_len, chunk_start + BLOCK_POOL) - chunk_start, 0)
    num_subs = tl.cdiv(chunk_visible, SUB_POOL)
    for sub in tl.range(num_subs):
        pool_offsets = chunk_start + sub * SUB_POOL + tl.arange(0, SUB_POOL)
        in_range = pool_offsets < max_pool_seq_len
        valid_pool = in_range & (pool_offsets < visible_pool_len)
        logical_pages = pool_offsets // pool_block_size
        page_offsets = pool_offsets % pool_block_size
        physical_blocks = tl.load(
            indexer_block_table_ptr + req_id * block_table_stride_req + logical_pages * block_table_stride_page,
            mask=in_range,
            other=0,
        ).to(tl.int64)
        # Clamp both sides: padded/stale block-table entries must never form
        # an out-of-range cache address, even though their loads are masked.
        physical_blocks = tl.minimum(tl.maximum(physical_blocks, 0), num_cache_blocks - 1)
        k_addrs = (
            physical_blocks[:, None] * cache_stride_block
            + page_offsets[:, None] * cache_stride_offset
            + dim_offsets[None, :] * cache_stride_d
        )
        k_tile = tl.load(indexer_cache_ptr + k_addrs, mask=valid_pool[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k_tile * qbar[None, :], axis=1)
        if OPTIMIZED_INF_FILL:
            neg_inf = tl.full((SUB_POOL,), -3.4028234663852886e38, dtype=tl.float32)
            scores = tl.where(valid_pool, scores, neg_inf)
            scores = tl.where(scores != scores, neg_inf, scores)
        else:
            scores = tl.where(valid_pool, scores, float("-inf"))
        tl.store(scores_ptr + local_token_idx * max_pool_seq_len + pool_offsets, scores, mask=in_range)
    if OPTIMIZED_INF_FILL:
        # The optimized wrapper passes an empty scratch tensor. Cover every
        # remaining cell in this chunk exactly once, including short requests.
        chunk_end = tl.minimum(chunk_start + BLOCK_POOL, max_pool_seq_len)
        num_subs_all = tl.cdiv(tl.maximum(chunk_end - chunk_start, 0), SUB_POOL)
        for sub in tl.range(num_subs, num_subs_all):
            pool_offsets = chunk_start + sub * SUB_POOL + tl.arange(0, SUB_POOL)
            tl.store(
                scores_ptr + local_token_idx * max_pool_seq_len + pool_offsets,
                tl.full((SUB_POOL,), -3.4028234663852886e38, dtype=tl.float32),
                mask=pool_offsets < max_pool_seq_len,
            )


def _legacy_indexer(
    query: torch.Tensor,
    indexer_cache: torch.Tensor,
    weights: torch.Tensor,
    cum_query_lens: torch.Tensor,
    indexer_seq_lens: torch.Tensor,
    indexer_block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    index_topk: int,
    index_kpool: int,
    max_pool_seq_len: int,
) -> torch.Tensor:
    pool_topk = index_topk // index_kpool
    output_width = index_topk + index_kpool - 1
    num_tokens = query.shape[0]
    if num_tokens == 0:
        return torch.empty(
            (0, 1, output_width),
            dtype=torch.int32,
            device=query.device,
        )

    output = torch.empty(
        (num_tokens, 1, output_width),
        dtype=torch.int32,
        device=query.device,
    )
    if max_pool_seq_len == 0:
        output.fill_(-1)
        tail_offsets = torch.arange(index_kpool - 1, device=query.device)
        tail_start = (positions + 1) // index_kpool * index_kpool
        tail = torch.where(
            tail_offsets[None, :] < (positions + 1 - tail_start)[:, None],
            tail_start[:, None] + tail_offsets[None, :],
            -1,
        )
        output[:, 0, index_topk:] = tail.to(torch.int32)
        return output
    block_pool = TRITON_POOL_CHUNK_SIZE
    num_chunks = (max_pool_seq_len + block_pool - 1) // block_pool
    topk = min(pool_topk, max_pool_seq_len)
    token_offsets = torch.arange(index_kpool, device=query.device)
    tail_offsets = torch.arange(index_kpool - 1, device=query.device)
    optimized_inf_fill = bool(envs.VLLM_ASCEND_ENABLE_INDEXER_INF_FILL)

    # Chunk the token dimension so the fp32 scores scratch stays bounded;
    # long-context prefill would otherwise need num_tokens x max_pool_seq_len
    # x 4 bytes (over 1GB at 128K context with a full prefill batch).
    token_chunk = max(1, TRITON_SCORES_CHUNK_BYTES // (max_pool_seq_len * 4))
    for token_start in range(0, num_tokens, token_chunk):
        token_end = min(token_start + token_chunk, num_tokens)
        rows = token_end - token_start
        # Head-weighted query, computed once here instead of per chunk program.
        qbar = (
            (query[token_start:token_end].float() * weights[token_start:token_end].float().unsqueeze(-1))
            .sum(dim=1)
            .contiguous()
        )
        if optimized_inf_fill:
            scores = torch.empty(
                (rows, max_pool_seq_len),
                dtype=torch.float32,
                device=query.device,
            )
        else:
            # Legacy path: initialize all cells, then rewrite -inf for TopKV2.
            scores = torch.full(
                (rows, max_pool_seq_len),
                float("-inf"),
                dtype=torch.float32,
                device=query.device,
            )
        _glm5_next_lightning_indexer_score_kernel[(rows, num_chunks)](
            qbar,
            indexer_cache,
            cum_query_lens,
            indexer_seq_lens,
            indexer_block_table,
            positions,
            scores,
            token_start,
            max_pool_seq_len,
            cum_query_lens.shape[0],
            indexer_cache.shape[0],
            indexer_cache.stride(0),
            indexer_cache.stride(1),
            indexer_cache.stride(3),
            indexer_block_table.stride(0),
            indexer_block_table.stride(1),
            indexer_cache.shape[1],
            next_power_of_2(max(1, cum_query_lens.shape[0])),
            query.shape[2],
            index_kpool,
            block_pool,
            TRITON_POOL_SUB_TILE_SIZE,
            optimized_inf_fill,
        )

        if not optimized_inf_fill:
            # TopKV2 faults on -inf at a non-aligned width. Keep this full-width
            # cleanup on the legacy path; the optimized kernel already wrote the
            # finite sentinel and removed NaNs while producing each cell.
            scores.nan_to_num_(nan=NEG_INF_SENTINEL, neginf=NEG_INF_SENTINEL)
        topk_vals, pool_ids = torch.topk(scores, topk, dim=1)
        pool_ids = torch.where(
            topk_vals <= NEG_INF_SENTINEL,
            torch.full_like(pool_ids, -1),
            pool_ids,
        )
        history = pool_ids.unsqueeze(-1) * index_kpool + token_offsets
        history = torch.where(
            pool_ids.unsqueeze(-1) >= 0,
            history,
            torch.full_like(history, -1),
        ).reshape(rows, topk * index_kpool)
        if topk < pool_topk:
            history = torch.nn.functional.pad(
                history,
                (0, (pool_topk - topk) * index_kpool),
                value=-1,
            )

        pos = positions[token_start:token_end]
        tail_start = (pos + 1) // index_kpool * index_kpool
        tail_count = pos + 1 - tail_start
        tail = torch.where(
            tail_offsets[None, :] < tail_count[:, None],
            tail_start[:, None] + tail_offsets[None, :],
            -1,
        )
        output[token_start:token_end, 0] = torch.cat([history, tail], dim=1).to(torch.int32)
    return output


@triton.jit(do_not_specialize=["token_start", "max_pool_seq_len", "num_reqs", "num_cache_blocks", "num_rows"])
def _glm5_indexer_gemm_score_kernel(
    qbar_ptr,
    indexer_cache_ptr,
    cum_query_lens_ptr,
    indexer_seq_lens_ptr,
    indexer_block_table_ptr,
    positions_ptr,
    scores_ptr,
    token_start,
    max_pool_seq_len,
    num_reqs,
    num_cache_blocks,
    num_rows,
    cache_stride_block: tl.constexpr,
    cache_stride_offset: tl.constexpr,
    cache_stride_d: tl.constexpr,
    block_table_stride_req: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    pool_block_size: tl.constexpr,
    REQ_POW2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INDEX_KPOOL: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
    OPTIMIZED_INF_FILL: tl.constexpr,
):
    req = tl.program_id(0)
    if req >= num_reqs:
        return
    sub_tile = tl.program_id(1)

    # chunk-local row range of this request; cum_query_lens is the per-request
    # END offset array (same layout as indexer_seq_lens), not a cu array.
    r_hi = tl.load(cum_query_lens_ptr + req)
    if req == 0:
        r_lo = r_hi * 0  # keep int64 type consistent across branches
    else:
        r_lo = tl.load(cum_query_lens_ptr + req - 1)
    lo = tl.maximum(r_lo - token_start, 0)
    hi = tl.minimum(r_hi - token_start, num_rows)
    if req == num_reqs - 1:
        hi = num_rows
    req_pool_len = tl.load(indexer_seq_lens_ptr + req).to(tl.int32)

    # gather this pool tile once for the whole request
    pool_offsets = sub_tile * BN + tl.arange(0, BN)
    in_range = pool_offsets < max_pool_seq_len
    logical_pages = pool_offsets // pool_block_size
    page_offsets = pool_offsets % pool_block_size
    physical_blocks = tl.load(
        indexer_block_table_ptr + req * block_table_stride_req + logical_pages * block_table_stride_page,
        mask=in_range & (pool_offsets < req_pool_len),
        other=0,
    ).to(tl.int64)
    physical_blocks = tl.minimum(tl.maximum(physical_blocks, 0), num_cache_blocks - 1)
    dim_offsets = tl.arange(0, HEAD_DIM)
    k_addrs = (
        physical_blocks[:, None] * cache_stride_block
        + page_offsets[:, None] * cache_stride_offset
        + dim_offsets[None, :] * cache_stride_d
    )
    b_tile = tl.load(indexer_cache_ptr + k_addrs, mask=(in_range & (pool_offsets < req_pool_len))[:, None], other=0.0).to(tl.float32)  # [BN, D]
    b_tile_t = tl.trans(b_tile)  # [D, BN]
    b_tile_t = tl.where(in_range[None, :], b_tile_t, 0.0)

    req_pool_len = tl.load(indexer_seq_lens_ptr + req).to(tl.int32)

    for m0 in tl.range(lo, hi, BM):
        offs_m = m0 + tl.arange(0, BM)
        m_m = offs_m < hi
        a = tl.load(
            qbar_ptr + offs_m[:, None] * HEAD_DIM + dim_offsets[None, :],
            mask=m_m[:, None],
            other=0.0,
        )  # [BM, D]
        c = tl.dot(a, b_tile_t, input_precision="ieee")  # [BM, BN] fp32
        pos = tl.load(positions_ptr + token_start + offs_m, mask=m_m, other=0).to(tl.int32)
        vis = tl.minimum((pos + 1) // INDEX_KPOOL, req_pool_len)
        valid = (pool_offsets[None, :] < vis[:, None]) & ((token_start + offs_m)[:, None] < r_hi)
        if OPTIMIZED_INF_FILL:
            c = tl.where(valid & (c == c), c, -3.4028234663852886e38)
        else:
            c = tl.where(valid, c, float("-inf"))
        tl.store(
            scores_ptr + offs_m[:, None] * max_pool_seq_len + pool_offsets[None, :],
            c,
            mask=m_m[:, None] & in_range[None, :],
        )




@triton.jit(do_not_specialize=["num_tokens"])
def _glm5_next_qbar_kernel(
    q_ptr,
    w_ptr,
    qbar_ptr,
    num_tokens,
    stride_qt,
    stride_qh,
    stride_wt,
    H: tl.constexpr,
    D: tl.constexpr,
):
    i_t = tl.program_id(0)
    if i_t >= num_tokens:
        return
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    b_q = tl.load(q_ptr + i_t * stride_qt + h[:, None] * stride_qh + d[None, :]).to(tl.float32)
    b_w = tl.load(w_ptr + i_t * stride_wt + h).to(tl.float32)
    b_qbar = tl.sum(b_q * b_w[:, None], axis=0)
    tl.store(qbar_ptr + i_t * D + d, b_qbar)


def _qbar_triton(query: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    num_tokens, H, D = query.shape
    qbar = torch.empty((num_tokens, D), dtype=torch.float32, device=query.device)
    _glm5_next_qbar_kernel[(num_tokens,)](
        query,
        weights,
        qbar,
        num_tokens,
        query.stride(0),
        query.stride(1),
        weights.stride(0),
        H=triton.next_power_of_2(H),
        D=triton.next_power_of_2(D),
        num_warps=4,
    )
    return qbar




@triton.jit(do_not_specialize=["token_start", "topk", "num_reqs"])
def _glm5_next_indexer_epilogue_kernel(
    vals_ptr, ids_ptr, positions_ptr, ends_ptr, out_ptr,
    token_start, topk, num_reqs,
    POOL_TOPK: tl.constexpr, POOL_POW2: tl.constexpr,
    POOL_OUT: tl.constexpr,
):
    t = tl.program_id(0)
    row = token_start + t
    pos = tl.load(positions_ptr + row).to(tl.int32)
    actual_end = tl.load(ends_ptr + num_reqs - 1)
    row_valid = row < actual_end
    jj = tl.arange(0, POOL_POW2)
    vals = tl.load(vals_ptr + t * topk + jj, mask=jj < topk, other=-3.4028234663852886e38)
    p = tl.load(ids_ptr + t * topk + jj, mask=jj < topk, other=-1).to(tl.int32)
    p = tl.where((jj < topk) & (vals > -3.4028234663852886e38) & row_valid, p, -1)
    tail_pool = (pos + 1) // 4
    tail_count = (pos + 1) % 4
    if POOL_OUT:
        base = row * (POOL_TOPK + 1)
        tail_col = tl.minimum(tail_pool, POOL_TOPK)
        # History and tail form one contiguous valid prefix for CANN SFA.
        p = tl.where(jj == tail_col, tl.where((tail_count > 0) & row_valid, tail_pool, -1), p)
        tl.store(out_ptr + base + jj, p, mask=jj < POOL_TOPK)
        tl.store(out_ptr + base + POOL_TOPK,
                 tl.where((tail_col == POOL_TOPK) & (tail_count > 0) & row_valid, tail_pool, -1))
    else:
        hist: tl.constexpr = POOL_TOPK * 4
        base = row * (hist + 3)
        x = p * 4
        expanded = tl.interleave(
            tl.interleave(tl.where(p >= 0, x, -1), tl.where(p >= 0, x + 2, -1)),
            tl.interleave(tl.where(p >= 0, x + 1, -1), tl.where(p >= 0, x + 3, -1)))
        cols = tl.arange(0, POOL_POW2 * 4)
        tail_start = tail_pool * 4
        tail_col = tl.minimum(tail_start, hist)
        tail_offset = cols - tail_col
        expanded = tl.where((tail_offset >= 0) & (tail_offset < 3),
                            tl.where((tail_offset < tail_count) & row_valid, tail_start + tail_offset, -1), expanded)
        tl.store(out_ptr + base + cols, expanded, mask=cols < hist)
        ti = tl.arange(0, 4)
        tl.store(out_ptr + base + hist + ti,
                 tl.where((tail_col == hist) & (ti < tail_count) & row_valid, tail_start + ti, -1), mask=ti < 3)


def _optimized_indexer(query, indexer_cache, weights, cum_query_lens,
                       indexer_seq_lens, indexer_block_table, positions, *,
                       index_topk, index_kpool, max_pool_seq_len):
    pool_out = indexer_pool_id_enabled(index_kpool)
    epilogue = bool(envs.VLLM_ASCEND_ENABLE_TRITON_INDEXER_EPILOGUE) and index_kpool == 4
    pool_topk = index_topk // index_kpool
    width = pool_topk + 1 if pool_out else index_topk + index_kpool - 1
    tokens = query.shape[0]
    output = torch.empty((tokens, 1, width), device=query.device, dtype=torch.int32)
    if tokens == 0:
        return output
    # No pools is a common first draft step; preserve the original early path.
    if max_pool_seq_len == 0:
        if not pool_out and not epilogue:
            return _legacy_indexer(query, indexer_cache, weights, cum_query_lens,
                indexer_seq_lens, indexer_block_table, positions,
                index_topk=index_topk, index_kpool=index_kpool, max_pool_seq_len=0)
        topk = 0
        vals = torch.empty((tokens, 0), dtype=torch.float32, device=query.device)
        ids = torch.empty((tokens, 0), dtype=torch.int64, device=query.device)
        if epilogue:
            _glm5_next_indexer_epilogue_kernel[(tokens,)](
                vals, ids, positions, cum_query_lens, output, 0, 0, cum_query_lens.shape[0],
                POOL_TOPK=pool_topk, POOL_POW2=next_power_of_2(pool_topk), POOL_OUT=pool_out)
        else:
            output.fill_(-1)
            tail = (positions + 1) // index_kpool
            output[:, 0, 0] = torch.where((positions + 1) % index_kpool != 0, tail, -1).to(torch.int32)
        return output
    num_reqs = cum_query_lens.shape[0]
    topk = min(pool_topk, max_pool_seq_len)
    token_chunk = max(1, TRITON_SCORES_CHUNK_BYTES // (max_pool_seq_len * 4))
    optimized_inf_fill = bool(envs.VLLM_ASCEND_ENABLE_INDEXER_INF_FILL)
    for token_start in range(0, tokens, token_chunk):
        token_end = min(token_start + token_chunk, tokens)
        rows = token_end - token_start
        if (envs.VLLM_ASCEND_ENABLE_TRITON_INDEXER_QBAR
                and query.shape[1] == next_power_of_2(query.shape[1])
                and query.shape[2] == next_power_of_2(query.shape[2])
                and query.stride(2) == weights.stride(1) == 1):
            qbar = _qbar_triton(query[token_start:token_end], weights[token_start:token_end])
        else:
            qbar = (query[token_start:token_end].float() * weights[token_start:token_end].float().unsqueeze(-1)).sum(1).contiguous()
        scores = (torch.empty((rows, max_pool_seq_len), device=query.device, dtype=torch.float32)
                  if optimized_inf_fill else torch.full((rows, max_pool_seq_len), float('-inf'), device=query.device, dtype=torch.float32))
        use_gemm = bool(envs.VLLM_ASCEND_ENABLE_INDEXER_GEMM_SCORE) and rows // max(1, num_reqs) >= 16
        common = (qbar, indexer_cache, cum_query_lens, indexer_seq_lens,
                  indexer_block_table, positions, scores, token_start,
                  max_pool_seq_len, num_reqs, indexer_cache.shape[0])
        strides = (indexer_cache.stride(0), indexer_cache.stride(1), indexer_cache.stride(3),
                   indexer_block_table.stride(0), indexer_block_table.stride(1),
                   indexer_cache.shape[1], next_power_of_2(max(1, num_reqs)), query.shape[2], index_kpool)
        if use_gemm:
            _glm5_indexer_gemm_score_kernel[(num_reqs, triton.cdiv(max_pool_seq_len, 64))](
                *common, rows, *strides, 64, 64, optimized_inf_fill, num_warps=4)
        else:
            _glm5_next_lightning_indexer_score_kernel[(rows, triton.cdiv(max_pool_seq_len, TRITON_POOL_CHUNK_SIZE))](
                *common, *strides, TRITON_POOL_CHUNK_SIZE, TRITON_POOL_SUB_TILE_SIZE, optimized_inf_fill)
        if not optimized_inf_fill:
            scores.nan_to_num_(nan=NEG_INF_SENTINEL, neginf=NEG_INF_SENTINEL)
        vals, ids = torch.topk(scores, topk, dim=1)
        if epilogue:
            _glm5_next_indexer_epilogue_kernel[(rows,)](
                vals, ids, positions, cum_query_lens, output, token_start, topk, num_reqs,
                POOL_TOPK=pool_topk, POOL_POW2=next_power_of_2(pool_topk), POOL_OUT=pool_out)
            continue
        ids = torch.where(vals > NEG_INF_SENTINEL, ids, -1)
        pos = positions[token_start:token_end]
        tail_pool = (pos + 1) // index_kpool
        if pool_out:
            pools = torch.nn.functional.pad(ids.to(torch.int32), (0, pool_topk + 1 - topk), value=-1)
            tail_col = tail_pool.clamp(max=pool_topk).long().unsqueeze(1)
            tail_val = torch.where((pos + 1) % index_kpool != 0, tail_pool, -1).to(torch.int32).unsqueeze(1)
            pools.scatter_(1, tail_col, tail_val)
            output[token_start:token_end, 0] = pools
        else:
            offsets = torch.arange(index_kpool, device=query.device)
            history = torch.where(ids[:, :, None] >= 0, ids[:, :, None] * index_kpool + offsets, -1).reshape(rows, -1)
            history = torch.nn.functional.pad(history, (0, (pool_topk - topk) * index_kpool), value=-1)
            offsets = torch.arange(index_kpool - 1, device=query.device)
            start = tail_pool * index_kpool
            tail = torch.where(offsets[None, :] < (pos + 1 - start)[:, None], start[:, None] + offsets, -1)
            output[token_start:token_end, 0] = torch.cat((history, tail), 1).to(torch.int32)
    return output


def glm5_next_lightning_indexer_triton(query, indexer_cache, weights, cum_query_lens,
        indexer_seq_lens, indexer_block_table, positions, *, index_topk, index_kpool,
        max_pool_seq_len):
    optimized = (envs.VLLM_ASCEND_ENABLE_TRITON_INDEXER_QBAR
                 or envs.VLLM_ASCEND_ENABLE_TRITON_INDEXER_EPILOGUE
                 or envs.VLLM_ASCEND_ENABLE_INDEXER_GEMM_SCORE
                 or indexer_pool_id_enabled(index_kpool))
    fn = _optimized_indexer if optimized else _legacy_indexer
    return fn(query, indexer_cache, weights, cum_query_lens, indexer_seq_lens,
              indexer_block_table, positions, index_topk=index_topk,
              index_kpool=index_kpool, max_pool_seq_len=max_pool_seq_len)
