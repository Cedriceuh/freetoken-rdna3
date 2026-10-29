# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""QSA block scoring over the paged compressed-key slab."""

from __future__ import annotations

import math

import os

import torch
import triton
import triton.language as tl

# FREETOKEN_QSA_SCORE_ROWS (default 1): a prefill of one request scores 16 query rows a program
# (_qsa_mqa_paged_rows_kernel), the same bits; 3.9-4.3x faster at a 16k-token chunk 150k deep. 0: one row a program
_SCORE_ROWS = os.environ.get("FREETOKEN_QSA_SCORE_ROWS", "1") == "1"
_ROWS_MIN = 64  # decode and short rows keep the one-row kernel
# FREETOKEN_QSA_SCORE_ROWS_STAGES / _TILES (default 1 / 16 on ROCm, 2 / 8 elsewhere): that kernel's pipeline stages and
# 64-column tiles a program. One stage keeps two programs' K tiles in the LDS one held, and 16 tiles load the 16 rows'
# queries once for twice the columns; the same K order and head sums: the same bits. 7900 XTX / XT, 30k-200k deep,
# synthetic queries: a 128 MiB block of scores 26-29 % / 19-22 % faster (in serving, a 210k-token read 0.6 % shorter)
_ROCM = torch.version.hip is not None
_ROWS_STAGES = int(os.environ.get("FREETOKEN_QSA_SCORE_ROWS_STAGES", "1" if _ROCM else "2"))
_ROWS_TILES = int(os.environ.get("FREETOKEN_QSA_SCORE_ROWS_TILES", "16" if _ROCM else "8"))


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    MAX_N: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, MAX_N)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    # Top-k is bounded by visible_blocks, so columns beyond it need no value.
    if tile_start * BLOCK_N >= visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    # Pad the small head axis to a tensor-core-compatible N dimension.
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(heads[None, :] < NUM_HEADS) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(heads[None, :] < NUM_HEADS, tl.maximum(scores, 0.0), 0.0)
        score = tl.sum(scores, axis=1) / score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(page_valid, score, -float("inf")),
            mask=live & (columns < num_columns),
        )


@triton.jit
def _qsa_mqa_paged_rows_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    # _qsa_mqa_paged_kernel for ROWS query rows of one request at a time: each tile of keys is loaded once for the
    # ROWS rows and multiplied with their NUM_HEADS heads together, without padding the heads. A score keeps its bits:
    # the same K order in the dot, and the heads summed as the padded reduction sums them, (h0 + h2) + (h1 + h3)
    r0 = tl.program_id(0) * ROWS
    rows = r0 + tl.arange(0, ROWS)
    row_ok = rows < num_rows
    request = tl.load(token_to_req_ptr + r0)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + rows, mask=row_ok, other=0)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum((query_position + 1) // COMPRESS_RATIO, sequence_length // COMPRESS_RATIO)
    visible = tl.where(row_ok, visible, 0)
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + rows, visible, mask=row_ok)
    most = tl.max(visible, axis=0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    if tile_start * BLOCK_N >= most:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(most, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    row_head = tl.arange(0, ROWS * NUM_HEADS)  # (row, head), row-major
    q_rows = r0 + row_head // NUM_HEADS
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + (row_head % NUM_HEADS)[None, :] * stride_q_head
        + dims[:, None] * stride_q_dim,
        mask=(q_rows[None, :] < num_rows) & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < most
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + logical_page * stride_table_page,
            mask=live,
            other=-1,
        )
        page_valid = live & (physical_page >= 0) & (physical_page < num_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=page_valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.maximum(tl.dot(keys, query, out_dtype=tl.float32), 0.0)
        even, odd = tl.split(tl.reshape(scores, (BLOCK_N, ROWS, 2, 2)))  # heads (0, 2) and (1, 3)
        h0, h2 = tl.split(even)
        h1, h3 = tl.split(odd)
        score = ((h0 + h2) + (h1 + h3)) / score_divisor
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            tl.where(page_valid[:, None], score, -float("inf")),
            mask=(columns[:, None] < visible[None, :]) & (columns[:, None] < num_columns),
        )


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    score_scale: float | None = None,
    one_request: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache. ``one_request``: every row belongs to one
    request (its page table), which lets FREETOKEN_QSA_SCORE_ROWS score several rows a program."""

    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if token_to_req.shape != (q.shape[0],) or query_positions.shape != (q.shape[0],):
        raise ValueError("QSA request mapping and positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    columns = logits.shape[1]
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    BLOCK_N = 64
    BLOCK_D = max(16, triton.next_power_of_2(q.shape[2]))
    MAX_N = max(16, triton.next_power_of_2(q.shape[1]))
    if one_request and _SCORE_ROWS and q.shape[1] == 4 and q.shape[0] >= _ROWS_MIN:
        _qsa_mqa_paged_rows_kernel[(triton.cdiv(q.shape[0], 16), triton.cdiv(columns, BLOCK_N * _ROWS_TILES))](
            q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, visible_blocks, logits,
            q.stride(0), q.stride(1), q.stride(2), k_cache.stride(0), k_cache.stride(1), k_cache.stride(3),
            page_table.stride(0), page_table.stride(1), logits.stride(0),
            q.shape[0], columns, k_cache.shape[0], page_table.shape[0], float(score_divisor),
            PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=page_table.shape[1], NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2],
            BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, ROWS=16, TILES_PER_PROG=_ROWS_TILES, STAGES=_ROWS_STAGES,
            COMPRESS_RATIO=compress_ratio,
            num_warps=8,
        )
        return logits, visible_blocks
    # Tuned on GB300: larger row batches provide enough parallelism to reuse Q.
    tiles_per_program = 1 if q.shape[0] <= 32 else 8
    _qsa_mqa_paged_kernel[
        (q.shape[0], triton.cdiv(columns, BLOCK_N * tiles_per_program))
    ](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        MAX_N=MAX_N,
        COMPRESS_RATIO=compress_ratio,
        num_warps=2,
    )
    return logits, visible_blocks


__all__ = ["qsa_mqa_paged"]
