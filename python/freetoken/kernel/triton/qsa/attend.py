# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""Sparse paged GQA over the QSA selection."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# FREETOKEN_QSA_MIN_WARPS (default 4 on ROCm, 0 = the profile's own): the prefill profile's 2 warps hold the
# [BLOCK_M, HEAD_DIM] accumulator and the K / V tiles in ~2.2 KB of scratch per lane on RDNA3 (head_dim 256); at 4
# warps none spills: the same bits (each lane's sums keep their order), the kernel ~40 % faster at 1k-16k rows, and no
# per-queue scratch (~60 MiB a queue on a 7900 XT) that the HIP runtime takes outside the torch allocator
_MIN_WARPS = int(os.environ.get("FREETOKEN_QSA_MIN_WARPS", "4")) if torch.version.hip is not None else 0
_ROCM = torch.version.hip is not None
# FREETOKEN_QSA_PREFILL_STAGES / _WARPS / _PREFETCH (default 1 / 4 / 1 on ROCm; elsewhere 2 stages, the profile's warps,
# 0): the one-split prefill profile (more than 512 rows). One pipeline stage, as a second doubles the K / V tiles' LDS
# (2 x 16 KB at head_dim 256, 32-token tiles) and halves the workgroups a CU holds; and each tile looks up the next
# tile's selected rows (the index, then the page) while its own K / V load and compute, instead of waiting on three
# dependent loads in a row. Each lane's loads and sums keep their order: the same bits. On selections dumped from a
# 96k-token read, 7900 XTX and XT: 30-41 % faster from 600 to 16k rows than 2 stages and 4 warps without the lookahead
_PREFILL_STAGES = int(os.environ.get("FREETOKEN_QSA_PREFILL_STAGES", "1" if _ROCM else "2"))
_PREFILL_WARPS = int(os.environ.get("FREETOKEN_QSA_PREFILL_WARPS", "4" if _ROCM else "0"))
_PREFILL_PREFETCH = os.environ.get("FREETOKEN_QSA_PREFILL_PREFETCH", "1" if _ROCM else "0") == "1"
# FREETOKEN_QSA_DECODE_HOST_STAGES (default 1 on ROCm, 2 elsewhere): pipeline stages of the decode profile when the
# K / V pages are host memory mapped into the GPU (FREETOKEN_QSA_KV_HOST): each lane's loads and sums unchanged, the
# same bits. EXL3 3.05 bpw and NVFP4 on xtx-xt-4x262k, one request: +1.7-2.1 % decode at short context, +3-3.8 % at
# 75-89k
_DECODE_HOST_STAGES = int(os.environ.get("FREETOKEN_QSA_DECODE_HOST_STAGES", "1" if _ROCM else "2"))
# the decode cache never holds a request's last tokens: written this step, or drafts rewritten after a rejection
_DC_MARGIN = 64


@triton.jit
def _tile_rows(indices_ptr, block_table_ptr, row, stride_indices_row, stride_table_req, request, safe_request,
               num_requests, num_cache_blocks, tile, column_offsets, TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr,
               PAGE_TABLE_WIDTH: tl.constexpr, BLOCK_N: tl.constexpr):
    """A tile's selected rows: (physical page, offset in the page, valid) of each of its BLOCK_N columns."""
    columns = tile * BLOCK_N + column_offsets
    logical_token = tl.load(
        indices_ptr + row * stride_indices_row + columns,
        mask=columns < TOPK,
        other=-1,
    )
    safe_token = tl.maximum(logical_token, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token % PAGE_SIZE
    valid = (
        (request >= 0)
        & (request < num_requests)
        & (logical_token >= 0)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request.to(tl.int64) * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    # physical_page * block stride can overflow int32 for large caches.
    return tl.maximum(physical_page, 0).to(tl.int64), page_offset, valid, safe_token


@triton.jit
def _dc_lookup(token, page, valid, dc_use, dc_limit, dc_set, dc_tokens, dc_cap, dc_slot_ptr, dc_tag_ptr):
    """A tile's tokens in the decode cache: (held in their slot, the slot). A slot's tag names the request's token
    and the physical page it was read from: a token whose page the request's table no longer points to (the prefix
    cache handing it the tree's copy) is read across PCIe again."""
    cacheable = valid & dc_use & (token < dc_limit)
    key = dc_set * dc_tokens + token
    slot = tl.load(dc_slot_ptr + key, mask=cacheable, other=-1).to(tl.int64)
    slot_ok = cacheable & (slot >= 0) & (slot < dc_cap)
    safe_slot = tl.minimum(tl.maximum(slot, 0), dc_cap - 1)
    tag = (page << 32) | key
    return slot_ok & (tl.load(dc_tag_ptr + safe_slot, mask=slot_ok, other=-1) == tag), safe_slot


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_s_block,
    stride_s_token,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    KV_INT8: tl.constexpr,
    PREFETCH_ROWS: tl.constexpr = False,
    dc_k_ptr=None,
    dc_v_ptr=None,
    dc_tag_ptr=None,
    dc_slot_ptr=None,
    dc_hand_ptr=None,
    dc_cnt_ptr=None,
    dc_ctl_ptr=None,
    seq_lens_ptr=None,
    ring_ptr=None,
    dc_requests=0,
    dc_tokens=0,
    dc_cap=0,
    num_programs=1,
    DCACHE: tl.constexpr = False,
    DC_MARGIN: tl.constexpr = 64,
) -> None:
    # row * stride can overflow int32 for large row counts.
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )
    if KV_INT8:
        # int8 codes and fp32 -> fp16 convert natively on RDNA3 (bf16 conversions are emulated), so the int8 path
        # runs its two dots in fp16; RMS-normed, roped queries are far inside the fp16 range
        query = query.to(tl.float16)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    if DCACHE:
        # decode cache (FREETOKEN_QSA_DECODE_CACHE): _dc_fill_kernel put the K/V rows of this call's selected tokens
        # in VRAM slots; a token is read there when the slot it maps to is tagged with it, else across PCIe
        ring = tl.load(ring_ptr + safe_request)
        dc_use = (request >= 0) & (request < num_requests) & (ring >= 0) & (ring < dc_requests)
        dc_limit = tl.load(seq_lens_ptr + safe_request) - DC_MARGIN
        dc_set = tl.minimum(tl.maximum(ring, 0), dc_requests - 1).to(tl.int64)

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    if PREFETCH_ROWS:
        next_page, next_offset, next_valid, next_token = _tile_rows(
            indices_ptr, block_table_ptr, row, stride_indices_row, stride_table_req, request, safe_request,
            num_requests, num_cache_blocks, split_tile_start, column_offsets, TOPK, PAGE_SIZE, PAGE_TABLE_WIDTH,
            BLOCK_N,
        )
    for tile in range(split_tile_start, split_tile_end):
        if PREFETCH_ROWS:
            # this tile's rows were looked up during the previous one; look up the next tile's now (the last
            # iteration repeats its own, unused), so a tile waits for its K / V loads only
            safe_page, page_offset, valid, token = next_page, next_offset, next_valid, next_token
            next_page, next_offset, next_valid, next_token = _tile_rows(
                indices_ptr, block_table_ptr, row, stride_indices_row, stride_table_req, request, safe_request,
                num_requests, num_cache_blocks, tl.minimum(tile + 1, split_tile_end - 1), column_offsets, TOPK,
                PAGE_SIZE, PAGE_TABLE_WIDTH, BLOCK_N,
            )
        else:
            safe_page, page_offset, valid, token = _tile_rows(
                indices_ptr, block_table_ptr, row, stride_indices_row, stride_table_req, request, safe_request,
                num_requests, num_cache_blocks, tile, column_offsets, TOPK, PAGE_SIZE, PAGE_TABLE_WIDTH, BLOCK_N,
            )
        k_ptrs = (k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
                  + kv_head * stride_k_head + dim_offsets[:, None])
        v_ptrs = (v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
                  + kv_head * stride_v_head + dim_offsets[None, :])
        if DCACHE:
            hit, safe_slot = _dc_lookup(token, safe_page, valid, dc_use, dc_limit, dc_set, dc_tokens, dc_cap,
                                        dc_slot_ptr, dc_tag_ptr)
            # two independent masked loads: one through pointers selected between the host slab (a storage past
            # 2 GB) and the cache would mix pointer-range hints that AMD's pointer canonicalization cannot merge, and
            # the host load as the cache load's `other` would make it wait for the host (an attention call 15 % slower)
            cached_row = safe_slot * stride_k_token + kv_head * HEAD_DIM
            from_host = valid & (hit == 0)
            keys = tl.where(hit[None, :], tl.load(dc_k_ptr + cached_row[None, :] + dim_offsets[:, None],
                                                  mask=hit[None, :], other=0.0),
                            tl.load(k_ptrs, mask=from_host[None, :], other=0.0))
            values = tl.where(hit[:, None], tl.load(dc_v_ptr + cached_row[:, None] + dim_offsets[None, :],
                                                    mask=hit[:, None], other=0.0),
                              tl.load(v_ptrs, mask=from_host[:, None], other=0.0))
        else:
            keys = tl.load(k_ptrs, mask=valid[None, :], other=0.0)
            values = tl.load(v_ptrs, mask=valid[:, None], other=0.0)
        if KV_INT8:
            # int8 K/V with one fp32 scale per (token, kv head). The codes convert to fp16 exactly
            # (|q| <= 127); the scales are per column of the score tile, so they apply after the QK dot and fold
            # into the probabilities before the PV dot instead of touching every K/V element.
            scale_offsets = safe_page * stride_s_block + page_offset * stride_s_token + kv_head
            k_scale = tl.load(k_scale_ptr + scale_offsets, mask=valid, other=0.0)
            v_scale = tl.load(v_scale_ptr + scale_offsets, mask=valid, other=0.0)
            keys = keys.to(query.dtype)
            values = values.to(query.dtype)
        scores = tl.dot(query, keys)
        if KV_INT8:
            scores *= k_scale[None, :]
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if KV_INT8:
            weights = probabilities * v_scale[None, :]
        else:
            weights = probabilities
        accumulator = tl.dot(
            weights.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if DCACHE:
        # the call's last attention program (the fill before it is done; nothing here reads the stamp or the hand):
        # the next call stamps its tokens anew, the hand moves past the slots this one took
        if tl.atomic_add(dc_ctl_ptr + 1, 1, sem="acq_rel") == num_programs - 1:
            tl.store(dc_ctl_ptr, tl.load(dc_ctl_ptr) + 1)
            tl.store(dc_ctl_ptr + 1, 0)
            tl.store(dc_hand_ptr, (tl.load(dc_hand_ptr) + tl.minimum(tl.load(dc_cnt_ptr), dc_cap)) % dc_cap)
            tl.store(dc_cnt_ptr, 0)
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _dc_fill_kernel(indices_ptr, block_table_ptr, token_to_req_ptr, seq_lens_ptr, ring_ptr, k_cache_ptr, v_cache_ptr,
                    dc_k_ptr, dc_v_ptr, dc_tag_ptr, dc_stamp_ptr, dc_slot_ptr, dc_hand_ptr, dc_cnt_ptr, dc_ctl_ptr,
                    stride_indices_row, stride_table_req, stride_block, num_requests, num_cache_blocks, dc_requests,
                    dc_tokens, dc_cap, TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
                    ROW: tl.constexpr, BLOCK_T: tl.constexpr, DC_MARGIN: tl.constexpr, TRIES: tl.constexpr,
                    CHUNK: tl.constexpr):
    """A block of a decode row's selected tokens, mapped as the attention maps them: each one no slot holds yet,
    once a request (the first of its rows to stamp it this call), gets the next slot of the layer's clock hand (the
    slots are shared by the requests) whose token this call has not stamped, and its K/V row from the host. Nothing
    reads a slot during the fill: a slot taken from a token stamped later in the call only costs that token a read
    across PCIe in the attention (its tag no longer names it)."""
    row = tl.program_id(0).to(tl.int64)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    ring = tl.load(ring_ptr + safe_request)
    use = (request >= 0) & (request < num_requests) & (ring >= 0) & (ring < dc_requests)
    cols = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    token = tl.load(indices_ptr + row * stride_indices_row + cols, mask=cols < TOPK, other=-1)
    safe_token = tl.maximum(token, 0)
    logical_page = safe_token // PAGE_SIZE
    valid = (token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
    page = tl.load(block_table_ptr + safe_request.to(tl.int64) * stride_table_req
                   + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1), mask=valid, other=-1)
    valid &= (page >= 0) & (page < num_cache_blocks)
    cacheable = valid & use & (token < tl.load(seq_lens_ptr + safe_request) - DC_MARGIN)
    stamp = tl.load(dc_ctl_ptr)
    # the request's token: also the index of its stamp and slot entries; the tag adds the physical page
    key = tl.minimum(tl.maximum(ring, 0), dc_requests - 1).to(tl.int64) * dc_tokens + safe_token
    page = tl.maximum(page, 0).to(tl.int64)
    tag = (page << 32) | key
    win = cacheable & (tl.atomic_xchg(dc_stamp_ptr + key, stamp, mask=cacheable) != stamp)
    if tl.sum(win.to(tl.int32), axis=0) > 0:
        slot = tl.load(dc_slot_ptr + key, mask=win, other=-1).to(tl.int64)
        slot_ok = win & (slot >= 0) & (slot < dc_cap)
        held = slot_ok & (tl.load(dc_tag_ptr + tl.minimum(tl.maximum(slot, 0), dc_cap - 1), mask=slot_ok,
                                  other=-1) == tag)
        todo = win & (held == 0)
        dest = tl.full((BLOCK_T,), -1, dtype=tl.int64)
        hand = tl.load(dc_hand_ptr).to(tl.int64)
        for _ in tl.static_range(TRIES):
            n = tl.sum(todo.to(tl.int32), axis=0)
            if n > 0:
                # the call's n-th slot from the hand: distinct slots within a call, never more than the capacity
                taken = (tl.atomic_add(dc_cnt_ptr, n) + tl.cumsum(todo.to(tl.int32), axis=0) - 1).to(tl.int64)
                todo = todo & (taken < dc_cap)
                cand = (hand + taken) % dc_cap
                old = tl.load(dc_tag_ptr + cand, mask=todo, other=-1)
                busy = todo & (old >= 0) & (tl.load(dc_stamp_ptr + (tl.maximum(old, 0) & 0xFFFFFFFF),
                                                    mask=todo & (old >= 0), other=0) == stamp)
                dest = tl.where(todo & (busy == 0), cand, dest)
                todo = todo & busy
        ok = dest >= 0
        if tl.sum(ok.to(tl.int32), axis=0) > 0:
            safe_dest = tl.maximum(dest, 0)
            src = (page * stride_block + (safe_token % PAGE_SIZE) * ROW)[:, None]
            dst = (safe_dest * ROW)[:, None]
            for c in tl.static_range(0, ROW, CHUNK):  # a row in chunks: fewer registers, more waves in flight
                d = c + tl.arange(0, CHUNK)[None, :]
                tl.store(dc_k_ptr + dst + d, tl.load(k_cache_ptr + src + d, mask=ok[:, None], other=0.0),
                         mask=ok[:, None])
                tl.store(dc_v_ptr + dst + d, tl.load(v_cache_ptr + src + d, mask=ok[:, None], other=0.0),
                         mask=ok[:, None])
            tl.store(dc_tag_ptr + safe_dest, tag, mask=ok)
            tl.store(dc_slot_ptr + key, safe_dest.to(tl.int32), mask=ok)


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    gate_ptr=None,
    stride_gate_row=0,
    stride_gate_head=0,
    HAS_GATE: tl.constexpr = False,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None].to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    if HAS_GATE:  # FREETOKEN_QSA_FUSED_GATE: _sigmoid_gate_mul_kernel's ops on the bf16 output, into [rows, heads * dim]
        o = merged.to(tl.bfloat16).to(tl.float32)
        g = tl.load(gate_ptr + row * stride_gate_row + head * stride_gate_head + dim_offsets).to(tl.float32)
        s = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        tl.store(output_ptr + (row * NUM_QUERY_HEADS + head) * HEAD_DIM + dim_offsets, (o * s).to(tl.bfloat16))
    else:
        tl.store(
            output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
            merged,
        )


@dataclass
class DecodeCache:
    """One layer's decode K/V cache (FREETOKEN_QSA_DECODE_CACHE, host K/V only), its slots shared by the requests:
    ``k`` / ``v`` ``[cap, row]`` (a token's K or V row, all K/V heads), ``tag`` int64 ``[cap]`` (the request slot,
    token and physical page a slot holds, ``page << 32 | ring * tokens + token``, -1 none), ``stamp`` / ``slot``
    int32 ``[requests, tokens]`` (the last call that selected a request's token; the slot holding it, checked against
    the tag), ``hand`` / ``cnt`` int32
    ``[1]`` (the next slot to reuse, the slots taken this call), ``ctl`` int32 ``[2]`` (this call's stamp, its
    attention programs done). ``seq_lens`` / ``ring``: each request's K/V length and request slot (table row)."""

    k: torch.Tensor
    v: torch.Tensor
    tag: torch.Tensor
    stamp: torch.Tensor
    slot: torch.Tensor
    hand: torch.Tensor
    cnt: torch.Tensor
    ctl: torch.Tensor
    seq_lens: torch.Tensor
    ring: torch.Tensor

    @property
    def requests(self) -> int:
        return self.stamp.shape[0]


def dc_fill(dcache: DecodeCache, logical_indices: torch.Tensor, block_table: torch.Tensor,
            token_to_req: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
    """Put a decode call's selected K/V rows in ``dcache`` (_dc_fill_kernel), before its attention reads them."""
    topk = logical_indices.shape[1]
    row = k_cache.shape[2] * k_cache.shape[3]
    cap, tokens = dcache.k.shape[0], dcache.slot.shape[-1]
    chunk = min(row, 128)
    assert row % chunk == 0 and dcache.requests * tokens < 1 << 31 and k_cache.shape[0] < 1 << 31
    block_t = 32  # 16-64 tokens a program, 2-8 warps, 32-128-element chunks: within 2 us of each other
    grid = (logical_indices.shape[0], triton.cdiv(topk, block_t))
    _dc_fill_kernel[grid](logical_indices, block_table, token_to_req, dcache.seq_lens, dcache.ring, k_cache, v_cache,
                          dcache.k, dcache.v, dcache.tag, dcache.stamp, dcache.slot, dcache.hand, dcache.cnt,
                          dcache.ctl, logical_indices.stride(0), block_table.stride(0), k_cache.stride(0),
                          block_table.shape[0], k_cache.shape[0], dcache.requests, tokens, cap, TOPK=topk,
                          PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1], ROW=row,
                          BLOCK_T=block_t, DC_MARGIN=_DC_MARGIN, TRIES=4, CHUNK=chunk, num_warps=4)


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    decode: bool = False,
    gate: torch.Tensor | None = None,
    kv_host: bool = False,
    dcache: DecodeCache | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches, or int8 ones with ``[pages, page_size, kv_heads]``
    fp32 scales (``k_scale`` / ``v_scale``, see kvcache/qsa_pool.py FREETOKEN_QSA_KV_INT8).

    ``gate`` (bf16, ``q``'s shape, any row and head strides): return ``sigmoid_gate_mul(out, gate)`` instead, ``[T,
    heads * head_dim]``, the gate applied in the split merge when there is one (the same bits, a launch fewer).

    ``decode``: pick the tile profile as for one query row, whatever the row count, so each decode row gets the
    bits it would get alone (the split count changes the merge order). Up to 4 rows per rank this is the profile
    the row count picks anyway; it matters for spec_decode verify steps of several requests.

    ``kv_host``: ``k_cache`` / ``v_cache`` are host memory mapped into the GPU (``FREETOKEN_QSA_KV_HOST``); a decode
    call then runs ``FREETOKEN_QSA_DECODE_HOST_STAGES`` pipeline stages.

    ``dcache``: the layer's decode K/V cache (:class:`DecodeCache`): this call's selected rows are put there first
    (:func:`dc_fill`), then read there instead of across PCIe."""

    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    kv_int8 = k_scale is not None
    if kv_int8:
        assert k_cache.dtype == v_cache.dtype == torch.int8 and v_scale is not None
        assert k_scale.shape == v_scale.shape == k_cache.shape[:3] and k_scale.dtype == v_scale.dtype == torch.float32
        assert k_scale.stride() == v_scale.stride() and k_scale.stride(2) == 1
    else:
        assert q.dtype == k_cache.dtype == v_cache.dtype
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.dtype == q.dtype and out.stride(2) == 1
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = (1 if decode else q.shape[0]) * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    partial_warps = max(partial_warps, _MIN_WARPS)
    wide = base_programs > 512  # the one-split prefill profile
    if wide and _PREFILL_WARPS > 0:
        partial_warps = _PREFILL_WARPS

    # RDNA (gfx11 / gfx12) has 64 KiB of LDS per workgroup; the profile above was
    # tuned on GB300. The K tile [head_dim, BLOCK_N] and the V tile [BLOCK_N, head_dim] alone
    # cost 2 * head_dim * block_n * itemsize, which at head_dim=256 and block_n=64 is exactly
    # 65536 B -- the whole budget -- and the launch then asks for 65792 and Triton refuses with
    # OutOfResources. Halve the tile until the pair fits with room for the index scratch.
    if torch.version.hip is not None:
        _lds_budget = 60 * 1024
        while block_n > 16 and 2 * head_dim * block_n * q.element_size() > _lds_budget:
            block_n //= 2

    # These tiles were tuned for the GB300's 228KB LDS; AMD parts have less and it varies by arch
    # (e.g. 64KB/CU on RDNA gfx11/gfx12), so k+v staging can overflow. Query the device's actual
    # shared-memory limit rather than assuming a fixed size, and shrink the KV tile only when it
    # doesn't fit -- so a larger-LDS arch keeps the wide tile. ROCm-gated so the NVIDIA tiles stay
    # byte-identical (a small NV part could report a 48KB default and would otherwise clamp).
    if torch.version.hip is not None:
        _props = torch.cuda.get_device_properties(q.device)
        _lds_budget = getattr(_props, "shared_memory_per_block", 64 * 1024) or 64 * 1024
        _kv_bytes = k_cache.element_size()
        while block_n > 16 and 2 * block_n * q.shape[2] * _kv_bytes + 2048 > _lds_budget:
            block_n //= 2

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    dc = {}
    if dcache is not None:
        cap, tokens = dcache.k.shape[0], dcache.slot.shape[-1]
        assert decode and not kv_int8 and dcache.k.dtype == k_cache.dtype
        assert dcache.k.shape[-1] == k_cache.shape[2] * head_dim and dcache.tag.dtype == torch.int64
        assert k_cache.is_contiguous() and v_cache.is_contiguous()
        assert tokens >= block_table.shape[1] * k_cache.shape[1]
        dc = dict(dc_k_ptr=dcache.k, dc_v_ptr=dcache.v, dc_tag_ptr=dcache.tag, dc_slot_ptr=dcache.slot,
                  dc_hand_ptr=dcache.hand, dc_cnt_ptr=dcache.cnt, dc_ctl_ptr=dcache.ctl,
                  seq_lens_ptr=dcache.seq_lens, ring_ptr=dcache.ring, dc_requests=dcache.requests, dc_tokens=tokens,
                  dc_cap=cap, num_programs=partial_grid[0] * partial_grid[1] * partial_grid[2], DCACHE=True,
                  DC_MARGIN=_DC_MARGIN)
        dc_fill(dcache, logical_indices, block_table, token_to_req, k_cache, v_cache)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        k_scale if kv_int8 else q,
        v_scale if kv_int8 else q,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        k_scale.stride(0) if kv_int8 else 0,
        k_scale.stride(1) if kv_int8 else 0,
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        KV_INT8=kv_int8,
        PREFETCH_ROWS=wide and _PREFILL_PREFETCH,
        num_warps=partial_warps,
        num_stages=_PREFILL_STAGES if wide else _DECODE_HOST_STAGES if decode and kv_host else 2,
        **dc,
    )
    if gate is not None:
        assert gate.shape == q.shape and gate.dtype == q.dtype == torch.bfloat16 and gate.stride(2) == 1
    if num_splits == 1:
        if gate is not None:
            from freetoken.kernel.triton.attn_gate import sigmoid_gate_mul

            return sigmoid_gate_mul(out, gate)
        return out

    gated = None if gate is None else torch.empty((q.shape[0], q.shape[1] * q.shape[2]), dtype=q.dtype, device=q.device)
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out if gate is None else gated,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        gate_ptr=gate,
        stride_gate_row=0 if gate is None else gate.stride(0),
        stride_gate_head=0 if gate is None else gate.stride(1),
        HAS_GATE=gate is not None,
        num_warps=2,
        num_stages=1,
    )
    return out if gate is None else gated


__all__ = ["DecodeCache", "dc_fill", "qsa_sparse_paged_attention"]
