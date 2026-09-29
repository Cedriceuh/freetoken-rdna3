"""Quantizing K/V store for the int8 QSA pool (kvcache/qsa_pool.py, FREETOKEN_QSA_KV_INT8=1).

One program per (token, kv head): symmetric int8 over the head_dim values with one fp32 scale
(amax / 127), round half away from zero. The attend kernel multiplies the int8 codes back by the
scale (kernel/triton/qsa/attend.py, KV_INT8).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_row(src_ptr, dst_ptr, scale_ptr, dim_offsets):
    x = tl.load(src_ptr + dim_offsets).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    y = x * inv
    y = tl.where(y >= 0.0, y + 0.5, y - 0.5)
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    tl.store(dst_ptr + dim_offsets, y.to(tl.int8))
    tl.store(scale_ptr, amax / 127.0)


@triton.jit
def _store_kv_int8_kernel(
    k_ptr, v_ptr, loc_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
    stride_k_row, stride_v_row, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    loc = tl.load(loc_ptr + row).to(tl.int64)
    dim_offsets = tl.arange(0, HEAD_DIM)
    cache_row = (loc * NUM_HEADS + head) * HEAD_DIM
    scale_row = loc * NUM_HEADS + head
    _quantize_row(k_ptr + row * stride_k_row + head * HEAD_DIM, k_cache_ptr + cache_row, k_scale_ptr + scale_row,
                  dim_offsets)
    _quantize_row(v_ptr + row * stride_v_row + head * HEAD_DIM, v_cache_ptr + cache_row, v_scale_ptr + scale_row,
                  dim_offsets)


def store_kv_int8(
    k: torch.Tensor,  # [T, heads * head_dim] (or [T, heads, head_dim]) compute dtype
    v: torch.Tensor,
    loc: torch.Tensor,  # [T] int, flat token slot
    k_cache: torch.Tensor,  # [slots, heads, head_dim] int8, contiguous
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,  # [slots, heads] fp32, contiguous
    v_scale: torch.Tensor,
) -> None:
    rows = loc.shape[0]
    if not rows:
        return
    _, heads, head_dim = k_cache.shape
    k = k.reshape(rows, heads * head_dim)
    v = v.reshape(rows, heads * head_dim)
    assert k.stride(1) == v.stride(1) == 1 and k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_scale.is_contiguous() and v_scale.is_contiguous() and k_scale.shape == (k_cache.shape[0], heads)
    _store_kv_int8_kernel[(rows, heads)](
        k, v, loc, k_cache, v_cache, k_scale, v_scale, k.stride(0), v.stride(0),
        NUM_HEADS=heads, HEAD_DIM=head_dim, num_warps=4,
    )


__all__ = ["store_kv_int8"]
