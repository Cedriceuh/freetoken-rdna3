"""The attention layer's per-head q and k RMSNorms and its 3-axis rope in one launch (FREETOKEN_QK_NORM_ROPE): one
program per (row, head), the norm as ``_rmsnorm_kernel`` computes it (the same code, tile and warps, so the same sum),
stored, then the rope of the head's first ``rotary_dim`` dims as ``_mrope_tiled`` computes it, in place. Three launches
become one; the bits do not change (tests/kernels/test_qk_norm_rope.py)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.norm import _num_warps


@triton.jit
def _qk_norm_rope_kernel(Q, K, QO, KO, WQ, WK, POS, CACHE, SEC, sqa, sqb, ska, skb, soqa, soqb, soka, sokb, nnz,
                         HEAD_Q, rotary_dim, half, eps: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
                         BLOCK_DHALF: tl.constexpr, GEMMA: tl.constexpr):
    a = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    if b < HEAD_Q:
        x_ptr = Q + a.to(tl.int64) * sqa + b * sqb
        o_ptr = QO + a.to(tl.int64) * soqa + b * soqb
        w_ptr = WQ
    else:
        x_ptr = K + a.to(tl.int64) * ska + (b - HEAD_Q) * skb
        o_ptr = KO + a.to(tl.int64) * soka + (b - HEAD_Q) * sokb
        w_ptr = WK
    # _rmsnorm_kernel's strided path
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if GEMMA:
        w = w + 1.0
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    y = x * inv * w
    tl.store(o_ptr + cols, y.to(QO.dtype.element_ty), mask=mask)
    tl.debug_barrier()  # the rope reads the stored row, halves crossing lanes
    # _mrope_tiled on this row and head
    d = tl.arange(0, BLOCK_DHALF)
    dmask = d < half
    sec = tl.load(SEC + d, mask=dmask, other=0).to(tl.int64)
    pos = tl.load(POS + sec * nnz + a, mask=dmask, other=0).to(tl.int64)
    cache_row = CACHE + pos * rotary_dim
    cos = tl.load(cache_row + d, mask=dmask, other=0.0)
    sin = tl.load(cache_row + half + d, mask=dmask, other=0.0)
    x0 = tl.load(o_ptr + d, mask=dmask, other=0.0).to(tl.float32)
    x1 = tl.load(o_ptr + half + d, mask=dmask, other=0.0).to(tl.float32)
    tl.store(o_ptr + d, (x0 * cos - x1 * sin).to(QO.dtype.element_ty), mask=dmask)
    tl.store(o_ptr + half + d, (x1 * cos + x0 * sin).to(QO.dtype.element_ty), mask=dmask)


def qk_norm_rope(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, eps: float, gemma: bool,
                 positions: torch.Tensor, cos_sin_cache: torch.Tensor, section_table: torch.Tensor):
    """``q`` [T, heads_q, H], ``k`` [T, heads_k, H] (any leading strides, unit last) -> new contiguous normalized and
    rotated ``(q, k)``: ``rmsnorm`` of each head (``gemma``: (1 + w)) then the 3-axis rope of positions [3, T]; None
    where the two norms would run at different warps (prefill row counts)."""
    T, hq, H = q.shape
    hk = k.shape[1]
    # the norms' warps: what _rmsnorm picks for each of the two calls this replaces
    warps = _num_warps(T * hq)
    if T == 0 or warps != _num_warps(T * hk):
        return None
    positions = positions.contiguous()  # as MRotaryEmbedding.forward
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and cos_sin_cache.is_contiguous()
    qo = torch.empty((T, hq, H), dtype=q.dtype, device=q.device)
    ko = torch.empty((T, hk, H), dtype=k.dtype, device=k.device)
    rotary_dim = cos_sin_cache.shape[1]
    half = rotary_dim // 2
    _qk_norm_rope_kernel[(T, hq + hk)](
        q, k, qo, ko, wq, wk, positions, cos_sin_cache, section_table, q.stride(0), q.stride(1), k.stride(0),
        k.stride(1), qo.stride(0), qo.stride(1), ko.stride(0), ko.stride(1), T, hq, rotary_dim, half, eps=eps, H=H,
        BLOCK=triton.next_power_of_2(H), BLOCK_DHALF=triton.next_power_of_2(half), GEMMA=gemma, num_warps=warps,
        num_stages=1)
    return qo, ko


__all__ = ["qk_norm_rope"]
