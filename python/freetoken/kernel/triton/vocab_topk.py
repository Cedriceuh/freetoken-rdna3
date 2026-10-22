"""The k largest logits of a vocabulary split over TP ranks, without gathering the logits (FREETOKEN_SPARSE_LOGITS).

Each rank takes the top k of its own vocabulary shard (two kernels up to k = 64: each 4096-wide block's k largest,
then the row's), ``pack_candidates`` writes them with their global ids into one fp32 row (ids < 2**24 are exact in
fp32), the ranks all-gather those rows, and
``merge_candidates`` keeps the tp * k candidates' k largest, equal values by ascending id: the set ``torch.topk`` of
the whole row selects (a global top-k element is in its own rank's top k; at the k-th value topk keeps the lowest
indices). Its first id is ``torch.argmax`` of the whole row (the lowest id among equal maxima). The set then goes back
into the order ``torch.topk`` returns, which on this build depends on the row count of its call: over one row, equal
values come by ascending index (the merge's own order); over several, its gather lays the values above the k-th out
in ascending index order, then those equal to the k-th, and sorts that layout (the same ``torch.sort`` call gives the
same order, equal values included). Both checked against topk over rows with many equal bf16 logits, 1 to 16 rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(v_ptr, i_ptr, out_ptr, start, stride_v, stride_i, K: tl.constexpr, K_PAD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, K_PAD)
    mask = offs < K
    idx = tl.load(i_ptr + row * stride_i + offs, mask=mask, other=0)
    v = tl.load(v_ptr + row * stride_v + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * (2 * K) + offs, v.to(tl.float32), mask=mask)
    tl.store(out_ptr + row * (2 * K) + K + offs, (idx + start).to(tl.float32), mask=mask)


@triton.jit
def _key(v, idx):
    """A descending sort key: larger value first (-0.0 as 0.0), then the smaller index."""
    b = (v + 0.0).to(tl.int32, bitcast=True)
    ordered = tl.where(b >= 0, b, b ^ 0x7FFFFFFF).to(tl.int64)
    return (ordered << 32) | (0xFFFFFFFF - idx.to(tl.int64))


@triton.jit
def _block_topk_kernel(x_ptr, keys_ptr, stride_x, W, NB, K: tl.constexpr, K_PAD: tl.constexpr,
                       BLOCK: tl.constexpr):
    """Each block of a row: its K largest (value, index) keys, the rest of the K_PAD padded with the smallest key."""
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < W
    v = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    key = tl.where(m, _key(v, offs), -9223372036854775807 - 1)
    o = tl.arange(0, K_PAD)
    out = tl.full((K_PAD,), -9223372036854775807 - 1, tl.int64)
    for i in tl.static_range(K):
        best = tl.max(key, axis=0)
        out = tl.where(o == i, best, out)
        key = tl.where(key == best, -9223372036854775807 - 1, key)
    tl.store(keys_ptr + (row * NB + blk) * K_PAD + o, out)


@triton.jit
def _rows_topk_kernel(keys_ptr, out_ptr, start, NB, K: tl.constexpr, K_PAD: tl.constexpr, C_PAD: tl.constexpr):
    """A row's K largest keys over its blocks, packed as ``[K values fp32 | K global ids as fp32]``."""
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C_PAD)
    key = tl.load(keys_ptr + row * NB * K_PAD + c, mask=c < NB * K_PAD, other=-9223372036854775807 - 1)
    o = tl.arange(0, K_PAD)
    vals = tl.zeros((K_PAD,), tl.float32)
    ids = tl.zeros((K_PAD,), tl.int64)
    for i in tl.static_range(K):
        best = tl.max(key, axis=0)
        ordv = (best >> 32).to(tl.int32)
        vals = tl.where(o == i, tl.where(ordv >= 0, ordv, ordv ^ 0x7FFFFFFF).to(tl.float32, bitcast=True), vals)
        ids = tl.where(o == i, 0xFFFFFFFF - (best & 0xFFFFFFFF), ids)
        key = tl.where(key == best, -9223372036854775807 - 1, key)
    tl.store(out_ptr + row * (2 * K) + o, vals, mask=o < K)
    tl.store(out_ptr + row * (2 * K) + K + o, (ids + start).to(tl.float32), mask=o < K)


# rows of at most this many candidates take the two kernels above instead of torch.topk: each pass is a max per
# candidate, so a wide k is cheaper through topk's radix select
_KERNEL_TOPK_MAX = 64


def pack_candidates(local: torch.Tensor, k: int, start: int, width: int) -> torch.Tensor:
    """``[N, 2k]`` fp32: this rank's k largest logits of ``local[:, :width]`` then their global ids (``start`` + local).
    Up to ``_KERNEL_TOPK_MAX``: two kernels, the k largest by (value, then the smaller index), the set torch.topk keeps
    (at the k-th value its gather takes the lowest indices) and the argmax first; the merge orders them."""
    assert width >= k, f"a vocabulary shard of {width} columns has no {k} candidates"
    x = local[:, :width]
    out = torch.empty((local.shape[0], 2 * k), dtype=torch.float32, device=local.device)
    if k <= _KERNEL_TOPK_MAX and x.stride(-1) == 1:
        block, k_pad = 4096, max(2, triton.next_power_of_2(k))
        nb = triton.cdiv(width, block)
        keys = torch.empty((local.shape[0], nb, k_pad), dtype=torch.int64, device=local.device)
        _block_topk_kernel[(local.shape[0], nb)](x, keys, x.stride(0), width, nb, K=k, K_PAD=k_pad, BLOCK=block,
                                                 num_warps=8)
        _rows_topk_kernel[(local.shape[0],)](keys, out, start, nb, K=k, K_PAD=k_pad,
                                             C_PAD=triton.next_power_of_2(nb * k_pad), num_warps=4)
        return out
    if k == 1:
        idx = torch.argmax(x, dim=-1, keepdim=True)
        vals = x.gather(-1, idx)
    else:
        vals, idx = torch.topk(x, k, dim=-1)
    _pack_kernel[(local.shape[0],)](
        vals, idx, out, start, vals.stride(0), idx.stride(0), K=k, K_PAD=max(2, triton.next_power_of_2(k)),
        num_warps=1,
    )
    return out


@triton.jit
def _merge_kernel(g_ptr, v_out, i_out, amax_out, N, TP: tl.constexpr, K: tl.constexpr, C_PAD: tl.constexpr,
                  STABLE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C_PAD)
    r = c // K
    j = c % K
    valid = c < TP * K
    base = g_ptr + (r.to(tl.int64) * N + row) * (2 * K) + j
    v = tl.load(base, mask=valid, other=0.0)
    ids = tl.load(base + K, mask=valid, other=0.0).to(tl.int64)
    key = tl.where(valid, _key(v, ids), -9223372036854775807 - 1)
    key = tl.sort(key, descending=True)
    top_ord = (key >> 32).to(tl.int32)
    top_id = 0xFFFFFFFF - (key & 0xFFFFFFFF)
    o = tl.arange(0, C_PAD)
    tl.store(amax_out + row + o, top_id, mask=o == 0)
    if STABLE:  # topk of a single row: equal values by ascending id, the order of this sort
        top_bits = tl.where(top_ord >= 0, top_ord, top_ord ^ 0x7FFFFFFF)
        tl.store(v_out + row * K + o, top_bits.to(tl.float32, bitcast=True), mask=o < K)
        tl.store(i_out + row * K + o, top_id, mask=o < K)
        return
    # torch.topk's layout before its sort: the values above the k-th in ascending id order, then the k-th's
    kth = tl.min(tl.where(o < K, top_ord, 2147483647), 0)
    # (above-k-th flag, id < 2**24) order the layout; the value rides in the low 32 bits (ids are distinct)
    lay = (tl.where(top_ord > kth, 0, 1).to(tl.int64) << 56) | (top_id << 32) | (top_ord.to(tl.int64) & 0xFFFFFFFF)
    lay = tl.sort(tl.where(o < K, lay, 9223372036854775807))
    lay_id = (lay >> 32) & 0xFFFFFF
    lay_ord = (lay & 0xFFFFFFFF).to(tl.int32)
    lay_bits = tl.where(lay_ord >= 0, lay_ord, lay_ord ^ 0x7FFFFFFF)
    keep = o < K
    tl.store(v_out + row * K + o, lay_bits.to(tl.float32, bitcast=True), mask=keep)
    tl.store(i_out + row * K + o, lay_id, mask=keep)


def merge_candidates(gathered: torch.Tensor, tp: int, k: int, rows: int = 2) -> tuple[torch.Tensor, torch.Tensor,
                                                                                    torch.Tensor]:
    """``gathered [tp * N, 2k]`` (every rank's ``pack_candidates``, rank-major) -> ``(vals [N, k] fp32, ids [N, k]
    int64, argmax [N] int64)``: the whole rows' ``torch.topk(k)`` (sorted, equal values in topk's order) and
    ``torch.argmax``. ``rows``: how many rows the topk call matched has, which sets its order of equal values (one
    row: ascending ids; more: its bitonic sort's)."""
    n = gathered.shape[0] // tp
    vals = torch.empty((n, k), dtype=torch.float32, device=gathered.device)
    ids = torch.empty((n, k), dtype=torch.int64, device=gathered.device)
    amax = torch.empty((n,), dtype=torch.int64, device=gathered.device)
    c_pad = max(2, triton.next_power_of_2(tp * k))
    stable = rows == 1
    _merge_kernel[(n,)](gathered, vals, ids, amax, n, TP=tp, K=k, C_PAD=c_pad, STABLE=stable,
                        num_warps=1 if c_pad <= 64 else 4)
    if k > 1 and not stable:
        vals, perm = torch.sort(vals, dim=-1, descending=True)
        ids = ids.gather(-1, perm)
    return vals, ids, amax


__all__ = ["merge_candidates", "pack_candidates"]
