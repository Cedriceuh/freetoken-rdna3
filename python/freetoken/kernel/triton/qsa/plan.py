"""The QSA layers' per-forward index-write plan (``QSASparseAttnBackend._plan_index_writes``) in two kernels instead of
~30 small torch ops: the same integer arithmetic, and the mrope cos/sin rows copied from the same table rows."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _plan_rope_store_kernel(out_loc, rope_pos, stride_rs, stride_rt, table):
    t = tl.program_id(0)
    row = tl.load(out_loc + t).to(tl.int64)
    for s in tl.static_range(3):
        tl.store(table + row * 3 + s, tl.load(rope_pos + s * stride_rs + t * stride_rt).to(tl.int32))


@triton.jit
def _plan_rows_kernel(out_loc, positions, token_to_req, ring_slots, cu_seqlens, cmp_rows, ring_rows,
                      ratio, ring_capacity, scratch_base,
                      rope_pos, stride_rs, stride_rt, table, section, cache, stride_c, q_cache, k_cache, rope_rows,
                      HALF: tl.constexpr, BLOCK: tl.constexpr, HAS_ROPE: tl.constexpr):
    t = tl.program_id(0)
    out = tl.load(out_loc + t).to(tl.int64)
    pos = tl.load(positions + t).to(tl.int64)
    req = tl.load(token_to_req + t).to(tl.int64)
    slot = tl.load(ring_slots + req).to(tl.int64)
    closing = out % ratio == ratio - 1
    tl.store(cmp_rows + t, tl.where(closing, out // ratio, scratch_base + slot).to(tl.int32))
    end = tl.load(cu_seqlens + req + 1).to(tl.int64)
    keep = t >= end - ring_capacity
    tl.store(ring_rows + t, tl.where(keep, slot * ring_capacity + pos % ring_capacity, -1).to(tl.int32))
    if HAS_ROPE:
        tl.store(rope_rows + t, t.to(tl.int32))
        i = tl.arange(0, BLOCK)
        m = i < HALF
        sec = tl.load(section + i, mask=m, other=0).to(tl.int64)
        pq = tl.load(rope_pos + sec * stride_rs + t * stride_rt, mask=m, other=0).to(tl.int64)
        group = out - out % ratio
        pk = tl.load(table + group * 3 + sec, mask=m, other=0).to(tl.int64)
        width = 2 * HALF
        tl.store(q_cache + t * width + i, tl.load(cache + pq * stride_c + i, mask=m), mask=m)
        tl.store(q_cache + t * width + HALF + i, tl.load(cache + pq * stride_c + HALF + i, mask=m), mask=m)
        tl.store(k_cache + t * width + i, tl.load(cache + pk * stride_c + i, mask=m), mask=m)
        tl.store(k_cache + t * width + HALF + i, tl.load(cache + pk * stride_c + HALF + i, mask=m), mask=m)


def qsa_plan_index_writes(out_loc: torch.Tensor, positions: torch.Tensor, token_to_req: torch.Tensor,
                          ring_slots: torch.Tensor, cu_seqlens: torch.Tensor, ratio: int, ring_capacity: int,
                          scratch_base: int, rope_positions: torch.Tensor | None = None,
                          rope_table: torch.Tensor | None = None, section_table: torch.Tensor | None = None,
                          cos_sin_cache: torch.Tensor | None = None):
    """Returns ``(cmp_rows, ring_rows, rope_rows, q_rope_cache, k_rope_cache)`` (the last three None without mrope);
    with mrope, writes ``rope_positions [3, T] or [T]`` into ``rope_table [slots, 3]`` at ``out_loc`` first."""
    n = out_loc.numel()
    dev = out_loc.device
    cmp_rows = torch.empty(n, dtype=torch.int32, device=dev)
    ring_rows = torch.empty(n, dtype=torch.int32, device=dev)
    has_rope = rope_positions is not None
    if has_rope:
        rp = rope_positions
        stride_rs, stride_rt = (rp.stride(0), rp.stride(1)) if rp.dim() == 2 else (0, rp.stride(0))
        half = cos_sin_cache.shape[1] // 2
        q_cache = torch.empty(n, 2 * half, dtype=cos_sin_cache.dtype, device=dev)
        k_cache = torch.empty(n, 2 * half, dtype=cos_sin_cache.dtype, device=dev)
        rope_rows = torch.empty(n, dtype=torch.int32, device=dev)
        if n:
            _plan_rope_store_kernel[(n,)](out_loc, rp, stride_rs, stride_rt, rope_table)
    else:
        rp, stride_rs, stride_rt, half = out_loc, 0, 0, 1
        q_cache = k_cache = rope_rows = None
    if n:
        _plan_rows_kernel[(n,)](
            out_loc, positions, token_to_req, ring_slots, cu_seqlens, cmp_rows, ring_rows, ratio, ring_capacity,
            scratch_base, rp, stride_rs, stride_rt, rope_table if has_rope else out_loc,
            section_table if has_rope else out_loc, cos_sin_cache if has_rope else out_loc,
            cos_sin_cache.stride(0) if has_rope else 0, q_cache if has_rope else cmp_rows,
            k_cache if has_rope else cmp_rows, rope_rows if has_rope else cmp_rows,
            HALF=half, BLOCK=triton.next_power_of_2(half), HAS_ROPE=has_rope, num_warps=1)
    return cmp_rows, ring_rows, rope_rows, q_cache, k_cache


__all__ = ["qsa_plan_index_writes"]
