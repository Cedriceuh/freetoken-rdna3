"""A decode MoE layer's slot-cache admission and the copy of the banks its GEMVs do not read from host memory
themselves, in one launch (FREETOKEN_MOE_ENSURE_COPY, default 1) instead of flashlib's ``lru_ensure`` then
``fast_index_copy_multi``.

Program 0 runs ``lru_ensure``'s sequential strategy (the same code: the same slots, victims, plan and clock), then
publishes the plan; the other programs wait for it and copy the missing experts' rows of the small banks, as the copy
kernel does. Integer bookkeeping and byte copies only, so the step's bits do not change. Program 0 is dispatched first
and the grid is a few programs, so the waiting ones never hold the CUs it needs.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache.triton.lru_ensure import (_STREAMING_MAX_K, _STREAMING_THRESHOLD, _num_warps_for,
                                                           _phase1)

ENABLED = os.environ.get("FREETOKEN_MOE_ENSURE_COPY", "1") == "1"
_BLOCKS_PER_BANK = 2
_CHUNK = 2048  # int32 words a program copies per iteration (8 KiB)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _ensure_copy_kernel(query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr, out_ptr, src_ptr,
                        dst_ptr, num_copy_ptr, K, num_cached, id_base, dst_base_ptr, src_base_ptr, feat_ptr, sync_ptr,
                        BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, USAGE_MAX: tl.constexpr, COPIERS: tl.constexpr,
                        BPB: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        # flashlib's _lru_ensure_kernel (sequential victims), statistics compiled out
        step = tl.load(lru_step_ptr) + 1
        tl.store(lru_step_ptr, step)
        q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
            query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K, id_base)
        if num_missing > 0:
            tl.debug_barrier()
            c = tl.arange(0, BLOCK_C)
            cmask = c < num_cached
            u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
            umax = tl.full([BLOCK_C], USAGE_MAX, u.dtype)
            u = tl.where((u == step) | (~cmask), umax, u)
            for i in tl.range(num_missing):
                victim = tl.argmin(u, axis=0).to(tl.int32)
                old = tl.load(id_of_slot_ptr + victim)
                if old >= 0:
                    tl.store(slot_of_id_ptr + old, -1)
                e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
                tl.store(id_of_slot_ptr + victim, e)
                tl.store(slot_of_id_ptr + e, victim)
                tl.store(lru_usage_ptr + victim, step)
                tl.store(dst_ptr + i, victim)
                tl.store(src_ptr + i, e - id_base)
                out = tl.where((rank == i) & miss, victim, out)
                u = tl.where(c == victim, umax, u)
        tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
        if COPIERS > 0:
            # every thread releases its own stores of the plan once the whole program has made them
            tl.debug_barrier()
            tl.atomic_xchg(sync_ptr, 1, sem="release", scope="gpu")
    else:
        while tl.atomic_add(sync_ptr, 0, sem="acquire", scope="gpu") == 0:
            pass
        cid = pid - 1
        b = cid // BPB
        part = cid % BPB
        n = tl.load(num_copy_ptr)
        feat = tl.load(feat_ptr + b)
        words = feat // 4
        dbase = tl.load(dst_base_ptr + b)
        sbase = tl.load(src_base_ptr + b)
        o = tl.arange(0, CHUNK)
        for i in range(n):
            d = tl.load(dst_ptr + i).to(tl.int64)
            s = tl.load(src_ptr + i).to(tl.int64)
            dp = (dbase + d * feat).to(tl.pointer_type(tl.int32))
            sp = (sbase + s * feat).to(tl.pointer_type(tl.int32))
            for off in range(part * CHUNK, words, BPB * CHUNK):
                m = off + o < words
                tl.store(dp + off + o, tl.load(sp + off + o, mask=m), mask=m)
        # the last copier resets the flags for the next launch (stream order)
        if tl.atomic_add(sync_ptr + 1, 1, sem="acq_rel", scope="gpu") == COPIERS - 1:
            tl.atomic_xchg(sync_ptr + 1, 0, sem="relaxed", scope="gpu")
            tl.atomic_xchg(sync_ptr, 0, sem="relaxed", scope="gpu")


def supported(query: torch.Tensor, id_of_slot: torch.Tensor) -> bool:
    """Shapes lru_ensure serves with its sequential strategy (the one this kernel carries)."""
    k, num_cached = query.numel(), id_of_slot.numel()
    return (ENABLED and query.dtype == torch.int32 and query.is_contiguous() and k <= 128
            and not (num_cached >= _STREAMING_THRESHOLD and k <= _STREAMING_MAX_K))


def ensure_copy(query: torch.Tensor, slot_of_id: torch.Tensor, id_of_slot: torch.Tensor, lru_usage: torch.Tensor,
                lru_step: torch.Tensor, src_indices: torch.Tensor, dst_indices: torch.Tensor, num_copy: torch.Tensor,
                id_base: int, copy_plan: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
                sync: torch.Tensor) -> None:
    """``lru_ensure(query, ..., out_indices=query)`` (in place), then ``fast_index_copy_multi(*copy_plan, dst_indices,
    src_indices, num_copy)``; ``copy_plan``: device (destination bases, source bases, row bytes, multiples of 16) of the
    banks to copy, None for none; ``sync``: two zeroed int32 words owned by the caller."""
    k, num_cached = query.numel(), id_of_slot.numel()
    block_c = triton.next_power_of_2(num_cached)
    nb = 0 if copy_plan is None else copy_plan[0].numel()
    dst_base, src_base, feat = copy_plan if copy_plan is not None else (sync, sync, sync)
    _ensure_copy_kernel[(1 + nb * _BLOCKS_PER_BANK,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step, query, src_indices, dst_indices, num_copy, k, num_cached,
        id_base, dst_base, src_base, feat, sync, BLOCK_K=triton.next_power_of_2(k), BLOCK_C=block_c,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max, COPIERS=nb * _BLOCKS_PER_BANK, BPB=_BLOCKS_PER_BANK, CHUNK=_CHUNK,
        num_warps=_num_warps_for(block_c))


__all__ = ["ENABLED", "ensure_copy", "supported"]
