"""A decode MoE layer's slot-cache admission and the copy of the banks its GEMVs do not read from host memory
themselves, in one launch (FREETOKEN_MOE_ENSURE_COPY, default 1) instead of flashlib's ``lru_ensure`` then
``fast_index_copy_multi``.

Program 0 runs ``lru_ensure``'s sequential strategy (the same code: the same slots, victims, plan and clock), or picks
the victims in one pass (FREETOKEN_MOE_ENSURE_STRIPES), then publishes the plan; the other programs wait for it and copy
the missing experts' rows of the small banks, as the copy kernel does. Integer bookkeeping and byte copies only, so the
step's bits do not change. Program 0 is dispatched first and the grid is a few programs, so the waiting ones never hold
the CUs it needs.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache.triton.lru_ensure import _STREAMING_MAX_K, _STREAMING_THRESHOLD, _num_warps_for

ENABLED = os.environ.get("FREETOKEN_MOE_ENSURE_COPY", "1") == "1"
# FREETOKEN_MOE_ENSURE_COPIERS (default 8): copier programs a bank part, each the misses i = slice (mod 8), so that the
# misses' small rows cross PCIe at once instead of one round trip after another (1: one copier a part, every miss)
_COPY_SLICES = max(1, int(os.environ.get("FREETOKEN_MOE_ENSURE_COPIERS", "8")))
# FREETOKEN_MOE_ENSURE_STRIPES (default 128): the victims picked in one pass, the oldest slot of each of that many
# interleaved stripes (the stripes taken from a point that moves with the clock), instead of one argmin over the whole
# cache a miss; the misses' count then costs nothing (0: the sequential argmins)
_STRIPES = int(os.environ.get("FREETOKEN_MOE_ENSURE_STRIPES", "128"))
if _STRIPES < 0 or _STRIPES & (_STRIPES - 1):
    raise ValueError(f"FREETOKEN_MOE_ENSURE_STRIPES must be 0 or a power of two, got {_STRIPES}")
# FREETOKEN_MOE_HIT_BONUS (default 480, about ten decode steps of calls): a hit stamps its slot that many calls ahead of
# the clock, so that an expert used again outlives one admitted once (0: plain LRU, flashlib's stamps)
_HIT_BONUS = max(0, int(os.environ.get("FREETOKEN_MOE_HIT_BONUS", "480")))
_BLOCKS_PER_BANK = 2
_CHUNK = 2048  # int32 words a program copies per iteration (8 KiB)


@triton.jit
def _phase1(query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K: tl.constexpr, id_base,
            HIT_BONUS: tl.constexpr):
    """flashlib's lru_ensure Phase 1 (dedup the query, split hit / miss, rank the misses), the hits stamped
    ``step + HIT_BONUS``."""
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=-1) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=-1)
    hit = kmask & (s >= 0)
    miss = kmask & (s == -1)
    same = (q[:, None] == q[None, :]) & (k[:, None] > k[None, :]) & kmask[:, None] & kmask[None, :]
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    smaller = (q[None, :] < q[:, None]) & first_miss[None, :]
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    tl.store(lru_usage_ptr + s, step + HIT_BONUS, mask=hit)
    return q, kmask, miss, first, first_miss, rank, num_missing, tl.where(hit, s, -1)


@triton.jit(do_not_specialize=["K", "num_cached", "id_base"])
def _ensure_copy_kernel(query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr, out_ptr, src_ptr,
                        dst_ptr, num_copy_ptr, K, num_cached, id_base, dst_base_ptr, src_base_ptr, feat_ptr, sync_ptr,
                        BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, USAGE_MAX: tl.constexpr, COPIERS: tl.constexpr,
                        BPB: tl.constexpr, CHUNK: tl.constexpr, SLICES: tl.constexpr = 1,
                        PER_SLICE: tl.constexpr = 1, STRIPES: tl.constexpr = 0, SLOT_BITS: tl.constexpr = 14,
                        HIT_BONUS: tl.constexpr = 0):
    pid = tl.program_id(0)
    if pid == 0:
        # flashlib's _lru_ensure_kernel (sequential victims), statistics compiled out, or the striped victims
        step = tl.load(lru_step_ptr) + 1
        tl.store(lru_step_ptr, step)
        q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
            query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K, id_base, HIT_BONUS)
        if num_missing > 0:
            tl.debug_barrier()
            striped = num_missing * 0
            if STRIPES > 0:
                if num_missing <= STRIPES:
                    # slot r * STRIPES + j is in stripe j; each stripe's oldest by (usage, slot), flashlib's packed key;
                    # this call's hits are stamped step + HIT_BONUS (step when 0): never victims
                    c2 = tl.arange(0, BLOCK_C // STRIPES)[:, None] * STRIPES + tl.arange(0, STRIPES)[None, :]
                    m2 = c2 < num_cached
                    u2 = tl.load(lru_usage_ptr + c2, mask=m2, other=USAGE_MAX)
                    key = tl.where(m2 & (u2 != step + HIT_BONUS), (u2.to(tl.int64) << SLOT_BITS) | c2.to(tl.int64),
                                   0x7FFFFFFFFFFFFFFF)
                    smin = tl.min(key, axis=0)
                    j = tl.arange(0, STRIPES)
                    # stripe j serves the miss of rank (j - step * 79) mod STRIPES
                    si = ((j - step * 79) % STRIPES + STRIPES) % STRIPES
                    take = si < num_missing
                    if tl.sum((take & (smin == 0x7FFFFFFFFFFFFFFF)).to(tl.int32)) == 0:
                        striped = num_missing
                        sv = (smin & ((1 << SLOT_BITS) - 1)).to(tl.int32)
                        # the plan in rank order, then each side reads the other's half of it back
                        tl.store(src_ptr + rank, q - id_base, mask=first_miss)
                        tl.store(dst_ptr + si, sv, mask=take)
                        tl.debug_barrier()
                        e = tl.load(src_ptr + si, mask=take, other=0) + id_base
                        old = tl.load(id_of_slot_ptr + sv, mask=take, other=-1)
                        tl.store(slot_of_id_ptr + old, -1, mask=take & (old >= 0))
                        tl.store(id_of_slot_ptr + sv, e, mask=take)
                        tl.store(slot_of_id_ptr + e, sv, mask=take)
                        tl.store(lru_usage_ptr + sv, step, mask=take)
                        out = tl.where(miss, tl.load(dst_ptr + rank, mask=miss, other=-1), out)
            if striped == 0:
                c = tl.arange(0, BLOCK_C)
                cmask = c < num_cached
                u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
                umax = tl.full([BLOCK_C], USAGE_MAX, u.dtype)
                u = tl.where((u == step + HIT_BONUS) | (~cmask), umax, u)
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
        sl = cid // PER_SLICE  # this program's misses: i = sl (mod SLICES)
        cid = cid % PER_SLICE
        b = cid // BPB
        part = cid % BPB
        n = tl.load(num_copy_ptr)
        feat = tl.load(feat_ptr + b)
        words = feat // 4
        dbase = tl.load(dst_base_ptr + b)
        sbase = tl.load(src_base_ptr + b)
        o = tl.arange(0, CHUNK)
        for i in range(sl, n, SLICES):
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
    copiers = nb * _BLOCKS_PER_BANK * _COPY_SLICES
    stripes = _STRIPES if 2 * _STRIPES <= block_c else 0
    _ensure_copy_kernel[(1 + copiers,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step, query, src_indices, dst_indices, num_copy, k, num_cached,
        id_base, dst_base, src_base, feat, sync, BLOCK_K=triton.next_power_of_2(k), BLOCK_C=block_c,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max, COPIERS=copiers, BPB=_BLOCKS_PER_BANK, CHUNK=_CHUNK,
        SLICES=_COPY_SLICES, PER_SLICE=max(nb * _BLOCKS_PER_BANK, 1), STRIPES=stripes,
        SLOT_BITS=max(block_c.bit_length() - 1, 1), HIT_BONUS=_HIT_BONUS, num_warps=_num_warps_for(block_c))


__all__ = ["ENABLED", "ensure_copy", "supported"]
