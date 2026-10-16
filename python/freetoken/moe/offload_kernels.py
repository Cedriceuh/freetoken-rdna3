from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import lru_ensure

# Hybrid backend: which of a step's missing experts to fetch (when capped below the miss
# count). "recency" (default) fetches the experts most-recently active before this step
# (LRU on the expert -> prioritizes recurring misses, lowering the steady miss rate);
# "lowest_id" fetches the smallest expert ids (the original, routing-blind heuristic).
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)


def ensure_experts(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids.

    Delegates to flashlib's slot cache. ``id_base`` maps this layer's expert ids into the
    flat ``layer * num_experts + expert`` space the cache indexes by, and maps
    ``src_indices`` back, so ``copy_missing`` still resolves against this layer's own host
    tensor. ``out_indices`` aliases the input, preserving the in-place rewrite every
    downstream GEMM depends on.
    """
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        cache.id_of_slot,
        cache.usage,
        cache.step,
        expert_ids,
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


def ensure_experts_hybrid(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, fetch_fraction: float = 0.0,
    out: torch.Tensor | None = None,
) -> None:
    """Capped-fetch variant of ``ensure_experts`` (hybrid backend).

    Identical LRU bookkeeping, but only the first ``max_fetch`` of this step's missing
    experts are given a slot and scheduled for copy; the overflow misses stay
    non-resident and their ``expert_ids`` positions are rewritten to ``-1`` (compute on
    the CPU). ``fetch_fraction`` > 0 replaces the fixed cap with the bandwidth-matched
    split (fraction = pcie_bw / cpu_bw): fetch ~fraction of the step's misses, rounded to
    the integer that makes the PCIe fetch and the CPU overflow compute finish closest to
    together. ``num_indices`` = capped fetch count (copy_missing); ``num_missing_full`` =
    pre-cap miss count (stats). ``out``: write the slot ids there and leave ``expert_ids`` as routed (saves the
    caller a copy of the routing)."""
    # Q16 fixed point so the GPU kernel and the CPU reference cap identically (no float).
    frac_q16 = min(1 << 16, max(0, round(fetch_fraction * (1 << 16))))
    if not expert_ids.is_cuda:
        if out is not None:
            expert_ids = out.copy_(expert_ids)
        return _ensure_experts_hybrid_cpu(cache, layer_id, expert_ids, max_fetch, frac_q16)
    _ensure_experts_hybrid_gpu(cache, layer_id, expert_ids, max_fetch, frac_q16, expert_ids if out is None else out)


def prefill_hit_compact(cache, layer_id: int, buffer_id: int) -> None:
    """Compact this layer's cache-resident experts into gather indices, device-side.

    hit = slot_for_id[layer_id][e] >= 2 * num_experts (the double buffer owns the
    slots below, so those bytes are volatile within a prefill chunk and classify
    as miss). Writes fixed-shape ``_prefill_hit_dst``/``_prefill_hit_src`` (buffer
    row / cache slot) and the count into ``_prefill_hit_num``; one launch on the
    current stream, no host sync. Safe against the concurrent buffer invalidation
    on the copy stream: that only rewrites entries already below the threshold."""
    num_experts = cache.num_experts
    _prefill_hit_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        buffer_id * num_experts,
        2 * num_experts,
        num_experts,
        BLOCK=triton.next_power_of_2(num_experts),
    )


def materialize_layer(cache, layer_id: int) -> None:
    _materialize_layer_gpu(cache, layer_id)


def reset_cache(cache) -> None:
    _reset_cache_gpu(cache)






def _ensure_experts_hybrid_gpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int, out: torch.Tensor
) -> None:
    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(cache.cache_size)
    num_warps = 8 if block_c >= 2048 else 4
    _ensure_experts_hybrid_kernel[(1,)](
        expert_ids,
        out,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.expert_recency,
        layer_id,
        expert_ids.numel(),
        int(max_fetch),
        int(frac_q16),
        cache.num_experts,
        cache.cache_size,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        BLOCK_K=max(16, triton.next_power_of_2(expert_ids.numel())),
        BY_RECENCY=_HYBRID_FETCH_BY_RECENCY,
        num_warps=num_warps,
    )


def _ensure_experts_hybrid_cpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int
) -> None:
    """CPU reference mirror of the hybrid kernel (eviction/fetch decisions bit-identical to
    the GPU path; see tests/test_offload_lru_kernels.py). Fetches at most ``max_fetch`` (or
    the bandwidth-matched ``~frac_q16/2^16 * misses`` when ``frac_q16`` > 0) of the missing
    experts; overflow misses are rewritten to -1. With ``BY_RECENCY`` the fetch set is the
    most-recently-active misses (ties -> lower id); else the lowest ids."""
    seen = []
    for expert in expert_ids.view(-1).tolist():
        if expert not in seen:
            seen.append(expert)

    cache.active_mask.zero_()
    step = int(cache.step.item()) + 1
    cache.step.fill_(step)
    for expert in seen:
        cache.active_mask[expert] = 1

    for expert in seen:
        slot = int(cache.slot_for_id[layer_id, expert].item())
        if slot != -1:
            cache.usage[slot] = step

    missing = [e for e in seen if int(cache.slot_for_id[layer_id, e].item()) == -1]
    if _HYBRID_FETCH_BY_RECENCY:
        rec = cache.expert_recency[layer_id].tolist()
        missing.sort(key=lambda e: (-rec[e], e))
    else:
        missing.sort()
    if frac_q16 > 0:
        m, q = len(missing), 1 << 16
        lo = (m * frac_q16) >> 16
        cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)  # noqa: E731
        max_fetch = lo if cost(lo) <= cost(lo + 1) else lo + 1
    num_fetch = min(len(missing), int(max_fetch))
    cache.num_missing_full.fill_(len(missing))
    cache.num_indices.fill_(num_fetch)

    usage = cache.usage.tolist()
    for idx in range(num_fetch):
        expert = missing[idx]
        victim = min(range(cache.cache_size), key=lambda s: (usage[s], s))
        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1
        cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
        cache.slot_for_id[layer_id, expert] = victim
        cache.usage[victim] = step
        usage[victim] = step
        cache.evict_slots[idx] = victim
        cache.src_indices[idx] = expert  # layer-local row

    if _HYBRID_FETCH_BY_RECENCY:
        for expert in seen:
            cache.expert_recency[layer_id, expert] = step

    # Overflow misses keep slot_for_id == -1, so the rewrite below yields -1 for them.
    flat = expert_ids.view(-1)
    for i in range(flat.numel()):
        flat[i] = int(cache.slot_for_id[layer_id, int(flat[i].item())].item())


def _materialize_layer_gpu(cache, layer_id: int) -> None:
    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def _reset_cache_gpu(cache) -> None:
    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


@triton.jit
def _reset_cache_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    num_indices_ptr,
    total_ids: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
    tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
    tl.store(usage_ptr + off, 0, mask=off < cache_size)
    tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
    if tl.program_id(0) == 0:
        tl.store(step_ptr, 0)
        tl.store(num_indices_ptr, 0)


@triton.jit
def _materialize_layer_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    layer_id: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    expert_mask = off < num_experts
    slot_mask = off < cache_size
    slot = off

    base = layer_id * num_experts
    old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
    # Flat ids make "belongs to this layer" a range check instead of a field compare.
    same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
    tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
    tl.store(usage_ptr + slot, 0, mask=same_layer)

    old_valid = expert_mask & (old_id >= 0) & (~same_layer)
    tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
    tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
    tl.store(usage_ptr + slot, step, mask=expert_mask)
    tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
    tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
    tl.store(num_indices_ptr, num_experts)




@triton.jit(do_not_specialize=["layer_id", "num_active", "max_fetch", "fetch_frac_q16"])
def _ensure_experts_hybrid_kernel(
    expert_ids_ptr,
    out_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    expert_recency_ptr,
    layer_id,
    num_active,
    max_fetch,
    fetch_frac_q16,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BY_RECENCY: tl.constexpr,
):
    """Capped-fetch timestamp-LRU (hybrid backend).

    Same as ``_ensure_experts_lru_v2_kernel`` but only ``min(num_missing, max_fetch)``
    missing experts are evicted-into / scheduled for copy; the overflow misses stay
    non-resident, so Phase 3 rewrites their positions to -1 (the layer computes those on
    the CPU). ``fetch_frac_q16`` > 0 (Q16 fixed point) replaces the fixed cap with the
    bandwidth-matched split ``~frac * num_missing`` (see the Phase-1 comment), computed
    in-kernel because ``num_missing`` only exists device-side (CUDA graph). ``num_indices``
    = the capped fetch count (copy_missing), ``num_missing_full`` = the pre-cap miss count
    (stats).

    Which misses to fetch is the cap policy. ``BY_RECENCY`` (default) fetches the experts
    most-recently active before this step (LRU on the expert, via ``expert_recency``),
    breaking ties toward the lower expert id -- this prioritizes *recurring* misses for
    caching, lowering the steady miss rate. Otherwise the lowest expert ids are fetched
    (``missing_rank``), the original routing-blind heuristic.

    The routes are read 16 at a time as vectors (a loop of dependent scalar loads per route cost several us a layer
    on RDNA3), and a slot used this step is told apart by its usage stamp (``step`` is unique to the call) instead of a
    compare of the whole cache per route."""
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts

    # ---- Phase 1: active + missing over experts ----
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    active = tl.zeros((BLOCK_E,), dtype=tl.int32)
    for k0 in tl.static_range(0, BLOCK_K, 16):
        kk = k0 + tl.arange(0, 16)
        ids = tl.load(expert_ids_ptr + kk, mask=kk < num_active, other=-1)
        active = tl.maximum(active, tl.max((ids[:, None] == off_e[None, :]).to(tl.int32), axis=0))
    is_active = active > 0
    tl.store(active_mask_ptr + off_e, active, mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & (slot == -1) & e_mask
    num_missing = tl.sum(is_missing.to(tl.int32))
    # Cap the fetches; the overflow misses are computed on the CPU (left non-resident).
    if fetch_frac_q16 > 0:
        # Bandwidth-matched split (fetch_frac = pcie_bw / cpu_bw): fetch time scales with
        # F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick
        # the integer neighbor that minimizes the slower (max) side of the overlap.
        lo = (num_missing * fetch_frac_q16) >> 16
        cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
        cost_hi = tl.maximum(
            (lo + 1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16
        )
        max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
    num_fetch = tl.minimum(num_missing, max_fetch)
    tl.store(num_missing_full_ptr, num_missing.to(tl.int64))
    tl.store(num_indices_ptr, num_fetch.to(tl.int64))
    is_hit = is_active & (slot >= 0)
    tl.store(usage_ptr + slot, step, mask=is_hit)

    # Fetch-selection priority: encode (recency desc, id asc) into one strictly-ordered
    # score so argmax has no ties (rec deltas are multiples of num_experts; the id term
    # spans only [0, num_experts), so it can only break exact-recency ties).
    if BY_RECENCY:
        rec = tl.load(expert_recency_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
        score = tl.where(
            is_missing, rec * num_experts + (num_experts - 1 - off_e), -1152921504606846976
        ).to(tl.int64)
    else:
        missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1

    # ---- Phase 2: evict victims by argmin(usage), only for the capped fetches ----
    if num_fetch > 0:
        tl.debug_barrier()  # the hits' usage stamps above, before the load below
        off_c = tl.arange(0, BLOCK_C)
        c_mask = off_c < cache_size
        u = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807).to(tl.int64)
        # a slot stamped this step holds an active expert of this layer (a hit): never a victim
        u = tl.where((u >= step) | (~c_mask), 9223372036854775807, u)
        for i in tl.range(num_fetch):
            victim = tl.argmin(u, axis=0).to(tl.int32)
            old_id = tl.load(id_of_slot_ptr + victim)
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            if BY_RECENCY:
                e = tl.argmax(score, axis=0).to(tl.int32)
                score = tl.where(off_e == e, -1152921504606846976, score)
            else:
                e = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
            tl.store(id_of_slot_ptr + victim, base + e)
            tl.store(slot_for_id_ptr + base + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            tl.store(src_indices_ptr + i, e)  # layer-local row
            u = tl.where(off_c == victim, 9223372036854775807, u)

    # ---- Phase 3: rewrite expert_ids -> slot id (hit/fetched) or -1 (overflow -> CPU) ----
    tl.debug_barrier()  # phase 2's slot_for_id stores, before the gathers below
    for k0 in tl.static_range(0, BLOCK_K, 16):
        kk = k0 + tl.arange(0, 16)
        km = kk < num_active
        e = tl.load(expert_ids_ptr + kk, mask=km, other=0)
        s = tl.load(slot_for_id_ptr + base + e, mask=km, other=-1)
        tl.store(out_ptr + kk, s, mask=km)

    # Bump every active expert's recency to this step (LRU on the expert): an overflow miss
    # computed on the CPU now ranks high if it recurs, so it gets fetched next time.
    if BY_RECENCY:
        step_vec = tl.zeros((BLOCK_E,), dtype=tl.int64) + step
        tl.store(expert_recency_ptr + base + off_e, step_vec, mask=is_active & e_mask)


@triton.jit(do_not_specialize=["buffer_base"])
def _prefill_hit_compact_kernel(
    slot_ptr,     # [num_experts] int32: this layer's slot_for_id row
    dst_ptr,      # [num_experts] int32 out: buffer rows, compacted
    src_ptr,      # [num_experts] int32 out: cache slots, compacted
    num_ptr,      # [1] int64 out: hit count
    buffer_base,  # buffer_id * num_experts
    threshold,    # 2 * num_experts
    num_experts,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    is_hit = lane & (slots >= threshold)
    pos = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(dst_ptr + pos, (buffer_base + offs).to(tl.int32), mask=is_hit)
    tl.store(src_ptr + pos, slots, mask=is_hit)
    tl.store(num_ptr, tl.sum(is_hit.to(tl.int64)))


def prefill_routed_plan(cache, layer_id: int, topk_ids: torch.Tensor, next_layer: int, predicted_ids: torch.Tensor | None,
                        slot_of_fix: torch.Tensor, slot_of_pf: torch.Tensor) -> None:
    """One prefill layer's slot bookkeeping (FREETOKEN_PREFILL_CACHED_MAX), device-side, three launches and a top-k:
    marks the experts ``topk_ids`` routes to (layer ``layer_id``) and, when ``next_layer`` >= 0, the ones
    ``predicted_ids`` predicts for ``next_layer``; usage = step for the resident ones (step + 1 first); then gives the
    absent ones the least recently used slots outside them (by (usage, slot) like ``lru_ensure``), the routed first.
    Writes the copy plans: decode's (``evict_slots``/``src_indices``/``num_indices``) for the routed misses, the
    prefetch's (``_prefill_pf_idx``/``_prefill_pf_num``) for the predicted ones, and each placed expert's slot
    (``slot_of_fix`` / ``slot_of_pf``, -1 elsewhere)."""
    E, C = cache.num_experts, cache.cache_size
    masks = cache._prefill_masks  # [2, E] int8, left zeroed by the plan kernel
    ids = topk_ids.reshape(-1)
    pred = predicted_ids.reshape(-1) if predicted_ids is not None else ids[:0]
    n = max(ids.numel(), pred.numel(), 1)
    _prefill_mark_kernel[(triton.cdiv(n, 1024),)](ids, ids.numel(), pred, pred.numel(), masks, E, BLOCK=1024)
    slot_bits = max(1, (C - 1).bit_length())
    _prefill_keys_kernel[(1,)](
        masks, cache.slot_for_id, cache.usage, cache.step, cache._prefill_keys, cache._prefill_miss,
        layer_id * E, max(next_layer, 0) * E, next_layer >= 0, E, C, slot_bits,
        BLOCK_E=triton.next_power_of_2(E), BLOCK_C=1024, num_warps=8,
    )
    cand = torch.topk(cache._prefill_keys, min(2 * E, C), largest=False, sorted=True).indices
    idx = cache._prefill_pf_idx
    _prefill_claim_kernel[(1,)](
        cache._prefill_miss, cand, cache.slot_for_id, cache.id_of_slot, cache.usage, cache.step,
        cache.evict_slots, cache.src_indices, cache.num_indices, idx, idx[E:], cache._prefill_pf_num,
        slot_of_fix, slot_of_pf, layer_id * E, max(next_layer, 0) * E, E, BLOCK_E=triton.next_power_of_2(E), num_warps=4,
    )


@triton.jit(do_not_specialize=["n_ids", "n_pred"])
def _prefill_mark_kernel(ids_ptr, n_ids, pred_ptr, n_pred, masks_ptr, E, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(ids_ptr + i, mask=i < n_ids, other=0)
    tl.store(masks_ptr + e, 1, mask=i < n_ids)
    p = tl.load(pred_ptr + i, mask=i < n_pred, other=0)
    tl.store(masks_ptr + E + p, 1, mask=i < n_pred)


@triton.jit(do_not_specialize=["base", "base_next", "has_next"])
def _prefill_keys_kernel(masks_ptr, slot_of_id_ptr, usage_ptr, step_ptr, keys_ptr, miss_ptr, base, base_next, has_next,
                         E, C, SLOT_BITS: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_C: tl.constexpr):
    e = tl.arange(0, BLOCK_E)
    em = e < E
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    r = tl.load(masks_ptr + e, mask=em, other=0) != 0
    p = (tl.load(masks_ptr + E + e, mask=em, other=0) != 0) & (has_next != 0)
    s_r = tl.load(slot_of_id_ptr + base + e, mask=em, other=-1)
    s_p = tl.load(slot_of_id_ptr + base_next + e, mask=em & (has_next != 0), other=-1)
    tl.store(usage_ptr + s_r, step, mask=em & r & (s_r >= 0))
    tl.store(usage_ptr + s_p, step, mask=em & p & (s_p >= 0))
    tl.store(miss_ptr + e, (r & (s_r < 0)).to(tl.int8), mask=em)
    tl.store(miss_ptr + E + e, (p & (s_p < 0)).to(tl.int8), mask=em)
    tl.store(masks_ptr + e, 0, mask=em)
    tl.store(masks_ptr + E + e, 0, mask=em)
    tl.debug_barrier()  # the bumps visible to the key pass
    for c0 in range(0, C, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)
        cm = c < C
        u = tl.load(usage_ptr + c, mask=cm, other=0)
        k = tl.where(u >= step, 0x7FFFFFFFFFFFFFFF, (u << SLOT_BITS) | c.to(tl.int64))
        tl.store(keys_ptr + c, k, mask=cm)


@triton.jit(do_not_specialize=["base", "base_next"])
def _prefill_claim_kernel(
    miss_ptr, cand_ptr, slot_of_id_ptr, id_of_slot_ptr, usage_ptr, step_ptr, fix_dst_ptr, fix_src_ptr, fix_num_ptr,
    pf_dst_ptr, pf_src_ptr, pf_num_ptr, slot_of_fix_ptr, slot_of_pf_ptr, base, base_next, E, BLOCK_E: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    em = e < E
    mr = (tl.load(miss_ptr + e, mask=em, other=0) != 0) & em
    mp = (tl.load(miss_ptr + E + e, mask=em, other=0) != 0) & em
    n_r = tl.sum(mr.to(tl.int32), 0)
    rank_r = tl.cumsum(mr.to(tl.int32), 0) - 1
    rank_p = tl.cumsum(mp.to(tl.int32), 0) - 1
    v_r = tl.load(cand_ptr + rank_r, mask=mr, other=0).to(tl.int32)
    v_p = tl.load(cand_ptr + n_r + rank_p, mask=mp, other=0).to(tl.int32)
    old_r = tl.load(id_of_slot_ptr + v_r, mask=mr, other=-1)
    old_p = tl.load(id_of_slot_ptr + v_p, mask=mp, other=-1)
    tl.debug_barrier()  # every old owner read before any map is written
    step = tl.load(step_ptr)
    tl.store(slot_of_id_ptr + old_r, -1, mask=mr & (old_r >= 0))
    tl.store(slot_of_id_ptr + old_p, -1, mask=mp & (old_p >= 0))
    tl.debug_barrier()  # evictions before the new owners (an evicted id may be a new owner's id of another layer)
    tl.store(id_of_slot_ptr + v_r, base + e, mask=mr)
    tl.store(slot_of_id_ptr + base + e, v_r, mask=mr)
    tl.store(id_of_slot_ptr + v_p, base_next + e, mask=mp)
    tl.store(slot_of_id_ptr + base_next + e, v_p, mask=mp)
    tl.store(usage_ptr + v_r, step, mask=mr)
    tl.store(usage_ptr + v_p, step, mask=mp)
    tl.store(fix_dst_ptr + rank_r, v_r, mask=mr)
    tl.store(fix_src_ptr + rank_r, e, mask=mr)
    tl.store(pf_dst_ptr + rank_p, v_p, mask=mp)
    tl.store(pf_src_ptr + rank_p, e, mask=mp)
    tl.store(slot_of_fix_ptr + e, tl.where(mr, v_r, -1), mask=em)
    tl.store(slot_of_pf_ptr + e, tl.where(mp, v_p, -1), mask=em)
    tl.store(fix_num_ptr, n_r.to(tl.int64))
    tl.store(pf_num_ptr, tl.sum(mp.to(tl.int32), 0).to(tl.int64))


def prefill_demote(cache, slot_ofs: list) -> None:
    """usage = 0 for the slots in ``slot_ofs`` ([E] int32 arrays, -1 = none): the next victims."""
    for so in slot_ofs:
        _prefill_demote_kernel[(1,)](so, cache.usage, so.numel(), BLOCK_E=triton.next_power_of_2(so.numel()))


@triton.jit
def _prefill_demote_kernel(slot_of_ptr, usage_ptr, E, BLOCK_E: tl.constexpr):
    e = tl.arange(0, BLOCK_E)
    s = tl.load(slot_of_ptr + e, mask=e < E, other=-1)
    tl.store(usage_ptr + s, 0, mask=(e < E) & (s >= 0))
