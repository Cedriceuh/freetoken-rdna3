"""Host orchestration of the EXL3 routed experts (layers/quantization/moe/exl3.py banks).

rotate-in (``H(x * suh)`` per route and projection) -> gate|up product -> mid (rotate out, silu * up, rotate into
down's domain) -> down product -> rotate out and routed sum. Decode runs the per-route split-K GEMVs (static shapes,
CUDA-graph safe); prefill runs expert-sorted grouped GEMMs on fp16 activations, in token blocks.
"""

from __future__ import annotations

import os

import torch

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.exl3 import exl3_gemm_down_out, exl3_gemv, exl3_mid, exl3_moe_gemm, exl3_out, exl3_rotate_in
from freetoken.kernel.triton.moe_align import moe_align_split
from freetoken.moe.fused import blocks_to_slots, moe_align_block_size
from freetoken.moe.prefill_hooks import BLOCK_DONE

# token block of the prefill path: its rotated inputs are [tokens * top_k, 2, hidden] fp16 and its products fp32,
# 540 MiB of temporaries at 4096 tokens (top-10, hidden 2560, a 384-wide slice), which the cache planner keeps free
# (TritonExl3MoEKernel.prefill_workspace_bytes). Swept on the 7900 XT with block_m (8192 tokens through one layer,
# I=384): 1024 / 64 5.43 ms per 1k tokens, 4096 / 32 3.43-3.49, 4096 / 64 3.54-3.63, 8192 / 64 3.25 for twice
# the memory. With 32-row tiles, a 4096-token block still feeds every expert ~80 routes on average.
EXL3_PREFILL_BLOCK = int(os.environ.get("FREETOKEN_EXL3_PREFILL_BLOCK", "4096"))
_PREFILL_BLOCK_M = 32
# short prefills (agent turns: ~5-10 routes an expert) take 16-row tiles: the same K order, so the same bits; the whole
# MoE block 11-19 % faster at 32-512 tokens on both cards, slower from ~1024 (I = 384 / 256, random routing)
_SHORT_PREFILL_BLOCK_M, _SHORT_PREFILL_TOKENS = 16, 512
# FREETOKEN_EXL3_PREFILL_TAIL_TILES (default 1): from 3072 tokens (an expert gets ~64 routes or more), the gate|up GEMM
# runs each expert's full 64-row tiles in one launch and its remaining rows in 32-row tiles in a second one (the down
# GEMM keeps its 32-row tiles: no gain there). The same K order, so the same bits. 4096-token block, random routing:
# 4.27 -> 3.74 ms (7900 XTX's 384-wide slice), 4.03 -> 3.67 ms (XT's 256); the whole MoE block with this model's
# routing (96 dumped 4096-token blocks): 919 -> 884 ms (XTX), 790 -> 762 ms (XT)
_TAIL_TILES = os.environ.get("FREETOKEN_EXL3_PREFILL_TAIL_TILES", "1") == "1"
_TAIL_MIN_TOKENS = 3072
# FREETOKEN_HYBRID_SKIP_CPU_ROUTES=0: under --moe-strategy hybrid the decode GEMVs still decode the routes the CPU
# computes (on slot 0, weight 0) instead of writing zeros for them without a read
SKIP_CPU_ROUTES = os.environ.get("FREETOKEN_HYBRID_SKIP_CPU_ROUTES", "1") != "0"
# FREETOKEN_EXL3_SHARED_ROUTES (default 4): a decode step of several rows (a verify step, several requests) gives the
# routes that read one expert (a third of them at four rows of one request) one GEMV program, up to this many: each
# trellis tile decoded once for all of them, the same sums. 1: one program a route
SHARED_ROUTES = int(os.environ.get("FREETOKEN_EXL3_SHARED_ROUTES", "4"))
if SHARED_ROUTES not in (1, 2, 4):
    raise ValueError(f"FREETOKEN_EXL3_SHARED_ROUTES must be 1, 2 or 4, got {SHARED_ROUTES}")


def fused_experts_exl3(
    hidden_states: torch.Tensor,
    gate_up: torch.Tensor,
    gate_up_suh: torch.Tensor,
    gate_up_svh: torch.Tensor,
    down: torch.Tensor,
    down_suh: torch.Tensor,
    down_svh: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    codebook: int,
    is_prefill: bool,
    num_experts: int | None = None,
    slot_map: torch.Tensor | None = None,
    misses: tuple | None = None,
) -> torch.Tensor:
    """``topk_ids`` index bank rows (cache slots in decode, the full-layer buffer's rows in prefill); a prefill with
    ``slot_map`` keeps expert ids in ``topk_ids`` and reads each expert from its slot of the cache. ``misses``
    (decode, OffloadMoeCache.miss_reads): the copy plan of the slots this step fills, which the GEMVs read from host
    memory and fill themselves (the cache did not copy their gate | up and down rows)."""
    M, _ = hidden_states.shape
    if is_prefill and M > EXL3_PREFILL_BLOCK > 0:
        out = torch.empty_like(hidden_states)
        done = BLOCK_DONE[0]
        for start in range(0, M, EXL3_PREFILL_BLOCK):
            end = min(start + EXL3_PREFILL_BLOCK, M)
            out[start:end] = fused_experts_exl3(
                hidden_states[start:end], gate_up, gate_up_suh, gate_up_svh, down, down_suh, down_svh,
                topk_weights[start:end], topk_ids[start:end], codebook=codebook, is_prefill=True,
                num_experts=num_experts, slot_map=slot_map,
            )
            if done is not None:
                done(start, end, out)
        return out
    top_k = topk_ids.shape[1]
    slots = topk_ids.reshape(-1)
    weights = topk_weights.contiguous()
    out = torch.empty_like(hidden_states)
    if is_prefill:
        assert num_experts is not None, "the prefill path sorts routes by expert: num_experts is required"
        block_m = _SHORT_PREFILL_BLOCK_M if M <= _SHORT_PREFILL_TOKENS else _PREFILL_BLOCK_M
        sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, block_m, num_experts)
        if slot_map is not None:
            slots = slot_map[slots.long()]
            expert_ids = blocks_to_slots(expert_ids, slot_map)
        a1 = exl3_rotate_in(hidden_states, gate_up_suh, slots, top_k, dtype=torch.float16)
        if _TAIL_TILES and block_m == _PREFILL_BLOCK_M and M >= _TAIL_MIN_TOKENS:
            c1 = None
            for (sid, eid, n), bm in zip(moe_align_split(topk_ids, 64, block_m, num_experts), (64, block_m)):
                eid = eid if slot_map is None else blocks_to_slots(eid, slot_map)
                c1 = exl3_moe_gemm(a1, gate_up, sid, eid, n, codebook, block_m=bm, out=c1)
        else:
            c1 = exl3_moe_gemm(a1, gate_up, sorted_ids, expert_ids, ntpp, codebook, block_m=block_m)
        del a1
        a2 = exl3_mid(c1, slots, gate_up_svh, down_suh, dtype=torch.float16)
        del c1
        # the rotation out runs as the down GEMM's epilogue (a program owns a 128-column block), then a plain sum
        y = exl3_gemm_down_out(a2, down, down_svh, weights.reshape(-1), sorted_ids, expert_ids, ntpp, codebook,
                               block_m=block_m, out_dtype=hidden_states.dtype)
        moe_sum_reduce_triton(y.view(M, top_k, -1), out)
        return out
    group = SHARED_ROUTES if M > 1 else 1  # one row's routes are distinct experts
    groups = torch.empty((slots.numel(), 4), dtype=torch.int32, device=slots.device) if group > 1 else None
    a1 = exl3_rotate_in(hidden_states, gate_up_suh, slots, top_k, groups=groups, group=group)
    plan, bank = (None, {}) if misses is None else misses
    if plan is not None:  # the host and slot rows share one layout: the kernel walks both with the slot strides
        assert gate_up[0].numel() * gate_up.element_size() == bank["gate_up"][1] == gate_up.stride(0) * 2, bank
        assert down[0].numel() * down.element_size() == bank["down"][1] == down.stride(0) * 2, bank
    c1 = exl3_gemv(a1, gate_up, slots, codebook, skip_inactive=SKIP_CPU_ROUTES, group=group, groups=groups,
                   misses=plan, bank=bank.get("gate_up", (0, 0))[0])
    a2 = exl3_mid(c1, slots, gate_up_svh, down_suh)
    c2 = exl3_gemv(a2, down, slots, codebook, skip_inactive=SKIP_CPU_ROUTES, group=group, groups=groups,
                   misses=plan, bank=bank.get("down", (0, 0))[0])
    return exl3_out(c2, slots, down_svh, weights, out)


__all__ = ["EXL3_PREFILL_BLOCK", "SHARED_ROUTES", "SKIP_CPU_ROUTES", "fused_experts_exl3"]
