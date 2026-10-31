"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from freetoken.kernel import moe_sum_reduce_triton, nvfp4_moe
from freetoken.kernel.backend import is_flashinfer_installed
from freetoken.kernel.triton.nvfp4_act_quant import quantize_nvfp4
from freetoken.layers import gated_act_and_mul
from freetoken.kernel.triton.moe_align import moe_align_split
from freetoken.moe.fused import blocks_to_slots, moe_align_block_size
from freetoken.moe.prefill_hooks import BLOCK_DONE
from freetoken.utils import init_logger

logger = init_logger(__name__)

# set by a caller that sums the decode experts' outputs itself (one element: mutable, read at call time)
PER_EXPERT_OUTPUT = [False]

# the launches live in kernel/nvfp4_moe.py; the RDNA3 benches and tests call them by these names
SKIP_CPU_ROUTES = nvfp4_moe.SKIP_CPU_ROUTES
# FREETOKEN_NVFP4_ACT_IN_GEMV (default 1): decode's gate | up GEMV (marlin) computes silu_and_mul itself, each program
# both halves of its columns with the GEMV's own tiles (the same bits, one launch fewer per MoE layer)
_ACT_IN_GEMV = os.environ.get("FREETOKEN_NVFP4_ACT_IN_GEMV", "1") != "0"
_decode_gemm = nvfp4_moe.decode_gemm
_decode_gemm_marlin = nvfp4_moe.decode_gemm_marlin
_prefill_gemm = nvfp4_moe.prefill_gemm


def _fused_experts_decode_nvfp4(
    gemm_fn,
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    apply_router_weight_on_input: bool,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    misses: tuple | None = None,
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`freetoken.kernel.nvfp4_moe.decode_gemm`), both with the same calling convention. ``misses``
    (marlin only, OffloadMoeCache.miss_reads): the copy plan of the slots this step fills, whose packed and scale rows
    the GEMVs read from host memory and fill themselves."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gu_miss = dn_miss = {}
    if misses is not None:
        assert gemm_fn is _decode_gemm_marlin, "only the marlin GEMVs read missing slots from host memory"
        plan, bank = misses
        for t, role in ((gate_up_packed, "gate_up"), (gate_up_scale, "gate_up_scale"), (down_packed, "down"),
                        (down_scale, "down_scale"), (gate_up_global, "gate_up_global"), (down_global, "down_global")):
            if role in bank:  # host and slot rows share one layout
                assert t[0].numel() * t.element_size() == bank[role][1] == t.stride(0) * t.element_size(), (role, bank)
        # the global scales' banks too when the GEMVs fill them (nvfp4_moe.MISS_GLOBALS: the kernel declares them)
        gu_miss = {"misses": plan, "banks": tuple(bank[r][0] for r in ("gate_up", "gate_up_scale", "gate_up_global")
                                                  if r in bank)}
        dn_miss = {"misses": plan, "banks": tuple(bank[r][0] for r in ("down", "down_scale", "down_global") if r in bank)}
    if (_ACT_IN_GEMV and gemm_fn is _decode_gemm_marlin and activation == "silu" and dt == torch.bfloat16
            and not is_flashinfer_installed()):
        # FREETOKEN_NVFP4_ACT_IN_GEMV: the gate | up GEMV and silu_and_mul in one launch, the bits of the Triton
        # activation (flashinfer's, used where it is installed, is another reference)
        nvfp4_moe.decode_gemm_marlin_act(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic2, topk_weights, topk_ids,
            apply_router_weight_on_input, **gu_miss,
        )
    else:
        ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
        gemm_fn(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
            ic1, topk_weights, topk_ids, apply_router_weight_on_input, False, **gu_miss,
        )
        gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True, **dn_miss,
    )
    if PER_EXPERT_OUTPUT[0]:  # the caller sums them in its own epilogue (FREETOKEN_FUSED_MOE_EPILOGUE)
        return ic3
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_decode_nvfp4_marlin(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    misses: tuple | None = None,
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit, misses=misses,
    )


def fused_experts_decode_nvfp4_serial(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Original LUT-gather decode (one program per route, full K reduction). Retained for
    A/B benchmarking against the marlin decode path; not on the production decode path."""
    return _fused_experts_decode_nvfp4(
        nvfp4_moe.decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


# FREETOKEN_NVFP4_PREFILL_TUNED=1: tiles swept on the 7900 XTX / XT for Qwen3.8-Flash-Next (512 experts, top-10,
# hidden 2560, intermediate 352 / 288 per rank; rdna3/bench/nvfp4_prefill_sweep.py). Only BLOCK_M / BLOCK_N / warps /
# stages change (BLOCK_KB stays 32), so each output keeps the same K accumulation: bit-identical results.
# The two GEMMs may use different BLOCK_M (then the tokens are aligned once per BLOCK_M). Swept again on upstream's
# rewritten prefill GEMM (one MoE block, interleaved, the same bits, each card on its own slice): 1600-2560 tokens
# 2.8-3.0x faster with 4 warps and 2 stages (XTX 9.7-10.3 -> 3.3-3.6 ms), 1280 1.4-1.5x, 32 1.3-1.4x; up to 6144 tokens
# the 64-row tile beats the 128-row one with tail tiles (6144: 9.1-9.2 -> 7.6 ms on both cards), from 7168 it does not.
NVFP4_PREFILL_TUNED = os.environ.get("FREETOKEN_NVFP4_PREFILL_TUNED", "0") == "1"
_TUNED_MID = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=4, num_stages=2)
_TUNED_BIG_GATE_UP = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=8,
                          num_stages=2)
# Short prefills (agent turns: ~5-25 rows an expert) run 4-warp tiles, bit-identical to the tiles they replace; on the
# prefill GEMM before upstream's rewrite, 7900 XTX / XT, random routing over 512 experts: 32-64 tokens 5-13x faster than
# the then-shipped 8-warp 4-stage tile, 128-512 tokens 1.7-2.5x and 768-1536 1.4-1.8x faster than the 8-warp 64-row one.
_TUNED_TINY_GATE_UP = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=1, num_warps=4,
                           num_stages=2)
_TUNED_TINY_DOWN = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32, GROUP_SIZE_M=1, num_warps=4, num_stages=2)
_TUNED_SMALL = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=1, num_warps=4, num_stages=1)
_TUNED_SHORT_GATE_UP = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=4,
                            num_stages=2)
_TUNED_SHORT_DOWN = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=4, num_stages=2)
# FREETOKEN_NVFP4_PREFILL_TAIL_TILES (default 1): on the 128-row tiles of a > 6144-token block, each expert's rows past
# its last full 128-row tile go to 32-row tiles of a second launch (both GEMMs), instead of a 128-row tile mostly
# padding (an expert gets ~160 rows of an 8192-token block). Only BLOCK_M / BLOCK_N / warps / stages change: the same
# bits.
# On the prefill GEMM before upstream's rewrite, 8192-token block, random routing, 7900 XTX / XT: gate_up 12.3 ->
# 9.4-9.9 ms, down (128-row tiles too) 7.0 -> 6.0-6.2 ms
_TAIL_TILES = os.environ.get("FREETOKEN_NVFP4_PREFILL_TAIL_TILES", "1") == "1"
_TAIL_GATE_UP = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=4, num_stages=2)
_TAIL_DOWN = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=8, num_stages=2)


def _prefill_configs(M: int, top_k: int, num_experts: int, device_index: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """(gate_up config, down config) for an M-token prefill block."""
    if not NVFP4_PREFILL_TUNED:
        cfg = nvfp4_moe.prefill_config(M, top_k, num_experts, device_index)
        return cfg, cfg
    if M <= 64:
        return _TUNED_TINY_GATE_UP, _TUNED_TINY_DOWN
    if M <= 640:
        return _TUNED_SMALL, _TUNED_SMALL
    if M <= 1536:
        return _TUNED_SHORT_GATE_UP, _TUNED_SHORT_DOWN
    if M > 6144:
        return _TUNED_BIG_GATE_UP, _TUNED_MID
    return _TUNED_MID, _TUNED_MID


# token block of the prefill MoE GEMMs (0 = the whole chunk at once); see fused_experts_nvfp4
NVFP4_PREFILL_BLOCK = int(os.environ.get("FREETOKEN_NVFP4_PREFILL_BLOCK", "4096"))


def fused_experts_nvfp4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
    slot_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill inline-NVFP4 MoE. ``topk_ids`` index rows of the bank tensors in
    ``[0, num_experts)``: full-layer banks with position == expert id (the
    materialized ``[:E]`` slot view or the overlap double buffer), raw ids; with ``slot_map`` (expert id -> slot) the
    banks are the slot cache and each expert's tiles read its slot.

    A chunk longer than NVFP4_PREFILL_BLOCK tokens runs block by block on the same banks: the [M, top_k, H]
    intermediate is ~840 MiB at a 16k-token chunk; every token's rows are independent, so the result is the same."""
    M, H = hidden_states.shape
    if M > NVFP4_PREFILL_BLOCK > 0:
        out = torch.empty_like(hidden_states)
        done = BLOCK_DONE[0]
        for start in range(0, M, NVFP4_PREFILL_BLOCK):
            end = min(start + NVFP4_PREFILL_BLOCK, M)
            out[start:end] = fused_experts_nvfp4(
                hidden_states[start:end], gate_up_packed, gate_up_scale, gate_up_global, down_packed, down_scale,
                down_global, topk_weights[start:end], topk_ids[start:end], num_experts, activation,
                apply_router_weight_on_input, act_alpha, act_limit, slot_map,
            )
            if done is not None:
                done(start, end, out)
        return out
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg_up, cfg_down = _prefill_configs(M, top_k, num_experts, hidden_states.device.index)

    slots = (lambda al: al) if slot_map is None else (lambda al: (al[0], blocks_to_slots(al[1], slot_map), al[2]))
    if _TAIL_TILES and cfg_up is _TUNED_BIG_GATE_UP:
        full, rest = (slots(al) for al in moe_align_split(topk_ids, 128, 32, num_experts))
        parts_up = ((full, _TUNED_BIG_GATE_UP), (rest, _TAIL_GATE_UP))
        parts_down = ((full, _TUNED_BIG_GATE_UP), (rest, _TAIL_DOWN))
    else:
        align_up = moe_align_block_size(topk_ids, cfg_up["BLOCK_SIZE_M"], num_experts)
        align_down = (align_up if cfg_down["BLOCK_SIZE_M"] == cfg_up["BLOCK_SIZE_M"]
                      else moe_align_block_size(topk_ids, cfg_down["BLOCK_SIZE_M"], num_experts))
        align_up = slots(align_up)
        align_down = align_up if cfg_down["BLOCK_SIZE_M"] == cfg_up["BLOCK_SIZE_M"] else slots(align_down)
        parts_up, parts_down = ((align_up, cfg_up),), ((align_down, cfg_down),)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    for align, cfg in parts_up:
        nvfp4_moe.prefill_gemm(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
            tw, *align, num_valid, top_k,
            apply_router_weight_on_input, cfg,
        )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    for align, cfg in parts_down:
        nvfp4_moe.prefill_gemm(
            ic2, down_packed, down_scale, down_global, ic3,
            tw, *align, num_valid, 1,
            not apply_router_weight_on_input, cfg,
        )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


def fused_experts_nvfp4_w4a4(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor,
    gate_up_scale: torch.Tensor,
    gate_up_global: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
    down_global: torch.Tensor,
    gate_up_input_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation: str = "silu",
    act_alpha: float = 1.702,
    act_limit: float = 7.0,
) -> torch.Tensor:
    """Prefill NVFP4 MoE with NVFP4 activations (W4A4) where the GPU and the batch favour it, else :func:`fused_experts_nvfp4`.

    ``*_input_scale`` are the per-expert static activation scales of the ``num_experts`` banks; each GEMM input is quantized under their max, as vLLM does.
    """
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    inter = gate_up_packed.shape[1] // 2
    cfg = nvfp4_moe.w4a4_prefill_config(M, top_k, num_experts, H, inter, activation, hidden_states.dtype, hidden_states.device.index)
    if cfg is None:
        return fused_experts_nvfp4(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, down_packed, down_scale, down_global,
            topk_weights, topk_ids, num_experts, activation, False, act_alpha, act_limit,
        )
    dev = hidden_states.device
    g = torch.stack((gate_up_input_scale[:num_experts], down_input_scale[:num_experts])).float().amax(dim=1)
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, cfg["BLOCK_SIZE_M"], num_experts)
    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()

    xq, xs = quantize_nvfp4(hidden_states, g[0:1])
    q = torch.empty((num_valid, inter // 2), device=dev, dtype=torch.uint8)
    qs = torch.empty((num_valid, inter // 16), device=dev, dtype=torch.float8_e4m3fn)
    nvfp4_moe.w4a4_gate_up_gemm(
        xq, xs, g[0:1], gate_up_packed, gate_up_scale, gate_up_global, q, qs, g[1:2], sorted_ids, expert_ids, ntpp,
        num_valid, top_k, activation, act_alpha, act_limit, cfg["gate_up"],
    )
    out = torch.zeros_like(hidden_states)
    nvfp4_moe.w4a4_down_gemm(
        q, qs, g[1:2], down_packed, down_scale, down_global, out, tw, sorted_ids, expert_ids, ntpp, num_valid, top_k,
        cfg["down"],
    )
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
    "fused_experts_nvfp4_w4a4",
]
