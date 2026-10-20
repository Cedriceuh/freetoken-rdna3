"""Host orchestration for the inline-dequant NVFP4 fused-MoE path.

Mirrors :mod:`freetoken.moe.fused` (gemm1 -> act -> gemm2 -> sum-reduce) but the two
grouped GEMMs read the NVFP4 expert cache directly and dequantize inside the K-loop,
so no BF16 copy of the experts is ever materialized.
"""

from __future__ import annotations

import os

from typing import Any, Dict

import torch
import triton
import triton.language as tl

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.kernel.triton.nvfp4_fused_moe import (
    _decode_nvfp4_marlin_kernel,
    _decode_nvfp4_moe_kernel,
    _e2m1_lut,
    _prefill_nvfp4_moe_kernel,
)
from freetoken.layers import gated_act_and_mul
from freetoken.kernel.triton.moe_align import moe_align_split
from freetoken.moe.fused import blocks_to_slots, moe_align_block_size
from freetoken.moe.prefill_hooks import BLOCK_DONE
from freetoken.utils import init_logger

logger = init_logger(__name__)

# set by a caller that sums the decode experts' outputs itself (one element: mutable, read at call time)
PER_EXPERT_OUTPUT = [False]
# FREETOKEN_HYBRID_SKIP_CPU_ROUTES=0: under --moe-strategy hybrid the decode GEMV still runs the routes the CPU computes
# (on slot 0, weight 0) instead of writing zeros for them without a read
SKIP_CPU_ROUTES = os.environ.get("FREETOKEN_HYBRID_SKIP_CPU_ROUTES", "1") != "0"

# Decode is captured into a CUDA graph, so the config must be fixed (no triton.autotune,
# which benchmarks at run time). Tuned offline against the NVFP4 decode kernels.
# These drive the original LUT-gather decode (_decode_gemm), kept only for A/B.
_DECODE_BLOCK_N = 64
_DECODE_BLOCK_KB = 128
_DECODE_WARPS = 4

# Marlin-style decode: every K-loop load must keep the word tile's register layout, so BLOCK_N * BLOCK_KW <= 128 * num_warps and BLOCK_KW <= 32 at 4 warps.
# Long K (gate_up) uses the narrow N tile for more programs at M=1; short K (down) uses the wide one.
_DECODE_MARLIN_BLOCK_N = 16
_DECODE_MARLIN_BLOCK_KW = 16
_DECODE_MARLIN_UNROLL = 2
_DECODE_MARLIN_WARPS = 4
_DECODE_MARLIN_DEEPK_BLOCK_N = 8
_DECODE_MARLIN_DEEPK_BLOCK_KW = 32
_DECODE_MARLIN_DEEPK_UNROLL = 2
_DECODE_MARLIN_DEEPK_THRESHOLD = 2048
# FREETOKEN_NVFP4_DECODE_TUNED=1: decode tiles swept on the 7900 XTX / XT for Qwen3.8-Flash-Next (per rank at
# FREETOKEN_TP_SPLIT=0.55: gate_up N 704 / 576, K 2560; down N 2560, K 352 / 288) inside a 48-launch CUDA graph
# (rdna3/bench/nvfp4_decode_sweep.py). Every entry is bit-identical to the shipped tiles. (N, K) -> (tiles for one
# decoding request, tiles for two or more): (BLOCK_N, BLOCK_KW, num_warps, UNROLL). gate_up keeps the shipped tiles,
# the best of the sweep; down takes one trip per tile (3 tiles of 16 words, which UNROLL 2 pads to 4): 1 request
# 14.7 -> 13.9 us (XTX) / 14.8 -> 13.4 us (XT), 2-4 requests x1.10-1.19.
NVFP4_DECODE_TUNED = os.environ.get("FREETOKEN_NVFP4_DECODE_TUNED", "0") == "1"
_DECODE_UNTUNED_SEEN: set[tuple[int, int]] = set()
_DECODE_TUNED: dict[tuple[int, int], tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = {
    (704, 2560): ((8, 32, 4, 2), (8, 32, 4, 2)),
    (576, 2560): ((8, 32, 4, 2), (8, 32, 4, 2)),
    (2560, 352): ((16, 16, 4, 1), (16, 16, 4, 1)),
    (2560, 288): ((16, 16, 2, 1), (16, 16, 2, 1)),
}


def _tl_dtype(dt: torch.dtype):
    if dt == torch.bfloat16:
        return tl.bfloat16
    if dt == torch.float16:
        return tl.float16
    return tl.float32


def _decode_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    grid = (total_routes, triton.cdiv(N, _DECODE_BLOCK_N))
    _decode_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights, topk_ids,
        _e2m1_lut(a.device.index),
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=_DECODE_BLOCK_N,
        BLOCK_SIZE_KB=_DECODE_BLOCK_KB,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        num_warps=_DECODE_WARPS,
    )


def _decode_gemm_marlin(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    a_row_is_route: bool,
) -> None:
    """Marlin-style decode GEMV: int32 wide loads + deferred reduction
    (:func:`_decode_nvfp4_marlin_kernel`). ``packed`` is the uint8 ``[S, N, K//2]`` bank;
    it is reinterpreted as int32 ``[S, N, K//8]`` (contiguous, K%8==0 for NVFP4)."""
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    packed_i32 = packed.view(torch.int32)  # [S, N, K // 8]
    scale = e4m3_kernel_view(scale)
    total_routes = M * top_k
    deep_k = K >= _DECODE_MARLIN_DEEPK_THRESHOLD
    block_n = _DECODE_MARLIN_DEEPK_BLOCK_N if deep_k else _DECODE_MARLIN_BLOCK_N
    block_kw = _DECODE_MARLIN_DEEPK_BLOCK_KW if deep_k else _DECODE_MARLIN_BLOCK_KW
    unroll = _DECODE_MARLIN_DEEPK_UNROLL if deep_k else _DECODE_MARLIN_UNROLL
    warps = _DECODE_MARLIN_WARPS
    tuned = _DECODE_TUNED.get((N, K)) if NVFP4_DECODE_TUNED else None
    if NVFP4_DECODE_TUNED and tuned is None and (N, K) not in _DECODE_UNTUNED_SEEN:
        _DECODE_UNTUNED_SEEN.add((N, K))  # the table is per (N, K) of one model and split: say when it does nothing
        logger.info(f"FREETOKEN_NVFP4_DECODE_TUNED: no tuned tiles for (N, K) = ({N}, {K}), shipped config used")
    if tuned is not None:
        block_n, block_kw, warps, unroll = tuned[0] if M == 1 else tuned[1]
    # int64 activation loads need contiguous 16-bit rows that start 16-byte aligned
    a_wide = (
        a.dtype in (torch.bfloat16, torch.float16)
        and a.stride(1) == 1 and a.stride(0) % 8 == 0 and a.data_ptr() % 16 == 0
    )
    grid = (total_routes, triton.cdiv(N, block_n))
    _decode_nvfp4_marlin_kernel[grid](
        a, packed_i32, scale, glob, c, topk_weights, topk_ids,
        total_routes, N, K,
        a.stride(0), a.stride(1),
        packed_i32.stride(0), packed_i32.stride(1), packed_i32.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        topk_weights.stride(0), topk_weights.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_KW=block_kw,
        UNROLL=unroll,
        TOP_K=top_k,
        A_ROW_IS_ROUTE=a_row_is_route,
        A_WIDE=a_wide,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=_tl_dtype(c.dtype),
        SKIP_INACTIVE=SKIP_CPU_ROUTES,
        num_warps=warps,
    )


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
) -> torch.Tensor:
    """Shared decode body (gemm1 -> act -> gemm2 -> sum-reduce); ``gemm_fn`` is either
    the marlin-style int32 GEMV (:func:`_decode_gemm_marlin`) or the original LUT-gather
    GEMV (:func:`_decode_gemm`), both with the same calling convention."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype

    ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
    gemm_fn(
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        ic1, topk_weights, topk_ids, apply_router_weight_on_input, False,
    )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    gemm_fn(
        ic2, down_packed, down_scale, down_global,
        ic3, topk_weights, topk_ids, not apply_router_weight_on_input, True,
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
) -> torch.Tensor:
    """Decode inline-NVFP4 MoE using the Marlin-style int32 wide-load GEMV."""
    return _fused_experts_decode_nvfp4(
        _decode_gemm_marlin,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
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
        _decode_gemm,
        hidden_states, gate_up_packed, gate_up_scale, gate_up_global,
        down_packed, down_scale, down_global,
        topk_weights, topk_ids, activation, apply_router_weight_on_input,
        act_alpha, act_limit,
    )


# FREETOKEN_NVFP4_PREFILL_TUNED=1: tiles swept on the 7900 XTX / XT for Qwen3.8-Flash-Next (512 experts, top-10,
# hidden 2560, intermediate 352 / 288 per rank; rdna3/bench/nvfp4_prefill_sweep.py). Only BLOCK_M / BLOCK_N / warps /
# stages change (BLOCK_KB stays 32), so each output keeps the same K accumulation: bit-identical results.
# Measured per MoE layer, 4096-token block: gate_up 12.1 -> 6.6 ms, down 7.2 -> 4.8 ms; 1536 tokens: 5.3 -> 4.1 and
# 3.2 -> 2.4 ms. The two GEMMs may use different BLOCK_M (then the tokens are aligned once per BLOCK_M).
NVFP4_PREFILL_TUNED = os.environ.get("FREETOKEN_NVFP4_PREFILL_TUNED", "0") == "1"
_TUNED_MID = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=8, num_stages=1)
_TUNED_BIG_GATE_UP = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=8,
                          num_stages=2)
# Short prefills (agent turns: ~5-25 rows an expert) run 4-warp tiles: per GEMM, bit-identical to the tiles they
# replace, 7900 XTX / XT, random routing over 512 experts: 32-64 tokens 5-13x faster than the shipped 8-warp 4-stage
# tile, 128-512 tokens 1.7-2.5x and 768-1536 1.4-1.8x faster than _TUNED_MID; past ~2048 tokens within 9 %, kept.
_TUNED_TINY_GATE_UP = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=1, num_warps=4,
                           num_stages=2)
_TUNED_TINY_DOWN = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32, GROUP_SIZE_M=1, num_warps=4, num_stages=1)
_TUNED_SMALL = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=1, num_warps=4, num_stages=1)
_TUNED_SHORT_GATE_UP = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=4,
                            num_stages=2)
_TUNED_SHORT_DOWN = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=4, num_stages=1)
# FREETOKEN_NVFP4_PREFILL_TAIL_TILES (default 1): on the 128-row tiles of a >= 3072-token block, each expert's rows past
# its last full 128-row tile go to 32-row tiles of a second launch (both GEMMs), instead of a 128-row tile mostly
# padding (an expert gets ~160 rows of an 8192-token block). Only BLOCK_M / BLOCK_N / warps / stages change: the same
# bits.
# 8192-token block, random routing, 7900 XTX / XT: gate_up 12.3 -> 9.4-9.9 ms, down (128-row tiles too) 7.0 ->
# 6.0-6.2 ms; the whole MoE block with this model's routing (96 dumped 8192-token blocks): 1921 -> 1696 ms (XTX's
# slice), 1892 -> 1682 ms (XT's)
_TAIL_TILES = os.environ.get("FREETOKEN_NVFP4_PREFILL_TAIL_TILES", "1") == "1"
_TAIL_GATE_UP = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=4, num_stages=2)
_TAIL_DOWN = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_KB=32, GROUP_SIZE_M=8, num_warps=8, num_stages=2)


def _prefill_configs(M: int) -> tuple[Dict[str, int], Dict[str, int]]:
    """(gate_up config, down config) for an M-token prefill block."""
    if not NVFP4_PREFILL_TUNED:
        cfg = _prefill_config(M)
        return cfg, cfg
    if M <= 64:
        return _TUNED_TINY_GATE_UP, _TUNED_TINY_DOWN
    if M <= 640:
        return _TUNED_SMALL, _TUNED_SMALL
    if M <= 1536:
        return _TUNED_SHORT_GATE_UP, _TUNED_SHORT_DOWN
    if M >= 3072:
        return _TUNED_BIG_GATE_UP, _TUNED_MID
    return _TUNED_MID, _TUNED_MID


def _prefill_config(M: int) -> Dict[str, int]:
    # ``BLOCK_SIZE_M`` is coupled to host-side ``moe_align_block_size`` (token padding),
    # so it cannot be picked by triton.autotune; these were chosen by an offline sweep
    # over (BLOCK_M, BLOCK_N, BLOCK_KB, num_warps, num_stages) for the MiniMax-M2 shapes.
    if M <= 64:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                    GROUP_SIZE_M=1, num_warps=8, num_stages=4)
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_KB=32,
                GROUP_SIZE_M=8, num_warps=8, num_stages=4)


def _prefill_gemm(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    glob: torch.Tensor,
    c: torch.Tensor,
    topk_weights_flat: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    kernel_top_k: int,
    mul_routed_weight: bool,
    cfg: Dict[str, Any],
) -> None:
    N = packed.shape[1]
    K = packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = e4m3_kernel_view(scale)
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _prefill_nvfp4_moe_kernel[grid](
        a, packed, scale, glob, c, topk_weights_flat, sorted_ids, expert_ids,
        num_tokens_post_padded,
        _e2m1_lut(a.device.index),
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2),
        scale.stride(0), scale.stride(1), scale.stride(2),
        glob.stride(0), glob.stride(1),
        c.stride(1), c.stride(2),
        topk_weights_flat.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype),
        **cfg,
    )


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
    cfg_up, cfg_down = _prefill_configs(M)

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
        _prefill_gemm(
            hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1,
            tw, *align, num_valid, top_k,
            apply_router_weight_on_input, cfg,
        )
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    for align, cfg in parts_down:
        _prefill_gemm(
            ic2, down_packed, down_scale, down_global, ic3,
            tw, *align, num_valid, 1,
            not apply_router_weight_on_input, cfg,
        )
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


__all__ = [
    "fused_experts_decode_nvfp4_marlin",
    "fused_experts_decode_nvfp4_serial",
    "fused_experts_nvfp4",
]
