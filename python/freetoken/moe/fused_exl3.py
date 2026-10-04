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
from freetoken.moe.fused import moe_align_block_size

# token block of the prefill path: its rotated inputs are [tokens * top_k, 2, hidden] fp16 and its products fp32,
# 540 MiB of temporaries at 4096 tokens (top-10, hidden 2560, a 384-wide slice), which the cache planner keeps free
# (TritonExl3MoEKernel.prefill_workspace_bytes). Swept on the 7900 XT with block_m (8192 tokens through one layer,
# I=384): 1024 / 64 5.43 ms per 1k tokens, 4096 / 32 3.43-3.49, 4096 / 64 3.54-3.63, 8192 / 64 3.25 for twice
# the memory. With 32-row tiles, a 4096-token block still feeds every expert ~80 routes on average.
EXL3_PREFILL_BLOCK = int(os.environ.get("FREETOKEN_EXL3_PREFILL_BLOCK", "4096"))
_PREFILL_BLOCK_M = 32


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
) -> torch.Tensor:
    """``topk_ids`` index bank rows (cache slots in decode, the full-layer buffer's rows in prefill)."""
    M, _ = hidden_states.shape
    if is_prefill and M > EXL3_PREFILL_BLOCK > 0:
        out = torch.empty_like(hidden_states)
        for start in range(0, M, EXL3_PREFILL_BLOCK):
            end = min(start + EXL3_PREFILL_BLOCK, M)
            out[start:end] = fused_experts_exl3(
                hidden_states[start:end], gate_up, gate_up_suh, gate_up_svh, down, down_suh, down_svh,
                topk_weights[start:end], topk_ids[start:end], codebook=codebook, is_prefill=True,
                num_experts=num_experts,
            )
        return out
    top_k = topk_ids.shape[1]
    slots = topk_ids.reshape(-1)
    weights = topk_weights.contiguous()
    out = torch.empty_like(hidden_states)
    if is_prefill:
        assert num_experts is not None, "the prefill path sorts routes by expert: num_experts is required"
        sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, _PREFILL_BLOCK_M, num_experts)
        a1 = exl3_rotate_in(hidden_states, gate_up_suh, slots, top_k, dtype=torch.float16)
        c1 = exl3_moe_gemm(a1, gate_up, sorted_ids, expert_ids, ntpp, codebook, block_m=_PREFILL_BLOCK_M)
        del a1
        a2 = exl3_mid(c1, slots, gate_up_svh, down_suh, dtype=torch.float16)
        del c1
        # the rotation out runs as the down GEMM's epilogue (a program owns a 128-column block), then a plain sum
        y = exl3_gemm_down_out(a2, down, down_svh, weights.reshape(-1), sorted_ids, expert_ids, ntpp, codebook,
                               block_m=_PREFILL_BLOCK_M, out_dtype=hidden_states.dtype)
        moe_sum_reduce_triton(y.view(M, top_k, -1), out)
        return out
    a1 = exl3_rotate_in(hidden_states, gate_up_suh, slots, top_k)
    c1 = exl3_gemv(a1, gate_up, slots, codebook)
    a2 = exl3_mid(c1, slots, gate_up_svh, down_suh)
    c2 = exl3_gemv(a2, down, slots, codebook)
    return exl3_out(c2, slots, down_svh, weights, out)


__all__ = ["EXL3_PREFILL_BLOCK", "fused_experts_exl3"]
