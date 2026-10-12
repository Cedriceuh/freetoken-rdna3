"""The PLE's m-row decode conv (``PLELayer._spec_decode_conv``) in one kernel instead of ~6 torch ops a row: the
taps of each row read off the sequence [state | rows], the snapshots after every row but the last and the rolled state
stored. The bits of the torch form: fp32 products rounded on their own (no FMA contraction, ``enable_fp_fusion=False``)
and summed as (p0 + p1) + (p2 + p3), the order torch's ``sum(-1)`` over 4 takes on gfx1100
(tests/kernels/test_decode_glue.py); SiLU stays a torch op."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _seq_load(state, rows, b, c, mask, i: tl.constexpr, L: tl.constexpr, s_sb, s_sc, s_rb, s_rm):
    if i < L:
        v = tl.load(state + b * s_sb + c * s_sc + i, mask=mask, other=0.0)
    else:
        v = tl.load(rows + b * s_rb + (i - L) * s_rm + c, mask=mask, other=0.0)
    return v


@triton.jit
def _ple_spec_conv_kernel(state, rows, weight, out, snaps, states, slots, C, s_sb, s_sc, s_rb, s_rm, s_nj, s_nb,
                          s_nc, s_tb, s_tc, L: tl.constexpr, M: tl.constexpr, K: tl.constexpr, DIL: tl.constexpr,
                          BLOCK: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < C
    w0 = tl.load(weight + c * K + 0, mask=mask, other=0.0)
    w1 = tl.load(weight + c * K + 1, mask=mask, other=0.0)
    w2 = tl.load(weight + c * K + 2, mask=mask, other=0.0)
    w3 = tl.load(weight + c * K + 3, mask=mask, other=0.0)
    for j in tl.static_range(M):
        p0 = _seq_load(state, rows, b, c, mask, j, L, s_sb, s_sc, s_rb, s_rm).to(tl.float32) * w0
        p1 = _seq_load(state, rows, b, c, mask, j + DIL, L, s_sb, s_sc, s_rb, s_rm).to(tl.float32) * w1
        p2 = _seq_load(state, rows, b, c, mask, j + 2 * DIL, L, s_sb, s_sc, s_rb, s_rm).to(tl.float32) * w2
        p3 = _seq_load(state, rows, b, c, mask, j + 3 * DIL, L, s_sb, s_sc, s_rb, s_rm).to(tl.float32) * w3
        acc = (p0 + p1) + (p2 + p3)
        tl.store(out + (b * M + j) * C + c, acc.to(out.dtype.element_ty), mask=mask)
        if j < M - 1:  # the state after row j: sequence columns j+1 .. j+L
            for i in tl.static_range(L):
                v = _seq_load(state, rows, b, c, mask, j + 1 + i, L, s_sb, s_sc, s_rb, s_rm)
                tl.store(snaps + j * s_nj + b * s_nb + c * s_nc + i, v, mask=mask)
    slot = tl.load(slots + b).to(tl.int64)
    for i in tl.static_range(L):
        v = _seq_load(state, rows, b, c, mask, M + i, L, s_sb, s_sc, s_rb, s_rm)
        tl.store(states + slot * s_tb + c * s_tc + i, v, mask=mask)



@functools.cache
def bits_checked(device: torch.device) -> bool:
    """Whether this GPU sums like the kernel: gfx11, where torch's ``sum(-1)`` over 4 values was checked bit for bit."""
    return torch.version.hip is not None and torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx11")


def ple_spec_conv(state: torch.Tensor, rows: torch.Tensor, weight: torch.Tensor, snaps: torch.Tensor,
                  states: torch.Tensor, slots: torch.Tensor, dilation: int) -> torch.Tensor:
    """``state [B, C, L]``, ``rows [B, M, C]`` (same dtype as ``states``), ``weight [C, K]`` fp32; returns the pre-SiLU
    conv output ``[B * M, C]`` in the rows' dtype; writes ``snaps[j, b]`` for j < M - 1 and ``states[slots[b]]``."""
    B, C, L = state.shape
    M = rows.shape[1]
    K = weight.shape[1]
    assert L == (K - 1) * dilation and K == 4 and state.stride(2) == 1 and rows.stride(2) == 1
    assert snaps.stride(3) == 1 and states.stride(2) == 1 and weight.is_contiguous()
    out = torch.empty(B * M, C, dtype=rows.dtype, device=rows.device)
    BLOCK = 256
    _ple_spec_conv_kernel[(B, triton.cdiv(C, BLOCK))](
        state, rows, weight, out, snaps, states, slots, C, state.stride(0), state.stride(1), rows.stride(0),
        rows.stride(1), snaps.stride(0), snaps.stride(1), snaps.stride(2), states.stride(0), states.stride(1),
        L=L, M=M, K=K, DIL=dilation, BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False)
    return out


__all__ = ["bits_checked", "ple_spec_conv"]
