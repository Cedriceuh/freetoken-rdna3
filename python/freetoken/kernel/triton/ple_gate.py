"""The PLE gate after its dot products (FREETOKEN_PLE_GATE_FUSED): torch's eight bf16 elementwise launches, each rounding to
bf16 as it stores -- the division by sqrt(hidden) (torch multiplies by the reciprocal of a CPU scalar, computed in
fp32), sign, abs, clamp_min(1e-6), sqrt, the product, sigmoid (1 / (1 + exp(-x)) with the libm exp), the product with
the value row -- in one launch, op for op."""

from __future__ import annotations

import math

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _ple_gate_kernel(s_ptr, v_ptr, out_ptr, stride_s, stride_v, stride_o, inv, eps, HC: tl.constexpr,
                     HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    g = tl.load(s_ptr + t * stride_s + h).to(tl.float32)
    g = _bf16(g * inv)
    sgn = tl.where(g > 0, 1.0, tl.where(g < 0, -1.0, 0.0))
    a = _bf16(tl.maximum(tl.abs(g), eps))
    a = tl.where(tl.abs(g) != tl.abs(g), tl.abs(g), a)  # clamp keeps a NaN
    r = _bf16(tl.sqrt_rn(a))
    x = _bf16(sgn * r)
    gate = _bf16(1.0 / (1.0 + libdevice.exp(-x)))
    cols = tl.arange(0, BLOCK)
    for c0 in tl.static_range(0, HIDDEN, BLOCK):
        c = c0 + cols
        m = c < HIDDEN
        v = tl.load(v_ptr + t * stride_v + c, mask=m, other=0.0).to(tl.float32)
        tl.store(out_ptr + t * stride_o + h * HIDDEN + c, (gate * v).to(tl.bfloat16), mask=m)


def ple_gate(s: torch.Tensor, value: torch.Tensor, hidden: int) -> torch.Tensor:
    """``s`` [T, hc] (or [T, hc, 1]) the gate's dot products (bf16), divided by sqrt(``hidden``); ``value`` [T, W] ->
    ``gated`` [T, hc * W]."""
    s = s.reshape(s.shape[0], -1)
    T, hc = s.shape
    W = value.shape[1]
    assert value.stride(1) == 1 and s.dtype == value.dtype == torch.bfloat16
    out = torch.empty((T, hc * W), dtype=value.dtype, device=value.device)
    inv = float(np.float32(1.0) / np.float32(math.sqrt(hidden)))
    eps = float(np.float32(1e-6))
    _ple_gate_kernel[(T, hc)](s, value, out, s.stride(0), value.stride(0), out.stride(0), inv, eps, HC=hc,
                              HIDDEN=W, BLOCK=min(1024, triton.next_power_of_2(W)), num_warps=4)
    return out
