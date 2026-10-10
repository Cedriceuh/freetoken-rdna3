"""Output gate of a gated attention: ``o * sigmoid(gate)`` in one kernel, reading the gate in place.

The gate is the second half of each head of the fused q|gate projection ([T, heads, 2 * head_dim]), so the eager
form needed a copy (``reshape`` of a strided slice), a sigmoid and a multiply: three launches. Same bits: the
sigmoid is rounded to bf16 before the product, as torch materializes it (1 / (1 + exp(-x)) in fp32 matches
torch.sigmoid on all 65280 finite bf16 inputs), and the product of two bf16 values is exact in fp32 before its rounding.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_mul_kernel(O, G, OUT, H: tl.constexpr, D: tl.constexpr, so_t, so_h, sg_t, sg_h,
                             BLOCK: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    o = tl.load(O + t.to(tl.int64) * so_t + h * so_h + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + t.to(tl.int64) * sg_t + h * sg_h + cols, mask=mask, other=0.0).to(tl.float32)
    s = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + (t.to(tl.int64) * H + h) * D + cols, (o * s).to(tl.bfloat16), mask=mask)


def sigmoid_gate_mul(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``o`` [T, H, D] (any strides, unit last), ``gate`` [T, H, D] (any strides, unit last), bf16 -> [T, H * D]
    contiguous: ``o * sigmoid(gate)`` with torch's bits."""
    T, H, D = gate.shape
    assert o.shape == gate.shape and o.dtype == gate.dtype == torch.bfloat16
    assert o.stride(-1) == 1 and gate.stride(-1) == 1
    out = torch.empty((T, H * D), dtype=torch.bfloat16, device=o.device)
    if T:
        _sigmoid_gate_mul_kernel[(T, H)](o, gate, out, H, D, o.stride(0), o.stride(1), gate.stride(0), gate.stride(1),
                                         BLOCK=triton.next_power_of_2(D), num_warps=1 if D <= 256 else 2)
    return out


__all__ = ["sigmoid_gate_mul"]
