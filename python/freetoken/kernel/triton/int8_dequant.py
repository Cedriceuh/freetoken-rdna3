"""Per-row int8 -> bf16 dequantization for a prefill's dense projections (FREETOKEN_INT8_PREFILL_DEQUANT).

``torch.mul(int8, fp32 scale[:, None], out=bf16)`` computes each element in fp32 and rounds it once to bf16; this kernel
does the same arithmetic (one fp32 product, round to nearest even) in one pass over the int8 bytes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_dequant_kernel(q_ptr, scale_ptr, out_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    m = (rows < N)[:, None] & (k < K)[None, :]
    s = tl.load(scale_ptr + rows, mask=rows < N, other=0.0)
    off = rows.to(tl.int64)[:, None] * K + k[None, :]
    q = tl.load(q_ptr + off, mask=m, other=0)
    tl.store(out_ptr + off, (q.to(tl.float32) * s[:, None]).to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=m)


def dequant_int8_rows(q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out[n, k] = bf16(fp32(q[n, k]) * scale[n])`` for contiguous ``q`` [N, K] int8, ``scale`` [N] fp32."""
    N, K = q.shape
    BLOCK_K = min(2048, triton.next_power_of_2(K))
    BLOCK_N = 2048 // BLOCK_K
    _int8_dequant_kernel[(triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))](q, scale, out, N, K, BLOCK_N=BLOCK_N,
                                                                              BLOCK_K=BLOCK_K, num_warps=4)
    return out
