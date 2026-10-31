"""FREETOKEN_INT8_EXACT_GEMV: four int8 decode GEMV tiles of kernel/triton/int8_gemv.py at 1-8 rows as HIP kernels with
the Triton kernels' sums (csrc/jit/int8_gemv_exact.cuh): the same bits, fewer instructions a row."""
from __future__ import annotations

from functools import lru_cache

import torch

from .utils import load_jit


@lru_cache(maxsize=None)
def _module():
    return load_jit("int8_gemv_exact", "v8", cuda_files=["int8_gemv_exact.cuh"],
                    cuda_wrappers=[("run", "&Int8GemvExact::run"), ("run_k2048", "&Int8GemvExactK2048::run"),
                                   ("run_n4w8", "&Int8GemvExactN4W8::run"),
                                   ("run_hc_mix", "&Int8HcMixExact::run")])


def gemv_int8_exact(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The (2, 4096, 1, 4) one-pass tile (the GDN projections, the attention output) at 1-8 rows."""
    M = x.shape[0]
    out = torch.empty((M, qweight.shape[0]), dtype=torch.bfloat16, device=x.device)
    _module().run(x, qweight, scale, out)
    return out


def gemv_int8_exact_k2048(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, silu_hc: int = 0,
                          silu_cols: int = 0) -> torch.Tensor:
    """The (1, 2048, 1, 8) tile (the HC down projection) at 1-8 rows, with its hc_silu epilogue."""
    M = x.shape[0]
    out = torch.empty((M, qweight.shape[0]), dtype=torch.bfloat16, device=x.device)
    _module().run_k2048(x, qweight, scale, out, silu_hc, silu_cols)
    return out


def gemv_int8_exact_n4w8(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The (4, 4096, 1, 8) one-pass tile (attention qkv) at 1-8 rows."""
    M = x.shape[0]
    out = torch.empty((M, qweight.shape[0]), dtype=torch.bfloat16, device=x.device)
    _module().run_n4w8(x, qweight, scale, out)
    return out


def gemv_int8_hc_mix_exact(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor,
                           rn: torch.Tensor) -> torch.Tensor:
    """gemv_int8_hc_mix (4 streams) at 1-8 rows, the same bits."""
    M = x.shape[0]
    out = torch.empty((M, qweight.shape[0] // 4), dtype=torch.bfloat16, device=x.device)
    _module().run_hc_mix(x, qweight, scale, rn, out)
    return out
