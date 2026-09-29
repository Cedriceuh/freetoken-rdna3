"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod


# rows (concurrent decoding requests) served by the row-looped GEMV; same knob and meaning as the int8 path
_GEMV_ROWS_MAX = max(1, int(os.environ.get("FREETOKEN_INT8_ROWS_MAX", "8")))


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            b = b.to(x.dtype) if b is not None else None
        if (b is None and x.is_cuda and x.dim() == 2 and 1 <= x.shape[0] <= _GEMV_ROWS_MAX
                and x.dtype is torch.bfloat16 and x.is_contiguous()):
            # decode on RDNA3 (one row per request, up to FREETOKEN_INT8_ROWS_MAX): a split-K Triton GEMV for the
            # shapes where it beats the BLAS pick; each row gets the bits it would get alone (batch-invariant)
            cfg = _decode_gemv_config(w)
            if cfg is not None:
                from freetoken.kernel.triton.dense_gemv import gemv_bf16

                return gemv_bf16(x, w, cfg)
        return F.linear(x, w, b)


@lru_cache(maxsize=None)
def _rdna3() -> bool:
    """The GEMV tile table was measured on gfx1100 (RDNA3): other ROCm GPUs (CDNA, RDNA4) keep F.linear."""
    from freetoken.kernel.backend import is_rocm

    if not is_rocm() or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0].startswith("gfx11")


def _decode_gemv_config(w: torch.Tensor):
    if not _rdna3():
        return None
    from freetoken.kernel.triton.dense_gemv import gemv_config

    return gemv_config(w)


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TorchLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)
