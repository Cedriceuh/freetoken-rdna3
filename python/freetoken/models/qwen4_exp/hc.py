"""Hyper-connection (gated residual) blocks for Qwen3.8-Flash-Next.

Every layer reads and writes ``hc_count`` residual streams packed as ``R [T, hc_count*hidden]``
(stream outer, hidden inner -- the checkpoint layout). On CUDA the mix/combine bodies are the
vendored vLLM Triton kernels (``kernel/triton/hc.py``: grouped_gemma_rmsnorm / hc_silu /
hc_gate_mix / hc_combine) around two ``F.linear`` GEMMs, or on int8 decode rows the norm and two
GEMVs that apply hc_silu and hc_gate_mix themselves; the pure-torch chain stays as the CPU
path and as the reference the kernels are diffed against. Both keep fp32 intermediates and cast
back at the store, so they agree to fp32 rounding.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import torch
import torch.nn.functional as F
from freetoken.kernel.triton.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_rmsnorm,
    hc_gate_mix,
    hc_silu,
)
from freetoken.layers import BaseOP, LinearReplicated

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# FREETOKEN_HC_INPLACE (default 1): a combine writes the new streams over the old ones, which nothing reads after it
# (the same bits; one [T, hc * hidden] buffer less: 320 MiB at a 16k-token prefill chunk)
_HC_INPLACE = os.environ.get("FREETOKEN_HC_INPLACE", "1") == "1"
# FREETOKEN_HC_SILU_EPILOGUE=0: hc_silu as its own kernel between the int8 down and up projections
# (GatedResidual._mix_int8)
_HC_SILU_EPILOGUE = os.environ.get("FREETOKEN_HC_SILU_EPILOGUE", "1") != "0"
# FREETOKEN_HC_MIX_IN_GEMV=0: hc_gate_mix as its own kernel after the int8 up projection (GatedResidual._mix_int8)
_HC_MIX_IN_GEMV = os.environ.get("FREETOKEN_HC_MIX_IN_GEMV", "1") != "0"


def grouped_plus_one_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    """RMSNorm each of ``num_groups`` equal slices of the last dim on its own fp32 statistic, then scale by (1+w)."""
    xf = x.float().unflatten(-1, (num_groups, -1))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf.flatten(-2) * (1.0 + weight.float())).to(x.dtype)


class GroupedPlusOneRMSNorm(BaseOP):
    """Per-stream RMSNorm of an ``[..., num_groups*group]`` tensor with one weight element per feature.

    HF ``Qwen4ExpTextRMSNorm(dim, group_size)``. The checkpoint weight is zero-centered and is
    loaded RAW: (1+w) is applied at runtime in fp32, never folded into the bf16 weight (the
    vendored Triton kernel does the same). ``ple.py`` reuses this class for norm_key /
    norm_query / norm_conv, so keep it exported.
    """

    def __init__(self, size: int, eps: float, num_groups: int) -> None:
        self.weight = torch.empty(size)
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the kernel is 2D-only, higher-rank callers keep the torch chain
        if x.is_cuda and x.dim() == 2:
            return grouped_gemma_rmsnorm(x, self.weight, self.eps, self.num_groups)
        return grouped_plus_one_rms_norm(x, self.weight, self.eps, self.num_groups)


class GatedResidual(BaseOP):
    """One hyper-connection block: ``mix`` reads the residual streams, ``combine`` writes a block output back.

    Frozen API (HF ``Qwen4ExpTextGatedResidual``, formulas at modeling_qwen4_exp.py:959-969)::

        x, s = hc.mix(R)          # R [T, hc_count*hidden] -> x [T, hidden], s [T, hc_count] or None
        y    = block(x)           # attention / GDN / MoE, plain [T, hidden] -> [T, hidden]
        R    = hc.combine(R, y, s)

        Rn      = groupRMSNorm(R) * (1 + hc_norm.weight)        # per hidden-size stream, fp32 stats
        lora, s = input_mix_weight_down_block_inject(Rn)        # merged GEMM: [lowrank | hc_count | pad]
        gate    = input_mix_weight_up(silu(lora / hc_count))
        x       = mean_i(sigmoid(gate_i) * Rn_i)
        R'_i    = R_i + 2*sigmoid(s_i / hc_count) * y

    ``s`` is the RAW inject logit slice of the merged GEMM (pre 2*sigmoid), which is what the
    vendored ``hc_combine`` kernel expects; ``combine`` applies the activation. The merged weight
    is ``[lowrank + hc_count + pad, hc_count*hidden]`` (Qwen3.8: 320 + 4 + 12 = 336 rows), the pad
    rows are zero and their GEMM output is dropped. ``use_combine=False`` is the top-level mixer:
    it owns the unmerged ``input_mix_weight_down``, returns ``s = None`` and has no ``combine``.

    Weight keys (checkpoint names, prefix stripped): ``hc_norm.weight``,
    ``input_mix_weight_down_block_inject.weight`` (loader: concat of
    ``input_mix_weight_down`` [lowrank, hc*hidden], ``block_inject_weight`` [hc_count, hc*hidden]
    and ``pad`` zero rows), ``input_mix_weight_up.weight``.

    Launch budget on CUDA: ``mix`` is 3 kernels around 2 GEMMs (on int8 decode rows, the norm and 2
    GEMVs), ``combine`` is 1.
    """

    def __init__(self, config: ModelConfig, use_combine: bool = True, *, prefix: str = "") -> None:
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.lowrank = args.hc_lowrank
        self.use_combine = use_combine
        width = args.ple_state_width
        self.hc_norm = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        if use_combine:
            # 16-row alignment for the merged skinny GEMM (vLLM hyperconnection.py:98)
            self.pad_size = (-(self.lowrank + self.hc_count)) % 16
            self.input_mix_weight_down_block_inject = LinearReplicated(
                width, self.lowrank + self.hc_count + self.pad_size, has_bias=False,
                quant_config=config.quant, prefix=f"{prefix}.input_mix_weight_down_block_inject",
            )
        else:
            self.pad_size = 0
            self.input_mix_weight_down = LinearReplicated(
                width, self.lowrank, has_bias=False,
                quant_config=config.quant, prefix=f"{prefix}.input_mix_weight_down",
            )
        self.input_mix_weight_up = LinearReplicated(
            self.lowrank, width, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.input_mix_weight_up",
        )

    def _down(self, rn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Run the down GEMM and split off the raw inject logits; the pad columns are dropped."""
        if not self.use_combine:
            return self.input_mix_weight_down.forward(rn), None
        down = self.input_mix_weight_down_block_inject.forward(rn)
        # both slices keep unit inner stride, so the kernels read them without a copy
        return down[:, : self.lowrank], down[:, self.lowrank : self.lowrank + self.hc_count]

    def _mix_kernel(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        return self.mix_normed(grouped_gemma_rmsnorm(R, self.hc_norm.weight, self.hc_norm.eps, self.hc_count))

    def mix_normed(self, rn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """``mix`` from the already normed streams (``combine_norm`` of the block before computed them)."""
        fused = self._mix_int8(rn)
        if fused is not None:
            return fused
        lora, s = self._down(rn)
        gate = self.input_mix_weight_up.forward(hc_silu(lora, self.hc_count))
        return hc_gate_mix(rn, gate, self.hc_count), s

    def _mix_int8(self, rn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None] | None:
        """Int8 decode rows: the down GEMV stores the lora columns through ``hc_silu`` (FREETOKEN_HC_SILU_EPILOGUE) and
        the up GEMV reads them in place and mixes its gates into the streams as it goes (FREETOKEN_HC_MIX_IN_GEMV), the
        same bits, two kernels less per hyper-connection; None otherwise."""
        if not (_HC_SILU_EPILOGUE and rn.is_cuda and rn.dim() == 2 and rn.dtype is torch.bfloat16
                and rn.stride(-1) == 1):
            return None
        from freetoken.kernel.triton.int8_gemv import ROWS_MAX, gemv_int8, gemv_int8_hc_mix
        from freetoken.layers.quantization.int8_weight_only import Int8WeightOnlyLinearMethod

        down = self.input_mix_weight_down_block_inject if self.use_combine else self.input_mix_weight_down
        up = self.input_mix_weight_up
        if not all(isinstance(op.quant_method, Int8WeightOnlyLinearMethod) and op.bias is None for op in (down, up)):
            return None
        m = rn.shape[0]
        if not (m == 1 or (1 < m <= ROWS_MAX and rn.is_contiguous())):  # the rows Int8WeightOnlyLinearMethod takes
            return None
        out = gemv_int8(rn, down.weight, down.weight_scale, silu_hc=self.hc_count, silu_cols=self.lowrank)
        s = out[:, self.lowrank : self.lowrank + self.hc_count] if self.use_combine else None
        lora = out[:, : self.lowrank]
        mixed = gemv_int8_hc_mix(lora, up.weight, up.weight_scale, rn, self.hc_count) if _HC_MIX_IN_GEMV else None
        if mixed is None:
            mixed = hc_gate_mix(rn, gemv_int8(lora, up.weight, up.weight_scale), self.hc_count)
        return mixed, s

    def _mix_torch(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        rn = grouped_plus_one_rms_norm(R, self.hc_norm.weight, self.hc_norm.eps, self.hc_count)
        lora, s = self._down(rn)
        lora = F.silu(lora.float() / self.hc_count)
        gate = self.input_mix_weight_up.forward(lora.to(R.dtype))
        mixed = torch.sigmoid(gate.float()).unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = mixed * rn.float().unflatten(-1, (self.hc_count, self.hidden_size))
        return mixed.mean(-2).to(R.dtype), s

    def _combine_torch(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        inject = 2.0 * torch.sigmoid(s.float() / self.hc_count)
        out = R.float().unflatten(-1, (self.hc_count, self.hidden_size))
        out = out + y.float().unsqueeze(-2) * inject.unsqueeze(-1)
        return out.flatten(-2).to(R.dtype)

    def mix(self, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Return the block input ``x [T, hidden]`` and the inject logits ``s [T, hc_count]`` (None if no combine)."""
        return self._mix_kernel(R) if R.is_cuda else self._mix_torch(R)

    def combine_norm(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor, nxt: "GatedResidual"):
        """``combine``, then the normed streams ``nxt.mix`` starts from, in one kernel: (R', Rn') for ``nxt.mix_normed``."""
        return hc_combine_rmsnorm(R, y, s, nxt.hc_norm.weight, nxt.hc_norm.eps, self.hc_count, inplace=_HC_INPLACE)

    def combine(self, R: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Inject the block output ``y [T, hidden]`` back into every stream of ``R``."""
        if R.is_cuda:
            return hc_combine(R, y, s, self.hc_count, inplace=_HC_INPLACE)
        return self._combine_torch(R, y, s)


__all__ = ["GatedResidual", "GroupedPlusOneRMSNorm", "grouped_plus_one_rms_norm"]
