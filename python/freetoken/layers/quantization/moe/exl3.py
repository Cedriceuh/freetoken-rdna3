"""EXL3 experts (exllamav3 trellis), served from the offload cache through the Triton kernels of kernel/triton/exl3.py.

Banks, per expert and per rank (``i`` = this rank's slice of the intermediate, ``h`` = hidden, K bits per weight):

* ``gate_up``     int16 ``[h/16, 2i/16, 16K]``: gate's then up's output tiles, both in their rotated domain;
* ``gate_up_suh`` fp16 ``[2, h]``: each projection's input scales (they differ: a gate|up input is rotated twice);
* ``gate_up_svh`` fp16 ``[2i]``: their output scales;
* ``down`` int16 ``[i/16, h/16, 16K]``, ``down_suh`` fp16 ``[i]``, ``down_svh`` fp16 ``[h]``.

A rank's slice must hold whole 128-wide Hadamard blocks of the intermediate (the dialect sets that split unit); the
checkpoint's tensors are cut on tile rows / columns, nothing is re-encoded.
"""

from __future__ import annotations

import torch

from freetoken.layers.quantization.exl3_codec import HAD_DIM

from ..registry import LayerKind, register_method
from ..scheme import QuantKind, exl3_params
from .base import BankSpec, ExpertView, MoEConfig, MoEKernel, MoEMethod

TILE = 16


class TritonExl3MoEKernel(MoEKernel):
    """Decode: per-route split-K GEMVs; prefill: expert-sorted grouped GEMMs; rotations fused around both."""

    name = "triton"
    cpu_format = None

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        reason = self._common_reject(cfg, resident_ok=False, tp_ok=True, cpu_ok=False, plain_silu_only=True)
        if reason:
            return f"triton exl3 MoE kernel: {reason}"
        if cfg.apply_router_weight_on_input:
            return "triton exl3 MoE kernel: the router weight applies to the output only"
        if cfg.hidden % HAD_DIM or cfg.intermediate % HAD_DIM:
            return f"triton exl3 MoE kernel: hidden {cfg.hidden} / intermediate {cfg.intermediate} not in {HAD_DIM}-blocks"
        lo, size = cfg.local_intermediate_range
        if lo % HAD_DIM or size % HAD_DIM:
            return (f"triton exl3 MoE kernel: this rank's intermediate slice [{lo}, {lo + size}) is not made of whole "
                    f"{HAD_DIM}-wide Hadamard blocks; choose FREETOKEN_TP_SPLIT so each rank gets a multiple of {HAD_DIM}")
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        bits, _ = exl3_params(cfg.scheme)
        i, h = cfg.local_intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((h // TILE, 2 * i // TILE, TILE * bits), torch.int16),
            "gate_up_suh": BankSpec((2, h), torch.float16),
            "gate_up_svh": BankSpec((2 * i,), torch.float16),
            "down": BankSpec((i // TILE, h // TILE, TILE * bits), torch.int16),
            "down_suh": BankSpec((i,), torch.float16),
            "down_svh": BankSpec((h,), torch.float16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        """Pieces: ``{gate,up,down}_{trellis,suh,svh}`` per expert, full width; this rank keeps its slice."""
        lo, i = cfg.local_intermediate_range
        t0, t1 = lo // TILE, (lo + i) // TILE
        cols = slice(lo, lo + i)
        out["gate_up"].copy_(torch.cat([pieces["gate_trellis"][:, :, t0:t1], pieces["up_trellis"][:, :, t0:t1]], dim=2))
        out["gate_up_suh"].copy_(torch.stack([pieces["gate_suh"], pieces["up_suh"]], dim=1))
        out["gate_up_svh"].copy_(torch.cat([pieces["gate_svh"][:, cols], pieces["up_svh"][:, cols]], dim=1))
        out["down"].copy_(pieces["down_trellis"][:, t0:t1])
        out["down_suh"].copy_(pieces["down_suh"][:, cols])
        out["down_svh"].copy_(pieces["down_svh"])
        return {}

    def prefill_workspace_bytes(self, cfg: MoEConfig, tokens: int) -> int:
        """The prefill path's peak (moe/fused_exl3.py), per token block: the rotated fp16 inputs of both projections
        with the fp32 gate|up products, or the fp16 mid with the routed bf16 down rows; plus the block's and the
        chunk's outputs. 620 MiB for 16384 tokens at top-10, hidden 2560, 384 intermediate (measured: 620 MiB)."""
        from freetoken.moe.fused_exl3 import EXL3_PREFILL_BLOCK

        block = min(tokens, EXL3_PREFILL_BLOCK) if EXL3_PREFILL_BLOCK > 0 else tokens
        routes, h, i = block * cfg.top_k, cfg.hidden, cfg.local_intermediate
        gate_up = routes * 2 * h * 2 + routes * 2 * i * 4
        down = routes * i * 2 + routes * h * 2
        return max(gate_up, down) + block * h * 2 + (tokens * h * 2 if tokens > block else 0)

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_exl3 import fused_experts_exl3

        t = view.tensors
        _, codebook = exl3_params(layer.quant_method.cfg.scheme)
        return fused_experts_exl3(
            x, t["gate_up"], t["gate_up_suh"], t["gate_up_svh"], t["down"], t["down_suh"], t["down_svh"],
            topk_weights, topk_ids, codebook=codebook, is_prefill=is_prefill, num_experts=view.n,
        )


@register_method(QuantKind.EXL3, LayerKind.MOE)
class Exl3MoEMethod(MoEMethod):
    candidates = (TritonExl3MoEKernel,)

    def create_weights(self, layer) -> None:
        raise NotImplementedError("EXL3 experts are served from the offload cache, not resident")

    def resident_view(self, layer) -> ExpertView:
        raise NotImplementedError("EXL3 experts are not resident")
