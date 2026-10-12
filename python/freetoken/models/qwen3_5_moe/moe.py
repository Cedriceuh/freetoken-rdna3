from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# FREETOKEN_SHARED_ACT_IN_GEMV=0: silu_and_mul as its own kernel after the shared expert's int8 gate | up projection
_ACT_IN_GEMV = os.environ.get("FREETOKEN_SHARED_ACT_IN_GEMV", "1") != "0"


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = "",
        local_intermediate: int | None = None,
    ):
        # local_intermediate: this rank's width when the caller splits the intermediate axis itself
        # (uneven TP); None splits it evenly
        local = None if local_intermediate is None else [local_intermediate, local_intermediate]
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            local_output_sizes=local, quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False, local_input_size=local_intermediate,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = self._gate_up_act_int8(x) if _ACT_IN_GEMV else None
        if act is None:
            act = silu_and_mul(self.gate_up_proj.forward(x))
        return self.down_proj.forward(act)

    def _gate_up_act_int8(self, x: torch.Tensor) -> torch.Tensor | None:
        """Int8 decode rows: the gate | up GEMV applies silu_and_mul as it stores (FREETOKEN_SHARED_ACT_IN_GEMV), the
        same bits as the Triton activation after it, one kernel less; None otherwise (flashinfer's activation is
        another reference)."""
        gu = self.gate_up_proj
        if not (x.is_cuda and x.dim() == 2 and x.dtype is torch.bfloat16 and x.stride(-1) == 1 and gu.bias is None):
            return None
        from freetoken.kernel.backend import is_flashinfer_installed
        from freetoken.kernel.triton.int8_gemv import ROWS_MAX, gemv_int8_act_mul
        from freetoken.layers.quantization.int8_weight_only import Int8WeightOnlyLinearMethod

        m = x.shape[0]
        if (not isinstance(gu.quant_method, Int8WeightOnlyLinearMethod) or is_flashinfer_installed()
                or not (m == 1 or (1 < m <= ROWS_MAX and x.is_contiguous()))):
            return None
        return gemv_int8_act_mul(x, gu.weight, gu.weight_scale)


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return (routed + shared).view(num_tokens, hidden_dim)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
