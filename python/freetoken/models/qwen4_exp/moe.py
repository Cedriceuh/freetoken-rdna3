from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.distributed.split import intermediate_partition, is_uneven
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE, _SharedExpert

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = "") -> None:
        super().__init__(config, layer_id, prefix=prefix)
        if is_uneven():
            # FREETOKEN_TP_SPLIT: the shared expert's intermediate follows the routed experts' uneven split
            # (the loader cuts its weights with the same intermediate_partition)
            _, local = intermediate_partition(config.shared_expert_intermediate_size)
            self.shared_expert = _SharedExpert(
                config, config.hidden_size, config.shared_expert_intermediate_size,
                prefix=f"{prefix}.shared_expert", local_intermediate=local,
            )
        # TP: the routed experts and the shared expert's down_proj each hold a partial sum; the gate is computed
        # from the replicated input, so sum_r(routed_r + g * shared_r) = routed + g * shared and ONE all-reduce of
        # the combined partial replaces two (48 fewer collectives per token on Qwen3.8-Flash-Next).
        self._tp_size = get_tp_info().size
        if self._tp_size > 1 and os.environ.get("FREETOKEN_FUSE_MOE_ALLREDUCE", "1") == "1":
            self.experts.defer_all_reduce = True
            self.shared_expert.down_proj.defer_all_reduce = True
            self._comm = DistributedCommunicator()
        else:
            self._comm = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        out = shared_gate_mul_add(routed, shared, gate)
        if self._comm is not None:
            out = self._comm.all_reduce(out)
        return out.view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
