from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.distributed.split import intermediate_partition, is_uneven
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid, shared_gate_sum_mul_add
from freetoken.layers.quantization.moe.exl3 import TritonExl3MoEKernel
from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel
from freetoken.moe import fused_nvfp4
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE, _SharedExpert

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# FREETOKEN_MOE_COPY_OVERLAP=1: in decode, the missing experts are copied on a side stream while the shared expert and
# its gate run (offload backend, GPU decode); same kernels and inputs, so the same bits
_COPY_OVERLAP = os.environ.get("FREETOKEN_MOE_COPY_OVERLAP", "0") == "1"
# FREETOKEN_FUSED_MOE_EPILOGUE (default 1): the routed sum, the shared-expert gate and the mul-add in one kernel
# (decode and prefill); the same arithmetic in the same order, so the same bits, two launches fewer per layer
_FUSED_EPILOGUE = os.environ.get("FREETOKEN_FUSED_MOE_EPILOGUE", "1") == "1"


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
        split = (_COPY_OVERLAP and not get_global_ctx().batch.is_prefill
                 and getattr(self.experts, "can_overlap_copy", lambda: False)())
        if split:
            topk_weights, topk_ids = self.experts.decode_select(hidden_states, router_logits)
        shared = self.shared_expert.forward(hidden_states)
        if _FUSED_EPILOGUE and not split and self._epilogue_fusable():
            fused_nvfp4.PER_EXPERT_OUTPUT[0] = True
            try:
                routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
            finally:
                fused_nvfp4.PER_EXPERT_OUTPUT[0] = False
            routed = routed if routed.dim() == 3 else routed.view(num_tokens, 1, hidden_dim)
            out = shared_gate_sum_mul_add(routed, shared, hidden_states, self.shared_expert_gate.weight.view(-1))
        else:
            gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
            if split:
                routed = self.experts.decode_finish(hidden_states, topk_weights, topk_ids)
            else:
                routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
            out = shared_gate_mul_add(routed, shared, gate)
        if self._comm is not None:
            out = self._comm.all_reduce(out)
        return out.view(num_tokens, hidden_dim)

    def _epilogue_fusable(self) -> bool:
        """The fused epilogue reads ``hidden_states`` after the routed experts and may get one output per expert: only
        the Triton NVFP4 / EXL3 expert kernels (they allocate their output; the bf16 one writes into its input) on the
        GPU decode path (the hybrid and CPU paths add partial sums), with no all-reduce inside the experts (TP=1, or the
        block's single all-reduce: the experts would otherwise reduce every expert's output, rounded another way)."""
        cache = getattr(self.experts, "offload_cache", None)
        kernel = getattr(self.experts.quant_method, "kernel", None)
        return (isinstance(kernel, (TritonNvfp4MoEKernel, TritonExl3MoEKernel)) and cache is not None
                and cache.decode_target != "hybrid" and not cache.is_cpu_layer(self.experts.layer_id)
                and (self._tp_size == 1 or self._comm is not None))


__all__ = ["Qwen4ExpMoE"]
