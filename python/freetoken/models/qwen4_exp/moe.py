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
from freetoken.moe import fused_nvfp4, prefill_hooks
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE, _SharedExpert

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# FREETOKEN_MOE_COPY_OVERLAP=1: in decode, the missing experts are copied on a side stream while the shared expert and
# its gate run (offload backend, GPU decode); same kernels and inputs, so the same bits
_COPY_OVERLAP = os.environ.get("FREETOKEN_MOE_COPY_OVERLAP", "0") == "1"
# FREETOKEN_PREFILL_COPY_OVERLAP: a short prefill that reads its experts from the slot cache copies the missing ones on
# the prefill copy stream while the shared expert runs; the same copies and GEMMs, so the same bits. "auto" (default):
# experts decoded on the GPU (offload; --hybrid was not measured), EXL3 ones with FREETOKEN_PREFILL_PREFETCH_DEFER: EXL3
# agent turns took 2 % less and one-word turns 4 % less, the decode after them unchanged (without the deferred plan,
# the EXL3 decode ran 1-2 % slower); NVFP4 one-word turns 5 % less, agent turns 2 %. "1" / "0": always / never
_PREFILL_COPY_OVERLAP = os.environ.get("FREETOKEN_PREFILL_COPY_OVERLAP", "auto")
# FREETOKEN_FUSED_MOE_EPILOGUE (default 1): the routed sum, the shared-expert gate and the mul-add in one kernel
# (decode and prefill); the same arithmetic in the same order, so the same bits, two launches fewer per layer
_FUSED_EPILOGUE = os.environ.get("FREETOKEN_FUSED_MOE_EPILOGUE", "1") == "1"
# FREETOKEN_PREFILL_PREDICT (default 1): when a prefill chunk reads its experts from the slot cache, layer L+1's router
# applied to layer L's MoE input predicts the experts layer L+1 routes to (top-k + _PREDICT_EXTRA per token), and the
# absent ones are copied in while layer L computes (only copies, so the same bits). Each token's own top-k (extra 0)
# copies about as many experts layer L+1 does not route to as it misses; wider predictions waste more PCIe than the
# misses they save (short agent turns: extra 0 -4 to -6 % against 4, 8 and 12 slower still, -2 slower)
_PREDICT = os.environ.get("FREETOKEN_PREFILL_PREDICT", "1") == "1"
_PREDICT_EXTRA = int(os.environ.get("FREETOKEN_PREFILL_PREDICT_EXTRA", "0"))
# FREETOKEN_PREFILL_PREDICT_MIN_VOTES: copy ahead only the experts at least this many tokens' predictions name (the
# others, when routed, go through the fix-up copy); only copies, so the same bits. "auto" (default): 2 with NVFP4
# experts, whose short turns keep PCIe busy throughout (agent turns -6 %, one-word turns -6 %), 1 otherwise (EXL3: 1-3 %
# slower at 2)
_PREDICT_MIN_VOTES = os.environ.get("FREETOKEN_PREFILL_PREDICT_MIN_VOTES", "auto")


def _block_tokens(kernel) -> int:
    """The token block of this layer's routed experts in a prefill (FREETOKEN_NVFP4_PREFILL_BLOCK /
    FREETOKEN_EXL3_PREFILL_BLOCK, 0: none): a chunk no longer runs in one block, and its all-reduce stays whole."""
    from freetoken.moe.fused_exl3 import EXL3_PREFILL_BLOCK

    return EXL3_PREFILL_BLOCK if isinstance(kernel, TritonExl3MoEKernel) else fused_nvfp4.NVFP4_PREFILL_BLOCK


def _cached_limit(experts) -> int:
    """The longest chunk the slot-cache path may take (FREETOKEN_PREFILL_CACHED_MAX), from the setting alone: whether a
    chunk takes it also depends on each rank's cache, so a longer one surely takes the full-layer path on every rank."""
    from freetoken.layers import moe as layers_moe

    if layers_moe._PREFILL_CACHED_SET is not None:
        return layers_moe._PREFILL_CACHED_SET
    return layers_moe.PREFILL_CACHED_DEFAULT if getattr(experts, "prefill_cached_default", False) else 0


def _min_votes(cache) -> int:
    if _PREDICT_MIN_VOTES == "auto":
        return 2 if cache is not None and cache.quant_format == "nvfp4" else 1
    return int(_PREDICT_MIN_VOTES)


def _voted(predicted: torch.Tensor, num_experts: int, min_votes: int) -> torch.Tensor:
    """The expert ids at least ``min_votes`` rows of ``predicted`` name, as one id a row (the most named one in the
    places of the others, so the shape stays fixed and nothing syncs)."""
    flat = predicted.reshape(-1)
    # index_add_, not bincount: CUDA's bincount reads the min and max back to the host
    votes = torch.zeros(num_experts, dtype=torch.int64, device=flat.device).index_add_(0, flat, torch.ones_like(flat))
    ids = torch.arange(num_experts, device=votes.device)
    return torch.where(votes >= min_votes, ids, votes.argmax()).view(-1, 1)
_BLOCKS: dict = {}  # MoE layer id -> block of the main model (the MTP head's is left out)
# FREETOKEN_PREFILL_AR_OVERLAP (default 1): a prefill chunk whose routed experts run in token blocks (NVFP4 / EXL3
# prefill blocks) finishes each block's rows (the fused epilogue, with the whole chunk's warps) and starts their
# all-reduce while the next block computes; every element is still routed + gate * shared, then the sum of the two
# ranks: the same bits
_AR_OVERLAP = os.environ.get("FREETOKEN_PREFILL_AR_OVERLAP", "1") == "1"


def _prefill_copy_overlap(cache) -> bool:
    """FREETOKEN_PREFILL_COPY_OVERLAP for this model's expert cache."""
    from freetoken.moe.offload_cache import _PREFETCH_DEFER

    return _PREFILL_COPY_OVERLAP == "1" or (_PREFILL_COPY_OVERLAP == "auto" and cache.decode_target == "gpu"
                                            and (cache.quant_format == "nvfp4" or _PREFETCH_DEFER))


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
        if layer_id is not None and layer_id < config.num_layers:
            _BLOCKS[self.experts.layer_id] = self
        self.experts.prefill_cached_default = _PREDICT
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
        cached = getattr(self.experts, "prefill_cached", None)  # offload experts only
        if cached is not None and get_global_ctx().batch.is_prefill:
            nxt = _BLOCKS.get(self.experts.layer_id + 1) if _PREDICT else None
            if nxt is not None and nxt is not self and cached(num_tokens):
                logits = nxt.gate.forward(hidden_states)
                width = max(1, min(self.experts.top_k + _PREDICT_EXTRA, logits.shape[-1]))
                predicted = torch.topk(logits, width, dim=-1).indices
                min_votes = _min_votes(getattr(self.experts, "offload_cache", None))
                if min_votes > 1:
                    predicted = _voted(predicted, logits.shape[-1], min_votes)
                self.experts.next_layer_prediction = (nxt.experts.layer_id, predicted)
        split = (_COPY_OVERLAP and not get_global_ctx().batch.is_prefill
                 and getattr(self.experts, "can_overlap_copy", lambda: False)())
        if split:
            topk_weights, topk_ids = self.experts.decode_select(hidden_states, router_logits)
        split_pf = (cached is not None and get_global_ctx().batch.is_prefill and cached(num_tokens)
                    and _prefill_copy_overlap(self.experts.offload_cache)
                    and self.experts.offload_cache.can_copy_async_prefill(self.experts.layer_id))
        if split_pf:
            routing = self.experts.prefill_select(hidden_states, router_logits)
        # decided the same way on every rank (the chunk length, the cache path decided at layer 0, the agreed layer)
        block = _block_tokens(getattr(self.experts.quant_method, "kernel", None))
        blocks = (_AR_OVERLAP and self._comm is not None and get_global_ctx().batch.is_prefill and block > 0
                  and num_tokens > max(block, _cached_limit(self.experts)) and self._ranks_agree_on_blocks())
        shared = self.shared_expert.forward(hidden_states)
        if _FUSED_EPILOGUE and not split and self._epilogue_fusable():
            gate_w = self.shared_expert_gate.weight.view(-1)
            works = None
            if blocks:
                out, works = torch.empty_like(hidden_states), []

                def block_done(start: int, end: int, routed: torch.Tensor) -> None:
                    rows = out[start:end]
                    shared_gate_sum_mul_add(routed[start:end].view(end - start, 1, hidden_dim), shared[start:end],
                                            hidden_states[start:end], gate_w, out=rows, warps_tokens=num_tokens)
                    works.append(self._comm.all_reduce_async(rows))

                prefill_hooks.BLOCK_DONE[0] = block_done
            fused_nvfp4.PER_EXPERT_OUTPUT[0] = True
            try:
                routed = (self.experts.prefill_finish(hidden_states, *routing) if split_pf
                          else self.experts.forward(hidden_states=hidden_states, router_logits=router_logits))
            finally:
                fused_nvfp4.PER_EXPERT_OUTPUT[0] = False
                prefill_hooks.BLOCK_DONE[0] = None
            if works:  # every block finished and all-reduced its rows
                for work in works:
                    if work is not None:
                        work.wait()
                        prefill_hooks.ASYNC_AR[0] = True
                return out.view(num_tokens, hidden_dim)
            routed = routed if routed.dim() == 3 else routed.view(num_tokens, 1, hidden_dim)
            out = shared_gate_sum_mul_add(routed, shared, hidden_states, gate_w)
        else:
            gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
            if split:
                routed = self.experts.decode_finish(hidden_states, topk_weights, topk_ids)
            elif split_pf:
                routed = self.experts.prefill_finish(hidden_states, *routing)
            else:
                routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
            out = shared_gate_mul_add(routed, shared, gate)
        if self._comm is not None:
            out = self._comm.all_reduce(out)
        return out.view(num_tokens, hidden_dim)

    def _ranks_agree_on_blocks(self) -> bool:
        """FREETOKEN_PREFILL_AR_OVERLAP: at two ranks only (a sum of two values does not depend on the blocks), and once
        both ranks found this layer on the fused path (their caches could put it apart, and they must issue the same
        all-reduces); asked once per layer, at its first long prefill."""
        agreed = self.__dict__.get("_blocks_agreed")
        if agreed is None:
            local = self._tp_size == 2 and _FUSED_EPILOGUE and self._epilogue_fusable()
            ok = torch.tensor([int(local)], dtype=torch.int32, device=torch.cuda.current_device())
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
            agreed = self._blocks_agreed = bool(ok.item())
        return agreed

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
