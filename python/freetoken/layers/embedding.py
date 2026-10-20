from __future__ import annotations

import os
from typing import Dict

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, nvtx_annotate

from .base import BaseOP
from .quantization import LayerKind, QuantConfig, quant_method_for


class VocabParallelEmbedding(BaseOP):
    """The vocab table, sharded over TP: its ``quant_method`` owns the table, looks rows up, and serves a tied LM head's product."""

    quant_layer_kind = LayerKind.EMBEDDING

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.embedding_dim = embedding_dim
        # the LM head subclass is a linear and builds its own method
        self.quant_method = None
        if self.quant_layer_kind is LayerKind.EMBEDDING:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            self.quant_method.create_weights(self)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.quant_method.embed(self, x)
        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


# FREETOKEN_EMBED_HOST (default 1 on ROCm, 0 elsewhere): an input embedding not tied to the LM head, looked up by the
# indexing kernel, keeps its table in pinned, device-mapped host memory (kernel/host_mapped.py) instead of VRAM. The same
# kernel reads the same rows, one per token across PCIe (5 KiB with 2560 bf16 columns); the VRAM goes to the pools sized
# after the weights (Qwen3.8-Flash-Next: 248,320 x 2560 bf16 rows, 606 MiB a rank at TP=2)
_EMBED_HOST = os.environ.get("FREETOKEN_EMBED_HOST", "1" if torch.version.hip is not None else "0") == "1"


def place_embeddings_on_host(root: BaseOP) -> int:
    """FREETOKEN_EMBED_HOST: move the eligible input tables under ``root`` to mapped host memory; returns the bytes."""
    from freetoken.kernel.host_mapped import host_mapped_empty
    from .quantization.embedding.unquantized import UnquantizedEmbeddingMethod

    if not _EMBED_HOST or not torch.cuda.is_available():
        return 0
    tables: list[VocabParallelEmbedding] = []
    tied: set[int] = set()
    seen: set[int] = set()

    def walk(op: BaseOP) -> None:
        if id(op) in seen:
            return
        seen.add(id(op))
        if isinstance(op, ParallelLMHead):
            if op.tied_embedding is not None:
                tied.add(id(op.tied_embedding))
        elif isinstance(op, VocabParallelEmbedding) and isinstance(op.quant_method, UnquantizedEmbeddingMethod):
            tables.append(op)
        for value in vars(op).values():
            for item in value if isinstance(value, (list, tuple)) else (value,):
                if isinstance(item, BaseOP):
                    walk(item)

    walk(root)
    moved = 0
    for op in tables:
        w = op.weight
        if id(op) in tied or not isinstance(w, torch.Tensor) or not w.is_cuda:
            continue
        host = host_mapped_empty(tuple(w.shape), w.dtype, w.device)
        host.copy_(w)
        op.weight = host
        moved += w.numel() * w.element_size()
    w = None  # the last table's VRAM goes back with the others before empty_cache
    if moved:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return moved


_DUMP_DIR = os.environ.get("FREETOKEN_DUMP_PROMPT_TOPK", "")
_DUMP_COUNT = 0


class ParallelLMHead(VocabParallelEmbedding):
    """The head is a linear layer over the vocab shard: its weights come from ``quant_method``
    unless they are tied to the input embedding."""

    quant_layer_kind = LayerKind.LINEAR

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.has_bias = bias
        self.prefix = prefix
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings
        self.in_features = embedding_dim
        self.out_features = self.num_embeddings_tp
        self.output_sizes = (self.num_embeddings_tp,)
        if tied_embedding is None:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            self.quant_method.create_weights(self)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None

    def finalize(self) -> None:
        if self.quant_method is not None:
            self.quant_method.finalize(self)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    def _dump_prompt_topk(self, x: torch.Tensor, k: int = 40, rows: int = 64) -> None:
        """FREETOKEN_DUMP_PROMPT_TOPK=<dir> (tests only): the top-k log-probabilities at every prompt position of a
        prefill chunk, saved by rank 0 as ``<dir>/prompt-<n>.pt`` -- teacher-forced comparison with a reference."""
        global _DUMP_COUNT
        ids, lps = [], []
        for a in range(0, x.shape[0], rows):
            part = x[a: a + rows].contiguous()
            local = self.tied_embedding.quant_method.logits(self.tied_embedding, part, self.bias) \
                if self.tied_embedding is not None else self.quant_method.apply(self, part)
            if self.tp_size > 1:
                shape = local.shape
                full = self._comm.all_gather(local).view((self.tp_size,) + shape).permute(1, 0, 2)
                local = full.reshape(shape[0], self.tp_size * shape[1])
            lp = torch.log_softmax(local[:, : self.num_embeddings].float(), dim=-1)
            top = lp.topk(k, dim=-1)
            ids.append(top.indices.cpu())
            lps.append(top.values.cpu())
        if get_tp_info().is_primary():
            import os

            os.makedirs(_DUMP_DIR, exist_ok=True)
            torch.save({"ids": torch.cat(ids), "lp": torch.cat(lps), "input_ids": get_global_ctx().batch.input_ids.cpu()},
                       os.path.join(_DUMP_DIR, f"prompt-{_DUMP_COUNT}.pt"))
        _DUMP_COUNT += 1

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if _DUMP_DIR and batch.is_prefill:
            self._dump_prompt_topk(x)
        if batch.is_prefill and x.shape[0] > bs:  # bs rows: one per request already (the MTP head picks its own)
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        if self.tied_embedding is not None:
            logits = self.tied_embedding.quant_method.logits(self.tied_embedding, x, self.bias)
        else:
            logits = self.quant_method.apply(self, x)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if input_shape[0] == 1:  # one row (decode rows can outnumber requests: spec_decode)
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]