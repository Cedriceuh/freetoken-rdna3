"""Qwen3.8-Flash-Next checkpoint reader (bf16, block-fp8, ModelOpt and llm-compressor exports).

Separate paths, because the checkpoint's weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_DenseReader``.
* :func:`load_ple_table` -- the n-gram table (47.7 GiB fp8 or 95.4 GiB bf16), 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named under the checkpoint's dialect, for the offload cache's expert reader.
* :func:`iter_expert_pieces` -- with the MTP head (``FREETOKEN_MTP=1``), those experts followed by the head's 512, quantized to NVFP4 from its stacked bf16 ``mtp.layers.0.mlp.experts.*``.

Dropped: ``mtp.*`` unless the MTP head is built; ``model.visual.*`` is kept only when the model built the tower.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.distributed.split import gdn_head_partition, intermediate_partition
from freetoken.models.qwen3_vl.weight import rename_vl_prefix

from freetoken.models.config import VISION_KEY_PREFIXES
from freetoken.models.loader import drop_page_cache, iter_weight_files, shard_tensor
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.layers.quantization import QuantConfig, QuantKind, QuantScheme, get_quant_config
from freetoken.models.register import get_model_spec
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, div_ceil, div_even, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.models.config import LinearGatedDeltaGroupConfig, ModelConfig

# Routed NVFP4 experts: per-expert, un-fused. Matched against the RAW weight_map key in nvfp4_banks.
# The ``model.language_model.`` anchor excludes the MTP head's stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = (
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>{kinds})$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
# role -> the expert bank reader's canonical (ModelOpt) tensor kind
_BANK_KINDS = {"weight": "weight", "weight_scale": "weight_scale", "weight_global": "weight_scale_2"}

# leaves the model builds as quant-aware Linear layers: their tensors are read under the QuantConfig
_LINEAR_LEAVES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "index_qk_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "input_mix_weight_down", "block_inject_weight", "input_mix_weight_up",
    "key_proj", "value_proj", "lm_head",
})
# activation scales of modules whose scheme carries no input_scale role
_DROPPED_SUFFIXES = frozenset({"input_scale", "input_global_scale"})
_ELEM_DTYPES = {"e4m3": torch.float8_e4m3fn, "e2m1": torch.uint8}
_QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8)
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# per-tensor quant scales of the MTP head's tensors: never read (the head's dense tensors load bf16)
_MTP_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# The per-layer HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM pads the merged rows to a multiple of 16 for cuBLAS (hyperconnection.py pad_size).
# The top-level hyper_connection_mixer has no injection and never fuses.
_PAD_TO = {"input_mix_weight_down_block_inject": 16}
_HC_WITH_INJECT = (".attn_hyper_connection", ".mlp_hyper_connection")


# EXL3 checkpoints keep the n-gram hash constants in their ngram_embedding.safetensors, under the table's prefix
_EXL3_PLE_CONSTANTS = {
    "ngram_embedding.head_offsets": "ngram_heads_offsets",
    "ngram_embedding.head_vocab_sizes": "ngram_heads_vocab_sizes",
    "ngram_embedding.layer_multipliers": "layer_multipliers",
}


_EXL3_VISION_QKV = re.compile(r"^model\.visual\.blocks\.\d+\.attn\.[qkv]_proj\.")


def _is_exl3() -> bool:
    try:
        quant = get_quant_config()
    except RuntimeError:  # no engine config installed (offline tools): not an EXL3 load
        return False
    return quant is not None and quant.dialect == "exl3"


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    for exl3_leaf, leaf in _EXL3_PLE_CONSTANTS.items():
        if raw_name.endswith(".ple.ple_embedding." + exl3_leaf):
            return rename_vl_prefix(raw_name[: -len(exl3_leaf)] + leaf)
    if raw_name.startswith("mtp."):
        from freetoken.spec_decode import MTP_ENABLED

        # the draft head's dense tensors keep their names (Qwen4ExpMTP); its stacked experts go to the banks
        if not MTP_ENABLED or ".mlp.experts." in raw_name or raw_name.endswith(_MTP_SCALE_SUFFIXES):
            return None
        return raw_name
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXL3_VISION_QKV.search(raw_name) and _is_exl3():
        return None  # EXL3 checkpoints add quantized q/k/v next to the tower's bf16 qkv, which the model reads
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    # static KV-cache scales of the quantizers; the KV cache runs in the engine's dtype
    if raw_name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
        return None
    return rename_vl_prefix(raw_name)


def _per_row_scale(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-tensor scalar or per-channel ``[rows, 1]`` scale -> fp32 ``[rows]``; any other count is refused rather than broadcast onto the wrong rows."""
    flat = scale.reshape(-1).to(torch.float32)
    if flat.numel() == 1:
        return flat.expand(rows).contiguous()
    if flat.numel() != rows:
        raise ValueError(f"scale has {flat.numel()} elements for a weight with {rows} output rows (shape {tuple(scale.shape)}); expected 1 or {rows}")
    return flat.contiguous()


class _DenseReader:
    """Routes each Linear tensor to the buffer its module's scheme declares; packed projections are concatenated per role once every part is in.

    The part table is the family's packed_modules_mapping. The QuantConfig picks the GDN in_proj layout and, through its dialect's storage names, which checkpoint tensor fills which role.
    """

    def __init__(self, quant: QuantConfig | None, packed: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
        self.quant = quant
        self.groups = {fused: parts for fused, parts in packed if fused != "experts"}  # experts: bank reader
        self.by_part: dict[str, list[tuple[str, int]]] = {}
        for fused, parts in self.groups.items():
            for idx, part in enumerate(parts):
                self.by_part.setdefault(part, []).append((fused, idx))
        # target module -> (part count, {part: {role: tensor}}, {part: the roles its module stores}, scheme)
        self.pending: dict[str, tuple[int, dict[int, dict[str, torch.Tensor]], dict[int, set[str]], QuantScheme | None]] = {}

    def scheme(self, module: str) -> QuantScheme | None:
        return None if self.quant is None else self.quant.scheme_for(module)

    def target(self, module: str) -> tuple[str, int, int]:
        """``(fused module, part index, part count)``; a standalone linear is its own single-part target."""
        parent, _, leaf = module.rpartition(".")
        candidates = self.by_part.get(leaf)
        if not candidates:
            return module, 0, 1
        if len(candidates) > 1:
            # GDN: quantized checkpoints split qkv|z from b|a; same test as gdn.py
            split = self.scheme(f"{parent}.in_proj_qkvz") is not None
            keep = {"in_proj_qkvz", "in_proj_ba"} if split else {"in_proj"}
            candidates = [c for c in candidates if c[0] in keep]
        fused, idx = candidates[0]
        if fused in _PAD_TO and not parent.endswith(_HC_WITH_INJECT):
            return module, 0, 1
        return f"{parent}.{fused}", idx, len(self.groups[fused])

    def add(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """Take one tensor; the emitted ``[(name, tensor)]`` once its target module is complete, ``[]`` before, None if ``name`` is not a Linear's tensor."""
        module, _, suffix = name.rpartition(".")
        if module.rpartition(".")[2] not in _LINEAR_LEAVES:
            return None
        scheme = self.scheme(module)
        roles = {"weight": "weight"} if scheme is None else {n: r for r, e in self.quant.storage(scheme).items() for n in e.names}
        role = roles.get(suffix)
        if role is None:
            if suffix in _DROPPED_SUFFIXES:
                return []
            raise ValueError(f"{name}: the checkpoint's quant config declares {module} {scheme or 'unquantized'}, stored as {sorted(roles)}")
        if role == "weight":
            self._check_weight(name, module, scheme, tensor)
        target, idx, count = self.target(module)
        _, parts, expected, _ = self.pending.setdefault(target, (count, {}, {}, scheme))
        part = parts.setdefault(idx, {})
        if role in part:
            raise ValueError(f"{name}: {module} already has its {role} from another checkpoint tensor")
        part[role] = tensor
        expected[idx] = set(roles.values())
        if scheme is not None and set(part) == expected[idx]:
            parts[idx] = self._finish(module, scheme, part)
        if len(parts) < count or any(set(parts[i]) != expected[i] for i in parts):
            return []
        del self.pending[target]
        return self._emit(target, [parts[i] for i in range(count)], scheme)

    def missing(self) -> list[str]:
        """One line per incomplete module: the roles its parts still lack."""
        lines = []
        for target, (count, parts, expected, _scheme) in sorted(self.pending.items()):
            lacking = sorted(set().union(*(expected[i] - set(parts[i]) for i in parts)))
            if len(parts) < count:
                lacking.append(f"{count - len(parts)} of {count} fused parts")
            lines.append(f"{target}: missing {lacking}")
        return lines

    @staticmethod
    def _check_weight(name: str, module: str, scheme: QuantScheme | None, tensor: torch.Tensor) -> None:
        if scheme is None:
            if tensor.dtype in _QUANT_DTYPES:
                raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} unquantized")
            return
        elem = _ELEM_DTYPES.get(scheme.weight.elem)
        if elem is None:
            raise ValueError(f"{name}: qwen4_exp has no dense reader for {scheme}")
        if tensor.dtype is not elem:
            raise ValueError(f"{name} is {tensor.dtype} but the checkpoint's quant config declares {module} {scheme}")
        rows, cols = tensor.shape[0], tensor.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        if block_rows > 1 and rows % block_rows or block_cols > 1 and cols % block_cols:
            raise ValueError(f"{name}: {rows}x{cols} is not a multiple of the {block_rows}x{block_cols} scale block of {module}")

    def _finish(self, module: str, scheme: QuantScheme, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate one complete part against ``scheme`` and put its scales in the layer's form."""
        storage = self.quant.storage(scheme)
        part = {role: 1.0 / t.to(torch.float32) if storage[role].reciprocal else t for role, t in part.items()}
        weight = part["weight"]
        rows, cols = weight.shape[0], weight.shape[1] * (2 if scheme.weight.elem == "e2m1" else 1)
        block_rows, block_cols = scheme.weight.group or (1, 1)
        scale_role = "weight_scale_inv" if "weight_scale_inv" in part else "weight_scale"
        out = dict(part)
        if block_cols < 0:
            try:
                out[scale_role] = _per_row_scale(part[scale_role], rows)
            except ValueError as exc:
                raise ValueError(f"{module}.{scale_role}: {exc}") from None
        else:
            want = (rows // block_rows, cols // block_cols)
            if tuple(part[scale_role].shape) != want:
                raise ValueError(f"{module}: {scale_role} is {tuple(part[scale_role].shape)}, expected {want} for {scheme}")
            if scheme.weight.scale == "e4m3" and part[scale_role].dtype is not torch.float8_e4m3fn:
                raise ValueError(f"{module}: {scale_role} is {part[scale_role].dtype} but {scheme} stores e4m3 scales")
        if "weight_global" in part:
            g = part["weight_global"].reshape(-1).to(torch.float32)
            if g.numel() != 1:
                raise ValueError(f"{module}: weight_global has {g.numel()} elements, expected one per-tensor scale")
            out["weight_global"] = g.expand(rows).contiguous()
        if "input_scale" in part:
            out["input_scale"] = part["input_scale"].reshape(()).to(torch.float32)
        return out

    @staticmethod
    def _emit(target: str, parts: list[dict[str, torch.Tensor]], scheme: QuantScheme | None) -> list[tuple[str, torch.Tensor]]:
        pad_to = _PAD_TO.get(target.rpartition(".")[2], 0)
        if pad_to and scheme is not None:
            raise ValueError(f"{target}: qwen4_exp pads this fusion to {pad_to} rows and reads it bf16 only, the checkpoint's quant config declares {scheme}")
        out = []
        for role in parts[0]:
            tensors = [part[role] for part in parts]
            if role == "input_scale":
                # fused parts read the same activation, so one range covers them: max is exact then and safe if they drift
                out.append((f"{target}.{role}", torch.stack(tensors).max()))
                continue
            pad = (-sum(t.shape[0] for t in tensors)) % pad_to if pad_to else 0
            if pad:
                tensors.append(torch.zeros(pad, *tensors[0].shape[1:], dtype=tensors[0].dtype, device=tensors[0].device))
            out.append((f"{target}.{role}", tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)))
        return out


# ======================================================================================
# Tensor parallelism (rewritten from lukascechovic/FreeToken rocm-gfx1201, d9fddf7 + 2433af3)
# ======================================================================================
#
# The dense tensors whose module is tensor-parallel are sharded HERE, on the checkpoint's own
# per-projection keys, BEFORE ``_DenseReader`` concatenates them. Everything not listed here or in
# the GDN tables below is replicated: the HC mixers, the PLE projections, the QSA indexer, every
# norm and the routers are ``LinearReplicated`` and every rank needs them whole. The vision tower is
# never sharded either: rank 0 alone holds it (its tensors arrive already renamed under ``visual.``).
#
# These five are fusion parts. A flat row chunk of the fused tensor has the right shape and is a
# different tensor (``qkv_proj`` is ``[2*qo | kv | kv]``), so each part is cut on its own axis;
# ``LinearColParallelMerged`` divides each declared output size on its own, which is the layout
# the per-part shards concatenate into.
_SHARD_BEFORE_FUSE = (
    ".self_attn.q_proj.weight",  # head-major, [q | gate] interleaved per head: a chunk is head-aligned
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
)
# Not fused: the row-parallel projections (sharded on their INPUT axis, the one their
# column-parallel producer already split) and the vocab-parallel pair.
_SHARD_UNFUSED = (
    ".self_attn.o_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
    "model.embed_tokens.weight",
    "lm_head.weight",
)
_TP_SHARDED = _SHARD_BEFORE_FUSE + _SHARD_UNFUSED
# The shared expert's intermediate axis: gate/up rows, down columns. Cut here rather than by
# shard_tensor so it can follow the uneven FREETOKEN_TP_SPLIT like the routed experts.
_SHARED_EXPERT_ROWS = (".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight")
_SHARED_EXPERT_COLUMNS = (".mlp.shared_expert.down_proj.weight",)
# Their width follows ``num_kv_heads``: below one head per rank ``shard_tensor`` replicates.
_KV_PROJ = (".self_attn.k_proj.weight", ".self_attn.v_proj.weight")
# ``VocabParallelEmbedding`` allocates ``div_ceil(V, tp)`` rows on every rank (``ParallelLMHead``
# all-gathers, which needs the same shape everywhere); ``shard_tensor`` hands the last rank the
# short real slice, so a vocab that does not divide needs zero rows appended.
_VOCAB_PARALLEL = ("model.embed_tokens.weight", "lm_head.weight")


def _pad_vocab_rows(local: torch.Tensor, full_rows: int, *, world_size: int) -> torch.Tensor:
    """``local`` grown to the module's ``div_ceil(V, tp)`` partition with zero rows (never read)."""
    per_rank = div_ceil(full_rows, world_size)
    pad = per_rank - local.shape[0]
    assert pad >= 0, f"vocab shard is {local.shape[0]} rows, wider than the {per_rank}-row buffer"
    if pad == 0:
        return local
    return torch.cat(
        [local, torch.zeros(pad, *local.shape[1:], dtype=local.dtype, device=local.device)], dim=0
    )


# The GDN tensors are composite one level deeper: ``in_proj_qkv`` and ``conv1d`` are laid out on
# ``conv_dim = [key | key | value]``, so each sub-block is split on its OWN head count. The whole
# GDN is sharded together: ``in_proj`` is a four-part fusion (``qkv | z | b | a``).
_GDN_INFIX = ".linear_attn."
_GDN_CONV_COMPOSITE = ("in_proj_qkv.weight", "conv1d.weight")  # dim 0, [key | key | value]
_GDN_VALUE_ROWS = ("in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight", "A_log", "dt_bias")
_GDN_VALUE_COLUMNS = ("out_proj.weight",)  # dim 1: row-parallel over the value heads
_GDN_REPLICATED = ("norm.weight",)  # head_v_dim wide: a per-head width, not a head count


def _heads_narrow(tensor: torch.Tensor, dim: int, num_heads: int, offset: int, count: int) -> torch.Tensor:
    """Heads ``[offset, offset + count)`` of ``dim``, which holds ``num_heads`` equal-width heads."""
    width = tensor.shape[dim]
    assert width % num_heads == 0, f"{width} does not divide into {num_heads} heads"
    per_head = width // num_heads
    return tensor.narrow(dim, offset * per_head, count * per_head).clone()


def _shard_gdn(
    leaf: str, tensor: torch.Tensor, group: LinearGatedDeltaGroupConfig, *, rank: int, world_size: int
) -> torch.Tensor:
    """This rank's slice of one GDN tensor, named by its leaf below ``.linear_attn.``.

    The heads come from :func:`gdn_head_partition`, the rule the module and the state pool size
    themselves with (even, or the uneven ``FREETOKEN_TP_SPLIT``).
    An unclassified leaf raises: loaded whole into a rank-local buffer it would be silently wrong.
    """
    if leaf in _GDN_REPLICATED:
        return tensor
    nk, nv = group.num_key_heads, group.num_value_heads
    k_off, k_n, v_off, v_n = gdn_head_partition(nk, nv, rank=rank, world_size=world_size)
    if leaf in _GDN_CONV_COMPOSITE:
        key_dim = group.num_key_heads * group.key_head_dim
        value_dim = group.num_value_heads * group.value_head_dim
        assert tensor.shape[0] == 2 * key_dim + value_dim, (
            f"GDN {leaf} is {tuple(tensor.shape)}, expected conv_dim "
            f"{2 * key_dim + value_dim} = 2*{key_dim} + {value_dim} rows"
        )
        q, k, v = torch.split(tensor, [key_dim, key_dim, value_dim], dim=0)
        return torch.cat(
            [
                _heads_narrow(q, 0, nk, k_off, k_n),
                _heads_narrow(k, 0, nk, k_off, k_n),
                _heads_narrow(v, 0, nv, v_off, v_n),
            ],
            dim=0,
        )
    if leaf in _GDN_VALUE_ROWS:
        return _heads_narrow(tensor, 0, nv, v_off, v_n)
    if leaf in _GDN_VALUE_COLUMNS:
        return _heads_narrow(tensor, 1, nv, v_off, v_n)
    raise NotImplementedError(
        f"GDN tensor {leaf!r} is not classified for tensor parallelism; add it to one of "
        f"_GDN_CONV_COMPOSITE / _GDN_VALUE_ROWS / _GDN_VALUE_COLUMNS / _GDN_REPLICATED"
    )


def _shard_for_rank(name: str, tensor: torch.Tensor, *, config: ModelConfig) -> torch.Tensor:
    """This rank's slice of ``name``, or the tensor whole when its module is replicated.

    Attention and the dense projections go through :func:`freetoken.models.loader.shard_tensor`
    (the rule every other model loads with, including its ``num_kv_heads < world_size`` branch);
    the GDN goes through :func:`_shard_gdn` because its axes are composite.
    """
    tp = get_tp_info()
    if tp.size == 1:
        return tensor
    if name.startswith(VISION_KEY_PREFIXES):
        return tensor  # the tower is never sharded: rank 0 holds it whole and encodes, the engine broadcasts
    if tensor.dtype in _FP8_DTYPES or name.endswith(".weight_scale_inv"):
        # The block-fp8 dense path is not TP-aware (a 128x128 scale grid does not follow a head split).
        raise NotImplementedError(f"{name}: block-fp8 dense weights are not supported at tp_size={tp.size}")
    leaf = name.split(_GDN_INFIX, 1)[1] if _GDN_INFIX in name else None
    if leaf is not None:
        return _shard_gdn(leaf, tensor, config.linear_attention_group(), rank=tp.rank, world_size=tp.size)
    if name.endswith(_SHARED_EXPERT_ROWS + _SHARED_EXPERT_COLUMNS):
        # intermediate_partition: the rule Qwen4ExpMoE sizes its shared expert with (even, or uneven)
        dim = 0 if name.endswith(_SHARED_EXPERT_ROWS) else 1
        full = tensor.shape[dim]
        assert full == config.shared_expert_intermediate_size, (name, tuple(tensor.shape))
        lo, size = intermediate_partition(full, rank=tp.rank, world_size=tp.size)
        return tensor.narrow(dim, lo, size).clone()
    if not name.endswith(_TP_SHARDED):
        return tensor
    local = shard_tensor(name, tensor, rank=tp.rank, world_size=tp.size, num_kv_heads=config.num_kv_heads)
    # shard_tensor returns an unrecognised key unchanged; a key declared tensor-parallel must come back
    # smaller, except on the kv replication branch where the whole projection IS the rank-local tensor.
    if not (name.endswith(_KV_PROJ) and config.num_kv_heads < tp.size):
        assert local.shape != tensor.shape, (
            f"{name!r} is declared tensor-parallel but shard_tensor returned it whole "
            f"({tuple(tensor.shape)}) at rank {tp.rank}/{tp.size}"
        )
    if name.endswith(_VOCAB_PARALLEL):
        local = _pad_vocab_rows(local, tensor.shape[0], world_size=tp.size)
    return local


# exllamav3 zero-pads a linear to multiples of 128 on both axes (its bias too); the vision tower's MLP intermediate
# (4304 -> 4352) is the only padded axis in this model
_EXL3_VISION_MLP = re.compile(r"^visual\.blocks\.\d+\.mlp\.linear_fc(?P<fc>[12])\.(?P<leaf>weight|bias)$")


def _exl3_unpad(name: str, tensor: torch.Tensor, config) -> torch.Tensor:
    match = _EXL3_VISION_MLP.match(name)
    if match is None or config.vision_config is None:
        return tensor
    inter = config.vision_config.intermediate_size
    if match.group("fc") == "1":  # [inter, hidden] weight, [inter] bias
        return tensor[:inter].contiguous() if tensor.shape[0] > inter else tensor
    if match.group("leaf") == "weight":  # fc2 [hidden, inter]; its bias is on the unpadded hidden axis
        return tensor[:, :inter].contiguous() if tensor.shape[1] > inter else tensor
    return tensor


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the model's state dict minus the routed experts.
    Each Linear is read under the scheme the checkpoint's QuantConfig gives it, the same answer the model built its buffers from, so bf16, per-channel / per-tensor fp8, block-fp8, MXFP8 and NVFP4 projections in any mix land as the model's state dict.
    Fusions, per role: attention q|k|v -> ``qkv_proj``; GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, or ``in_proj_qkvz`` + ``in_proj_ba`` when qkv|z is quantized; shared-expert gate|up -> ``gate_up_proj``; each per-layer HC's ``input_mix_weight_down`` | ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.
    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the routed experts are NVFP4 and always come from the offload cache's expert reader.
    """
    # TP: the modules shard their own head counts, so the tensors backing them arrive rank-local;
    # ``_shard_for_rank`` runs before ``reader.add`` so the fusion parts are cut on their own axes.
    if not include_non_moe:
        return

    from .config import parse_config

    hf_config = cached_load_hf_config(model_path)
    config = parse_config(hf_config)
    spec = get_model_spec(hf_config.architectures[0])
    quant = get_quant_config()
    # EXL3: every dense linear is a trellis, served bf16 (only the routed experts stay EXL3)
    exl3 = None
    if quant is not None and quant.dialect == "exl3":
        from freetoken.models.exl3_weights import Exl3DenseReconstructor, checkpoint_tensor_names

        # codebook markers from every file: a module whose marker is not seen would decode with the wrong codebook
        exl3 = Exl3DenseReconstructor(checkpoint_tensor_names(download_hf_weight(model_path)), rename=_rename)
        quant = None  # the reconstructed weights are bf16: nothing for the reader to route by the dialect
    reader = _DenseReader(quant, spec.packed_modules_mapping)

    def place(name: str, tensor: torch.Tensor):
        if exl3 is not None:
            tensor = _exl3_unpad(name, tensor, config)
        tensor = _shard_for_rank(name, tensor, config=config)
        emitted = None if name.startswith(VISION_KEY_PREFIXES) else reader.add(name, tensor)
        if emitted is None:
            yield name, tensor
        else:
            yield from emitted

    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                if not include_vision and name.startswith(VISION_KEY_PREFIXES):
                    continue
                tensor = f.get_tensor(raw_name)
                rebuilt = exl3.feed(name, tensor) if exl3 is not None else None
                if rebuilt is None:
                    yield from place(name, tensor)
                else:
                    for weight_name, weight in rebuilt:
                        yield from place(weight_name, weight)

    if exl3 is not None:
        exl3.check_done()
    if reader.pending:
        lines = reader.missing()
        shown = "\n  ".join(lines[:8]) + (f"\n  ... {len(lines) - 8} more" if len(lines) > 8 else "")
        raise ValueError(f"checkpoint is missing tensors the quant config declares for {len(lines)} modules:\n  {shown}")


def iter_vision_weights(model_path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """The vision tower alone, named as iter_weights names it."""
    if _is_exl3():
        raise NotImplementedError("the encoder-only reader does not decode EXL3 tensors; load the whole checkpoint")
    for file in iter_weight_files(model_path):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is not None and name.startswith(VISION_KEY_PREFIXES):
                    yield name, f.get_tensor(raw_name)


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor scale (1 for a bf16 table)."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` view of the bank in the stored dtype."""
        return self.bank.tensor


_PLE_ST_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16}


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the PLE n-gram table tensors, and only those, into ``ple-table-*.safetensors`` next to an FTW checkpoint.

    The table is served from safetensors files in the checkpoint dir (see load_ple_table), not from FTW entries."""
    from safetensors.torch import save_file

    folder = download_hf_weight(model_path)
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush():
        nonlocal batch, size
        if batch:
            name = f"ple-table-{len(written):05d}.safetensors"
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(folder):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_FILE_BYTES:
                    flush()
    flush()
    return written


def _ple_row_shard(folder: str, bank_rows: int) -> tuple[int, int]:
    """This rank's ``[lo, hi)`` row range of the flat n-gram table.

    The table is sharded on the hash-head axis: rank ``r`` owns heads ``[r * H/tp, (r + 1) * H/tp)``,
    and each head's prime-sized vocab is laid out back to back, so those heads are one contiguous
    row range. The split point comes from the checkpoint's own ``ngram_heads_offsets`` (the tensor
    ``NGramEmbedding.local_row_base`` rebases lookups against); the last rank absorbs the padding.
    """
    from freetoken.models.loader import safetensors_weight_map

    info = get_tp_info()
    if info.size == 1:
        return 0, bank_rows
    offsets = None
    for name, shard in safetensors_weight_map(folder).items():
        if name.endswith(".ple.ple_embedding.ngram_heads_offsets"):
            with safetensors.safe_open(os.path.join(folder, shard), framework="pt", device="cpu") as f:
                offsets = f.get_tensor(name).tolist()
            break
    if offsets is None:
        raise ValueError("PLE table cannot be TP-sharded: no ngram_heads_offsets in the checkpoint")
    per_rank = div_even(len(offsets), info.size)
    lo = int(offsets[per_rank * info.rank])
    hi = bank_rows if info.rank == info.size - 1 else int(offsets[per_rank * (info.rank + 1)])
    return lo, hi


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB (fp8) and must not also sit in the page cache while the bank holds the same bytes.
    An fp8 table needs the checkpoint's scalar ``weight_scale``; a bf16 table without one is unscaled.
    """
    folder = download_hf_weight(model_path)
    if os.path.exists(os.path.join(folder, "ngram_embedding.safetensors")):
        raise ValueError("this EXL3 checkpoint's n-gram table is trellis-coded: serve it with --ple-backend disk")
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    dtype: torch.dtype | None = None
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            shard_dtype = _PLE_ST_DTYPES.get(meta["dtype"])
            if shard_dtype is None or dtype not in (None, shard_dtype):
                raise ValueError(f"PLE table shard {key} has dtype {meta['dtype']}, expected one of {sorted(_PLE_ST_DTYPES)} for every shard")
            dtype = shard_dtype
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        if dtype is torch.float8_e4m3fn:
            raise ValueError("PLE table has no weight_scale")
        scale = torch.tensor(1.0, dtype=torch.bfloat16)

    row_bytes = cols * dtype.itemsize
    shard_bytes = rows * row_bytes
    row_lo, row_hi = _ple_row_shard(folder, expected * rows)
    bank = HostBank((row_hi - row_lo, cols), dtype)
    bar = byte_bar((row_hi - row_lo) * row_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            # This shard covers global rows [lo, hi); keep only its overlap with this rank's range.
            lo, hi = shard * rows, (shard + 1) * rows
            take_lo, take_hi = max(lo, row_lo), min(hi, row_hi)
            if take_lo >= take_hi:
                continue
            take = (take_hi - take_lo) * row_bytes
            read_range_into(buf, path, file_offset=offset + (take_lo - lo) * row_bytes, nbytes=take,
                            dest_offset=(take_lo - row_lo) * row_bytes, workers=workers, chunk=chunk)
            bar.update(take)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def nvfp4_expert_spec(model_path: str, config) -> Nvfp4ExpertSourceSpec:
    """The per-expert NVFP4 layout under the checkpoint's dialect names (ModelOpt or llm-compressor)."""
    quant = get_quant_config()
    stored = quant.stored_tensors(QuantKind.NVFP4)
    kind_map = {stored[role].name: kind for role, kind in _BANK_KINDS.items()}
    return Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(_EXPERT_KEY_RE.format(kinds="|".join(map(re.escape, kind_map)))),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, config: layer,  # every layer is MoE
        desc=f"Qwen3.8-Flash-Next NVFP4 experts ({quant.dialect})",
        kind_map=kind_map,
        global_reciprocal=stored["weight_global"].reciprocal,
    )


class _CheckpointMoELayers:
    """``config`` as the checkpoint's own expert reader sees it: without the MTP head's extra bank layers."""

    def __init__(self, config) -> None:
        self._config = config
        self.num_moe_layers = config.num_moe_layers - config.mtp_layers

    def __getattr__(self, name):
        return getattr(self._config, name)


def iter_expert_pieces(model_path: str, config, kind, *, parallel: bool = False, workers: int = 8,
                       chunk: int = 8 << 20):
    """The family hook of moe/expert_pieces.iter_expert_pieces: None (the generic readers) unless the MTP head is
    built, in which case the checkpoint's NVFP4 experts are followed by the MTP's 512, quantized to NVFP4 here."""
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.nvfp4_banks import iter_nvfp4_expert_pieces

    if kind is QuantKind.EXL3:
        return _exl3_expert_pieces(model_path, config, parallel=parallel, workers=workers, chunk=chunk)
    if getattr(config, "mtp_layers", 0) and kind is not QuantKind.NVFP4:
        raise ValueError("FREETOKEN_MTP=1 needs an NVFP4 checkpoint (the MTP experts join its NVFP4 expert banks)")
    if not getattr(config, "mtp_layers", 0):
        # official FP8 checkpoints share qwen3_5_moe's block-fp8 expert layout; None leaves the generic readers
        from freetoken.models.qwen3_5_moe.weight import iter_expert_pieces as qwen3_5_pieces

        return qwen3_5_pieces(model_path, config, kind, parallel=parallel, workers=workers, chunk=chunk)
    main = iter_nvfp4_expert_pieces(model_path, _CheckpointMoELayers(config),
                                    nvfp4_expert_spec(model_path, config),
                                    parallel=parallel, workers=workers, chunk=chunk)
    return itertools.chain(main, _mtp_expert_pieces(model_path, bank_layer=config.num_moe_layers - 1))


_EXL3_EXPERT_RE = re.compile(
    r"^(?:model\.language_model\.layers\.(?P<layer>\d+)|mtp\.layers\.(?P<mtp>\d+))\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.\w+$"
)


def _exl3_expert_pieces(model_path: str, config, *, parallel: bool, workers: int, chunk: int):
    """EXL3 checkpoints store every expert per tensor, the MTP head's 512 too; with the head built they fill its bank
    layer (the last one), as quantized: no NVFP4 requantization."""
    from freetoken.models.exl3_weights import iter_exl3_expert_pieces

    mtp_bank = config.num_moe_layers - 1 if getattr(config, "mtp_layers", 0) else None
    roles = {"gate_proj": "gate", "up_proj": "up", "down_proj": "down"}

    def locate(name: str):
        match = _EXL3_EXPERT_RE.match(name)
        if match is None:
            return None
        if match.group("mtp") is not None:
            if mtp_bank is None or int(match.group("mtp")) != 0:
                return None
            layer = mtp_bank
        else:
            layer = int(match.group("layer"))
        return layer, int(match.group("expert")), roles[match.group("proj")]

    return iter_exl3_expert_pieces(model_path, locate, expected_experts=config.num_moe_layers * config.num_experts,
                                   parallel=parallel, workers=workers, chunk=chunk,
                                   desc="Qwen3.8-Flash-Next EXL3 experts")


def _mtp_expert_pieces(model_path: str, *, bank_layer: int):
    """One piece per MTP expert in the checkpoint experts' form (``gate`` / ``up`` / ``down`` e2m1 codes, their e4m3
    ``_scale`` and fp32 ``_global``), quantized from the stacked bf16 ``mtp.layers.0.mlp.experts.*`` (on the GPU)."""
    from freetoken.models.loader import safetensors_weight_map

    from .mtp import quantize_nvfp4

    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    names = {leaf: f"mtp.layers.0.mlp.experts.{leaf}" for leaf in ("gate_up_proj", "down_proj")}
    device = torch.device("cuda", torch.cuda.current_device())
    stacked = {}
    for leaf, name in names.items():
        if name not in weight_map:
            raise ValueError(f"FREETOKEN_MTP=1: no {name} in this checkpoint (the head's experts are read as stacked "
                             f"bf16, as RadixArk/Qwen3.8-Flash-Next-NVFP4 ships them)")
        with safetensors.safe_open(os.path.join(folder, weight_map[name]), framework="pt", device="cpu") as f:
            stacked[leaf] = f.get_tensor(name)
        if stacked[leaf].dtype is not torch.bfloat16:
            raise ValueError(f"FREETOKEN_MTP=1: {name} is {stacked[leaf].dtype}, the head's experts are read as bf16")
    gate_up, down = stacked["gate_up_proj"], stacked["down_proj"]
    inter = gate_up.shape[1] // 2
    for e in range(gate_up.shape[0]):
        piece = {}
        gu = gate_up[e].to(device)
        for role, w in (("gate", gu[:inter]), ("up", gu[inter:]), ("down", down[e].to(device))):
            codes, scale, glob = quantize_nvfp4(w)
            piece[role] = codes.cpu().unsqueeze(0)
            piece[role + "_scale"] = scale.cpu().unsqueeze(0)
            piece[role + "_global"] = glob.to(torch.float32).cpu().reshape(1, 1)
        yield bank_layer, e, e + 1, piece


__all__ = [
    "iter_expert_pieces",
    "nvfp4_expert_spec",
    "PleTable",
    "iter_weights",
    "load_ple_table",
]
