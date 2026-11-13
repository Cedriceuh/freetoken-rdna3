"""Uneven tensor-parallel split for mismatched cards (opt-in, ``FREETOKEN_TP_SPLIT``).

Every TP axis is split evenly by default. ``FREETOKEN_TP_SPLIT`` gives the ranks relative shares
instead, for the axes whose consumers accept any split: the routed experts' intermediate (NVFP4
Triton expert kernels), and on models that wire it (qwen4_exp) the GatedDeltaNet heads and the shared
expert's intermediate. Only qwen4_exp is tested. Attention, the vocab-parallel pair and (qwen4_exp) the PLE heads stay even
(``FREETOKEN_ATTN_RANK`` moves the attention whole to one rank: ``attn_owner``).
Unequal cards also need ``FREETOKEN_TP_ALLOW_IMBALANCE=1`` (plan memory on the smaller one). The value is either rank 0's fraction at TP=2
("0.55") or one weight per rank ("11:9"). A faster / larger rank 0 then does more of the sharded
work and holds bigger expert slices, and the smaller card's expert slots shrink, so the two
ranks' expert caches fill both cards instead of stopping at the smaller one.

Each axis is cut on multiples of its ``unit`` (1 head, 16 intermediate rows: one NVFP4 scale
block, and a whole number of int32 words of packed codes), rounding the cumulative boundaries so
the sizes always add up to the full axis.

``FREETOKEN_TP_SPLIT_DENSE`` (with ``FREETOKEN_TP_SPLIT``, same syntax) gives the axes other than the routed experts
(GatedDeltaNet heads, shared expert, LM head rows) shares of their own. An expert format with wide blocks cannot be
halved (EXL3: five 128-wide blocks, 3 + 2 on two ranks): on identical cards the rank holding 3 bounds the expert
cache, and the other axes the other way round (``0.6`` with ``0.4``) leave it more memory and even out the work.
"""

from __future__ import annotations

import os
from functools import lru_cache

from freetoken.distributed.info import get_tp_info
from freetoken.utils import div_even

ENV = "FREETOKEN_TP_SPLIT"
ENV_DENSE = "FREETOKEN_TP_SPLIT_DENSE"
ENV_ATTN_RANK = "FREETOKEN_ATTN_RANK"


@lru_cache(maxsize=None)
def _parse_shares(raw: str, world_size: int, env: str = ENV) -> tuple[float, ...]:
    if ":" in raw:
        weights = [float(x) for x in raw.split(":")]
        if len(weights) != world_size:
            raise ValueError(f"{env}={raw!r} gives {len(weights)} weights for tp_size={world_size}")
    else:
        if world_size != 2:
            raise ValueError(f"{env}={raw!r}: a single fraction needs tp_size=2, got {world_size}")
        first = float(raw)
        weights = [first, 1.0 - first]
    if any(w <= 0 for w in weights):
        raise ValueError(f"{env}={raw!r}: every share must be positive")
    total = sum(weights)
    return tuple(w / total for w in weights)


def tp_shares(world_size: int | None = None, *, dense: bool = False) -> tuple[float, ...] | None:
    """Per-rank shares from ``FREETOKEN_TP_SPLIT``, or None for the default even split. ``dense``: the shares of the
    axes other than the routed experts, ``FREETOKEN_TP_SPLIT_DENSE`` when it is set."""
    raw = os.environ.get(ENV, "").strip()
    world_size = get_tp_info().size if world_size is None else world_size
    if not raw or world_size == 1:
        return None
    own = os.environ.get(ENV_DENSE, "").strip() if dense else ""
    if own:
        return _parse_shares(own, world_size, ENV_DENSE)
    return _parse_shares(raw, world_size)


def is_uneven() -> bool:
    return tp_shares() is not None


def tp_partition(
    total: int, unit: int = 1, *, rank: int | None = None, world_size: int | None = None, dense: bool = False
) -> tuple[int, int]:
    """``(offset, size)`` of ``rank``'s slice of an axis of ``total`` elements.

    Even split (``div_even`` semantics, exact division required) without ``FREETOKEN_TP_SPLIT``.
    """
    info = get_tp_info() if rank is None or world_size is None else None
    rank = info.rank if rank is None else rank
    world_size = info.size if world_size is None else world_size
    shares = tp_shares(world_size, dense=dense)
    if shares is None:
        size = div_even(total, world_size)
        return rank * size, size
    assert total % unit == 0, f"axis of {total} is not a multiple of the split unit {unit}"
    units = total // unit
    bounds = [0]
    acc = 0.0
    for share in shares[:-1]:
        acc += share
        bounds.append(round(units * acc))
    bounds.append(units)
    sizes = [b - a for a, b in zip(bounds, bounds[1:])]
    if min(sizes) < 1:
        raise ValueError(f"{ENV} leaves a rank no {unit}-wide unit of an axis of {total}: {sizes}")
    return bounds[rank] * unit, sizes[rank] * unit


def gdn_head_partition(
    num_k_heads: int, num_v_heads: int, *, rank: int | None = None, world_size: int | None = None
) -> tuple[int, int, int, int]:
    """``(k_offset, k_heads, v_offset, v_heads)`` of a rank's GatedDeltaNet heads.

    The k heads are split; each k head keeps its ``num_v_heads / num_k_heads`` v heads (GQA
    grouping), so the v slice follows. With fewer heads than ranks every rank holds one shared
    head (the ``div_even(..., allow_replicate=True)`` rule), evenly.
    """
    info = get_tp_info() if rank is None or world_size is None else None
    rank = info.rank if rank is None else rank
    world_size = info.size if world_size is None else world_size

    def replicated_or_even(heads: int) -> tuple[int, int]:
        if world_size > heads:
            assert world_size % heads == 0, f"{world_size} ranks must be divisible by {heads} heads"
            return rank * heads // world_size, 1
        size = div_even(heads, world_size)
        return rank * size, size

    if world_size > num_k_heads or tp_shares(world_size) is None:
        k_off, k_n = replicated_or_even(num_k_heads)
        v_off, v_n = replicated_or_even(num_v_heads)
        return k_off, k_n, v_off, v_n
    assert num_v_heads % num_k_heads == 0, f"{num_v_heads} v heads do not group over {num_k_heads} k heads"
    group = num_v_heads // num_k_heads
    k_off, k_n = tp_partition(num_k_heads, 1, rank=rank, world_size=world_size, dense=True)
    return k_off, k_n, k_off * group, k_n * group


def attn_owner(world_size: int | None = None) -> int | None:
    """``FREETOKEN_ATTN_RANK=<r>``: rank ``r`` holds every attention head and the whole K/V, the other ranks none (they
    add zeros to the output's all-reduce). None: each rank its share of the heads. For a model whose K/V heads cannot
    follow an uneven split (Qwen3.8-Flash-Next has two): the K/V leaves the rank whose expert slices bound the cache."""
    raw = os.environ.get(ENV_ATTN_RANK, "").strip()
    world_size = get_tp_info().size if world_size is None else world_size
    if not raw or world_size == 1:
        return None
    owner = int(raw)
    if not 0 <= owner < world_size:
        raise ValueError(f"{ENV_ATTN_RANK}={raw!r} names no rank of tp_size={world_size}")
    if tp_shares(world_size) is None:
        # the ranks' K/V then differ in size: each must plan its own memory, which the uneven split's planner does
        raise ValueError(f"{ENV_ATTN_RANK} needs {ENV} (the ranks plan their own memory under the uneven split)")
    return owner


def attn_local_heads(
    num_heads: int, *, rank: int | None = None, world_size: int | None = None, allow_replicate: bool = False
) -> int:
    """This rank's attention heads (query or K/V): its even share, or all / none under ``FREETOKEN_ATTN_RANK``."""
    info = get_tp_info() if rank is None or world_size is None else None
    rank = info.rank if rank is None else rank
    world_size = info.size if world_size is None else world_size
    owner = attn_owner(world_size)
    if owner is None:
        return div_even(num_heads, world_size, allow_replicate=allow_replicate)
    return num_heads if rank == owner else 0


# the split unit of the expert / shared-expert intermediate axis (see the module docstring). Example:
# Qwen3.8-Flash-Next's 640 at 0.55 -> 352/288 (7900 XTX + 7900 XT). The routed-expert split applies where the expert
# kernel slices by rank range (NVFP4 Triton, layers/quantization/moe/nvfp4.py); the GDN-head and shared-expert splits to the models that ask for them
INTERMEDIATE_UNIT = 16
_intermediate_unit = INTERMEDIATE_UNIT


def set_intermediate_unit(unit: int) -> None:
    """A coarser intermediate split unit for expert formats with wider blocks (EXL3: the 128-wide Hadamard blocks);
    set by the checkpoint's QuantConfig before the model sizes anything."""
    global _intermediate_unit
    _intermediate_unit = max(INTERMEDIATE_UNIT, unit)


def intermediate_partition(total: int, *, rank: int | None = None, world_size: int | None = None) -> tuple[int, int]:
    return tp_partition(total, _intermediate_unit, rank=rank, world_size=world_size)


def shared_intermediate_partition(total: int, *, rank: int | None = None, world_size: int | None = None) -> tuple[int, int]:
    """The shared expert's slice: the routed experts' unit, the dense axes' shares (FREETOKEN_TP_SPLIT_DENSE)."""
    return tp_partition(total, _intermediate_unit, rank=rank, world_size=world_size, dense=True)


# FREETOKEN_LMHEAD_SPLIT=1 (opt-in, with FREETOKEN_TP_SPLIT, on the models that wire it: qwen4_exp): the LM head's
# vocabulary rows follow the ranks' shares instead of an even split. Each logit is one rank's dot product, so its bits
# do not depend on the split (the int8 GEMV keys its K tiles on the head's shape: kernel/triton/int8_gemv.py)
LMHEAD_SPLIT = os.environ.get("FREETOKEN_LMHEAD_SPLIT", "0") == "1"
VOCAB_UNIT = 64


def vocab_shards(total: int, world_size: int | None = None) -> list[tuple[int, int]] | None:
    """Every rank's ``(offset, rows)`` of an uneven LM head (FREETOKEN_LMHEAD_SPLIT), or None for the even split."""
    world_size = get_tp_info().size if world_size is None else world_size
    if not LMHEAD_SPLIT or tp_shares(world_size) is None or total % VOCAB_UNIT:
        return None
    return [tp_partition(total, VOCAB_UNIT, rank=r, world_size=world_size, dense=True) for r in range(world_size)]


__all__ = [
    "ENV",
    "ENV_ATTN_RANK",
    "ENV_DENSE",
    "attn_local_heads",
    "attn_owner",
    "INTERMEDIATE_UNIT",
    "gdn_head_partition",
    "intermediate_partition",
    "is_uneven",
    "set_intermediate_unit",
    "shared_intermediate_partition",
    "tp_partition",
    "tp_shares",
    "vocab_shards",
]
