"""Uneven tensor-parallel split for mismatched cards (opt-in, ``FREETOKEN_TP_SPLIT``).

Every TP axis is split evenly by default. ``FREETOKEN_TP_SPLIT`` gives the ranks relative shares
instead, for the axes whose consumers accept any split: the routed experts' intermediate (NVFP4
Triton expert kernels), and on models that wire it (qwen4_exp) the GatedDeltaNet heads and the shared
expert's intermediate. Only qwen4_exp is tested. Attention, the vocab-parallel pair and (qwen4_exp) the PLE heads stay even.
Unequal cards also need ``FREETOKEN_TP_ALLOW_IMBALANCE=1`` (plan memory on the smaller one). The value is either rank 0's fraction at TP=2
("0.55") or one weight per rank ("11:9"). A faster / larger rank 0 then does more of the sharded
work and holds bigger expert slices, and the smaller card's expert slots shrink, so the two
ranks' expert caches fill both cards instead of stopping at the smaller one.

Each axis is cut on multiples of its ``unit`` (1 head, 16 intermediate rows: one NVFP4 scale
block, and a whole number of int32 words of packed codes), rounding the cumulative boundaries so
the sizes always add up to the full axis.
"""

from __future__ import annotations

import os
from functools import lru_cache

from freetoken.distributed.info import get_tp_info
from freetoken.utils import div_even

ENV = "FREETOKEN_TP_SPLIT"


@lru_cache(maxsize=None)
def _parse_shares(raw: str, world_size: int) -> tuple[float, ...]:
    if ":" in raw:
        weights = [float(x) for x in raw.split(":")]
        if len(weights) != world_size:
            raise ValueError(f"{ENV}={raw!r} gives {len(weights)} weights for tp_size={world_size}")
    else:
        if world_size != 2:
            raise ValueError(f"{ENV}={raw!r}: a single fraction needs tp_size=2, got {world_size}")
        first = float(raw)
        weights = [first, 1.0 - first]
    if any(w <= 0 for w in weights):
        raise ValueError(f"{ENV}={raw!r}: every share must be positive")
    total = sum(weights)
    return tuple(w / total for w in weights)


def tp_shares(world_size: int | None = None) -> tuple[float, ...] | None:
    """Per-rank shares from ``FREETOKEN_TP_SPLIT``, or None for the default even split."""
    raw = os.environ.get(ENV, "").strip()
    world_size = get_tp_info().size if world_size is None else world_size
    if not raw or world_size == 1:
        return None
    return _parse_shares(raw, world_size)


def is_uneven() -> bool:
    return tp_shares() is not None


def tp_partition(
    total: int, unit: int = 1, *, rank: int | None = None, world_size: int | None = None
) -> tuple[int, int]:
    """``(offset, size)`` of ``rank``'s slice of an axis of ``total`` elements.

    Even split (``div_even`` semantics, exact division required) without ``FREETOKEN_TP_SPLIT``.
    """
    info = get_tp_info() if rank is None or world_size is None else None
    rank = info.rank if rank is None else rank
    world_size = info.size if world_size is None else world_size
    shares = tp_shares(world_size)
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
    k_off, k_n = tp_partition(num_k_heads, 1, rank=rank, world_size=world_size)
    return k_off, k_n, k_off * group, k_n * group


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


__all__ = [
    "ENV",
    "INTERMEDIATE_UNIT",
    "gdn_head_partition",
    "intermediate_partition",
    "is_uneven",
    "set_intermediate_unit",
    "tp_partition",
    "tp_shares",
]
