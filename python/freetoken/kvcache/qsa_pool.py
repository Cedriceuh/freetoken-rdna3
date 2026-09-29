"""QSA compressed-block sparse KV pool: paged GQA K/V + compressed index keys + pending ring.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups instead of single tokens, so
its indexer slab holds ONE compressed key row per group, addressed by ``slot //
index_ratio``. Because ``page_size % index_ratio == 0``, a group's tokens always live in one
page at consecutive slots, which makes that division well-defined: the compressed rows are a
1/ratio shadow of the K/V pages and follow page sharing and eviction for free -- no
allocator, no free, no clear (SGLang qsa_kv_pool / vLLM compressed-region precedent).

Two tiers ride alongside the shadow slab and are NOT per-token:
- ``pending_ring``: the last ``ring_capacity`` pre-RoPE index keys of each running request (sized by ``ring_capacity_for``), indexed by ``Req.table_idx``. A group that straddles two forwards (chunked prefill, and
  every decode step) reads its already-consumed members from here. Never cleared: a new
  tenant of a table_idx starts at a group boundary (cached_len is 0 or a page multiple), so
  its first closing group takes every member from its own forward.
- scratch rows at ``cmp_scratch_base``: one row per request slot, the write target for rows
  whose group does not close in this forward, so the compress kernel scatters unconditionally
  with no negative index and no cross-row conflict (DSV4 precedent).

The slab is amortized into the per-token KV price (``unit_bytes``); the ring and scratch are
fixed and priced through ``kv_cost``'s ``fixed_cache_size``.

``FREETOKEN_QSA_KV_INT8=1`` (EXPERIMENTAL, NOT RECOMMENDED: on a long-context agentic bench it cut the
success rate by about a third and changed the agent's behaviour, although short scored benches and needle
tests saw no loss -- one scale per (token, kv head) over 256 dims lets the keys' outlier channels crush the
others) stores the paged K/V as symmetric int8 with one fp32 scale per
(token, kv head) (kernel/triton/qsa/kv_int8.py writes it, the attend kernel dequantizes the
selected tokens): 520 instead of 1024 bytes per token per layer at head_dim 256, priced as such
in ``kv_cost`` so the planner hands the freed memory to the MoE expert cache. The index slab and
the ring stay in the compute dtype.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import torch

from .mha_pool import MHAKVCache

# The index tiers are always 2-byte (compute dtype); spec_kv_bytes_per_token budgets the same.
_INDEX_DTYPE_BYTES = 2
# t/h/w int32 rope position kept per KV slot on mrope models
_ROPE_POS_BYTES = 3 * 4
KV_INT8_ENV = "FREETOKEN_QSA_KV_INT8"
_KV_SCALE_BYTES = 4  # fp32 scale per (token, kv head), K and V each


def kv_int8_enabled() -> bool:
    return os.environ.get(KV_INT8_ENV) == "1"


class QSAKVCache(MHAKVCache):
    """MHA paged pool + the compressed index-key slab + the per-request pending ring.

    ``cmp_k_cache(slot)`` is row-flat ``[num_pages * page_size // index_ratio + num_req_slots,
    index_head_dim]``: row ``r < cmp_scratch_base`` holds the compressed key of the token group
    whose K/V slots are ``[r * index_ratio, (r + 1) * index_ratio)``, and the rows from
    ``cmp_scratch_base`` on are the per-request-slot scratch sinks. ``slot`` is the sparse
    layer's order in the attention backend, same convention as BSAKVCache/DSAKVCache.
    """

    # host KV tier: K/V (+ int8 scales), the compressed index rows and the mrope positions, all copied per page
    host_tier_supported = True

    @classmethod
    def ring_capacity_for(cls, index_ratio: int, num_speculative_tokens: int = 0) -> int:
        """Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing)."""
        return index_ratio * math.ceil((index_ratio + num_speculative_tokens) / index_ratio)

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        index_ratio: int,
        num_req_slots: int,
        ring_capacity: int | None = None,
        layer_ids: Sequence[int] | None = None,
        mrope: bool = False,
    ) -> None:
        if index_ratio < 1 or page_size % index_ratio != 0:
            # slot // index_ratio only names one group when a group never straddles a page.
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        if ring_capacity is None:
            ring_capacity = self.ring_capacity_for(index_ratio)
        if ring_capacity < index_ratio:
            # A closing group reads up to index_ratio - 1 past members plus this forward's.
            raise ValueError(
                f"QSA needs ring_capacity ({ring_capacity}) >= index_ratio ({index_ratio})"
            )
        # Index keys ride the compute dtype (the model's index_k is engine-dtype). The KV cost
        # model budgets 2 bytes per token per index layer for the slab
        # (base.spec_kv_bytes_per_token); keep the two in lockstep.
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._ring_capacity = ring_capacity
        self._index_dtype = dtype
        self._page_size = page_size
        self._mrope = mrope
        self._kv_int8 = kv_int8_enabled()
        self._kv_scale: torch.Tensor | None = None
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=torch.int8 if self._kv_int8 else dtype,
            device=device,
            layer_ids=layer_ids,
        )
        self._alloc_kv_scales()
        self._zero_kv_slabs()
        self._alloc_index_tiers(num_pages)

    def _alloc_kv_scales(self) -> None:
        if self._kv_int8:
            # [2, storage_layers, pages, page_size, kv_heads]: the K/V buffer without its head_dim axis
            self._kv_scale = torch.zeros(self._kv_buffer.shape[:5], dtype=torch.float32, device=self._device)

    def _zero_kv_slabs(self) -> None:
        # Defense-in-depth: the attend kernels pos-mask every K/V load (the real fix for
        # torch.empty's recycled NaN/Inf bit patterns), but a zeroed slab keeps any future
        # unmasked read finite instead of model-poisoning. One memset per (re)allocation.
        self._kv_buffer.zero_()
        if self._kv_scale is not None:
            self._kv_scale.zero_()

    def _alloc_index_tiers(self, num_pages: int) -> None:
        # ZERO-initialized: the score kernel reads whole rows of blocks unmasked and relies on
        # never-written tail rows dotting to a finite 0. Written rows are never cleared again,
        # so the kernel must clamp visible blocks to kvlen // index_ratio.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        self._cmp_k_buffer = torch.zeros(
            self._num_index_layers,
            self._cmp_scratch_base + self._num_req_slots,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        self._pending_ring = torch.zeros(
            self._num_req_slots,
            self._num_index_layers,
            self._ring_capacity,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        # 3-axis rope position of every stored token: a compressed group ropes at its first token, which under mrope is not derivable from the logical position
        self._rope_positions = (
            torch.zeros(num_pages * self._page_size, 3, dtype=torch.int32, device=self._device)
            if self._mrope
            else None
        )

    def rebuild(self, num_pages: int) -> None:
        # Free the index tiers BEFORE the K/V realloc (super().rebuild frees + syncs +
        # empty_cache), then re-derive them at the new page count. If the index alloc itself
        # fails (OOM), null the K/V slab too and re-raise: a pool with a grown K/V slab and no
        # index slab would mis-serve silently. Rebuild is idle-only, so zeroing the ring here
        # cannot drop a live request's pending members.
        self._cmp_k_buffer = None
        self._pending_ring = None
        self._rope_positions = None
        self._kv_scale = None
        super().rebuild(num_pages)
        self._alloc_kv_scales()
        self._zero_kv_slabs()
        try:
            self._alloc_index_tiers(num_pages)
        except Exception:
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            raise

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token
        from freetoken.attention import AttnType

        num_req_slots = config.max_running_req + 1
        per_token = 0
        fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            per_token += spec_kv_bytes_per_token(spec, config)
            if spec.attn_type is AttnType.QSA and kv_int8_enabled():
                # int8 codes + one fp32 scale per (token, kv head) instead of compute-dtype K/V
                from freetoken.utils import div_even

                heads = div_even(spec.num_kv_heads, config.tp_info.size, allow_replicate=True)
                slabs = 2 * heads * spec.num_layers
                per_token += slabs * (spec.head_dim * 1 + _KV_SCALE_BYTES) - slabs * spec.head_dim * config.dtype.itemsize
            if spec.attn_type is AttnType.QSA:
                # One index-key row = all index layers at one position.
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
                fixed += num_req_slots * row * (cls.ring_capacity_for(spec.index_ratio) + 1)
                if config.model_config.model_is_mrope:
                    per_token += _ROPE_POS_BYTES
        return per_token * config.page_size, fixed, config.page_size, 0

    def unit_bytes(self) -> tuple[int, int]:
        # Only the shadow slab scales with pages, and only its non-scratch rows; the ring and
        # the scratch rows are the fixed term kv_cost reports separately.
        kv, swa = super().unit_bytes()
        tokens = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        if self._kv_scale is not None:
            kv += int(self._kv_scale.numel() * self._kv_scale.element_size()) // tokens
        slab = (
            self._num_index_layers
            * self._cmp_scratch_base
            * self._index_head_dim
            * self._index_dtype.itemsize
        )
        return kv + slab // tokens + (_ROPE_POS_BYTES if self._mrope else 0), swa

    @property
    def dtype(self) -> torch.dtype:
        """The compute dtype (the index tiers'); the K/V slab itself is int8 under FREETOKEN_QSA_KV_INT8."""
        return self._index_dtype

    @property
    def kv_int8(self) -> bool:
        return self._kv_int8

    def k_scale(self, layer_id: int) -> torch.Tensor:
        """``[pages, page_size, kv_heads]`` fp32 scales of one layer's int8 keys."""
        assert self._kv_scale is not None, f"K/V scales exist only with {KV_INT8_ENV}=1"
        return self._kv_scale[0, self._dense(layer_id)]

    def v_scale(self, layer_id: int) -> torch.Tensor:
        assert self._kv_scale is not None, f"K/V scales exist only with {KV_INT8_ENV}=1"
        return self._kv_scale[1, self._dense(layer_id)]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        if not self._kv_int8:
            return super().store_kv(k, v, out_loc, layer_id)
        from freetoken.kernel.triton.qsa.kv_int8 import store_kv_int8

        dense = self._dense(layer_id)
        slots, heads, _ = self._storage_shape
        store_kv_int8(
            k, v, out_loc,
            self._k_buffer[dense].view(self._storage_shape),
            self._v_buffer[dense].view(self._storage_shape),
            self._kv_scale[0, dense].view(slots, heads),
            self._kv_scale[1, dense].view(slots, heads),
        )

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer: ``[rows, index_head_dim]``."""
        return self._cmp_k_buffer[slot]

    def pending_ring(self, slot: int) -> torch.Tensor:
        """One sparse layer's pending ring: ``[num_req_slots, ring_capacity, index_head_dim]``."""
        return self._pending_ring[:, slot]

    @property
    def rope_positions(self) -> torch.Tensor:
        """``[num_tokens, 3]`` int32 t/h/w rope position per KV slot (written by the QSA backend)."""
        assert self._rope_positions is not None, "rope positions are only kept on mrope models"
        return self._rope_positions

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row of ``cmp_k_cache``; row ``cmp_scratch_base + table_idx`` sinks a
        forward whose group does not close."""
        return self._cmp_scratch_base

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    @property
    def num_req_slots(self) -> int:
        return self._num_req_slots


__all__ = ["QSAKVCache"]
