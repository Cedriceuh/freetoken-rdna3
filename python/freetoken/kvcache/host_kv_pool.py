"""Host-RAM second tier for the hybrid radix cache (``FREETOKEN_HOST_KV=1``).

When device KV pages run short, ``HybridRadixCache`` evicts its least-recently-used leaves.
Without this tier their KV and GDN snapshot are simply dropped, and a returning conversation
(an agent coming back after other agents filled the pool) re-prefills its whole context. With
it, the evicted node is DEMOTED instead: its KV pages and GDN snapshot are copied to pinned host
memory and the node stays in the tree, marked ``on_host``. A later match that reaches a host
snapshot deeper than the device one PROMOTES the path back (fresh device pages + GDN slot,
host-to-device copies) instead of recomputing it. The copies are exact, so a promoted prefix
reads the same bits a never-evicted one would.

This class only owns the host buffers and the copies; the tree bookkeeping lives in
``HybridRadixCache``. Layout:

* KV: one host page = every byte the device pool keeps for one device page (K/V of all paged
  layers, the int8 scales if any, the QSA compressed index rows, the mrope positions), packed
  into one ``page_bytes`` row of a pinned ``uint8 [num_pages, page_bytes]`` buffer.
* GDN snapshots: one row of a pinned ``uint8 [num_snaps, snap_bytes]`` buffer per snapshot
  (conv + recurrent + declared slot states of every linear layer).

Device pages go through a small device staging buffer (gather, then one copy per run of
consecutive host pages). All copies run on a private stream that waits for the caller's current
stream and is waited on by it, so they are ordered after every earlier write of the evicted pages
and before every later use of the reused pages, whichever stream the scheduler is on.

Capacities are counted in pages and snapshots, never bytes, so both tensor-parallel ranks (whose
shards differ in size) take identical eviction and promotion decisions.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

HOST_KV_ENV = "FREETOKEN_HOST_KV"
HOST_KV_TOKENS_ENV = "FREETOKEN_HOST_KV_TOKENS"
HOST_KV_SNAPSHOTS_ENV = "FREETOKEN_HOST_KV_SNAPSHOTS"
HOST_KV_LOG_ENV = "FREETOKEN_HOST_KV_LOG"
_STAGING_PAGES = 32  # device pages per staging round (~24 MB at 12 KB/token/rank, page 64)
_ALIGN = 64


def host_kv_enabled() -> bool:
    return os.environ.get(HOST_KV_ENV) == "1"


def host_kv_log() -> bool:
    return os.environ.get(HOST_KV_LOG_ENV) == "1"


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class _Section:
    """One device tensor viewed as ``[num_pages or num_slots, *unit_shape]`` along ``axis``."""

    def __init__(self, name: str, tensor: torch.Tensor, axis: int, offset: int) -> None:
        self.name = name
        self.tensor = tensor
        self.axis = axis
        self.offset = offset
        shape = list(tensor.shape)
        del shape[axis]
        self.unit_shape = tuple(shape)
        numel = 1
        for s in shape:
            numel *= s
        self.nbytes = numel * tensor.element_size()

    def view(self, buf: torch.Tensor) -> torch.Tensor:
        """This section of ``buf`` (``uint8 [rows, row_bytes]``) as ``[rows, *unit_shape]``."""
        return buf[:, self.offset : self.offset + self.nbytes].view(self.tensor.dtype).unflatten(1, self.unit_shape)


def _runs(host_pages: Sequence[int]) -> List[Tuple[int, int, int]]:
    """``(position, first_host_page, count)`` for each run of consecutive host pages."""
    runs: List[Tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(host_pages) + 1):
        if i == len(host_pages) or host_pages[i] != host_pages[i - 1] + 1:
            runs.append((start, host_pages[start], i - start))
            start = i
    return runs


class HostKVPool:
    def __init__(
        self,
        kv_pool,
        linear_state_pool,
        page_size: int,
        num_pages: int,
        num_snaps: int,
        device: torch.device,
    ) -> None:
        self.page_size = page_size
        self.device = device
        self.num_pages = num_pages
        self.num_snaps = num_snaps
        self._kv_pool = kv_pool
        self._linear_pool = linear_state_pool
        self._pin = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self._pin else None
        self._build_page_sections()
        self._build_snap_sections()
        self.page_buf = _host_buffer(num_pages, self.page_bytes, self._pin)
        self.snap_buf = _host_buffer(num_snaps, self.snap_bytes, self._pin)
        self.staging = torch.empty(
            (_STAGING_PAGES, self.page_bytes), dtype=torch.uint8, device=device)
        self._free_pages: List[int] = list(range(num_pages))
        self._free_pages_sorted = True
        self._free_snaps: List[int] = list(range(num_snaps))
        self.stats = {"demoted_tokens": 0, "promoted_tokens": 0, "dropped_tokens": 0,
                      "demoted_snaps": 0, "promoted_snaps": 0, "adopted_tokens": 0, "released_snaps": 0}
        gib = (self.page_buf.numel() + self.snap_buf.numel()) / 2**30
        logger.info(
            f"host KV tier: {num_pages} pages x {self.page_bytes} B ({num_pages * page_size} tokens)"
            f" + {num_snaps} GDN snapshots x {self.snap_bytes} B = {gib:.2f} GiB pinned")

    # ------------------------------------------------------------------ layout
    def _page_tensors(self) -> List[Tuple[str, torch.Tensor, int]]:
        """``(name, tensor, page_axis)`` for every per-page device tensor of the KV pool."""
        pool = self._kv_pool
        ps = self.page_size
        out = [("kv", pool._kv_buffer, 2)]
        scale = getattr(pool, "_kv_scale", None)
        if scale is not None:
            out.append(("kv_scale", scale, 2))
        cmp = getattr(pool, "_cmp_k_buffer", None)
        if cmp is not None:
            base = pool.cmp_scratch_base
            pages = int(pool._kv_buffer.shape[2])
            rows = ps // pool.index_ratio
            assert base == pages * rows, (base, pages, rows)
            out.append(("cmp", cmp[:, :base].unflatten(1, (pages, rows)), 1))
        rope = getattr(pool, "_rope_positions", None)
        if rope is not None:
            out.append(("rope", rope.unflatten(0, (-1, ps)), 0))
        return out

    def _build_page_sections(self) -> None:
        self.page_sections: List[_Section] = []
        off = 0
        for name, t, axis in self._page_tensors():
            sec = _Section(name, t, axis, off)
            self.page_sections.append(sec)
            off = _align(off + sec.nbytes)
        self.page_bytes = off

    def _build_snap_sections(self) -> None:
        pool = self._linear_pool
        tensors = [("conv", pool.conv_states), ("recurrent", pool.recurrent_states)]
        tensors += [(f"slot:{k}", v) for k, v in pool.slot_states.items()]
        self.snap_sections: List[_Section] = []
        off = 0
        for name, t in tensors:
            sec = _Section(name, t, 1, off)
            self.snap_sections.append(sec)
            off = _align(off + sec.nbytes)
        self.snap_bytes = off

    def refresh(self) -> None:
        """Re-point the sections after a pool rebuild (the device tensors were replaced) and
        drop every host entry (the tree that referenced them is discarded too)."""
        old_page, old_snap = self.page_bytes, self.snap_bytes
        self._build_page_sections()
        self._build_snap_sections()
        assert (self.page_bytes, self.snap_bytes) == (old_page, old_snap), (
            "host KV tier: pool geometry changed on rebuild")
        self.reset()

    def reset(self) -> None:
        self._free_pages = list(range(self.num_pages))
        self._free_pages_sorted = True
        self._free_snaps = list(range(self.num_snaps))

    # ------------------------------------------------------------------ allocation
    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def free_snaps(self) -> int:
        return len(self._free_snaps)

    def alloc_pages(self, n: int) -> List[int]:
        assert n <= len(self._free_pages), (n, len(self._free_pages))
        if not self._free_pages_sorted:
            self._free_pages.sort()
            self._free_pages_sorted = True
        out, self._free_pages = self._free_pages[:n], self._free_pages[n:]
        return out

    def free_page_list(self, pages: Sequence[int]) -> None:
        if pages:
            self._free_pages.extend(pages)
            self._free_pages_sorted = False

    def alloc_snap(self) -> int:
        return self._free_snaps.pop()

    def free_snap(self, s: int) -> None:
        self._free_snaps.append(s)

    # ------------------------------------------------------------------ copies
    def _enter(self):
        if self.stream is None:
            return None
        cur = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(cur)
        return cur

    def _leave(self, cur) -> None:
        if cur is not None:
            cur.wait_stream(self.stream)

    def _ctx(self):
        return torch.cuda.stream(self.stream) if self.stream is not None else _Null()

    def save_pages(self, dev_pages: torch.Tensor, host_pages: Sequence[int]) -> None:
        """Copy device pages ``dev_pages`` (page ids) to host pages ``host_pages``."""
        n = len(host_pages)
        assert int(dev_pages.numel()) == n
        cur = self._enter()
        with self._ctx():
            idx_all = dev_pages.to(device=self.device, dtype=torch.long)
            for s in range(0, n, _STAGING_PAGES):
                c = min(_STAGING_PAGES, n - s)
                idx = idx_all[s : s + c]
                stage = self.staging[:c]
                for sec in self.page_sections:
                    sec.view(stage).copy_(sec.tensor.index_select(sec.axis, idx).movedim(sec.axis, 0))
                for pos, h0, cnt in _runs(host_pages[s : s + c]):
                    self.page_buf[h0 : h0 + cnt].copy_(stage[pos : pos + cnt], non_blocking=True)
        self._leave(cur)

    def load_pages(self, host_pages: Sequence[int], dev_pages: torch.Tensor) -> None:
        """Copy host pages ``host_pages`` into device pages ``dev_pages`` (page ids)."""
        n = len(host_pages)
        assert int(dev_pages.numel()) == n
        cur = self._enter()
        with self._ctx():
            idx_all = dev_pages.to(device=self.device, dtype=torch.long)
            for s in range(0, n, _STAGING_PAGES):
                c = min(_STAGING_PAGES, n - s)
                idx = idx_all[s : s + c]
                stage = self.staging[:c]
                for pos, h0, cnt in _runs(host_pages[s : s + c]):
                    stage[pos : pos + cnt].copy_(self.page_buf[h0 : h0 + cnt], non_blocking=True)
                for sec in self.page_sections:
                    sec.tensor.index_copy_(sec.axis, idx, sec.view(stage).movedim(0, sec.axis))
        self._leave(cur)

    def save_snap(self, slot: int, hs: int) -> None:
        cur = self._enter()
        with self._ctx():
            row = self.snap_buf[hs : hs + 1]
            for sec in self.snap_sections:
                dst = sec.view(row)[0]
                for layer in range(sec.tensor.shape[0]):
                    dst[layer].copy_(sec.tensor[layer, slot], non_blocking=True)
        self._leave(cur)

    def load_snap(self, hs: int, slot: int) -> None:
        cur = self._enter()
        with self._ctx():
            row = self.snap_buf[hs : hs + 1]
            for sec in self.snap_sections:
                src = sec.view(row)[0]
                for layer in range(sec.tensor.shape[0]):
                    sec.tensor[layer, slot].copy_(src[layer], non_blocking=True)
        self._leave(cur)


_REGISTERED: List[torch.Tensor] = []  # registered host buffers stay alive for the process


def _host_buffer(rows: int, row_bytes: int, pin: bool) -> torch.Tensor:
    """``uint8 [rows, row_bytes]`` host buffer, page-locked when ``pin``.

    Pinned through ``hipHostRegister`` on a plain allocation rather than ``pin_memory=True``:
    PyTorch's caching host allocator rounds every block up to a power of two, which would turn a
    6 GiB tier into 8 GiB of locked RAM. Falls back to the caching allocator if registering fails.
    """
    size = rows * row_bytes
    if not pin:
        return torch.empty((rows, row_bytes), dtype=torch.uint8)
    raw = torch.empty(size + 4096, dtype=torch.uint8)
    skip = (-raw.data_ptr()) % 4096
    buf = raw[skip : skip + size]
    err = torch._C._cudart.cudaHostRegister(buf.data_ptr(), size, 0)
    if int(err) != 0:
        logger.warning(f"host KV tier: hipHostRegister failed ({err}), using pin_memory")
        return torch.empty((rows, row_bytes), dtype=torch.uint8, pin_memory=True)
    _REGISTERED.append(raw)
    return buf.view(rows, row_bytes)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_host_kv_pool(kv_pool, linear_state_pool, page_size: int, device: torch.device):
    """The host tier from the env knobs, or None when disabled or unsupported."""
    if not host_kv_enabled():
        return None
    # A pool class opts in by declaring host_tier_supported itself (not inherited): its every per-page tensor must be
    # one HostKVPool._page_tensors copies (K/V, int8 scales, QSA index rows, mrope positions). Any other pool would be
    # promoted with state missing, so the tier refuses it.
    if (kv_pool is None or linear_state_pool is None
            or not type(kv_pool).__dict__.get("host_tier_supported", False)):
        logger.warning(f"FREETOKEN_HOST_KV=1 ignored: needs a GDN state pool and a KV pool that declares "
                       f"host_tier_supported (got {type(kv_pool).__name__})")
        return None
    # Defaults sized for Qwen3.8-Flash-Next at TP=2: ~12.8 KB of KV per token and ~60 MB per GDN
    # snapshot per rank -> 262144 tokens + 24 snapshots ~= 4.3-4.6 GiB of locked RAM per rank on xtx-xt.
    tokens = int(os.environ.get(HOST_KV_TOKENS_ENV, str(256 * 1024)))
    snaps = int(os.environ.get(HOST_KV_SNAPSHOTS_ENV, "24"))
    pages = max(1, tokens // page_size)
    return HostKVPool(kv_pool, linear_state_pool, page_size, pages, max(1, snaps), device)


__all__ = ["HostKVPool", "make_host_kv_pool", "host_kv_enabled", "host_kv_log"]
