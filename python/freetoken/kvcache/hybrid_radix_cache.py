"""Hybrid (full-attn KV + GDN linear-state) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (Option C) that REUSES the shared ``RadixTreeNode``
and walk/split logic, so the production KV radix is untouched (zero risk to non-hybrid models).
It adds a second "currency": an optional GDN state snapshot (``node.mamba_value`` = a
LinearStatePool slot id) attached at chunk/page-aligned boundary nodes, with its own LRU
eviction. Mirrors sglang ``MambaRadixCache`` (donate-not-copy, dual eviction, internal-node
tombstone, ``full_ref >= mamba_ref``) on FreeToken's tree.

Currency seam: the secondary value + its eviction is the slot a future SWA component plugs
into. This class is pool-agnostic -- it stores/returns slot ids and KV page indices; the
caller (CacheManager / scheduler) does the actual LinearStatePool / KV-pool free.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down, init_logger

from .base import BaseCacheHandle
from .host_kv_pool import host_kv_log
from .radix_cache import RadixTreeNode, _get_key_fn

logger = init_logger(__name__)


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable
    KV page indices. ``cached_len`` is already truncated to the deepest live-snapshot boundary.
    Plugs into PrefillAdder (reads ``.cached_len`` / ``.get_matched_indices()``) like the plain
    RadixCacheHandle; the restore slot rides on ``MatchResult.mamba_value``."""

    node: RadixTreeNode
    kv_indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free


class HybridRadixCache:
    def __init__(self, device: torch.device, page_size: int) -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE
        # Snapshots land on ×CHUNK_SIZE boundaries; require them to be page-aligned so the KV
        # node boundary and the GDN-state boundary coincide (page_size in {1,2,4,8,16,32,64}).
        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.mamba_evictable = 0     # number of live, unlocked snapshots
        self.mamba_protected = 0
        # Host tier (FREETOKEN_HOST_KV=1, attached by CacheManager): evicted leaves are DEMOTED
        # to host memory instead of dropped, and promoted back on a deeper host match. None =
        # the original drop-on-evict behaviour, bit for bit.
        self.host = None
        self.host_tokens = 0         # tokens whose KV lives in host memory (not in full_*)

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        """Match the token prefix, then truncate the reusable length to the deepest node on
        the path that still owns a LIVE snapshot (a continuation can only resume the GDN
        recurrence from a checkpointed boundary)."""
        node, _ = self._walk(input_ids)
        # walk up to the deepest node whose END boundary has a live snapshot
        cur, end_len = node, self._path_len(node)
        while not cur.is_root():
            if cur.mamba_value is not None:
                return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: int) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. Returns (dup_len, mamba_exist): dup_len is the length the tree already held on
        the DEVICE (the caller frees its own pages for that span as duplicates; with the host tier, host
        nodes on the path adopt the caller's pages instead, so they do not count). If the boundary node
        already owns a live snapshot, returns mamba_exist=True and does not attach (caller frees the
        donated slot -- dedup)."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = self._walk(input_ids)
        # ``dup_len``: the length whose KV the tree already held on the DEVICE -- the caller frees
        # its own pages for that span as duplicates and keeps the rest as tree-owned.
        dup_len = prefix_len
        if self.host_tokens:
            # A host node on the path holds exactly these tokens, but only in host memory: ADOPT
            # the caller's freshly computed pages as its device KV (dropping the host copy), and
            # report the duplicate span as ending where the device part ended.
            dup_len = self._adopt_host_path(node, kv_indices)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], kv_indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            node = new_node
        if node.is_root():
            return dup_len, True      # root can't hold a snapshot; report exist so caller frees it
        if node.mamba_value is not None:
            return dup_len, True                    # dedup: caller frees its donated slot
        node.mamba_value = mamba_value              # fills a fresh node or a tombstone
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        return dup_len, False

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> None:
        """Protect a matched node's snapshot (mamba ref on the node) and its KV path
        (full ref node..root). Enforces full_ref >= mamba_ref: using a snapshot at N pins the
        whole root..N KV chain."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too -- with the host tier,
        demotes the needed tail of the node (KV + snapshot) to host memory instead."""
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, mamba, freed = [], [], 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if (node.ref_count != 0 or node.on_host or not self._is_dev_leaf(node)
                    or node.is_root()):
                continue
            if self.host is not None:
                # a turn's KV is ONE node (only a prefill's last chunk commits): demote just the tail this
                # eviction needs, not a whole 100k-token context that then overflows the host tier
                node = self._split_for_eviction(node, num_tokens - freed)
            freed += node.length
            parent = self._evict_node(node, kv, mamba)
            parent, casc = self._cascade_tombstone_leaves(parent, kv, mamba)
            freed += casc
            if (self._is_dev_leaf(parent) and parent.ref_count == 0 and not parent.is_root()
                    and not parent.on_host):
                heapq.heappush(leaves, parent)
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    def evict_mamba(self, num: int) -> EvictResult:
        """Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes
        too. Internal node -> TOMBSTONE (free the slot, keep KV + children). Leaf node -> free
        both KV and slot and unlink, then cascade-delete any KV-only tombstone leaves it exposes
        upward (so a leaf always carries a live snapshot -- mirrors sglang)."""
        cands = [n for n in self._snapshot_nodes() if n.mamba_ref_count == 0]
        heapq.heapify(cands)
        kv, mamba, freed = [], [], 0
        while freed < num and cands:
            node = heapq.heappop(cands)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            if self._is_dev_leaf(node) and node.ref_count == 0:
                freed += 1
                if self.host is not None:
                    # a GDN slot is wanted, not KV room: demote the snapshot with a one-page tail, the rest of
                    # the context stays on the device as its (tombstone) prefix
                    node = self._split_for_eviction(node, self.page_size)
                self._cascade_tombstone_leaves(self._evict_node(node, kv, mamba), kv, mamba)
            else:
                self._free_node_mamba(node, mamba)  # tombstone internal (or locked-KV) node
                freed += 1
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    @property
    def size_info(self):
        """KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage).
        The GDN-snapshot currency is reported via mamba_evictable_size."""
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        # Structural: every snapshot-bearing node holds a real slot id; ref counts non-negative.
        # (KV/page conservation is checked by CacheManager.check_integrity.)
        for n in self._snapshot_nodes():
            assert n.mamba_value is not None and n.mamba_ref_count >= 0 and n.ref_count >= 0
        if self.host is not None:
            tokens = snaps = 0
            stack = [self.root]
            while stack:
                n = stack.pop()
                stack.extend(n.children.values())
                if n.on_host:
                    assert n.ref_count == 0 and n.mamba_value is None, "locked/live host node"
                    assert all(c.on_host for c in n.children.values()), "device below host"
                    tokens += n.length
                    snaps += n.host_mamba is not None
                else:
                    assert n.host_mamba is None, "host snapshot on a device node"
            assert tokens == self.host_tokens, (tokens, self.host_tokens)
            h = self.host
            assert h.free_pages + tokens // self.page_size == h.num_pages, "host page leak"
            assert h.free_snaps + snaps == h.num_snaps, "host snapshot leak"

    # ---------------------------------------------------------------- helpers
    def _free_node_mamba(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_value is not None:
            out.append(node.mamba_value)
            node.mamba_value = None
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_tombstone_leaves(self, parent: RadixTreeNode, kv_out: List[torch.Tensor],
                                  mamba_out: List[int]):
        """After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes
        upward (mamba_value None, no children, unlocked): free their KV and unlink, walking up.
        Keeps the 'a leaf always carries a live snapshot' invariant (sglang
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed_tokens).
        Host tier: a tombstone whose host children still need it as their prefix stays on the device (a
        promotion reads it in place); eviction still takes it under real pressure."""
        freed = 0
        # `not parent.children`: a tombstone that is the prefix of host nodes is not reclaimed eagerly (a promotion
        # needs it; eviction still takes it under real pressure). Without the host tier every device leaf is
        # childless, so this is the original condition.
        while (parent.mamba_value is None and not parent.on_host and not parent.children
               and parent.ref_count == 0 and not parent.is_root()):
            freed += parent.length
            parent = self._evict_node(parent, kv_out, mamba_out)
        return parent, freed

    def _split_for_eviction(self, node: RadixTreeNode, need: int) -> RadixTreeNode:
        """Host tier: split an unlocked device leaf so that only its last ``need`` tokens (page-rounded) are evicted.
        Returns the tail (keeps the snapshot and any host children); the head stays on the device as its prefix."""
        want = -(-need // self.page_size) * self.page_size
        if want >= node.length:
            return node
        node.split_at(node.length - want)  # the new node is the head; `node` keeps the tail
        return node

    # ---------------------------------------------------------------- host tier
    def _is_dev_leaf(self, node: RadixTreeNode) -> bool:
        """No DEVICE child. Host children depend on this node's KV as their prefix, but do not
        pin it on the device: it can be demoted after them. Equals ``is_leaf`` without a tier."""
        return all(c.on_host for c in node.children.values())

    def _evict_node(self, node: RadixTreeNode, kv_out: List[torch.Tensor],
                    mamba_out: List[int]) -> RadixTreeNode:
        """Evict one unlocked device leaf: its device pages (and slot) go to the caller either
        way; the node is demoted to host memory when the tier is on and can hold it, else it is
        unlinked (with any host subtree, which cannot outlive its prefix). Returns the parent."""
        kv_out.append(node.value)
        self.full_evictable -= node.length
        if self.host is not None and self._demote(node, mamba_out):
            return node.parent
        if node.children and host_kv_log():
            logger.info_rank0(f"host KV: demotion refused for a {node.length}-token node with host children (snapshot "
                        f"{node.mamba_value is not None}); host free pages {self.host.free_pages} snaps "
                        f"{self.host.free_snaps} -> its host subtree is dropped")
        self._free_node_mamba(node, mamba_out)
        if node.children:
            self._drop_host_subtree(node)
        return self._unlink(node)

    def _host_pages_of(self, node: RadixTreeNode) -> List[int]:
        return (node.value[:: self.page_size] // self.page_size).tolist()

    def _host_slots(self, pages: List[int]) -> torch.Tensor:
        ps = self.page_size
        base = torch.tensor(pages, dtype=torch.int32).unsqueeze(1) * ps
        return (base + torch.arange(ps, dtype=torch.int32)).flatten()

    def _demote(self, node: RadixTreeNode, mamba_out: List[int]) -> bool:
        """Copy ``node``'s KV pages (+ its GDN snapshot) to host memory and keep it in the tree as
        a host node. Worth it only if something can resume from it: its own snapshot, or host
        children that need it as prefix. False = not worth it or no room (caller drops it)."""
        host = self.host
        has_snap = node.mamba_value is not None
        if not has_snap and not node.children:
            return False
        # An intermediate snapshot (the node already has host children, i.e. a deeper resume point) is only kept
        # while there are free snapshot rows: never let it push out a context.
        keep_snap = has_snap and (not node.children or host.free_snaps > 0)
        pages = node.length // self.page_size
        if not self._host_make_room(pages, 1 if keep_snap else 0):
            return False
        if has_snap and not keep_snap:
            # making room may have freed a row, or dropped the host children that made the snapshot optional
            keep_snap = host.free_snaps > 0 or (not node.children and self._host_make_room(pages, 1))
        if not keep_snap and not node.children:     # nothing could resume from it on the host
            return False
        hp = host.alloc_pages(pages)
        host.save_pages(node.value[:: self.page_size] // self.page_size, hp)
        if keep_snap:
            hs = host.alloc_snap()
            host.save_snap(node.mamba_value, hs)
            node.host_mamba = hs
            host.stats["demoted_snaps"] += 1
        self._free_node_mamba(node, mamba_out)
        node._value = self._host_slots(hp)
        node.on_host = True
        self.host_tokens += node.length
        host.stats["demoted_tokens"] += node.length
        return True

    def _host_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.on_host:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _host_leaves(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.on_host and not n.children:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _host_make_room(self, pages: int, snaps: int) -> bool:
        """Free host pages/snapshot rows. Snapshot rows first come from INTERMEDIATE host snapshots (a host node with
        exactly one child: every continuation through it can resume from a deeper snapshot below), oldest first;
        then LRU host leaves are dropped (never a pinned one, i.e. a path being promoted), cascading through the
        snapshot-less host nodes they expose. A context with many chunk snapshots (one per prefill chunk) thus
        costs one row, not one per chunk, before another context has to go."""
        host = self.host
        if pages > host.num_pages or snaps > host.num_snaps:
            return False
        if host.free_pages >= pages and host.free_snaps >= snaps:
            return True
        unpinned = [n for n in self._host_nodes() if n.host_pin == 0]
        if (host.free_pages + sum(n.length for n in unpinned) // self.page_size < pages
                or host.free_snaps + sum(n.host_mamba is not None for n in unpinned) < snaps):
            return False  # cannot succeed: fail before dropping anything
        if host.free_snaps < snaps:
            inter = [n for n in unpinned
                     if n.host_mamba is not None and len(n.children) == 1 and n.host_pin == 0]
            heapq.heapify(inter)
            while host.free_snaps < snaps and inter:
                n = heapq.heappop(inter)
                host.free_snap(n.host_mamba)
                n.host_mamba = None
                host.stats["released_snaps"] += 1
        if host.free_pages >= pages and host.free_snaps >= snaps:
            return True
        cands = [n for n in self._host_leaves() if n.host_pin == 0]
        if host_kv_log():
            logger.info_rank0(f"host KV: make room for {pages} pages / {snaps} snapshots: free {host.free_pages} / "
                        f"{host.free_snaps}, host holds {self.host_tokens} tokens, {len(cands)} unpinned host leaves")
        heapq.heapify(cands)
        while (host.free_pages < pages or host.free_snaps < snaps) and cands:
            n = heapq.heappop(cands)
            if not n.on_host or n.children or n.host_pin:
                continue
            parent = self._delete_host_node(n)
            while (parent.on_host and not parent.children and parent.host_mamba is None
                   and parent.host_pin == 0):
                parent = self._delete_host_node(parent)
            if parent.on_host and not parent.children and parent.host_pin == 0:
                heapq.heappush(cands, parent)
        return host.free_pages >= pages and host.free_snaps >= snaps

    def _release_host(self, node: RadixTreeNode) -> None:
        self.host.free_page_list(self._host_pages_of(node))
        if node.host_mamba is not None:
            self.host.free_snap(node.host_mamba)
            node.host_mamba = None
        self.host_tokens -= node.length
        node.on_host = False

    def _delete_host_node(self, node: RadixTreeNode) -> RadixTreeNode:
        self.host.stats["dropped_tokens"] += node.length
        self._release_host(node)
        return self._unlink(node)

    def _drop_host_subtree(self, node: RadixTreeNode) -> None:
        """Drop every (host) descendant of a device node about to be unlinked."""
        stack = list(node.children.values())
        while stack:
            n = stack.pop()
            assert n.on_host and n.host_pin == 0, "device or pinned node below an evicted leaf"
            stack.extend(n.children.values())
            self.host.stats["dropped_tokens"] += n.length
            self._release_host(n)
        node.children.clear()

    def _adopt_host_path(self, node: RadixTreeNode, kv_indices: torch.Tensor) -> int:
        """insert() through host nodes: give each host node on root..node the caller's device
        pages for its span and release its host copy. Returns the end of the device part."""
        path = []
        n = node
        while not n.is_root():
            path.append(n)
            n = n.parent
        path.reverse()
        start, dup_len = 0, None
        for n in path:
            if n.on_host:
                if dup_len is None:
                    dup_len = start
                self._release_host(n)
                n._value = kv_indices[start : start + n.length].clone()
                self.full_evictable += n.length
                self.host.stats["adopted_tokens"] += n.length
            start += n.length
        return start if dup_len is None else dup_len

    def plan_promotion(self, input_ids: torch.Tensor):
        """The host part of the match: ``(device_parent, host_nodes)`` down to the deepest host
        node that owns a host snapshot, or None when the device match is as good as it gets."""
        if not self.host_tokens:
            return None
        node, _ = self._walk(input_ids)
        path = []
        while not node.is_root():
            path.append(node)
            node = node.parent
        path.reverse()
        hosts = [n for n in path if n.on_host]
        last = max((i for i, n in enumerate(hosts) if n.host_mamba is not None), default=None)
        if last is None:
            return None
        return hosts[0].parent, hosts[: last + 1]

    def host_pin(self, nodes: List[RadixTreeNode], delta: int) -> None:
        for n in nodes:
            n.host_pin += delta

    def promote_node(self, node: RadixTreeNode, dev_slots: torch.Tensor, slot: Optional[int] = None) -> None:
        """Copy one host node back into device token slots ``dev_slots`` (and, for the promotion target, its snapshot
        into GDN slot ``slot``); it becomes a device node again, unlocked. A path is promoted top-down one node at a
        time (CacheManager.maybe_promote), so the host copy of each node is released before the next node's device
        pages are taken -- the demotions that allocation causes then need host room for one node, not for the whole
        context on top of it. An intermediate host snapshot is released rather than restored (a GDN slot each)."""
        host = self.host
        ps = self.page_size
        assert node.on_host and len(dev_slots) == node.length
        host.load_pages(self._host_pages_of(node), dev_slots[::ps] // ps)
        if slot is not None:
            host.load_snap(node.host_mamba, slot)
        self._release_host(node)
        node._value = dev_slots.clone()
        self.full_evictable += node.length
        host.stats["promoted_tokens"] += node.length
        if slot is not None:
            node.mamba_value = slot
            self.mamba_evictable += 1
            host.stats["promoted_snaps"] += 1

    def _path_len(self, node: RadixTreeNode) -> int:
        n, total = node, 0
        while not n.is_root():
            total += n.length
            n = n.parent
        return total

    def _collect_kv(self, node: RadixTreeNode) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        n = node
        while not n.is_root():
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.empty

    def _leaves(self) -> List[RadixTreeNode]:
        """Device leaves: device nodes without a device child (host subtrees are skipped)."""
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            dev = [c for c in n.children.values() if not c.on_host]
            if not dev:
                if not n.is_root():
                    out.append(n)
            else:
                stack.extend(dev)
        return out

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_value is not None and not n.is_root():
                out.append(n)
            stack.extend(c for c in n.children.values() if not c.on_host)  # host nodes carry no device snapshot
        return out

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = time.monotonic_ns()
        while prefix_len < total:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "HybridCacheHandle"]
