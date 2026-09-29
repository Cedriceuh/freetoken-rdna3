"""Host KV tier (FREETOKEN_HOST_KV=1): demote-on-evict / promote-on-match, on CPU.

The pools are small fakes with the real tensor layouts (paged K/V ``[2, L, P, ps, h, d]``, the
QSA compressed index slab, GDN conv/recurrent state slots). Every device page and snapshot is
filled with a value derived from the token PREFIX it encodes, so after any sequence of
evictions (to host), host drops and promotions (back to device) the test can check that each
device node still holds exactly the bytes of its own prefix, and the accounting of both tiers.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

PS = 4          # page size
RATIO = 2       # QSA index ratio (compressed rows per page = PS // RATIO)
LAYERS = 2
HEADS, DIM = 1, 3
IDX_LAYERS, IDX_DIM = 2, 2
GDN_LAYERS = 2


def _prefix_hash(ids) -> float:
    h = 0
    for t in ids:
        h = (h * 31 + int(t) + 7) % 1_000_003
    return float(h)


class FakeKV:
    host_tier_supported = True  # the real layout of a QSA pool's per-page tensors

    def __init__(self, num_pages: int) -> None:
        pages = num_pages + 1
        self._kv_buffer = torch.zeros(2, LAYERS, pages, PS, HEADS, DIM)
        self._kv_scale = None
        self.index_ratio = RATIO
        self.cmp_scratch_base = pages * PS // RATIO
        self._cmp_k_buffer = torch.zeros(IDX_LAYERS, self.cmp_scratch_base + 3, IDX_DIM)
        self._rope_positions = None

    def write(self, slots: torch.Tensor, ids, start: int) -> None:
        """Write the KV of tokens ``ids[start:]`` at token slots ``slots``."""
        for j, slot in enumerate(slots.tolist()):
            pos = start + j
            v = _prefix_hash(ids[: pos + 1])
            p, o = divmod(slot, PS)
            for kv in range(2):
                for layer in range(LAYERS):
                    self._kv_buffer[kv, layer, p, o] = v + kv * 0.5 + layer * 0.25
            if (pos + 1) % RATIO == 0:
                self._cmp_k_buffer[:, slot // RATIO] = v

    def check(self, slots: torch.Tensor, ids) -> None:
        for pos, slot in enumerate(slots.tolist()):
            v = _prefix_hash(ids[: pos + 1])
            p, o = divmod(slot, PS)
            got = self._kv_buffer[:, :, p, o]
            want = torch.tensor([[v + kv * 0.5 + layer * 0.25 for layer in range(LAYERS)]
                                 for kv in range(2)])
            assert torch.equal(got[..., 0, 0], want), (pos, slot)
            if (pos + 1) % RATIO == 0:
                assert torch.all(self._cmp_k_buffer[:, slot // RATIO] == v), (pos, slot)


class FakeLinear:
    def __init__(self, num_slots: int) -> None:
        self.conv_states = torch.zeros(GDN_LAYERS, num_slots, 3, 2)
        self.recurrent_states = torch.zeros(GDN_LAYERS, num_slots, 2, 2, 2)
        self.slot_states = {}
        self._num_slots = num_slots
        self._free = list(range(1, num_slots))

    @property
    def num_free_slots(self) -> int:
        return len(self._free)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def alloc(self, n: int = 1):
        assert n <= len(self._free)
        return [self._free.pop() for _ in range(n)]

    def free(self, slots) -> None:
        if isinstance(slots, int):
            slots = [slots]
        self._free.extend(int(s) for s in slots)

    def write(self, slot: int, ids) -> None:
        v = _prefix_hash(ids)
        self.conv_states[:, slot] = v
        self.recurrent_states[:, slot] = v + 1

    def check(self, slot: int, ids) -> None:
        v = _prefix_hash(ids)
        assert torch.all(self.conv_states[:, slot] == v), slot
        assert torch.all(self.recurrent_states[:, slot] == v + 1), slot


def _make(monkeypatch, num_pages=24, host_tokens=48, host_snaps=4, slots=16):
    from freetoken.scheduler.cache import CacheManager

    monkeypatch.setenv("FREETOKEN_HOST_KV", "1")
    monkeypatch.setenv("FREETOKEN_HOST_KV_TOKENS", str(host_tokens))
    monkeypatch.setenv("FREETOKEN_HOST_KV_SNAPSHOTS", str(host_snaps))
    kv, lin = FakeKV(num_pages), FakeLinear(slots)
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(num_pages, PS, pt, "hybrid_radix", linear_state_pool=lin, kv_pool=kv)
    assert cm.host_kv is not None
    return cm, kv, lin


def _prefill(cm, kv, lin, ids) -> None:
    """Admit + prefill + finish ``ids`` the way the scheduler does, minus the forward: match
    (with promotion), allocate the new pages, write their KV, commit KV + final snapshot."""
    req = SimpleNamespace(input_ids=torch.tensor(ids + [0], dtype=torch.int32),
                          input_len=len(ids) + 1, output_len=0)
    mr = cm.match_req(req)
    mr = cm.maybe_promote(req, mr, lambda cached: (len(ids) - cached))
    h = mr.cuda_handle
    cached = h.cached_len
    ids_t = torch.tensor(ids, dtype=torch.int32)
    # a resumed prefix must read its own bytes back (device-held or promoted)
    kv.check(h.get_matched_indices(), ids[:cached])
    if cached:
        lin.check(mr.mamba_value, ids[:cached])
    cm.lock(h)
    new_pages = (len(ids) - cached) // PS
    slots = cm._page_to_token(cm._allocate(new_pages)) if new_pages else torch.empty(0, dtype=torch.int32)
    kv.write(slots, ids, cached)
    if lin.num_free_slots < 1:
        cm.ensure_mamba_slots(1)
    live = lin.alloc(1)[0]
    lin.write(live, ids)
    all_slots = torch.cat([h.get_matched_indices(), slots])
    dup_len, exist = cm.prefix_cache.insert(ids_t, all_slots, live)
    cm.unlock(h)
    cm._free(all_slots[cached:dup_len])
    if exist:
        lin.free([live])


def _check_tree(cm, kv, lin) -> None:
    pc = cm.prefix_cache
    cm.check_integrity()
    stack = [(pc.root, [])]
    while stack:
        n, prefix = stack.pop()
        ids = prefix + ([] if n.is_root() else n._key.tolist())
        if not n.is_root() and not n.on_host:
            start = len(prefix)
            full = torch.cat([pc._collect_kv(n.parent), n.value]) if not n.parent.is_root() else n.value
            kv.check(full, ids)
            if n.mamba_value is not None:
                lin.check(n.mamba_value, ids)
            assert start == len(ids) - n.length
        for c in n.children.values():
            stack.append((c, ids))


def _seq(rng, prefixes, pages):
    head = rng.choice(prefixes)
    return head + [rng.randrange(1, 50) for _ in range(pages * PS - len(head))]


def test_evicted_context_comes_back_from_host(monkeypatch):
    cm, kv, lin = _make(monkeypatch)
    a = list(range(1, 1 + 5 * PS))
    _prefill(cm, kv, lin, a)
    # fill the device pool with other contexts until A is evicted (demoted, not dropped)
    for k in range(4):
        _prefill(cm, kv, lin, [100 + k] + list(range(200, 200 + 5 * PS - 1)))
    _check_tree(cm, kv, lin)
    assert cm.prefix_cache.host_tokens > 0
    assert cm.host_kv.stats["demoted_tokens"] > 0  # only the tail eviction needed goes (split on evict)
    # A returns with one more turn: its whole context is promoted, not recomputed
    req = SimpleNamespace(input_ids=torch.tensor(a + [9, 9, 9, 9, 0], dtype=torch.int32),
                          input_len=len(a) + 5, output_len=0)
    before = cm.match_req(req).cuda_handle.cached_len
    mr = cm.maybe_promote(req, cm.match_req(req), lambda cached: len(a) + 4 - cached)
    assert before < len(a) and mr.cuda_handle.cached_len == len(a)
    kv.check(mr.cuda_handle.get_matched_indices(), a)
    lin.check(mr.mamba_value, a)
    _check_tree(cm, kv, lin)


def test_no_promotion_when_admission_cannot_fit(monkeypatch):
    cm, kv, lin = _make(monkeypatch)
    a = list(range(1, 1 + 5 * PS))
    _prefill(cm, kv, lin, a)
    for k in range(4):
        _prefill(cm, kv, lin, [100 + k] + list(range(200, 200 + 5 * PS - 1)))
    req = SimpleNamespace(input_ids=torch.tensor(a + [0], dtype=torch.int32),
                          input_len=len(a) + 1, output_len=0)
    mr0 = cm.match_req(req)
    mr = cm.maybe_promote(req, mr0, lambda cached: 10**9)
    assert mr is mr0 and cm.host_kv.stats["promoted_tokens"] == 0
    _check_tree(cm, kv, lin)


@pytest.mark.parametrize("seed", range(12))
def test_random_lifecycle(monkeypatch, seed):
    rng = random.Random(seed)
    cm, kv, lin = _make(monkeypatch, num_pages=20, host_tokens=12 * PS, host_snaps=3, slots=12)
    prefixes = [[], list(range(60, 60 + 2 * PS)), list(range(80, 80 + 3 * PS))]
    history = []
    for _ in range(60):
        if history and rng.random() < 0.5:
            base = rng.choice(history)
            ids = base + [rng.randrange(1, 50) for _ in range(rng.randrange(0, 3) * PS)]
        else:
            ids = _seq(rng, prefixes, rng.randrange(3, 7))
        if len(ids) // PS > 16:
            ids = ids[: 16 * PS]
        _prefill(cm, kv, lin, ids)
        history.append(ids)
        _check_tree(cm, kv, lin)
    st = cm.host_kv.stats
    assert st["demoted_tokens"] > 0


def test_promotion_path_survives_host_pressure(monkeypatch):
    """The promotion's own device allocation demotes another leaf into a FULL host tier: the
    path being promoted is pinned, so the other leaf is dropped instead of overwriting it."""
    cm, kv, lin = _make(monkeypatch, num_pages=10, host_tokens=5 * PS, host_snaps=1)
    a = [1] + list(range(10, 10 + 5 * PS - 1))
    _prefill(cm, kv, lin, a)
    _prefill(cm, kv, lin, [2] + list(range(10, 10 + 5 * PS - 1)))
    _prefill(cm, kv, lin, [3] + list(range(10, 10 + 5 * PS - 1)))   # evicts a -> host (full)
    assert cm.prefix_cache.host_tokens == len(a)
    req = SimpleNamespace(input_ids=torch.tensor(a + [9, 9, 9, 9, 0], dtype=torch.int32),
                          input_len=len(a) + 5, output_len=0)
    mr = cm.maybe_promote(req, cm.match_req(req), lambda cached: len(a) + 4 - cached)
    assert mr.cuda_handle.cached_len == len(a)
    kv.check(mr.cuda_handle.get_matched_indices(), a)
    lin.check(mr.mamba_value, a)
    _check_tree(cm, kv, lin)


def test_chunk_snapshots_do_not_push_whole_contexts_out(monkeypatch):
    """A long context carries one snapshot per prefill chunk. With few host snapshot rows, demoting other contexts
    must release A's INTERMEDIATE host snapshots, not drop A: A comes back whole from its leaf snapshot."""
    cm, kv, lin = _make(monkeypatch, num_pages=16, host_tokens=40 * PS, host_snaps=2, slots=16)
    a = [1] + list(range(10, 10 + 12 * PS - 1))
    for pages in (4, 8, 12):                     # three chunks -> three snapshot nodes on A's path
        _prefill(cm, kv, lin, a[: pages * PS])
    for k in range(3):                           # three other contexts push A, then B, out of the device
        _prefill(cm, kv, lin, [2 + k] + list(range(10, 10 + 8 * PS - 1)))
    _check_tree(cm, kv, lin)
    st = cm.host_kv.stats
    assert st["dropped_tokens"] == 0 and st["released_snaps"] >= 1, st
    req = SimpleNamespace(input_ids=torch.tensor(a + [9, 9, 9, 9, 0], dtype=torch.int32),
                          input_len=len(a) + 5, output_len=0)
    mr = cm.maybe_promote(req, cm.match_req(req), lambda cached: len(a) + 4 - cached)
    assert mr.cuda_handle.cached_len == len(a)
    kv.check(mr.cuda_handle.get_matched_indices(), a)
    lin.check(mr.mamba_value, a)
    _check_tree(cm, kv, lin)


def _chunked_turn(cm, kv, lin, ids, chunk_pages, commit_every_chunk=False):
    """One turn like the scheduler runs it: match (+ promotion), 3 GDN slots, prefill in chunks, commit (donate a
    snapshot and re-lock there, CacheManager._cache_req_hybrid) -- FreeToken commits only a prefill's LAST chunk (a
    ChunkedReq never commits), so a turn is one node; ``commit_every_chunk`` commits each chunk. Returns the cached
    length."""
    from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle

    pc = cm.prefix_cache
    req = SimpleNamespace(input_ids=torch.tensor(ids + [0], dtype=torch.int32), input_len=len(ids) + 1, output_len=0)
    mr = cm.maybe_promote(req, cm.match_req(req), lambda cached: len(ids) - cached)
    h = mr.cuda_handle
    cached = h.cached_len
    kv.check(h.get_matched_indices(), ids[:cached])
    if cached:
        lin.check(mr.mamba_value, ids[:cached])
    cm.lock(h)
    if lin.num_free_slots < 3:
        cm.ensure_mamba_slots(3)
    live, pp = lin.alloc(1)[0], lin.alloc(1)[0]
    table = h.get_matched_indices().clone()
    pos = cached
    while pos < len(ids):
        end = min(pos + chunk_pages * PS, len(ids))
        new = cm._page_to_token(cm._allocate((end - pos) // PS))
        kv.write(new, ids, pos)
        table = torch.cat([table, new])
        if end < len(ids) and not commit_every_chunk:
            pos = end
            continue
        lin.write(pp, ids[:end])
        dup, exist = pc.insert(torch.tensor(ids[:end], dtype=torch.int32), table[:end], pp)
        cm.unlock(h)
        cm._free(table[h.cached_len:dup])
        m = pc.match_prefix(torch.tensor(ids[:end], dtype=torch.int32))
        table = torch.cat([m.kv_indices, table[m.cached_len:]])
        h = HybridCacheHandle(m.cached_len, m.node, m.kv_indices)
        cm.lock(h)
        if not exist:
            cm.ensure_mamba_slots(1)
            pp = lin.alloc(1)[0]
        pos = end
    cm.unlock(h)
    cm._free(table[h.cached_len:])
    lin.free([live, pp])
    return cached


@pytest.mark.parametrize("commit_every_chunk", [False, True])
def test_four_long_agents_taking_turns_keep_their_contexts(monkeypatch, commit_every_chunk):
    """Four long agents of 20-27 pages (one snapshot per 4-page chunk) take turns on a 64-page device
    pool + a 64-page host tier, GDN slots as scarce as a 4-request server's (25). Every later turn must resume from its whole
    context (all but its new page). Two ways the host overflowed and dropped other agents' contexts (cached 0):
    promoting a whole path before releasing its host copy, and evicting a whole one-node turn (tens of pages) when
    the allocation needed a few."""
    cm, kv, lin = _make(monkeypatch, num_pages=64, host_tokens=64 * PS, host_snaps=24, slots=25)
    ctxs = [[100 + a] + [(a * 7 + i) % 50 + 1 for i in range(p * PS - 1)] for a, p in enumerate((22, 20, 20, 27))]
    for a in range(4):
        _chunked_turn(cm, kv, lin, ctxs[a], 4, commit_every_chunk)
    got, want = 0, 0
    for r in range(1, 4):
        for a in range(4):
            ctxs[a] = ctxs[a] + [(r * 13 + a + i) % 50 + 1 for i in range(PS)]
            got += _chunked_turn(cm, kv, lin, ctxs[a], 4, commit_every_chunk)
            want += len(ctxs[a]) - PS
            _check_tree(cm, kv, lin)
    assert got >= want - 2 * PS, (got, want, cm.host_kv.stats)


def test_gdn_slot_pressure_moves_one_page_not_the_context(monkeypatch):
    """KV room to spare, GDN slots scarce: freeing a snapshot slot must demote the snapshot with a one-page tail,
    not the whole one-node context (which would copy tens of pages to the host, and overflow it at scale)."""
    cm, kv, lin = _make(monkeypatch, num_pages=64, host_tokens=64 * PS, host_snaps=8, slots=8)
    ctxs = [[100 + a] + [(a * 5 + i) % 50 + 1 for i in range(12 * PS - 1)] for a in range(3)]
    for a in range(3):
        _chunked_turn(cm, kv, lin, ctxs[a], 4)
    got, want = 0, 0
    for r in range(1, 3):
        for a in range(3):
            ctxs[a] = ctxs[a] + [(r * 11 + a + i) % 50 + 1 for i in range(4 * PS)]  # multi-page turn nodes
            got += _chunked_turn(cm, kv, lin, ctxs[a], 4)
            want += len(ctxs[a]) - 4 * PS
            _check_tree(cm, kv, lin)
    st = cm.host_kv.stats
    assert got == want, (got, want, st)
    assert st["demoted_snaps"] > 0 and st["demoted_tokens"] <= st["demoted_snaps"] * PS, st
