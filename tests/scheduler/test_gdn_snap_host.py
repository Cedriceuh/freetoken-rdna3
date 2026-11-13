"""FREETOKEN_GDN_SNAP_HOST: the tree's GDN snapshots in host slots of the LinearStatePool, the device pool holding the
requests' working slots only. CPU, real pool + CacheManager, hand-built Reqs (the host slots are plain tensors here)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import (
    LinearStatePool, _linear_pool_min_slots, _linear_pool_num_slots, _linear_pool_snap_slots,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=3, snap_slots=6):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1, snap_slots=snap_slots)


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def _fill(pool, slot, value):
    pool.conv_states[:, slot] = value
    pool.recurrent_states[:, slot] = value


def _conserved(cm, pool):
    """Every host snapshot slot is free or held by a tree node (the pages here are hand-made, so not the page check)."""
    pc = cm.prefix_cache
    pc.check_integrity()
    assert pool.num_free_snaps + pc.mamba_evictable_size + pc.mamba_protected == pool.num_snap_slots


def _admit(cm, pool, ids, table_idx, cached_len, pages):
    mr = cm.match_req(_pend(ids))
    assert cm.reserve_state_slots()
    live, pp = cm.alloc_state_slots()
    cm.page_table[table_idx, :cached_len] = torch.tensor(pages, dtype=torch.int32)
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx, cached_len=cached_len,
              output_len=1, uid=table_idx, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    return req, mr


def test_snapshot_ids_follow_the_device_slots_and_copies_cross_both_ways():
    pool = _pool()
    assert pool.host_snapshots and pool.num_snap_slots == 6 and pool.num_free_snaps == 6
    assert pool.num_free_slots == 2
    live = pool.alloc(1)[0]
    snap = pool.alloc_snap()
    assert snap >= pool.num_slots
    _fill(pool, live, 3.0)
    pool.copy_from(live, snap)
    _fill(pool, live, 5.0)
    other = pool.alloc(1)[0]
    pool.copy_from(snap, other)
    assert bool((pool.recurrent_states[:, other] == 3.0).all()) and bool((pool.conv_states[:, other] == 3.0).all())
    assert bool((pool.recurrent_states[:, live] == 5.0).all())
    pool.free([snap, other])
    assert pool.num_free_snaps == 6 and pool.num_free_slots == 1


def test_chunk_commit_donates_the_track_slot_and_replaces_it_with_a_host_slot():
    pool = _pool()
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool)
    req, _ = _admit(cm, pool, [1, 2, 3, 4, 5], 0, 4, [100, 101, 102, 103])
    pp = req.mamba_ping_pong
    assert min(pp) >= pool.num_slots and pool.num_free_slots == pool.num_slots - 2   # only the live slot is a device one
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = 4
    # what a prefill's track write does: rows of the track tensors, counted from the first host slot
    pool.track_recurrent(0)[pp[0] - pool.track_base] = 7.0
    pool.track_recurrent(1)[pp[0] - pool.track_base] = 7.0
    pool.track_conv(0)[pp[0] - pool.track_base] = 7.0
    pool.track_conv(1)[pp[0] - pool.track_base] = 7.0
    cm.cache_req(req, finished=False)
    assert req.mamba_ping_pong[0] != pp[0] and req.mamba_ping_pong[0] >= pool.num_slots
    assert pool.num_free_snaps == pool.num_snap_slots - 3              # the pair + the donated one
    mr = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mr.cuda_handle.cached_len == 4 and mr.mamba_value == pp[0]
    dst = pool.alloc(1)[0]
    pool.copy_from(mr.mamba_value, dst)                                # a prefix hit's restore
    assert bool((pool.recurrent_states[:, dst] == 7.0).all()) and bool((pool.conv_states[:, dst] == 7.0).all())


def test_finish_gives_the_tree_a_copy_and_every_working_slot_comes_back():
    pool = _pool()
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool)
    req, _ = _admit(cm, pool, [7, 8, 9, 10], 1, 3, [200, 201, 202])
    _fill(pool, req.linear_slot_idx, 2.0)
    cm.cache_req(req, finished=True)
    assert pool.num_free_slots == pool.num_slots - 1 and pool.num_free_snaps == pool.num_snap_slots - 1
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr.cuda_handle.cached_len == 3 and mr.mamba_value >= pool.num_slots
    dst = pool.alloc(1)[0]
    pool.copy_from(mr.mamba_value, dst)
    assert bool((pool.conv_states[:, dst] == 2.0).all())
    _conserved(cm, pool)


def test_a_duplicate_donation_releases_its_copy():
    pool = _pool()
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool)
    a, _ = _admit(cm, pool, [7, 8, 9, 10], 0, 3, [200, 201, 202])
    cm.cache_req(a, finished=True)
    b, mr = _admit(cm, pool, [7, 8, 9, 10], 1, 3, [200, 201, 202])
    assert mr.cuda_handle.cached_len == 3
    cm.cache_req(b, finished=True)                                     # same boundary: the tree keeps its snapshot
    assert pool.num_free_snaps == pool.num_snap_slots - 1 and pool.num_free_slots == pool.num_slots - 1
    _conserved(cm, pool)


def test_a_full_snapshot_store_evicts_the_oldest_unlocked_snapshot():
    pool = _pool(snap_slots=3)   # the third request's pair needs one of the two snapshots
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool)
    for i, first in enumerate((10, 20, 30)):
        req, _ = _admit(cm, pool, [first, 2, 3, 4], 0, 3, [100 * (i + 1) + j for j in range(3)])
        cm.cache_req(req, finished=True)
        _conserved(cm, pool)
    assert pool.num_free_snaps == 1
    assert cm.match_req(_pend([10, 2, 3, 4])).mamba_value is None      # the first conversation's snapshot went
    assert cm.match_req(_pend([30, 2, 3, 4])).mamba_value is not None
    assert pool.num_free_slots == pool.num_slots - 1


def test_admission_waits_for_a_live_slot_and_takes_its_pair_from_the_tree():
    pool = _pool(num_slots=2, snap_slots=3)                            # one live slot; a pair + one snapshot
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool)
    req, _ = _admit(cm, pool, [7, 8, 9, 10], 0, 3, [200, 201, 202])
    assert not cm.reserve_state_slots()                                # the live slot is taken
    cm.cache_req(req, finished=True)
    assert pool.num_free_snaps == 2 and cm.reserve_state_slots()
    live, pp = cm.alloc_state_slots()
    assert live < pool.num_slots <= min(pp)
    assert cm.match_req(_pend([7, 8, 9, 10])).mamba_value is not None  # the snapshot was not needed for the pair


def test_rebuild_reclaims_both_kinds_of_slots():
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    req, _ = _admit(cm, pool, [7, 8, 9, 10], 1, 3, [200, 201, 202])
    cm.cache_req(req, finished=True)
    cm.rebuild(64, pt)
    assert pool.num_free_slots == pool.num_slots - 1 and pool.num_free_snaps == pool.num_snap_slots


def test_pool_sizing(monkeypatch):
    cfg = SimpleNamespace(max_running_req=4, cache_type="hybrid_radix", linear_state_cache_ratio=2.0)
    monkeypatch.delenv("FREETOKEN_GDN_SNAP_HOST", raising=False)
    assert (_linear_pool_num_slots(cfg), _linear_pool_snap_slots(cfg), _linear_pool_min_slots(cfg)) == (25, 0, 17)
    monkeypatch.setenv("FREETOKEN_GDN_SNAP_HOST", "1")
    assert (_linear_pool_num_slots(cfg), _linear_pool_snap_slots(cfg), _linear_pool_min_slots(cfg)) == (5, 20, 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the host slots are mapped host memory on a GPU")
def test_gpu_host_slots_hold_what_kernels_write_and_give_it_back():
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1, 2), num_key_heads=2, num_value_heads=4,
        key_head_dim=64, value_head_dim=64, conv_kernel_dim=4, output_gate="silu",
    )
    dev = torch.device("cuda")
    pool = LinearStatePool(group=g, num_slots=3, dtype=torch.bfloat16, device=dev, tp_size=1, snap_slots=5)
    torch.manual_seed(0)
    assert pool.track_recurrent(0).is_cuda and pool.track_base == 3
    # a prefill's track write: rows of the track tensors, two requests at once
    dst = torch.tensor([4, 1], device=dev)
    rec = [torch.randn(2, *pool.recurrent_states.shape[2:], device=dev) for _ in range(3)]
    conv = [torch.randn(2, *pool.conv_states.shape[2:], device=dev).to(torch.bfloat16) for _ in range(3)]
    for li in range(3):
        pool.track_recurrent(li).index_copy_(0, dst, rec[li])
        pool.track_conv(li).index_copy_(0, dst, conv[li])
    # a prefix hit's restore into live slots, then a finished request's live slot into another host slot and back
    pool.copy_from(3 + 4, 1)
    pool.copy_from(3 + 1, 2)
    for li in range(3):
        assert torch.equal(pool.recurrent_states[li, 1], rec[li][0]) and torch.equal(pool.recurrent_states[li, 2], rec[li][1])
        assert torch.equal(pool.conv_states[li, 1], conv[li][0]) and torch.equal(pool.conv_states[li, 2], conv[li][1])
    pool.recurrent_states[:, 1].add_(1.0)
    want = pool.recurrent_states[:, 1].clone()
    pool.copy_from(1, 3 + 0)
    pool.recurrent_states[:, 1].zero_()
    pool.copy_from(3 + 0, 2)
    torch.cuda.synchronize()
    assert torch.equal(pool.recurrent_states[:, 2], want)
    assert torch.equal(pool.slot_rows("recurrent", 3 + 0).cpu(), want.cpu())   # a device-to-host copy of a host slot


@pytest.mark.parametrize("cheap_tokens", [8, 0])
def test_a_long_idle_conversation_keeps_its_snapshot_while_another_takes_turns(monkeypatch, cheap_tokens):
    """Conversation C (12 tokens read at once) waits while A takes turn after turn of 2 tokens, each leaving a
    snapshot. The store fills: A's turn ends, 2 tokens from the snapshot before them, go before C's, which costs 12
    tokens to read again. Plain LRU (0) drops C first, its KV with it."""
    import freetoken.kvcache.hybrid_radix_cache as hrc

    monkeypatch.setattr(hrc, "_EVICT_CHEAP_TOKENS", cheap_tokens)
    pool = _pool(num_slots=2, snap_slots=6)
    cm = CacheManager(256, 1, torch.zeros(4, 256, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool)
    c = [9] * 13
    req, _ = _admit(cm, pool, c, 0, 12, list(range(200, 212)))
    cm.cache_req(req, finished=True)
    a = [5] * 13
    for turn in range(8):
        n = len(a) - 1
        req, mr = _admit(cm, pool, a, 1, n, list(range(10, 10 + n)))
        assert mr.cuda_handle.cached_len == (0 if turn == 0 else n - 2)   # A always finds its last turn's end
        cm.cache_req(req, finished=True)
        _conserved(cm, pool)
        a = a + [6 + turn, 7 + turn]
    kept = cm.match_req(_pend(c))
    assert (kept.cuda_handle.cached_len == 12) == (cheap_tokens > 0)
    assert (kept.mamba_value is not None) == (cheap_tokens > 0)


def test_cheap_snapshots_go_oldest_first_and_a_merged_span_counts(monkeypatch):
    """Evicting a turn's end makes the next one down the path dearer (it now stands for both turns)."""
    import freetoken.kvcache.hybrid_radix_cache as hrc

    monkeypatch.setattr(hrc, "_EVICT_CHEAP_TOKENS", 5)
    pool = _pool(num_slots=2, snap_slots=8)
    cm = CacheManager(256, 1, torch.zeros(4, 256, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool)
    a = [5] * 4
    for turn in range(4):                                   # snapshots at 3, 5, 7, 9 tokens
        n = len(a) - 1
        req, _ = _admit(cm, pool, a, 1, n, list(range(10, 10 + n)))
        cm.cache_req(req, finished=True)
        a = a + [6 + turn, 7 + turn]
    pc = cm.prefix_cache
    costs = sorted(pc._snapshot_costs().values())
    assert costs == [2, 2, 2, 3]
    er = pc.evict_mamba(2)                                   # the two oldest cheap ones: at 3 and 5 tokens
    pool.free(er.mamba_slots)
    assert sorted(pc._snapshot_costs().values()) == [2, 7]   # the one at 7 now costs the 7 tokens above it
    er = pc.evict_mamba(1)                                   # the cheap one left (at 9), not the dear one
    pool.free(er.mamba_slots)
    m = cm.match_req(_pend(a[:10]))
    assert m.cuda_handle.cached_len == 7 and m.mamba_value is not None
