"""The boundary carry -- a turn's GDN boundary must survive the prefill chunk seam.

A prefill forward tracks a GDN snapshot boundary only when ITS OWN chunk is at least
``CHUNK_SIZE + 1`` tokens (``attention/linear.py:_build_track_metadata``), and only the FINAL
chunk of a turn commits (``scheduler.py`` skips ``ChunkedReq``). Before the carry
``scheduler/prefill.py`` rebuilt each continuation ``Req`` WITHOUT ``mamba_last_track_seqlen``,
so an earlier chunk's boundary was discarded at the seam: a turn whose last chunk was short
donated no reusable resume point at all, and the next turn resumed from the last turn that did
(measured exact on 18 banked loads).

CPU only: no GPU, no engine, no kernels. The tests drive the real ``PrefillAdder`` chunking,
the real ``_build_track_metadata`` bookkeeping and the real ``CacheManager`` /
``HybridRadixCache`` commit, and stand in for the GDN kernel by writing a per-boundary MARKER
into the slot ``_build_track_metadata`` itself nominated (``track_dst``).

That marker is what makes this a correctness test and not a plumbing test: the tree's answer
for prefix ``[0, L)`` is checked by CONTENT, so a snapshot captured at one boundary and
committed against a different one fails loudly instead of passing quietly. A snapshot
donated at ``L`` claims the GDN state at ``L`` is what a future request restores from -- get
that wrong and the row serves wrong continuations with no error and no crash.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

CHUNK = 64  # freetoken.kernel.fla.chunk.CHUNK_SIZE; asserted in test_chunk_size_is_what_the_rule_assumes
PAGE = 64   # the served hybrid shape: prefill_chunk_align == page_size (cache.py:77-81)


# --------------------------------------------------------------------------- harness


def _pool(num_slots=32):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _rig(num_pages=64, width=1024, max_running=4, num_slots=32):
    pool = _pool(num_slots)
    pt = torch.zeros(max_running, width, dtype=torch.int32)
    cm = CacheManager(num_pages, PAGE, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid and cm.prefill_chunk_align == PAGE
    return cm, TableManager(max_running_reqs=max_running, page_table=pt), pool


def _pend(uid, ids, max_tokens=1):
    return PendingReq(uid, torch.tensor(ids, dtype=torch.int32),
                      SamplingParams(max_tokens=max_tokens))


def _marker(pool, slot):
    """The one fp32 value standing in for a whole GDN state. recurrent_states is fp32
    (ssm_state_dtype default), so an integer boundary round-trips exactly."""
    return float(pool.recurrent_states[0, slot].flatten()[0].item())


def _write_marker(pool, slot, value):
    pool.recurrent_states[:, slot] = float(value)
    pool.conv_states[:, slot] = float(value)


def _forward_track(pool, reqs):
    """Run the REAL ``_build_track_metadata`` over one prefill batch, then stand in for the
    GDN kernel: write a marker equal to the boundary into the slot it nominated.

    Returns ``[(req, dst_slot, boundary), ...]`` for the requests that tracked. Asserts the
    pairing invariant the carry leans on -- that after the call
    ``ping_pong[1 - mamba_next_track_idx]`` IS the slot the metadata named as ``track_dst``.
    """
    import freetoken.core as core
    from freetoken.attention.linear import _build_track_metadata

    cu = torch.tensor([0, *[r.extend_len for r in reqs]], dtype=torch.int64).cumsum_(0)
    before_idx = [r.mamba_next_track_idx for r in reqs]
    saved = core._GLOBAL_CTX
    core._GLOBAL_CTX = SimpleNamespace(linear_state_pool=pool)
    try:
        track = _build_track_metadata(reqs, cu, torch.device("cpu"), {"device": "cpu"})
    finally:
        core._GLOBAL_CTX = saved

    named = [] if track["track_dst"] is None else track["track_dst"].tolist()
    tracked, k = [], 0
    for r, b in zip(reqs, before_idx):
        if r.mamba_next_track_idx == b:
            continue                      # this request did not track this forward
        dst = r.mamba_ping_pong[1 - r.mamba_next_track_idx]
        assert dst == named[k], (
            f"pairing invariant broken: metadata wrote slot {named[k]} but "
            f"ping_pong[1 - next_track_idx] is {dst}"
        )
        k += 1
        _write_marker(pool, dst, r.mamba_last_track_seqlen)   # the kernel's write, stood in for
        tracked.append((r, dst, r.mamba_last_track_seqlen))
    assert k == len(named), "a nominated track_dst belongs to no request that flipped"
    return tracked


def _run_turn(cm, tm, pending, budget, *, sample=1):
    """One turn of prefill, chunk by chunk, exactly as the scheduler drives it.

    ``PrefillAdder`` chunks -> ``allocate_paged`` -> the track bookkeeping + the stood-in
    kernel write -> ``complete_one`` -> commit. The commit mirrors ``_process_last_data``:
    an intermediate ``ChunkedReq`` is SKIPPED (the ``ChunkedReq`` check in ``_process_last_data``, "the full prompt is cached
    once when the final chunk is processed"); only the final ``Req`` reaches ``cache_req``.

    Returns ``(final_req, [chunk lengths], [(boundary, dst) tracked])``.
    """
    lens, tracked, req = [], [], None
    while True:
        adder = PrefillAdder(token_budget=budget, reserved_size=0,
                             cache_manager=cm, table_manager=tm)
        req = adder.try_add_one(pending)
        assert req is not None, "the adder stalled: budget too small for one aligned unit"
        lens.append(req.extend_len)
        cm.allocate_paged([req])
        for _, dst, boundary in _forward_track(cm.linear_state_pool, [req]):
            tracked.append((boundary, dst))
        req.complete_one()                        # engine.py:929, inside the forward
        if isinstance(req, ChunkedReq):
            pending.chunked_req = req             # prefill.py:349
            continue                              # scheduler.py:346-352 -- no commit
        pending.chunked_req = None
        cm.cache_req(req, finished=False)         # scheduler.py:424
        return req, lens, tracked


def _finish(cm, req):
    """End the turn: the scheduler's finish-donate of the live full-sequence state."""
    cm.cache_req(req, finished=True)


# --------------------------------------------------------------------------- the rule


def test_chunk_size_is_what_the_rule_assumes():
    """The rule -- 'a turn donates iff its final chunk is >= 65 tokens' -- is CHUNK+1, and
    CHUNK is the kernel's. Pin it: a CHUNK change silently re-prices every claim of the carry."""
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    assert CHUNK_SIZE == CHUNK


def test_a_short_final_chunk_tracks_nothing_and_a_long_one_does():
    """The defect's precondition, from the real bookkeeping: ``c = (extend_len - 1) // CHUNK``."""
    cm, tm, pool = _rig()
    pending = _pend(0, list(range(1, 226)))                  # 225 = 192 + 33
    req, lens, tracked = _run_turn(cm, tm, pending, budget=192)
    assert lens == [192, 33], lens                           # final chunk 33 < CHUNK + 1
    assert [b for b, _ in tracked] == [128], tracked         # chunk 1 tracked, the final one did not


# --------------------------------------------------------------------------- the red one


def test_a_short_final_chunk_no_longer_throws_away_the_turns_boundary():
    """THE RED ONE. Turn 1 chunks 192 + 33: chunk 1 tracks a boundary at 128, the 33-token
    final chunk tracks nothing. Before the carry the seam dropped chunk 1's boundary, the
    commit hit ``if L is None: return`` and turn 2 resumed from 0 -- re-prefilling everything.

    Checked by CONTENT, not by ``cached_len``: the slot the tree hands back for prefix
    ``[0, 128)`` must carry the marker written AT 128."""
    cm, tm, pool = _rig()
    ids = list(range(1, 226))
    req, lens, tracked = _run_turn(cm, tm, _pend(0, ids), budget=192)
    assert lens == [192, 33] and [b for b, _ in tracked] == [128]

    m = cm.match_req(_pend(1, ids[:128] + [9001, 9002]))
    assert m.cuda_handle.cached_len == 128, (
        f"turn 2 resumed at {m.cuda_handle.cached_len}, not 128: the boundary chunk 1 tracked "
        "was thrown away at the chunk seam (the boundary carry)"
    )
    assert m.mamba_value is not None, "prefix matched but no GDN snapshot came with it"
    assert _marker(pool, m.mamba_value) == 128.0, (
        f"the tree returned a snapshot holding state {_marker(pool, m.mamba_value)} for prefix "
        "[0, 128): the frozen slot does not correspond to the committed boundary"
    )
    _finish(cm, req)
    cm.check_integrity()


def test_the_carried_boundary_survives_two_seams():
    """Three chunks, the last short: chunk 1 tracks 128, chunk 2 tracks 320 and OVERWRITES the
    other ping-pong slot, chunk 3 (33) tracks nothing. The committed snapshot must be chunk
    2's -- the deepest -- and hold chunk 2's state, not chunk 1's stale one."""
    cm, tm, pool = _rig()
    ids = list(range(1, 418))                                 # 417 = 192 + 192 + 33
    req, lens, tracked = _run_turn(cm, tm, _pend(0, ids), budget=192)
    assert lens == [192, 192, 33], lens
    assert [b for b, _ in tracked] == [128, 320], tracked
    assert tracked[0][1] != tracked[1][1], "chunk 2 reused chunk 1's ping-pong slot"

    m = cm.match_req(_pend(1, ids[:320] + [9001]))
    assert m.cuda_handle.cached_len == 320, m.cuda_handle.cached_len
    assert _marker(pool, m.mamba_value) == 320.0, (
        "the tree returned the state of the WRONG boundary across two seams"
    )
    _finish(cm, req)
    cm.check_integrity()


def test_a_tracking_final_chunk_is_unchanged():
    """The regression arm. When the final chunk DOES track, the carried value is overwritten
    and behaviour must be exactly what it was before the carry: commit at the final chunk's
    own boundary, holding the final chunk's own state."""
    cm, tm, pool = _rig()
    ids = list(range(1, 386))                                 # 385 = 256 + 129, final >= CHUNK+1
    req, lens, tracked = _run_turn(cm, tm, _pend(0, ids), budget=256)
    assert lens == [256, 129], lens
    assert [b for b, _ in tracked] == [192, 384], tracked     # both chunks tracked

    m = cm.match_req(_pend(1, ids[:384] + [9001]))
    assert m.cuda_handle.cached_len == 384, m.cuda_handle.cached_len
    assert _marker(pool, m.mamba_value) == 384.0
    _finish(cm, req)
    cm.check_integrity()


def test_an_unchunked_turn_is_unchanged():
    """A prompt that fits one chunk has no seam; a fresh admit must still start with no carried
    boundary (the field is per-turn, never inherited across requests)."""
    cm, tm, pool = _rig()
    ids = list(range(1, 194))
    req, lens, tracked = _run_turn(cm, tm, _pend(0, ids), budget=256)
    assert lens == [193] and [b for b, _ in tracked] == [192], (lens, tracked)
    m = cm.match_req(_pend(1, ids[:192] + [9001]))
    assert m.cuda_handle.cached_len == 192 and _marker(pool, m.mamba_value) == 192.0
    _finish(cm, req)
    cm.check_integrity()


# --------------------------------------------------------------------------- the carry itself


def test_a_continuation_chunk_inherits_the_tracked_boundary():
    """The one-line half of the carry, isolated: ``PrefillAdder`` must rebuild the
    continuation ``Req`` with the boundary its predecessor tracked -- as a PAIR with
    ``mamba_next_track_idx``, which it already carried."""
    cm, tm, _ = _rig()
    pending = _pend(0, list(range(1, 226)))
    first = PrefillAdder(token_budget=192, reserved_size=0,
                         cache_manager=cm, table_manager=tm).try_add_one(pending)
    assert isinstance(first, ChunkedReq)
    first.mamba_last_track_seqlen = 128
    first.mamba_next_track_idx = 1
    first.complete_one()
    pending.chunked_req = first

    second = PrefillAdder(token_budget=192, reserved_size=0,
                          cache_manager=cm, table_manager=tm).try_add_one(pending)
    assert not isinstance(second, ChunkedReq)
    assert second.mamba_last_track_seqlen == 128, (
        "the continuation chunk dropped the boundary its predecessor tracked "
    )
    assert second.mamba_next_track_idx == 1, "the boundary and its slot index must travel together"
    assert second.mamba_ping_pong == first.mamba_ping_pong


def test_a_fresh_admit_carries_no_boundary():
    """The other side of the same rule: only a continuation inherits. A new request starts at
    None, or it would commit a previous request's snapshot against its own prefix."""
    cm, tm, _ = _rig()
    req = PrefillAdder(token_budget=192, reserved_size=0, cache_manager=cm,
                       table_manager=tm).try_add_one(_pend(0, list(range(1, 100))))
    assert req.mamba_last_track_seqlen is None


# --------------------------------------------------------------------------- the guard


def _staged(cm, tm, pool, prompt_len=225):
    """A request at its final chunk, everything real, commit not yet called: pages allocated, a
    ping-pong pair held, one boundary tracked at the seam."""
    pending = _pend(0, list(range(1, prompt_len + 1)))
    adder = PrefillAdder(token_budget=192, reserved_size=0, cache_manager=cm, table_manager=tm)
    first = adder.try_add_one(pending)
    cm.allocate_paged([first])
    _forward_track(pool, [first])
    first.complete_one()
    pending.chunked_req = first
    adder = PrefillAdder(token_budget=192, reserved_size=0, cache_manager=cm, table_manager=tm)
    last = adder.try_add_one(pending)
    cm.allocate_paged([last])
    last.complete_one()
    return last


def test_the_guard_refuses_a_boundary_past_what_is_committed():
    """``insert()`` reads ``page_indices[:L]`` off a page-table row only ``req.cached_len``
    long while keying on ``cache_ids[:L]``, which decode has already grown -- so an L past the
    commit inserts a key and a page run of DIFFERENT lengths. Out of reach for a carried
    boundary by construction; enforced rather than argued.

    256, not ``cached_len + PAGE``: the pre-existing page-alignment check (cache.py:424)
    already refuses an unaligned L, which would hide this one."""
    cm, tm, pool = _rig()
    req = _staged(cm, tm, pool)                             # cached_len 225
    L = 256                                                 # page-aligned AND past the commit
    assert L > req.cached_len
    req.mamba_last_track_seqlen = L
    before = cm.linear_state_pool.num_free_slots
    cm.cache_req(req, finished=False)
    assert req.mamba_last_track_seqlen is None, "the out-of-window boundary was not dropped"
    assert cm.linear_state_pool.num_free_slots == before, "a slot moved on a refused commit"
    assert cm.match_req(_pend(1, list(range(1, 226)))).cuda_handle.cached_len == 0, (
        "a boundary past the commit reached the tree"
    )


def test_the_guard_refuses_a_boundary_below_the_requests_own_prefix():
    """Below the handle's own matched prefix the dedup ``_free`` slice and the page-table
    re-point both invert, and ``unlock(old_handle)`` drops the LONGER lock while taking a
    shorter one -- leaving the request's own page-table row naming pages nothing protects."""
    cm, tm, pool = _rig()
    ids = list(range(1, 386))                                   # 385 = 256 + 129
    seed, lens, _ = _run_turn(cm, tm, _pend(0, ids), budget=256)
    assert lens == [256, 129]                                   # final chunk tracks -> donates at 384
    _finish(cm, seed)

    pending = _pend(1, ids[:384] + list(range(9000, 9100)))
    adder = PrefillAdder(token_budget=512, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert req.cache_handle.cached_len == 384, req.cache_handle.cached_len
    cm.allocate_paged([req])
    req.complete_one()
    req.mamba_last_track_seqlen = 320                           # aligned, but below its own prefix
    cm.cache_req(req, finished=False)
    assert req.mamba_last_track_seqlen is None
    assert req.cache_handle.cached_len == 384, (
        f"the handle shrank to {req.cache_handle.cached_len}: a boundary below the request's "
        "own matched prefix unlocked the longer node it is still sitting on"
    )
    _finish(cm, req)
    cm.check_integrity()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
