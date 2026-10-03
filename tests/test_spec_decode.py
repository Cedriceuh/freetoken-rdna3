"""spec_decode rollback: every registered state ends where the request's last accepted row left it."""
from __future__ import annotations

import torch

import freetoken.spec_decode as spec


def test_rollback_picks_the_snapshot_of_the_last_accepted_row(monkeypatch):
    monkeypatch.setattr(spec, "SPEC_M", 3)
    monkeypatch.setattr(spec, "_TARGETS", {})
    state = torch.arange(10, dtype=torch.float32).view(5, 2)  # 5 slots, final values after row m-1
    snaps = torch.tensor([[[100.0, 101.0], [110.0, 111.0]],   # after row 0, requests 0 / 1
                          [[200.0, 201.0], [210.0, 211.0]]])  # after row 1
    spec.register("s", spec.RollbackTarget(state, lambda step, b: snaps[step, b], padding_slot=4))
    final = state.clone()
    slots = torch.tensor([3, 1])
    spec.apply_rollback(slots, torch.tensor([1, 3]))  # request 0 kept row 0 only, request 1 kept all 3 rows
    assert state[3].tolist() == [100.0, 101.0]
    assert state[1].tolist() == final[1].tolist()
    spec.apply_rollback(slots, torch.tensor([2, 2]))
    assert state[3].tolist() == [200.0, 201.0] and state[1].tolist() == [210.0, 211.0]
    untouched = [0, 2]  # slot 4 is the padding sink: the request that kept every row wrote there
    assert state[untouched].tolist() == final[untouched].tolist()


def test_stacked_layers_roll_back_in_one_copy(monkeypatch):
    monkeypatch.setattr(spec, "SPEC_M", 2)
    monkeypatch.setattr(spec, "_TARGETS", {})
    state = torch.zeros(3, 4, 2)  # [layers, slots, ...]
    snaps = torch.arange(3 * 2 * 2, dtype=torch.float32).view(3, 2, 1, 2).expand(3, 2, 1, 2)  # [layers, reqs, m-1, ...]
    spec.register("gdn", spec.RollbackTarget(state, lambda step, b: snaps[:, b, step], dim=1))
    spec.apply_rollback(torch.tensor([2, 3]), torch.tensor([1, 2]), padding_slot=0)
    assert torch.equal(state[:, 2], snaps[:, 0, 0]) and torch.equal(state[:, 3], torch.zeros(3, 2))


def _depth_env(monkeypatch, m=4, costs=None, dynamic=True, bs_max=1):
    monkeypatch.setattr(spec, "SPEC_M", m)
    monkeypatch.setattr(spec, "SPEC_MS", tuple(range(1, m + 1)))
    monkeypatch.setattr(spec, "SPEC_DYNAMIC", dynamic)
    monkeypatch.setattr(spec, "SPEC_BS_MAX", bs_max)
    monkeypatch.setattr(spec, "SPEC_COSTS", costs or {1: 18.0, 2: 25.0, 3: 31.0, 4: 37.0})


class _Req:
    def __init__(self, drafts=None, stats=None, remain=None):
        if drafts is not None:
            self.spec_drafts = drafts
        if stats is not None:
            self.spec_stats = stats
        if remain is not None:
            self.remain_len = remain


def test_depth_stats_update_only_the_tried_positions(monkeypatch):
    _depth_env(monkeypatch)
    s = spec.DepthStats()
    s.update(3, 2)  # 2 drafts tried, the first kept, the second refused; position 2 untried
    assert s.rate[0] > spec._PRIOR and s.rate[1] < spec._PRIOR
    assert s.rate[2] == spec._PRIOR + spec._RELAX * (s.rate[1] - spec._PRIOR)  # drifts toward the deepest tried
    before = list(s.rate)
    s.update(4, 1)  # the first draft refused: positions 1 and 2 were not reached
    assert s.rate[0] < before[0] and s.rate[1:] == before[1:]


def test_best_rows_follow_the_acceptance(monkeypatch):
    _depth_env(monkeypatch)
    s = spec.DepthStats()
    s.rate = [0.95, 0.9, 0.85]
    assert s.best_rows() == 4  # 1 + .95 + .855 + .727 = 3.53 tokens for 37 ms beats 2.81 for 31
    s.rate = [0.85, 0.5, 0.3]
    assert s.best_rows() == 2  # 1.85 / 25 beats 2.275 / 31
    s.rate = [0.0, 0.0, 0.0]
    assert s.best_rows() == 2  # never 1 row: the stats would stop updating


def test_rows_for_caps_by_requests_and_drafts(monkeypatch):
    _depth_env(monkeypatch)
    high = spec.DepthStats()
    high.rate = [0.95, 0.9, 0.85]
    assert spec.rows_for([_Req(drafts=3, stats=high)]) == 4
    assert spec.rows_for([_Req(drafts=1, stats=high)]) == 2  # one guess after a prompt
    assert spec.rows_for([_Req(drafts=0, stats=high)]) == 1  # the head only wrote its KV
    assert spec.rows_for([_Req(drafts=3, stats=high), _Req(drafts=3, stats=high)]) == 1  # past SPEC_BS_MAX
    assert spec.draft_depth([_Req(stats=high)]) == 3 and spec.draft_depth([_Req(), _Req()]) == 0
    _depth_env(monkeypatch, dynamic=False)
    assert spec.rows_for([_Req()]) == 4  # no draft model: every row a placeholder, SPEC_M rows


def test_rows_for_skips_row_counts_without_graphs(monkeypatch):
    _depth_env(monkeypatch)
    monkeypatch.setattr(spec, "SPEC_MS", (1, 4))
    assert spec.rows_for([_Req(drafts=2)]) == 1  # 3 rows would fit the drafts, but only 1 and 4 are captured


def test_rows_and_drafts_stay_inside_what_the_request_may_emit(monkeypatch):
    """Near max_device_len (the page table's width when the request may fill the context) a step runs no more rows
    than the tokens left, and the chained drafts' KV ends at max_device_len - 1."""
    _depth_env(monkeypatch)
    high = spec.DepthStats()
    high.rate = [0.95, 0.9, 0.85]
    for remain in (1, 2, 3, 4, 9):
        req = _Req(drafts=3, stats=high, remain=remain)
        m = spec.rows_for([req])
        assert m == min(4, remain)
        depth = spec.draft_depth([req], m)
        # the step's rows sit at L .. L+m-1 and the chain's at most at L+m-1 + depth-1, with L = max_device_len-remain-1
        assert m - 1 + max(depth - 1, 0) <= remain and depth >= 1
    monkeypatch.setattr(spec, "SPEC_MS", (2, 4))  # no 1-row graphs: the step that fits runs eagerly
    assert spec.rows_for([_Req(drafts=3, stats=high, remain=1)]) == 1


def test_one_card_costs_favour_two_rows(monkeypatch):
    """On one card the extra rows' missing experts cross PCIe: with its measured costs the adaptive depth keeps to 2 rows
    even at high acceptance, while two cards go deeper."""
    _depth_env(monkeypatch)
    high = spec.DepthStats()
    high.rate = [0.95, 0.9, 0.85]
    monkeypatch.setattr(spec, "SPEC_COSTS", spec._costs(1))
    assert high.best_rows() == 2
    monkeypatch.setattr(spec, "SPEC_COSTS", spec._costs(2))
    assert high.best_rows() == 4
    monkeypatch.setenv("FREETOKEN_SPEC_COSTS", "4:30")
    assert spec._costs(1)[4] == 30.0 and spec._costs(1)[2] == 53.0


def test_runtime_cache_rebuild_refused_with_verify_rows(monkeypatch):
    """The rollback targets hold the state pools registered at the first capture: a runtime rebuild that replaced
    those pools would leave the rollback writing into the old tensors, so it is refused before anything is freed."""
    from types import SimpleNamespace

    import pytest

    from freetoken.engine import engine as eng
    from freetoken.kvcache.base import CacheRebuildRejected

    monkeypatch.setattr(eng, "SPEC_M", 4)
    monkeypatch.setattr(eng, "tp_shares", lambda size: None)
    fake = SimpleNamespace(config=SimpleNamespace(tp_info=SimpleNamespace(size=1)))
    with pytest.raises(CacheRebuildRejected, match="SPEC_VERIFY_M"):
        eng.Engine.rebuild_runtime_cache(fake, num_mamba_slots=8)
