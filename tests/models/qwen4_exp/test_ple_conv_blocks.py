"""PLE prefill forward over time blocks (PLE_CONV_BLOCK, PLELayer._forward_blocks) == the packed forward: output,
next conv state and the GDN track snapshot, for one request and for several packed together.

CPU only; the block path only changes how the [T, width] temporaries are sized, never the result.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.models.qwen4_exp.ple as ple


@pytest.mark.parametrize("block", [4, 8, 13])
def test_blocked_forward_matches_the_packed_forward(monkeypatch, block):
    """PLELayer.forward for one long prefill request: time blocks == one pass (output and next conv state)."""
    from .test_ple import EOS, _config, _forward, _make_layer, _meta

    torch.manual_seed(21)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    tokens = [3, 4, EOS, 5, 6, 8, 9, 2, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3]
    R = torch.randn(len(tokens), args.ple_state_width)
    states = torch.randn(1, args.ple_state_width, args.ple_conv_state_len) * 0.1

    monkeypatch.setattr(ple, "PLE_CONV_BLOCK", 0)
    ref_states = states.clone()
    ref = _forward(layer, R, _meta([tokens], [[21, 22]]), ref_states)
    monkeypatch.setattr(ple, "PLE_CONV_BLOCK", block)
    got_states = states.clone()
    got = _forward(layer, R, _meta([tokens], [[21, 22]]), got_states)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(got_states, ref_states, rtol=1e-5, atol=1e-6)


def test_blocked_forward_writes_the_same_track_snapshot(monkeypatch):
    """The prefix-cache snapshot of the conv history (GDN track boundary) is the same through the blocked path."""
    from .test_ple import _config, _make_layer, _meta

    torch.manual_seed(22)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    tokens = list(range(3, 3 + 30))
    R = torch.randn(len(tokens), args.ple_state_width)
    fla = SimpleNamespace(track_boundary_row=torch.tensor([17]), track_dst=torch.tensor([3]))
    batch = SimpleNamespace(fla_metadata=fla)
    results = []
    for block in (0, 8):
        monkeypatch.setattr(ple, "PLE_CONV_BLOCK", block)
        states = torch.zeros(4, args.ple_state_width, args.ple_conv_state_len)
        out = layer.forward(R, batch, meta=_meta([tokens], [[21, 22]], slots=[1]), conv_states=states)
        results.append((out, states))
    (ref, ref_states), (got, got_states) = results
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(got_states[3], ref_states[3], rtol=1e-5, atol=1e-6) and ref_states[3].abs().sum() > 0
    assert torch.allclose(got_states[1], ref_states[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("block", [2, 4, 8, 13])
def test_blocked_forward_several_requests_matches_the_packed_forward(monkeypatch, block):
    """PLELayer.forward for several requests prefilled together (a multi-request server batches up to
    --max-prefill-length tokens of them): time blocks, each request's conv over its own history, == one packed pass (output and next conv states),
    requests shorter than a block and than the conv history included."""
    from .test_ple import EOS, _config, _forward, _make_layer, _meta

    torch.manual_seed(23)
    config = _config()
    args = config.qwen4_args
    layer = _make_layer(config)
    sequences = [[3, 4, EOS, 5, 6, 8, 9, 2, 4, 5, 6, 7, 11, 12, 13], [2, EOS, 11], [9], list(range(3, 23))]
    contexts = [[EOS, EOS], [21, 22], [EOS, 31], [5, 6]]
    total = sum(len(s) for s in sequences)
    R = torch.randn(total, args.ple_state_width)
    states = torch.randn(6, args.ple_state_width, args.ple_conv_state_len) * 0.1
    slots = [4, 1, 5, 2]

    monkeypatch.setattr(ple, "PLE_CONV_BLOCK", 0)
    ref_states = states.clone()
    ref = _forward(layer, R, _meta(sequences, contexts, slots=slots, fresh=[False, True, False, False]), ref_states)
    monkeypatch.setattr(ple, "PLE_CONV_BLOCK", block)
    got_states = states.clone()
    got = _forward(layer, R, _meta(sequences, contexts, slots=slots, fresh=[False, True, False, False]), got_states)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(got_states, ref_states, rtol=1e-5, atol=1e-6)
