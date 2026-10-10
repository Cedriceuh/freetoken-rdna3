"""FREETOKEN_GDN_PREFILL_BLOCK: the blocked GDN prefill gives the bits of the one-call prefill (GPU), output, final
states and the tracked h rows, for packed requests that span several blocks."""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_blocked_gdn_prefill_matches_one_call():
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla
    from freetoken.models.qwen4_exp import gdn as G

    torch.manual_seed(0)
    dev = "cuda"
    lens = [9000, 5000]  # two requests, neither a multiple of the block
    T, HK, HV, D = sum(lens), 7, 21, 128  # one rank's heads at the uneven split
    q = torch.randn(1, T, HK, D, device=dev).to(torch.bfloat16)
    k = torch.randn(1, T, HK, D, device=dev).to(torch.bfloat16)
    v = torch.randn(1, T, HV, D, device=dev).to(torch.bfloat16)
    g = -torch.rand(1, T, HV, device=dev) * 0.1
    beta = torch.rand(1, T, HV, device=dev)
    state0 = torch.randn(4, HV, D, D, device=dev) * 0.01
    idx = torch.tensor([1, 3], device=dev, dtype=torch.int32)
    cu = torch.tensor([0, lens[0], T], device=dev, dtype=torch.int64)
    # track the state after 100 and after 70 chunks of 64 tokens (rows into the packed per-chunk h)
    track = [(0, 100), (1, 70)]
    boh = [0, (lens[0] + 63) // 64]
    s1 = state0.clone()
    o1, h = gdn_prefill_chunk_fla(q, k, v, g, beta, state_source=s1, indices=idx, cu_seqlens=cu, scale=D ** -0.5,
                                  return_h=True)
    rows1 = torch.stack([h[0, boh[i] + c] for i, c in track])
    fla = types.SimpleNamespace(seq_bounds=[(0, lens[0]), (lens[0], T)], cache_indices=idx, track_seq_chunk=track)
    s2 = state0.clone()
    o2, rows2 = G._gdn_prefill_blocked(q, k, v, g, beta, s2, fla, D ** -0.5, True)
    assert torch.equal(o1, o2)
    assert torch.equal(s1, s2)
    assert torch.equal(rows1, rows2)
