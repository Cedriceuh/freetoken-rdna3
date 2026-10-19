"""FREETOKEN_GDN_AR_OVERLAP: the blocked delta rule hands each block's output over as it finishes, the same rows the
assembled output holds, and the gated norm of a block with the whole chunk's rows per program gives the rows of the
whole call. (The output projection's blocks are checked at run time against hipBLASLt; the all-reduce is a sum of two
values per element.)"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_blocked_delta_rule_hands_over_the_assembled_rows(monkeypatch):
    from freetoken.models.qwen4_exp import gdn

    monkeypatch.setattr(gdn, "_GDN_PREFILL_BLOCK", 128)
    T, H, HV, K = 512, 4, 12, 128
    g = torch.Generator(device="cuda").manual_seed(3)

    def r(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)

    q, k, v = r(1, T, H, K), r(1, T, H, K), r(1, T, HV, K)
    gg = -torch.rand(1, T, HV, device="cuda", generator=g)
    beta = torch.rand(1, T, HV, device="cuda", generator=g)
    state0 = r(2, HV, K, K, dtype=torch.float32, scale=0.1)

    def fla():
        return SimpleNamespace(seq_bounds=[(0, T)], cache_indices=torch.tensor([1], device="cuda"), track_seq_chunk=[])

    state = state0.clone()
    o, _ = gdn._gdn_prefill_blocked(q, k, v, gg, beta, state, fla(), K ** -0.5, False)
    state2, seen = state0.clone(), []
    gdn._gdn_prefill_blocked(q, k, v, gg, beta, state2, fla(), K ** -0.5, False,
                             block_done=lambda a, b, res: seen.append((a, b, res.clone())))
    assert [(a, b) for a, b, _ in seen] == [(0, 128), (128, 256), (256, 384), (384, 512)]
    for a, b, res in seen:
        assert torch.equal(res, o[a:b])
    assert torch.equal(state2, state)


@pytest.mark.parametrize("tokens,block,strided", [(16384, 4096, True), (512, 128, True), (16384, 4096, False)])
def test_gated_norm_by_blocks_with_the_whole_rows_per_program(tokens, block, strided):
    from freetoken.kernel.fla.layernorm_gated import calc_rows_per_block
    from freetoken.layers import GatedRMSNorm

    HV, V = 27, 128
    norm = GatedRMSNorm(V, eps=1e-6, activation="sigmoid")
    g = torch.Generator(device="cuda").manual_seed(tokens)
    norm.weight = torch.randn(V, device="cuda", generator=g).bfloat16()
    x = torch.randn(tokens * HV, V, device="cuda", generator=g).bfloat16()
    zfull = torch.randn(tokens, HV + 5, V, device="cuda", generator=g).bfloat16()
    # FREETOKEN_FEWER_COPIES: the projection's strided slice, else its rows copied out flat
    z = zfull[:, :HV] if strided else zfull[:, :HV].reshape(-1, V)
    whole = norm.forward(x, z)
    rpb = calc_rows_per_block(tokens * HV, x.device)
    for a in range(0, tokens, block):
        b = a + block
        part = norm.forward(x[a * HV:b * HV], z[a:b] if strided else z[a * HV:b * HV], rows_per_block=rpb)
        assert torch.equal(part, whole[a * HV:b * HV])
