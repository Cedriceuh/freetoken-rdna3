"""``hc_combine_rmsnorm`` gives exactly what ``hc_combine`` then ``grouped_gemma_rmsnorm`` give (both outputs)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 37, 1500])
@pytest.mark.parametrize("shared_weight", [False, True])
def test_combine_rmsnorm_is_bit_exact(tokens, shared_weight):
    from freetoken.kernel.triton.hc import grouped_gemma_rmsnorm, hc_combine, hc_combine_rmsnorm

    hc, hidden = 4, 2560
    gen = torch.Generator(device="cuda").manual_seed(tokens * 7 + shared_weight)
    res = torch.randn(tokens, hc * hidden, device="cuda", generator=gen).bfloat16()
    block = torch.randn(tokens, hidden, device="cuda", generator=gen).bfloat16()
    inj = torch.randn(tokens, hc, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(hidden if shared_weight else hc * hidden, device="cuda", generator=gen) * 0.1).bfloat16()

    combined = hc_combine(res, block, inj, hc)
    normed = grouped_gemma_rmsnorm(combined, w, 1e-6, hc)
    got_combined, got_normed = hc_combine_rmsnorm(res, block, inj, w, 1e-6, hc)
    assert torch.equal(got_combined, combined)
    assert torch.equal(got_normed, normed)
