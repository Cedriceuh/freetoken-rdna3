"""FREETOKEN_QK_NORM_ROPE: the attention layer's q / k RMSNorms and 3-axis rope in one launch give the bits of the
three launches they replace (GemmaPlusOneRMSNorm.forward_strided twice, then MRotaryEmbedding)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("hq,hk", [(12, 1), (24, 2)])
@pytest.mark.parametrize("seed", range(3))
def test_qk_norm_rope_matches_the_three_launches(rows, hq, hk, seed):
    from freetoken.kernel.triton.norm import gemma_rmsnorm
    from freetoken.kernel.triton.qk_norm_rope import qk_norm_rope
    from freetoken.kernel.triton.rope import apply_mrope_with_cos_sin_cache_inplace
    from freetoken.layers.rotary import build_section_table

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 100 + rows)
    H, rotary_dim = 256, 64
    # the projection's output: [rows, q heads x 2H (q | gate)] then k, as the layer slices it
    qkv = torch.randn(rows, hq * 2 * H + hk * H, generator=g, device=dev).to(torch.bfloat16) * 3
    q = qkv[:, : hq * 2 * H].view(rows, hq, 2 * H)[..., :H]
    k = qkv[:, hq * 2 * H:].view(rows, hk, H)
    wq = (torch.randn(H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    wk = (torch.randn(H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    pos = torch.randint(0, 200000, (3, rows), generator=g, device=dev, dtype=torch.int32)
    inv = 1.0 / (10000000 ** (torch.arange(0, rotary_dim, 2, device=dev, dtype=torch.float32) / rotary_dim))
    t = torch.arange(262144, device=dev, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()
    section = build_section_table((11, 11, 10), "interleaved").to(dev)
    eps = 1e-6
    # the three launches
    q_ref = torch.empty(q.shape, dtype=q.dtype, device=dev)
    k_ref = torch.empty(k.shape, dtype=k.dtype, device=dev)
    gemma_rmsnorm(q, wq, eps, out=q_ref)
    gemma_rmsnorm(k, wk, eps, out=k_ref)
    apply_mrope_with_cos_sin_cache_inplace(pos, q_ref.view(rows, -1), k_ref.view(rows, -1), H, cache, section)
    got = qk_norm_rope(q, k, wq, wk, eps, True, pos, cache, section)
    assert got is not None
    assert torch.equal(got[0], q_ref) and torch.equal(got[1], k_ref)
