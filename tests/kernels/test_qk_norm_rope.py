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


@pytest.mark.parametrize("rows", [1, 3, 4, 16])
@pytest.mark.parametrize("hq,hk", [(12, 1), (24, 2)])
@pytest.mark.parametrize("seed", range(3))
def test_kv_store_matches_store_cache(rows, hq, hk, seed):
    """FREETOKEN_QK_STORE_KV: the launch's K/V store leaves the caches store_cache leaves, q and k unchanged."""
    from freetoken.kernel import store_cache
    from freetoken.kernel.triton.qk_norm_rope import qk_norm_rope
    from freetoken.layers.rotary import build_section_table

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 100 + rows + 7)
    H, rotary_dim, slots = 256, 64, 4096
    qkv = torch.randn(rows, hq * 2 * H + 2 * hk * H, generator=g, device=dev).to(torch.bfloat16) * 3
    q = qkv[:, : hq * 2 * H].view(rows, hq, 2 * H)[..., :H]
    k = qkv[:, hq * 2 * H: hq * 2 * H + hk * H].view(rows, hk, H)
    v = qkv[:, hq * 2 * H + hk * H:].view(rows, hk, H)
    wq = (torch.randn(H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    wk = (torch.randn(H, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    pos = torch.randint(0, 200000, (3, rows), generator=g, device=dev, dtype=torch.int32)
    inv = 1.0 / (10000000 ** (torch.arange(0, rotary_dim, 2, device=dev, dtype=torch.float32) / rotary_dim))
    freqs = torch.outer(torch.arange(262144, device=dev, dtype=torch.float32), inv)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()
    section = build_section_table((11, 11, 10), "interleaved").to(dev)
    loc = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[:rows].to(dev)
    if seed == 1:
        loc = loc.to(torch.int32)
    kc0 = torch.randn(slots // 64, 64, hk, H, generator=g, device=dev).to(torch.bfloat16)
    vc0 = torch.randn(slots // 64, 64, hk, H, generator=g, device=dev).to(torch.bfloat16)
    ref = qk_norm_rope(q, k, wq, wk, 1e-6, True, pos, cache, section)
    kc_ref, vc_ref = kc0.clone(), vc0.clone()
    store_cache(kc_ref.view(slots, hk, H), vc_ref.view(slots, hk, H), loc, ref[1].view(rows, -1), v.reshape(rows, -1))
    kc, vc = kc0.clone(), vc0.clone()
    got = qk_norm_rope(q, k, wq, wk, 1e-6, True, pos, cache, section,
                       kv_store=(kc.view(slots, -1), vc.view(slots, -1), loc, v))
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    assert torch.equal(kc, kc_ref) and torch.equal(vc, vc_ref)
