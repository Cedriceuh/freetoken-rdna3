"""FREETOKEN_ROUTER_SHARED: the router GEMV and the shared expert's gate | up GEMV with its activation in one launch give
the bits of the two launches (gemv_bf16 with its in-kernel reduce, gemv_int8_act_mul), at the warps they run at."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# the shared expert's gate | up rows a rank holds: NVFP4 even split (640) and uneven (704 / 576), EXL3 3.05 bpw (768,
# 4 warps by the default config)
SHARED_ROWS = [640, 704, 576, 768]


@pytest.mark.parametrize("n2", SHARED_ROWS)
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_one_launch_matches_the_two(n2, rows, seed, monkeypatch):
    from freetoken.kernel.triton import router_shared
    from freetoken.kernel.triton.dense_gemv import _BF16_CONFIGS, gemv_bf16, gemv_counters
    from freetoken.kernel.triton.int8_gemv import gemv_int8_act_mul

    monkeypatch.setattr(router_shared, "ENABLED", True)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 10 + rows)
    x = (torch.randn(rows, 2560, generator=g, device=dev) * (0.3 + seed)).to(torch.bfloat16)
    gate_w = (torch.randn(512, 2560, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    q = torch.randint(-127, 128, (n2, 2560), generator=g, device=dev, dtype=torch.int8)
    scale = torch.rand(n2, generator=g, device=dev) * 0.01
    cfg = _BF16_CONFIGS[(512, 2560)]
    ref_logits = gemv_bf16(x, gate_w, cfg, counters=gemv_counters(gate_w, cfg))
    ref_act = gemv_int8_act_mul(x, q, scale)
    got = router_shared.router_and_shared_act(x, gate_w, cfg, gemv_counters(gate_w, cfg), q, scale)
    assert got is not None
    assert torch.equal(got[0], ref_logits) and torch.equal(got[1], ref_act)


@pytest.mark.parametrize("dk", [n // 2 for n in SHARED_ROWS])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_down_and_topk_match_the_two(dk, rows, seed, monkeypatch):
    """The shared expert's down projection and the router's softmax top-k in one launch: the bits of gemv_int8 and
    fused_topk_softmax (ties included: integer-valued logits)."""
    from freetoken.kernel.triton.int8_gemv import gemv_int8
    from freetoken.kernel.triton.moe_router import fused_topk_softmax
    from freetoken.kernel.triton import router_shared
    from freetoken.kernel.triton.router_shared import shared_down_and_topk

    monkeypatch.setattr(router_shared, "ENABLED", True)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 10 + rows)
    act = (torch.randn(rows, dk, generator=g, device=dev) * (0.5 + seed)).to(torch.bfloat16)
    q = torch.randint(-127, 128, (2560, dk), generator=g, device=dev, dtype=torch.int8)
    scale = torch.rand(2560, generator=g, device=dev) * 0.01
    logits = torch.randn(rows, 512, generator=g, device=dev) * 3
    if seed % 3 == 2:
        logits = logits.round()
    logits = logits.to(torch.bfloat16)
    ref = gemv_int8(act, q, scale), *fused_topk_softmax(logits, 10, True)
    got = shared_down_and_topk(act, q, scale, logits, 10, True)
    assert got is not None
    assert all(torch.equal(a, b) for a, b in zip(got, ref))
