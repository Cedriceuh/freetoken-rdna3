"""FREETOKEN_NVFP4_ACT_IN_GEMV: decode's gate | up GEMV with silu_and_mul in the same kernel gives, bit for bit, the
marlin GEMV's output then the activation kernel's (both halves of a program's columns on the GEMV's own tiles), at the
model's per-rank shapes, 1-8 decoding rows, the shipped and the tuned tiles, routes the CPU computes (slot -1)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("shape", [(704, 2560), (576, 2560), (256, 512)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("routed_weight", [False, True])
def test_act_in_gemv_matches_gemv_then_activation(monkeypatch, shape, rows, tuned, routed_weight):
    import freetoken.kernel.nvfp4_moe as nm
    from freetoken.layers import gated_act_and_mul

    monkeypatch.setattr(nm, "NVFP4_DECODE_TUNED", tuned)
    N, K = shape
    dev, S, top_k = "cuda", 24, 10
    g = torch.Generator(device=dev).manual_seed(N + K + rows + 2 * tuned + routed_weight)
    packed = torch.randint(0, 256, (S, N, K // 2), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)
    scale = torch.randint(0, 0x7E, (S, N, K // 16), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)
    glob = torch.rand(S, N, device=dev, generator=g) * 1e-3
    hidden = torch.randn(rows, K, device=dev, generator=g).to(torch.bfloat16)
    ids = torch.randint(-1, S, (rows, top_k), device=dev, generator=g, dtype=torch.int32)
    if not nm.SKIP_CPU_ROUTES:
        ids = ids.clamp(min=0)
    w = torch.rand(rows, top_k, device=dev, generator=g)
    ic1 = torch.empty((rows, top_k, N), device=dev, dtype=torch.bfloat16)
    nm.decode_gemm_marlin(hidden, packed, scale, glob, ic1, w, ids, routed_weight, False)
    want = torch.empty((rows * top_k, N // 2), device=dev, dtype=torch.bfloat16)
    gated_act_and_mul("silu", ic1.view(-1, N), want)
    got = torch.empty_like(want)
    nm.decode_gemm_marlin_act(hidden, packed, scale, glob, got, w, ids, routed_weight)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
