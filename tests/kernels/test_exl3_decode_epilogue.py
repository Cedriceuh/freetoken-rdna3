"""FREETOKEN_EXL3_DECODE_EPILOGUE: exl3_out with the layer's epilogue (the gate from the HIP router | shared launch)
gives the bits of exl3_out followed by shared_gate_sum_mul_add, at a decode step's 1-8 rows (GPU, ROCm)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="needs a ROCm GPU")


def _spread(shape, g, dev):
    v = torch.randn(shape, generator=g, device=dev)
    return v * torch.exp2(torch.randint(-8, 8, shape, generator=g, device=dev).float())


@pytest.mark.parametrize("n2", [768, 640])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("seed", range(4))
def test_epilogue_in_the_out_kernel_matches_the_separate_one(n2, rows, seed, monkeypatch):
    from freetoken.kernel.router_shared_exact import router_and_shared_act_exact
    from freetoken.kernel.triton.exl3 import exl3_out
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sum_mul_add

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 10 + rows + n2)
    mk = _spread if seed % 2 else (lambda shape, g, dev: torch.randn(shape, generator=g, device=dev))
    H, top_k, S = 2560, 10, 64
    x = mk((rows, H), g, dev).bfloat16()
    if seed == 2:
        x[0, :32] = 0.0
    router_w = (torch.randn(512, H, generator=g, device=dev) * 0.02).bfloat16()
    q = torch.randint(-127, 128, (n2, H), generator=g, device=dev, dtype=torch.int8)
    scale = torch.rand(n2, generator=g, device=dev) * 0.01
    gate_w = (mk((H,), g, dev) * 0.05).bfloat16()
    _, _, gate = router_and_shared_act_exact(x, router_w, q, scale, n2 == 768, gate_w)
    shared = mk((rows, H), g, dev).bfloat16()
    c = torch.randn(1, rows * top_k, H, device=dev, generator=g) * (4.0 if seed % 2 else 1.0)
    if seed == 2:
        c[:, :, :64] = 0.0  # routed zeros (signed) through the epilogue
    slots = torch.randint(0, S, (rows * top_k,), device=dev, dtype=torch.int32, generator=g)
    svh = torch.rand(S, H, device=dev, generator=g).half()
    w = torch.rand(rows, top_k, device=dev, generator=g)
    routed = exl3_out(c, slots, svh, w, torch.empty(rows, H, device=dev, dtype=torch.bfloat16))
    ref = shared_gate_sum_mul_add(routed.view(rows, 1, H), shared, x, gate_w)
    got = exl3_out(c, slots, svh, w, torch.empty(rows, H, device=dev, dtype=torch.bfloat16), epilogue=(shared, gate))
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
