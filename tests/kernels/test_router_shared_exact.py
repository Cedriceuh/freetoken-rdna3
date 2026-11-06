"""FREETOKEN_ROUTER_SHARED_HIP: the HIP router | shared gate-up kernel gives the Triton launch's bits (both outputs),
at the rows a rank's shared expert holds and 1-8 rows, on values spread over many binades (sums in another order
would round differently)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="needs a ROCm GPU")


def _spread(shape, g, dev, lo=-8, hi=8):
    v = torch.randn(shape, generator=g, device=dev)
    return v * torch.exp2(torch.randint(lo, hi, shape, generator=g, device=dev).float())


@pytest.mark.parametrize("n2", [640, 704, 576, 768])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("seed", range(6))
def test_hip_matches_the_triton_launch(n2, rows, seed, monkeypatch):
    from freetoken.kernel.triton import router_shared
    from freetoken.kernel.triton.dense_gemv import _BF16_CONFIGS, gemv_counters

    monkeypatch.setattr(router_shared, "ENABLED", True)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 10 + rows)
    if seed % 2:
        x = _spread((rows, 2560), g, dev).to(torch.bfloat16)
        gate_w = (_spread((512, 2560), g, dev) * 0.02).to(torch.bfloat16)
    else:
        x = (torch.randn(rows, 2560, generator=g, device=dev) * (0.3 + seed)).to(torch.bfloat16)
        gate_w = (torch.randn(512, 2560, generator=g, device=dev) * 0.02).to(torch.bfloat16)
    if seed == 4:
        x[0] = 0  # signed-zero products
    q = torch.randint(-127, 128, (n2, 2560), generator=g, device=dev, dtype=torch.int8)
    scale = torch.rand(n2, generator=g, device=dev) * (0.01 if seed < 3 else 1.0)
    cfg = _BF16_CONFIGS[(512, 2560)]
    monkeypatch.setattr(router_shared, "HIP", False)
    ref = router_shared.router_and_shared_act(x, gate_w, cfg, gemv_counters(gate_w, cfg), q, scale)
    monkeypatch.setattr(router_shared, "HIP", True)
    got = router_shared.router_and_shared_act(x, gate_w, cfg, gemv_counters(gate_w, cfg), q, scale)
    assert ref is not None and got is not None
    assert torch.equal(got[0].view(torch.int16), ref[0].view(torch.int16))
    assert torch.equal(got[1].view(torch.int16), ref[1].view(torch.int16))



@pytest.mark.parametrize("n2", [640, 768])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("seed", range(4))
def test_hip_gate_matches_shared_gate_sigmoid(n2, rows, seed):
    """The shared expert's gate the launch also computes (FREETOKEN_EXL3_DECODE_EPILOGUE) has shared_gate_sigmoid's bits,
    at both block sizes (the K-blocked and the one-pass gate | up tiles), and leaves the other outputs unchanged."""
    from freetoken.kernel.router_shared_exact import router_and_shared_act_exact
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sigmoid

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 10 + rows + n2)
    mk = _spread if seed % 2 else (lambda shape, g, dev: torch.randn(shape, generator=g, device=dev))
    x = mk((rows, 2560), g, dev).bfloat16()
    if seed == 2:
        x[0, :40] = 0.0
    gate_w = (torch.randn(512, 2560, generator=g, device=dev) * 0.02).bfloat16()
    q = torch.randint(-127, 128, (n2, 2560), generator=g, device=dev, dtype=torch.int8)
    scale = torch.rand(n2, generator=g, device=dev) * 0.01
    shared_gate_w = (mk((2560,), g, dev) * (0.002 if seed % 2 else 0.05)).bfloat16()
    logits, act, gate = router_and_shared_act_exact(x, gate_w, q, scale, n2 == 768, shared_gate_w)
    logits0, act0, none = router_and_shared_act_exact(x, gate_w, q, scale, n2 == 768)
    assert none is None
    assert torch.equal(gate.view(torch.int32), shared_gate_sigmoid(x, shared_gate_w).view(torch.int32))
    assert torch.equal(logits.view(torch.int16), logits0.view(torch.int16))
    assert torch.equal(act.view(torch.int16), act0.view(torch.int16))
