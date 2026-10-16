"""The EXL3 prefill GEMMs run two pipeline stages and 16-block groups (gate | up) since 2026-10-14: only the launch order
and the loads' prefetch change, so the routed experts' output must keep the bits of one stage and 8-block groups."""

from __future__ import annotations

import functools

import pytest
import torch

from tests.moe.test_cpu_moe_exl3 import _cache, _experts

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
@pytest.mark.parametrize("M", [300, 2000])
def test_two_stages_keep_the_bits(M, monkeypatch):
    from freetoken.layers.quantization.exl3_codec import CB_MUL1
    from freetoken.moe import fused_exl3

    dev = torch.device("cuda")
    E, H, I, top_k = 16, 256, 256, 4
    bank = {k: v[0].to(dev) for k, v in _cache(_experts(E, H, I, 3, seed=5), 0, I).bank_sources.items()}
    gen = torch.Generator().manual_seed(M)
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(M)]).to(torch.int32).to(dev)
    w = torch.rand(M, top_k, generator=gen).to(dev)
    names = ("gate_up", "gate_up_suh", "gate_up_svh", "down", "down_suh", "down_svh")

    def run():
        return fused_exl3.fused_experts_exl3(x, *(bank[n] for n in names), w, ids, codebook=CB_MUL1, is_prefill=True,
                                             num_experts=E)

    out = run()
    monkeypatch.setattr(fused_exl3, "exl3_moe_gemm", functools.partial(fused_exl3.exl3_moe_gemm, num_stages=1,
                                                                        group_m=8))
    monkeypatch.setattr(fused_exl3, "exl3_gemm_down_out", functools.partial(fused_exl3.exl3_gemm_down_out,
                                                                             num_stages=1))
    assert torch.equal(out, run())
