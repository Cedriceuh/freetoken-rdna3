"""The short-prefill expert tiles give the bits of the tiles they replace (the same K order): NVFP4's 4-warp tiles up to
1536 tokens against the 8-warp ones, EXL3's 16-row tiles up to 512 tokens against 32-row ones."""

from __future__ import annotations

import pytest
import torch

from tests.moe.test_cpu_moe_exl3 import _cache, _experts
from tests.moe.test_nvfp4_backends import E as NV_E
from tests.moe.test_nvfp4_backends import H as NV_H
from tests.moe.test_nvfp4_backends import TOPK as NV_TOPK
from tests.moe.test_nvfp4_backends import _make_native_sources

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("M", [32, 300, 1280])
def test_nvfp4_short_tiles_keep_the_bits(monkeypatch, M):
    from freetoken.moe import fused_nvfp4 as fm

    dev = torch.device("cuda")
    sources = _make_native_sources(dev, seed=5)
    names = ("gate_up_packed", "gate_up_scale", "gate_up_global", "down_packed", "down_scale", "down_global")
    bank = [sources[n][1].to(dev) for n in names]
    gen = torch.Generator().manual_seed(M)
    x = (torch.randn(M, NV_H, generator=gen) / 4).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(NV_E, generator=gen)[:NV_TOPK] for _ in range(M)]).to(torch.int32).to(dev)
    w = torch.rand(M, NV_TOPK, generator=gen).to(dev)
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", True)
    new = fm.fused_experts_nvfp4(x, *bank, w, ids, NV_E, "silu", False)
    # the tiles these sizes took before: the shipped 16-row 8-warp tile up to 64 tokens, _TUNED_MID above
    monkeypatch.setattr(fm, "_prefill_configs",
                        lambda m: (fm._prefill_config(m),) * 2 if m <= 64 else (fm._TUNED_MID, fm._TUNED_MID))
    old = fm.fused_experts_nvfp4(x, *bank, w, ids, NV_E, "silu", False)
    assert torch.equal(new, old)


@pytest.mark.parametrize("M", [16, 300, 512])
def test_exl3_short_tiles_keep_the_bits(monkeypatch, M):
    from freetoken.layers.quantization.exl3_codec import CB_MUL1
    from freetoken.moe import fused_exl3 as fe

    dev = torch.device("cuda")
    E, H, I, top_k = 16, 256, 256, 4
    bank = {k: v[0].to(dev) for k, v in _cache(_experts(E, H, I, 3, seed=12), 0, I).bank_sources.items()}
    names = ("gate_up", "gate_up_suh", "gate_up_svh", "down", "down_suh", "down_svh")
    gen = torch.Generator().manual_seed(M)
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(M)]).to(torch.int32).to(dev)
    w = torch.rand(M, top_k, generator=gen).to(dev)
    new = fe.fused_experts_exl3(x, *(bank[n] for n in names), w, ids, codebook=CB_MUL1, is_prefill=True, num_experts=E)
    monkeypatch.setattr(fe, "_SHORT_PREFILL_TOKENS", 0)  # 32-row tiles at every size
    old = fe.fused_experts_exl3(x, *(bank[n] for n in names), w, ids, codebook=CB_MUL1, is_prefill=True, num_experts=E)
    assert torch.equal(new, old)
