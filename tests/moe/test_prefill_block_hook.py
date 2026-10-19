"""FREETOKEN_PREFILL_AR_OVERLAP: the routed experts of a long prefill call ``BLOCK_DONE`` once per token block, with
those rows final; the MoE epilogue computed block by block with the whole chunk's warps gives the bits of one call.
(The all-reduce of a block is the elementwise sum of the two ranks' rows, the same per element however the rows are
split.)"""

from __future__ import annotations

import pytest
import torch

from tests.moe.test_nvfp4_backends import E as NV_E
from tests.moe.test_nvfp4_backends import H as NV_H
from tests.moe.test_nvfp4_backends import TOPK as NV_TOPK
from tests.moe.test_nvfp4_backends import _make_native_sources

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_nvfp4_prefill_calls_block_done_with_final_rows(monkeypatch):
    from freetoken.moe import fused_nvfp4, prefill_hooks

    dev = torch.device("cuda")
    sources = _make_native_sources(dev, seed=5)
    names = ("gate_up_packed", "gate_up_scale", "gate_up_global", "down_packed", "down_scale", "down_global")
    bank = {n: sources[n][0].to(dev) for n in names}
    gen = torch.Generator().manual_seed(9)
    M = 300
    x = (torch.randn(M, NV_H, generator=gen) / 4).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(NV_E, generator=gen)[:NV_TOPK] for _ in range(M)]).to(torch.int32).to(dev)
    w = torch.rand(M, NV_TOPK, generator=gen).to(dev)
    monkeypatch.setattr(fused_nvfp4, "NVFP4_PREFILL_BLOCK", 128)
    seen = []

    def done(start, end, out):
        seen.append((start, end, out[start:end].clone()))

    prefill_hooks.BLOCK_DONE[0] = done
    try:
        out = fused_nvfp4.fused_experts_nvfp4(x, *(bank[n] for n in names), w, ids, NV_E, "silu", False)
    finally:
        prefill_hooks.BLOCK_DONE[0] = None
    assert [(s, e) for s, e, _ in seen] == [(0, 128), (128, 256), (256, 300)]
    for s, e, rows in seen:
        assert torch.equal(rows, out[s:e])


@cuda
def test_exl3_prefill_calls_block_done_with_final_rows(monkeypatch):
    from freetoken.layers.quantization.exl3_codec import CB_MUL1
    from freetoken.moe import fused_exl3, prefill_hooks
    from tests.moe.test_cpu_moe_exl3 import _cache, _experts

    dev = torch.device("cuda")
    E, H, I, top_k, M = 4, 256, 256, 2, 300
    bank = {k: v[0].to(dev) for k, v in _cache(_experts(E, H, I, 3, seed=5), 0, I).bank_sources.items()}
    gen = torch.Generator().manual_seed(9)
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(M)]).to(torch.int32).to(dev)
    w = torch.rand(M, top_k, generator=gen).to(dev)
    monkeypatch.setattr(fused_exl3, "EXL3_PREFILL_BLOCK", 128)
    seen = []
    prefill_hooks.BLOCK_DONE[0] = lambda start, end, out: seen.append((start, end, out[start:end].clone()))
    try:
        out = fused_exl3.fused_experts_exl3(x, bank["gate_up"], bank["gate_up_suh"], bank["gate_up_svh"], bank["down"],
                                            bank["down_suh"], bank["down_svh"], w, ids, codebook=CB_MUL1,
                                            is_prefill=True, num_experts=E)
    finally:
        prefill_hooks.BLOCK_DONE[0] = None
    assert [(s, e) for s, e, _ in seen] == [(0, 128), (128, 256), (256, 300)]
    for s, e, rows in seen:
        assert torch.equal(rows, out[s:e])


def test_the_layer_blocks_by_its_own_expert_format():
    """A chunk is split for the all-reduce only past the block its experts' format runs (NVFP4 8192 on the NVFP4
    profiles, EXL3 4096): an NVFP4 chunk of 4097-8192 tokens runs in one block, with one all-reduce."""
    from freetoken.layers.quantization.moe.exl3 import TritonExl3MoEKernel
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel
    from freetoken.models.qwen4_exp import moe
    from freetoken.moe import fused_exl3, fused_nvfp4

    assert moe._block_tokens(object.__new__(TritonExl3MoEKernel)) == fused_exl3.EXL3_PREFILL_BLOCK
    assert moe._block_tokens(object.__new__(TritonNvfp4MoEKernel)) == fused_nvfp4.NVFP4_PREFILL_BLOCK


@cuda
@pytest.mark.parametrize("M,block", [(3000, 1024), (2049, 2048), (300, 128)])
def test_epilogue_by_blocks_matches_one_call(M, block):
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sum_mul_add

    H = 2560
    g = torch.Generator(device="cuda").manual_seed(M)
    routed = torch.randn(M, 1, H, device="cuda", generator=g).bfloat16()
    shared = torch.randn(M, H, device="cuda", generator=g).bfloat16()
    hidden = torch.randn(M, H, device="cuda", generator=g).bfloat16()
    gate_w = (torch.randn(H, device="cuda", generator=g) / 50).bfloat16()
    ref = shared_gate_sum_mul_add(routed, shared, hidden, gate_w)
    out = torch.empty_like(ref)
    for s in range(0, M, block):
        e = min(s + block, M)
        shared_gate_sum_mul_add(routed[s:e], shared[s:e], hidden[s:e], gate_w, out=out[s:e], warps_tokens=M)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
