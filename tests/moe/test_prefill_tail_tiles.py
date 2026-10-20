"""Prefill tail tiles: a >= 3072-token NVFP4 block runs each expert's full 128-row tiles in one launch and its remaining
rows in 32-row tiles in a second one (FREETOKEN_NVFP4_PREFILL_TAIL_TILES), EXL3's gate|up GEMM full 64-row tiles and
32-row ones (FREETOKEN_EXL3_PREFILL_TAIL_TILES); ``moe_align_split`` places every routed row exactly once, and the
outputs keep the bits of the one-launch tiles (the same K order)."""

from __future__ import annotations

import pytest
import torch

from tests.moe.test_nvfp4_backends import E as NV_E
from tests.moe.test_nvfp4_backends import H as NV_H
from tests.moe.test_nvfp4_backends import _make_native_sources

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _routing(M: int, E: int, top_k: int, skew: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    pref = torch.rand(E, generator=gen) * skew  # skew > 0: a few experts take most rows, some none
    return (torch.rand(M, E, generator=gen) + pref).topk(top_k, dim=1).indices.to(torch.int32)


@pytest.mark.parametrize("skew", [0.0, 3.0])
def test_split_alignment_places_every_row_once(skew):
    from freetoken.kernel.triton.moe_align import moe_align_split

    E, M, top_k = 64, 3000, 4
    ids = _routing(M, E, top_k, skew, seed=1).cuda()
    (fs, fe, fn), (rs, re_, rn) = moe_align_split(ids, 128, 32, E)
    flat = ids.flatten().long().cpu()
    counts = torch.bincount(flat, minlength=E)
    nf, nr = int(fn), int(rn)
    full, rest = fs[:nf].long().cpu(), rs[:nr].long().cpu()
    assert nf == int(((counts // 128) * 128).sum()) and nf % 128 == 0 and nr % 32 == 0
    assert bool((full < flat.numel()).all())  # full tiles hold no padding
    got = torch.cat([full, rest[rest < flat.numel()]])
    assert torch.equal(got.sort().values, torch.arange(flat.numel()))  # every row exactly once
    for blocks, eids, bs in ((full, fe, 128), (rest, re_, 32)):
        e_of_block = eids[: blocks.numel() // bs].long().cpu()
        rows = blocks.view(-1, bs)
        ok = rows < flat.numel()
        assert bool((flat[rows[ok]] == e_of_block.repeat_interleave(bs).view(-1, bs)[ok]).all())
    per_expert_rest = torch.bincount(flat[rest[rest < flat.numel()]], minlength=E)
    assert torch.equal(per_expert_rest, counts % 128)


@pytest.mark.parametrize("M,skew", [(3072, 0.0), (5000, 3.0), (8192, 1.0)])
@pytest.mark.parametrize("slots", [False, True])
def test_tail_tiles_keep_the_bits(monkeypatch, M, skew, slots):
    from freetoken.moe import fused_nvfp4 as fm

    dev = torch.device("cuda")
    sources = _make_native_sources(dev, seed=7)
    names = ("gate_up_packed", "gate_up_scale", "gate_up_global", "down_packed", "down_scale", "down_global")
    bank = [sources[n][1].to(dev) for n in names]
    gen = torch.Generator().manual_seed(M)
    x = (torch.randn(M, NV_H, generator=gen) / 4).bfloat16().to(dev)
    ids = _routing(M, NV_E, 2, skew, seed=M).to(dev)
    w = torch.rand(M, 2, generator=gen).to(dev)
    slot_map, by_id = None, bank
    if slots:  # the banks are a slot cache, expert e in slot (e + 3) % E
        perm = (torch.arange(NV_E) + 3) % NV_E
        bank = [t[torch.argsort(perm)] if t.dim() > 1 else t for t in bank]
        slot_map = perm.to(torch.int32).to(dev)
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", True)
    monkeypatch.setattr(fm, "NVFP4_PREFILL_BLOCK", 0)
    monkeypatch.setattr(fm, "_TAIL_TILES", True)
    new = fm.fused_experts_nvfp4(x, *bank, w, ids, NV_E, "silu", False, slot_map=slot_map)
    monkeypatch.setattr(fm, "_TAIL_TILES", False)
    old = fm.fused_experts_nvfp4(x, *bank, w, ids, NV_E, "silu", False, slot_map=slot_map)
    assert torch.equal(new, old)
    if slots:  # and the slot cache reads each expert's own weights: the bits of the banks indexed by expert id
        monkeypatch.setattr(fm, "_TAIL_TILES", True)
        assert torch.equal(new, fm.fused_experts_nvfp4(x, *by_id, w, ids, NV_E, "silu", False))
    # and the old path is the one-launch 128 / 64-row tiles
    assert fm._prefill_configs(M) == (fm._TUNED_BIG_GATE_UP, fm._TUNED_MID)


@pytest.mark.parametrize("skew", [0.0, 3.0])
@pytest.mark.parametrize("slots", [False, True])
def test_exl3_tail_tiles_keep_the_bits(monkeypatch, skew, slots):
    from freetoken.layers.quantization.exl3_codec import CB_MUL1
    from freetoken.moe import fused_exl3 as fe
    from tests.moe.test_cpu_moe_exl3 import _cache, _experts

    dev = torch.device("cuda")
    E, H, I, top_k, M = 16, 256, 256, 4, 3072  # ~768 routes an expert: full 64-row tiles and remainders
    bank = {k: v[0].to(dev) for k, v in _cache(_experts(E, H, I, 3, seed=13), 0, I).bank_sources.items()}
    names = ("gate_up", "gate_up_suh", "gate_up_svh", "down", "down_suh", "down_svh")
    gen = torch.Generator().manual_seed(int(skew * 10) + slots)
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16().to(dev)
    ids = _routing(M, E, top_k, skew, seed=21).to(dev)
    w = torch.rand(M, top_k, generator=gen).to(dev)
    slot_map, tensors = None, [bank[n] for n in names]
    by_id = tensors
    if slots:  # the banks are a slot cache, expert e in slot (e + 5) % E
        perm = (torch.arange(E) + 5) % E
        tensors = [t[torch.argsort(perm).to(dev)] for t in tensors]
        slot_map = perm.to(torch.int32).to(dev)
    monkeypatch.setattr(fe, "EXL3_PREFILL_BLOCK", 0)
    run = lambda: fe.fused_experts_exl3(x, *tensors, w, ids, codebook=CB_MUL1, is_prefill=True, num_experts=E,
                                        slot_map=slot_map)
    monkeypatch.setattr(fe, "_TAIL_TILES", True)
    new = run()
    monkeypatch.setattr(fe, "_TAIL_TILES", False)
    assert torch.equal(new, run())
    if slots:  # the slot cache reads each expert's own weights
        monkeypatch.setattr(fe, "_TAIL_TILES", True)
        assert torch.equal(new, fe.fused_experts_exl3(x, *by_id, w, ids, codebook=CB_MUL1, is_prefill=True,
                                                      num_experts=E))
