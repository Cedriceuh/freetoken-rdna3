"""Prefill through the slot cache (FREETOKEN_PREFILL_CACHED_MAX): the grouped GEMMs read each routed expert from its
cache slot instead of a full-layer buffer where position == expert id. The tiles are those of the expert ids, so the
output must be bit-identical; ``ensure_prefill_routed`` must leave every routed expert resident with its bytes, keep
the hits and evict the least recently used other slots."""

from __future__ import annotations

import pytest
import torch

from tests.moe.test_cpu_moe_exl3 import _cache, _experts
from tests.moe.test_nvfp4_backends import E as NV_E
from tests.moe.test_nvfp4_backends import H as NV_H
from tests.moe.test_nvfp4_backends import TOPK as NV_TOPK
from tests.moe.test_nvfp4_backends import _make_native_sources

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _scatter_to_slots(full: torch.Tensor, perm: torch.Tensor, slots: int, gen: torch.Generator) -> torch.Tensor:
    """A slot bank of ``slots`` rows of random bytes, with expert e's row copied to slot ``perm[e]``."""
    out = torch.empty((slots, *full.shape[1:]), dtype=full.dtype, device=full.device)
    raw = out.view(torch.uint8)
    raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8, generator=gen))
    out[perm.long()] = full
    return out


@cuda
@pytest.mark.parametrize("M", [1, 7, 90, 300])
def test_exl3_prefill_from_slots_is_bit_identical(M):
    from freetoken.layers.quantization.exl3_codec import CB_MUL1
    from freetoken.moe.fused_exl3 import fused_experts_exl3

    dev = torch.device("cuda")
    E, H, I, top_k, S = 16, 256, 256, 4, 40
    bank = {k: v[0].to(dev) for k, v in _cache(_experts(E, H, I, 3, seed=11), 0, I).bank_sources.items()}
    gen = torch.Generator().manual_seed(M)
    perm = torch.randperm(S, generator=gen)[:E].to(dev).to(torch.int32)
    slots = {k: _scatter_to_slots(v, perm, S, gen) for k, v in bank.items()}
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(M)]).to(torch.int32).to(dev)
    w = torch.rand(M, top_k, generator=gen).to(dev)
    names = ("gate_up", "gate_up_suh", "gate_up_svh", "down", "down_suh", "down_svh")
    ref = fused_experts_exl3(x, *(bank[n] for n in names), w, ids, codebook=CB_MUL1, is_prefill=True, num_experts=E)
    out = fused_experts_exl3(x, *(slots[n] for n in names), w, ids, codebook=CB_MUL1, is_prefill=True, num_experts=E,
                             slot_map=perm)
    assert torch.equal(out, ref)


@cuda
@pytest.mark.parametrize("M", [1, 9, 200])
def test_nvfp4_prefill_from_slots_is_bit_identical(M):
    from freetoken.moe.fused_nvfp4 import fused_experts_nvfp4

    dev = torch.device("cuda")
    S = 21
    sources = _make_native_sources(dev, seed=3)
    names = ("gate_up_packed", "gate_up_scale", "gate_up_global", "down_packed", "down_scale", "down_global")
    bank = {n: sources[n][1].to(dev) for n in names}
    gen = torch.Generator().manual_seed(100 + M)
    perm = torch.randperm(S, generator=gen)[:NV_E].to(dev).to(torch.int32)
    slots = {n: _scatter_to_slots(v, perm, S, gen) for n, v in bank.items()}
    x = (torch.randn(M, NV_H, generator=gen) / 4).bfloat16().to(dev)
    ids = torch.stack([torch.randperm(NV_E, generator=gen)[:NV_TOPK] for _ in range(M)]).to(torch.int32).to(dev)
    w = torch.rand(M, NV_TOPK, generator=gen).to(dev)
    ref = fused_experts_nvfp4(x, *(bank[n] for n in names), w, ids, NV_E, "silu", False)
    out = fused_experts_nvfp4(x, *(slots[n] for n in names), w, ids, NV_E, "silu", False, slot_map=perm)
    assert torch.equal(out, ref)


@cuda
def test_ensure_prefill_routed_keeps_hits_and_evicts_lru():
    from freetoken.moe.offload_cache import OffloadMoeCache

    dev = torch.device("cuda")
    L, E, S = 3, 8, 12
    g = torch.Generator().manual_seed(0)
    sources = {
        "gate_up": [torch.randn(E, 32, 8, generator=g).bfloat16().pin_memory() for _ in range(L)],
        "down": [torch.randn(E, 8, 16, generator=g).bfloat16().pin_memory() for _ in range(L)],
    }
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=S, device=dev, prefill_overlap=False)
    cache.set_bank_sources(sources)
    cache.reset()
    assert cache.prefill_cached_ok(1)

    def seed(layer, expert, slot, usage):
        cache.slot_for_id[layer, expert] = slot
        cache.id_of_slot[slot] = layer * E + expert
        cache.usage[slot] = usage
        for name, per_layer in sources.items():
            cache.bank_caches[name][slot].copy_(per_layer[layer][expert])

    # layer 1 expert 3 (routed, a hit) in slot 5; other layers' experts with increasing usage; slots 9-11 empty
    seed(1, 3, 5, 2)
    for slot, (layer, expert, usage) in enumerate([(0, 0, 7), (0, 1, 3), (2, 4, 9), (2, 5, 1), (0, 6, 8)]):
        seed(layer, expert, slot, usage)
    seed(1, 6, 6, 4)  # layer 1, not routed: evictable
    seed(2, 7, 7, 6)
    seed(0, 2, 8, 5)
    cache.step.fill_(10)
    topk = torch.tensor([[3, 0], [5, 3], [0, 7], [2, 5]], dtype=torch.int32, device=dev)  # routed: 0 2 3 5 7
    row = cache.ensure_prefill_routed(1, topk)

    assert int(cache.step) == 11
    assert int(row[3]) == 5  # the hit stays put
    routed = [0, 2, 3, 5, 7]
    for e in routed:
        s = int(row[e])
        assert s >= 0 and int(cache.id_of_slot[s]) == E + e
        for name, per_layer in sources.items():
            assert torch.equal(cache.bank_caches[name][s].cpu(), per_layer[1][e])
    new_slots = sorted(int(row[e]) for e in routed if e != 3)
    # 4 misses: the 3 empty slots (usage 0) first, then the oldest resident (slot 3, usage 1)
    assert new_slots == [3, 9, 10, 11]
    assert int(cache.slot_for_id[2, 5]) == -1
    # every map entry agrees with its reverse
    for s in range(S):
        i = int(cache.id_of_slot[s])
        if i >= 0:
            assert int(cache.slot_for_id.view(-1)[i]) == s
    assert int(cache.usage[5]) == 11 and all(int(cache.usage[s]) == 11 for s in new_slots)
    assert int(cache.usage[6]) == 4  # layer 1's unrouted expert keeps its usage
    # once the GEMMs are queued, the experts the prefill brought in are the next victims; the hit stays recent
    cache.demote_prefill_slots()
    assert all(int(cache.usage[s]) == 0 for s in new_slots) and int(cache.usage[5]) == 11


@cuda
def test_prefill_split_copy_matches_sources():
    """The hit/miss-split prefetch of a full layer into the double buffer (the small banks as whole layers, the misses of
    the big ones through the batched copy engine): every buffer row equals its source, including an expert left in a
    buffer slot (< 2E) whose bytes are poisoned. On ROCm too (the upstream tests of this path need CUDA >= 13)."""
    from freetoken.moe import offload_cache as oc

    dev = torch.device("cuda")
    L, E, S = 3, 8, 24
    g = torch.Generator().manual_seed(1)

    def rand(*shape, dtype=torch.uint8):  # random bytes viewed as dtype
        n = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g).view(dtype).view(*shape)

    sources = {  # the native NVFP4 banks: the packed rows are big (256 KiB, gathered and copied by rows), the rest small
        "gate_up_packed": [rand(E, 1024, 256) for _ in range(L)],
        "gate_up_scale": [rand(E, 1024, 32) for _ in range(L)],
        "gate_up_global": [rand(E, 1024, dtype=torch.float16) for _ in range(L)],
        "down_packed": [rand(E, 512, 512) for _ in range(L)],
        "down_scale": [rand(E, 512, 64) for _ in range(L)],
        "down_global": [rand(E, 512, dtype=torch.float16) for _ in range(L)],
    }
    sources = {n: [t.pin_memory() for t in v] for n, v in sources.items()}
    cache = oc.OffloadMoeCache(num_layers=L, num_experts=E, cache_size=S, device=dev, quant_format="nvfp4",
                               prefill_overlap=True, prefill_hit_d2d=True)
    cache.set_bank_sources(sources)

    def seed(layer, expert, slot):
        cache.slot_for_id[layer, expert] = slot
        cache.id_of_slot[slot] = layer * E + expert
        for name, per_layer in sources.items():
            cache.bank_caches[name][slot].copy_(per_layer[layer][expert])

    seed(1, 1, 17)
    seed(1, 3, 20)
    seed(1, 0, 2)
    seed(2, 5, 23)
    for name in sources:
        cache.bank_caches[name][2].view(torch.uint8).fill_(0xFF)
    cache.begin_prefill()
    if not cache._prefill_hit_d2d_active:
        pytest.skip("hit-D2D split unavailable (no batch memcpy)")
    for layer_id in range(L):
        cache.prefetch_prefill_layer(layer_id)
        cache.prefetch_prefill_layer(layer_id + 1)
        views = cache.wait_prefill_layer(layer_id)
        torch.cuda.synchronize()
        for view, name in zip(views, cache.bank_schema):
            assert torch.equal(view.cpu().view(torch.uint8), sources[name][layer_id].view(torch.uint8)), (layer_id, name)
        cache.release_prefill_layer(layer_id)


@cuda
def test_predicted_prefetch_then_fixup_keeps_every_routed_expert():
    """A layer's routed experts are resident with their bytes after the predicted prefetch (copy stream) and the
    fixup copy of what the prediction missed; the prefetch never takes a slot of the layer still computing; the
    prefill's own slots end demoted, and the maps stay each other's inverse."""
    from freetoken.moe import offload_cache as oc

    dev = torch.device("cuda")
    L, E, S = 3, 16, 40
    g = torch.Generator().manual_seed(4)

    def rand(*shape, dtype=torch.uint8):
        n = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g).view(dtype).view(*shape)

    sources = {
        "gate_up_packed": [rand(E, 1024, 256) for _ in range(L)],
        "gate_up_scale": [rand(E, 1024, 32) for _ in range(L)],
        "gate_up_global": [rand(E, 1024, dtype=torch.float16) for _ in range(L)],
        "down_packed": [rand(E, 512, 512) for _ in range(L)],
        "down_scale": [rand(E, 512, 64) for _ in range(L)],
        "down_global": [rand(E, 512, dtype=torch.float16) for _ in range(L)],
    }
    sources = {n: [t.pin_memory() for t in v] for n, v in sources.items()}
    cache = oc.OffloadMoeCache(num_layers=L, num_experts=E, cache_size=S, device=dev, quant_format="nvfp4",
                               prefill_overlap=True, prefill_hit_d2d=True)
    cache.set_bank_sources(sources)
    cache.reset()

    def check(layer, experts):
        row = cache.slot_for_id[layer]
        for e in experts:
            s = int(row[e])
            assert s >= 0, (layer, e)
            for name in cache.bank_schema:
                assert torch.equal(cache.bank_caches[name][s].cpu().view(torch.uint8),
                                   sources[name][layer][e].view(torch.uint8)), (layer, e, name)

    def ids(experts):
        return torch.tensor(experts, dtype=torch.int32, device=dev).view(-1, 1)

    routed0 = [0, 1, 2, 3, 5, 8, 13]
    predicted = torch.tensor([[1, 2, 4, 6, 7], [9, 10, 11, 12, 14]], dtype=torch.int64, device=dev)
    row0 = cache.ensure_prefill_routed(0, ids(routed0), (1, predicted))
    slots0 = {int(row0[e]) for e in routed0}
    pf_slots = {int(s) for s in cache._prefill_prefetch[1][1] if int(s) >= 0}
    assert len(pf_slots) == 10  # none of layer 1's experts was resident
    assert not pf_slots & slots0  # layer 0's routed slots are in use by its GEMMs
    check(0, routed0)
    cache.demote_prefill_slots()
    routed1 = [1, 2, 3, 4, 7, 15]  # 3 and 15 were not predicted
    cache.ensure_prefill_routed(1, ids(routed1))
    assert int(cache.num_indices) == 2
    torch.cuda.synchronize()
    check(1, routed1)
    cache.demote_prefill_slots()
    assert all(int(cache.usage[int(cache.slot_for_id[1, e])]) == 0 for e in routed1)  # all brought in by this prefill
    assert not cache._prefill_prefetch
    for s in range(S):
        i = int(cache.id_of_slot[s])
        if i >= 0:
            assert int(cache.slot_for_id.view(-1)[i]) == s
    for layer in range(L):
        for e in range(E):
            s = int(cache.slot_for_id[layer, e])
            if s >= 0:
                assert int(cache.id_of_slot[s]) == layer * E + e


def test_path_is_decided_once_per_forward_for_every_layer(monkeypatch):
    """The first MoE layer decides from the chunk length; later layers, the MTP head's (one row a request) included,
    follow it: after a full-layer chunk the last layer has queued the head's experts into a double buffer."""
    from types import SimpleNamespace

    from freetoken.layers import moe
    from freetoken.layers.moe import OffloadMoELayer

    monkeypatch.setattr(moe, "_PREFILL_CACHED_SET", None)  # the default threshold, whatever FREETOKEN_PREFILL_CACHED_MAX says

    all_ok = [True]
    cache = SimpleNamespace(prefill_chunk_cached=False, prefill_cached_all_ok=lambda: all_ok[0])
    method = SimpleNamespace(kernel=SimpleNamespace(supports_prefill_slot_map=True))

    def layer(i):
        return SimpleNamespace(offload_cache=cache, layer_id=i, prefill_cached_default=True, quant_method=method)

    decide = OffloadMoELayer.prefill_cached
    assert not decide(layer(0), 4096) and not decide(layer(47), 4096) and not decide(layer(48), 1)
    assert decide(layer(0), 200) and decide(layer(47), 200) and decide(layer(48), 1)
    all_ok[0] = False  # a CPU or unpinned layer anywhere: no layer takes it
    assert not decide(layer(0), 200) and not decide(layer(48), 1)
