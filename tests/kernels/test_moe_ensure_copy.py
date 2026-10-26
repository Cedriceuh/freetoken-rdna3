"""FREETOKEN_MOE_ENSURE_COPY: the one-launch admission + small-bank copy (kernel/triton/ensure_copy.py) leaves the
slot cache, its plan and the copied rows exactly as flashlib's lru_ensure followed by fast_index_copy_multi do, step
after step (hits, misses, duplicate routes, evictions, several layers sharing one pool)."""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

E, LAYERS = 512, 3
FEATS = (10240, 1536, 768, 5120)  # EXL3's suh / svh rows at TP=2, bytes


def _state(cache_size: int, dev):
    return dict(
        slot_of_id=torch.full((LAYERS * E,), -1, dtype=torch.int32, device=dev),
        id_of_slot=torch.full((cache_size,), -1, dtype=torch.int32, device=dev),
        usage=torch.zeros((cache_size,), dtype=torch.int64, device=dev),
        step=torch.zeros((), dtype=torch.int64, device=dev),
        src=torch.zeros((max(E, cache_size),), dtype=torch.int32, device=dev),
        dst=torch.zeros((max(E, cache_size),), dtype=torch.int32, device=dev),
        num=torch.zeros((1,), dtype=torch.int64, device=dev),
        banks=[torch.zeros((cache_size, f), dtype=torch.uint8, device=dev) for f in FEATS],
    )


@requires_cuda
@pytest.mark.parametrize("cache_size", [700, 11675])
@pytest.mark.parametrize("rows", [1, 4, 8, 12])  # 12 rows: 120 routes, the widest query the one launch takes
def test_one_launch_matches_ensure_then_copy(cache_size, rows):
    from flashlib.kernels.slot_cache import lru_ensure

    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    from freetoken.kernel.triton.ensure_copy import ensure_copy, supported

    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(cache_size + rows)
    host = [[torch.randint(0, 256, (E, f), dtype=torch.uint8, generator=g).pin_memory() for f in FEATS]
            for _ in range(LAYERS)]
    ref, got = _state(cache_size, dev), _state(cache_size, dev)
    sync = torch.zeros((2,), dtype=torch.int32, device=dev)

    def plan(st, layer):
        return (torch.tensor([b.data_ptr() for b in st["banks"]], dtype=torch.int64, device=dev),
                torch.tensor([h.data_ptr() for h in host[layer]], dtype=torch.int64, device=dev),
                torch.tensor(FEATS, dtype=torch.int64, device=dev))

    hot = torch.randperm(E, generator=g)[:40]
    for step in range(60):
        layer = step % LAYERS
        # a skewed routing: mostly a hot set (hits), some fresh experts (misses), duplicates across rows
        ids = torch.where(torch.rand(rows, 10, generator=g) < 0.8, hot[torch.randint(0, 40, (rows, 10), generator=g)],
                          torch.randint(0, E, (rows, 10), generator=g)).to(torch.int32)
        q_ref, q_got = ids.clone().view(-1).to(dev), ids.clone().view(-1).to(dev)
        lru_ensure(q_ref, ref["slot_of_id"], ref["id_of_slot"], ref["usage"], ref["step"], q_ref, ref["src"],
                   ref["dst"], ref["num"], id_base=layer * E)
        fast_index_copy_multi_jit(*plan(ref, layer), ref["dst"], ref["src"], ref["num"])
        assert supported(q_got, got["id_of_slot"])
        ensure_copy(q_got, got["slot_of_id"], got["id_of_slot"], got["usage"], got["step"], got["src"], got["dst"],
                    got["num"], layer * E, plan(got, layer), sync)
        torch.cuda.synchronize()
        assert torch.equal(q_ref, q_got), f"step {step}: slots differ"
        n = int(ref["num"])
        assert int(got["num"]) == n
        assert torch.equal(ref["src"][:n], got["src"][:n]) and torch.equal(ref["dst"][:n], got["dst"][:n])
        for k in ("slot_of_id", "id_of_slot", "usage", "step"):
            assert torch.equal(ref[k], got[k]), f"step {step}: {k} differs"
        for b in range(len(FEATS)):
            assert torch.equal(ref["banks"][b], got["banks"][b]), f"step {step}: bank {b} differs"
        assert sync.tolist() == [0, 0]


@requires_cuda
def test_without_banks_to_copy():
    from flashlib.kernels.slot_cache import lru_ensure

    from freetoken.kernel.triton.ensure_copy import ensure_copy

    dev = torch.device("cuda")
    ref, got = _state(300, dev), _state(300, dev)
    sync = torch.zeros((2,), dtype=torch.int32, device=dev)
    g = torch.Generator().manual_seed(3)
    for step in range(20):
        ids = torch.randint(0, E, (20,), generator=g).to(torch.int32)
        q_ref, q_got = ids.clone().to(dev), ids.clone().to(dev)
        lru_ensure(q_ref, ref["slot_of_id"], ref["id_of_slot"], ref["usage"], ref["step"], q_ref, ref["src"],
                   ref["dst"], ref["num"], id_base=E)
        ensure_copy(q_got, got["slot_of_id"], got["id_of_slot"], got["usage"], got["step"], got["src"], got["dst"],
                    got["num"], E, None, sync)
        torch.cuda.synchronize()
        assert torch.equal(q_ref, q_got)
        for k in ("slot_of_id", "id_of_slot", "usage", "step"):
            assert torch.equal(ref[k], got[k])
