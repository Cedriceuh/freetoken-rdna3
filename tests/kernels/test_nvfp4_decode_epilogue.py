"""FREETOKEN_NVFP4_DECODE_EPILOGUE: the NVFP4 decode down GEMV with the layer's epilogue (the last of a token's route
programs over a row block adds the routes and applies the gate from the HIP router launch) gives the bits of the down
GEMV followed by shared_gate_sum_mul_add, its per-route outputs unchanged, and leaves its counters at zero (graph
replays reuse them): the per-rank shapes, the shipped and tuned tiles, 1-8 rows, misses read from pinned host banks
(the bulk and the tile-by-tile paths), a route the CPU computes (GPU, ROCm)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="needs a ROCm GPU")


def _host(t):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    h = alloc_pinned_tensor(*t.shape, dtype=t.dtype)
    h.copy_(t.cpu())
    return h


@pytest.mark.parametrize("K", [352, 288])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_down_epilogue_matches_down_then_shared_gate_sum_mul_add(monkeypatch, K, rows, tuned, bulk, seed):
    import freetoken.kernel.nvfp4_moe as nm
    from freetoken.kernel.pinned import device_ptr
    from freetoken.kernel.router_shared_exact import router_and_shared_act_exact
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_sum_mul_add

    monkeypatch.setattr(nm, "NVFP4_DECODE_TUNED", tuned)
    monkeypatch.setattr(nm, "MISS_BULK", bulk)
    N, dev, S, E, top_k = 2560, "cuda", 64, 64, 10
    g = torch.Generator(device=dev).manual_seed(K + 10 * rows + 100 * seed + 1000 * tuned + 2000 * bulk)
    packed = torch.randint(0, 256, (S, N, K // 2), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)
    scale = torch.randint(0, 0x7E, (S, N, K // 16), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)
    glob = torch.rand(S, N, device=dev, generator=g) * 1e-3
    h_packed = _host(torch.randint(0, 256, (E, N, K // 2), device=dev, generator=g, dtype=torch.int32).to(torch.uint8))
    h_scale = _host(torch.randint(0, 0x7E, (E, N, K // 16), device=dev, generator=g, dtype=torch.int32).to(torch.uint8))
    h_glob = _host(torch.rand(E, N, device=dev, generator=g) * 1e-3)
    ptrs = torch.tensor([device_ptr(h_glob), device_ptr(h_packed), 0, device_ptr(h_scale)], dtype=torch.int64,
                        device=dev)
    banks = (1, 3, 0) if bulk else (1, 3)
    ids = torch.stack([torch.randperm(S, device=dev, generator=g)[:top_k] for _ in range(rows)]).int()
    if seed == 2:
        ids[0, 4] = -1  # a route the CPU computes: zeros, still counted
    if seed == 1 and rows > 1:
        ids[1, 3] = ids[0, 2]  # one expert routed by two rows
    w = torch.rand(rows, top_k, device=dev, generator=g)
    a = (torch.randn(rows * top_k, K, device=dev, generator=g) * (8.0 if seed else 1.0)).to(torch.bfloat16)
    x = torch.randn(rows, N, device=dev, generator=g).to(torch.bfloat16)
    shared = torch.randn(rows, N, device=dev, generator=g).to(torch.bfloat16)
    gate_w = (torch.randn(N, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    router_w = (torch.randn(512, N, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    q = torch.randint(-127, 128, (704, N), generator=g, device=dev, dtype=torch.int8)
    _, _, gate = router_and_shared_act_exact(x, router_w, q, torch.rand(704, generator=g, device=dev) * 0.01, False,
                                             gate_w)
    used = ids[ids >= 0].unique()
    n_miss = (0, 1, used.numel())[seed]
    ev = torch.full((rows * top_k,), -7, dtype=torch.int32, device=dev)
    src = torch.zeros((rows * top_k,), dtype=torch.int32, device=dev)
    pick = used[torch.randperm(used.numel(), device=dev, generator=g)[:n_miss]]
    ev[:n_miss] = pick.int()
    src[:n_miss] = torch.randperm(E, device=dev, generator=g)[:n_miss].int()
    plan = (ev, src, torch.tensor([n_miss], dtype=torch.int64, device=dev), ptrs)
    c_ref = torch.empty((rows, top_k, N), device=dev, dtype=torch.bfloat16)
    rp, rs, rg = packed.clone(), scale.clone(), glob.clone()
    assert not nm.decode_gemm_marlin(a, rp, rs, rg, c_ref, w, ids, True, True, misses=plan, banks=banks)
    want = shared_gate_sum_mul_add(c_ref, shared, x, gate_w)
    for _ in range(2):  # the second call reuses the counters the first left
        c = torch.empty_like(c_ref)
        out = torch.full((rows, N), 7.0, device=dev, dtype=torch.bfloat16)
        gp, gs, gg = packed.clone(), scale.clone(), glob.clone()
        assert nm.decode_gemm_marlin(a, gp, gs, gg, c, w, ids, True, True, misses=plan, banks=banks,
                                     epilogue=(shared, gate, out))
        assert torch.equal(c.view(torch.int16), c_ref.view(torch.int16))
        assert torch.equal(out.view(torch.int16), want.view(torch.int16))
        assert torch.equal(gp, rp) and torch.equal(gs, rs) and torch.equal(gg, rg)
    assert int(nm._EPILOGUE_COUNTERS[c.device.index].abs().sum()) == 0
