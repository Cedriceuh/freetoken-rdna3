"""FREETOKEN_MOE_MISS_READS on NVFP4: the decode marlin GEMVs (gate | up with and without the SiLU, down) read the
packed and scale rows of the slots the expert cache fills this step from the layer's pinned host banks and write them
into the slots on the way (tile by tile, or the program's rows first, FREETOKEN_NVFP4_MISS_BULK, then with the global
scales' rows too, FREETOKEN_NVFP4_MISS_GLOBALS). The outputs are those of copying the rows first and reading the
slots, and the slots end up holding the copied rows; at the per-rank shapes, the shipped and tuned tiles, CPU routes
(-1) among the routes, 0 to all slots missing."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _host(t):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    h = alloc_pinned_tensor(*t.shape, dtype=t.dtype)
    h.copy_(t.cpu())
    return h


def _plan(ids, n_miss, E, g, dev):
    used = ids[ids >= 0].unique()
    pick = used[torch.randperm(used.numel(), device=dev, generator=g)[:n_miss]]
    ev = torch.full((ids.numel(),), -7, dtype=torch.int32, device=dev)
    src = torch.zeros((ids.numel(),), dtype=torch.int32, device=dev)
    ev[:pick.numel()] = pick.int()
    src[:pick.numel()] = torch.randperm(E, device=dev, generator=g)[:pick.numel()].int()
    return ev, src, torch.tensor([pick.numel()], dtype=torch.int64, device=dev), pick.numel()


@pytest.mark.parametrize("shape", [(704, 2560), (576, 2560), (2560, 352), (2560, 288)])
@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("bulk, globals_", [(False, False), (True, False), (True, True)])
def test_marlin_reading_misses_from_host_matches_copy_then_gemv(monkeypatch, shape, rows, tuned, bulk, globals_):
    import freetoken.kernel.nvfp4_moe as nm
    from freetoken.kernel.pinned import device_ptr

    monkeypatch.setattr(nm, "NVFP4_DECODE_TUNED", tuned)
    monkeypatch.setattr(nm, "MISS_BULK", bulk)
    N, K = shape
    dev, S, E, top_k = "cuda", 24, 40, 10
    g = torch.Generator(device=dev).manual_seed(N + K + rows + tuned)
    packed = torch.randint(0, 256, (S, N, K // 2), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)
    scale = torch.randint(0, 0x7E, (S, N, K // 16), device=dev, generator=g, dtype=torch.int32).to(torch.uint8)
    glob = torch.rand(S, N, device=dev, generator=g) * 1e-3
    h_packed = _host(torch.randint(0, 256, (E, N, K // 2), device=dev, generator=g, dtype=torch.int32).to(torch.uint8))
    h_scale = _host(torch.randint(0, 0x7E, (E, N, K // 16), device=dev, generator=g, dtype=torch.int32).to(torch.uint8))
    h_glob = _host(torch.rand(E, N, device=dev, generator=g) * 1e-3)
    ptrs = torch.tensor([device_ptr(h_glob), device_ptr(h_packed), 0, device_ptr(h_scale)], dtype=torch.int64,
                        device=dev)
    banks = (1, 3, 0) if globals_ else (1, 3)  # with FREETOKEN_NVFP4_MISS_GLOBALS the global scales' bank too
    route_rows = N == 2560  # the down GEMV: one activation row a route
    for cpu_routes in (False, True):
        ids = torch.stack([torch.randperm(S, device=dev, generator=g)[:top_k] for _ in range(rows)]).int()
        if cpu_routes and nm.SKIP_CPU_ROUTES:
            ids[torch.rand(ids.shape, device=dev, generator=g) < 0.15] = -1
        w = torch.rand(rows, top_k, device=dev, generator=g)
        a = torch.randn(rows * top_k if route_rows else rows, K, device=dev, generator=g).to(torch.bfloat16)
        n_used = ids[ids >= 0].unique().numel()
        for n_miss in sorted({0, 1, n_used // 2, n_used}):
            ev, src, num, k = _plan(ids.view(-1), n_miss, E, g, dev)
            rp, rs, rg = packed.clone(), scale.clone(), glob.clone()
            rp[ev[:k].long()] = h_packed[src[:k].cpu().long()].to(dev)
            rs[ev[:k].long()] = h_scale[src[:k].cpu().long()].to(dev)
            if globals_:
                rg[ev[:k].long()] = h_glob[src[:k].cpu().long()].to(dev)
            want = torch.empty((rows, top_k, N), device=dev, dtype=torch.bfloat16)
            nm.decode_gemm_marlin(a, rp, rs, rg, want, w, ids, True, route_rows)
            gp, gs, gg = packed.clone(), scale.clone(), (glob.clone() if globals_ else rg)
            got = torch.full_like(want, 7.0)
            nm.decode_gemm_marlin(a, gp, gs, gg, got, w, ids, True, route_rows, misses=(ev, src, num, ptrs),
                                  banks=banks)
            assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (n_miss, cpu_routes)
            assert torch.equal(gp, rp) and torch.equal(gs, rs) and torch.equal(gg, rg), (n_miss, cpu_routes)
            if route_rows:
                continue
            act_want = torch.empty((rows * top_k, N // 2), device=dev, dtype=torch.bfloat16)
            nm.decode_gemm_marlin_act(a, rp, rs, rg, act_want, w, ids, False)
            gp, gs, gg = packed.clone(), scale.clone(), (glob.clone() if globals_ else rg)
            act_got = torch.full_like(act_want, 7.0)
            nm.decode_gemm_marlin_act(a, gp, gs, gg, act_got, w, ids, False, misses=(ev, src, num, ptrs),
                                      banks=banks)
            assert torch.equal(act_got.view(torch.int16), act_want.view(torch.int16)), (n_miss, cpu_routes)
            assert torch.equal(gp, rp) and torch.equal(gs, rs) and torch.equal(gg, rg), (n_miss, cpu_routes)

