"""The host all-reduce region's two-rank kernels on two GPUs (ROCm): FREETOKEN_HOST_ALLGATHER gives
all_gather_into_tensor's result for several dtypes and shapes, interleaved with all-reduces; FREETOKEN_HOST_AR_HC_COMBINE
gives the bits of the plain host all-reduce then hc_combine_rmsnorm (the reduced tensor, the combined streams in place,
the normed streams) at 1-16 rows and both norm weight shapes, eagerly and replayed in a CUDA graph."""

from __future__ import annotations

import os
import socket

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None or torch.cuda.device_count() < 2,
    reason="needs two ROCm GPUs",
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gather(rank, ar, dev, dist):
    ok = True
    for it in range(10):
        g = torch.Generator(device=dev).manual_seed(it * 2 + rank)
        for shape, dt in (((3, 37), torch.int64), ((4, 16), torch.float32), ((5, 2560), torch.bfloat16)):
            x = (torch.randint(-1000, 1000, shape, generator=g, device=dev) if not dt.is_floating_point
                 else torch.randn(shape, generator=g, device=dev)).to(dt)
            got = ar.gather(x)
            parts = [torch.empty_like(x.cpu()) for _ in range(2)]
            dist.all_gather(parts, x.cpu())
            ok &= torch.equal(got.view(torch.uint8), torch.cat(parts).to(dev).view(torch.uint8))
            y = torch.randn(4, 2560, generator=g, device=dev).to(torch.bfloat16)
            ref = y.float().cpu()
            dist.all_reduce(ref)
            ar(y)
            ok &= torch.equal(y, ref.to(torch.bfloat16).to(dev))
    return ok


def _combine(rank, ar, dev):
    from freetoken.kernel.triton.hc import hc_combine_rmsnorm

    HD, ok = 2560, True
    for it in range(8):
        for rows in (1, 2, 3, 4, 8, 12, 16):
            g = torch.Generator(device=dev).manual_seed(it * 100 + rows * 10 + rank)
            gs = torch.Generator(device=dev).manual_seed(it * 100 + rows)  # the same on both ranks
            x = (torch.randn(rows, HD, generator=g, device=dev) * (1 + it % 5)).to(torch.bfloat16)
            res = (torch.randn(rows, 4 * HD, generator=gs, device=dev) * (0.5 + it % 3)).to(torch.bfloat16)
            prev = (torch.randn(rows, 336, generator=gs, device=dev) * 3).to(torch.bfloat16)
            inj = prev[:, 320:324]
            w = (torch.randn(HD if it % 2 else 4 * HD, generator=gs, device=dev) * 0.3).to(torch.bfloat16)
            x_ref, res_ref = x.clone(), res.clone()
            ar(x_ref)
            comb_ref, rn_ref = hc_combine_rmsnorm(res_ref, x_ref, inj, w, 1e-6, 4, inplace=True)
            x_got, res_got = x.clone(), res.clone()
            rn = ar.hc_combine(x_got, res_got, inj, w, 1e-6)
            ok &= torch.equal(x_got.view(torch.int16), x_ref.view(torch.int16))
            ok &= torch.equal(res_got.view(torch.int16), comb_ref.view(torch.int16))
            ok &= torch.equal(rn.view(torch.int16), rn_ref.view(torch.int16))
    # replayed in a graph with new inputs
    rows = 3
    x = torch.zeros(rows, HD, dtype=torch.bfloat16, device=dev)
    res = torch.zeros(rows, 4 * HD, dtype=torch.bfloat16, device=dev)
    prev = torch.zeros(rows, 336, dtype=torch.bfloat16, device=dev)
    w = torch.ones(HD, dtype=torch.bfloat16, device=dev)
    out = [None]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        ar.hc_combine(x, res, prev[:, 320:324], w, 1e-6)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=s):
            out[0] = ar.hc_combine(x, res, prev[:, 320:324], w, 1e-6)
    return ok, (x, res, prev, w, out, graph)


def _run(rank, port, results):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    torch.cuda.set_device(rank)
    from freetoken.kernel.host_allreduce import HostAllReduce
    from freetoken.kernel.triton.hc import hc_combine_rmsnorm

    ar = HostAllReduce(rank, 2, dist.group.WORLD)
    dev = torch.device("cuda", rank)
    ok = _gather(rank, ar, dev, dist)
    ok_c, (x, res, prev, w, out, graph) = _combine(rank, ar, dev)
    ok &= ok_c
    for it in range(5):
        g = torch.Generator(device=dev).manual_seed(500 + it * 2 + rank)
        gs = torch.Generator(device=dev).manual_seed(500 + it)
        xin = torch.randn(x.shape, generator=g, device=dev).to(torch.bfloat16)
        rin = torch.randn(res.shape, generator=gs, device=dev).to(torch.bfloat16)
        pin = (torch.randn(prev.shape, generator=gs, device=dev) * 3).to(torch.bfloat16)
        x.copy_(xin)
        res.copy_(rin)
        prev.copy_(pin)
        x_ref, res_ref = xin.clone(), rin.clone()
        ar(x_ref)
        comb_ref, rn_ref = hc_combine_rmsnorm(res_ref, x_ref, pin[:, 320:324], w, 1e-6, 4, inplace=True)
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        ok &= torch.equal(x, x_ref) and torch.equal(res, comb_ref) and torch.equal(out[0], rn_ref)
    results[rank] = bool(ok)
    ar.close()
    dist.destroy_process_group()


def test_host_allgather_and_hc_combine_two_ranks():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    results = ctx.Manager().dict()
    port = _free_port()
    procs = [ctx.Process(target=_run, args=(r, port, results)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    assert results.get(0) and results.get(1), dict(results)
