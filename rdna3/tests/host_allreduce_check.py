"""GPU check of the host-memory all-reduce (FREETOKEN_HOST_ALLREDUCE) against RCCL, on both cards.

  python rdna3/tests/host_allreduce_check.py [--iters 2000]

Two processes (rank r on HIP device r). Checks: bit-identical results to dist.all_reduce for bf16 / fp32 at
decode-like and larger sizes, a misaligned view, 96 all-reduces captured in a CUDA graph and replayed; then the time
per call of both paths for a 5 KB message (one decode hidden state). A desynchronized peer traps the kernel after
FREETOKEN_HOST_ALLREDUCE_TIMEOUT_S, so run it under `timeout`.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, iters: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    cpu = dist.new_group(backend="gloo")
    from freetoken.kernel.host_allreduce import HostAllReduce

    ar = HostAllReduce(rank, 2, cpu)
    say = print if rank == 0 else (lambda *a, **k: None)
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    ok_all = True
    # 1. bit-identical to RCCL
    for dtype in (torch.bfloat16, torch.float32):
        for n in (1, 7, 2560, 4 * 2560, 32768, ar.slot_bytes // torch.tensor([], dtype=dtype).element_size()):
            bad = 0
            for _ in range(50):
                x = torch.randn(n, device="cuda", dtype=dtype, generator=gen)
                ref = x.clone()
                dist.all_reduce(ref)
                ar(x)
                bad += int(not torch.equal(x, ref))
            ok_all &= bad == 0
            say(f"{str(dtype):15s} n={n:6d}: {50 - bad}/50 identical to RCCL")
    # 2. misaligned view (element path on this rank only)
    buf = torch.randn(2562, device="cuda", dtype=torch.bfloat16, generator=gen)
    x = buf[1:2561]
    ref = x.clone()
    dist.all_reduce(ref)
    ar(x)
    ok_all &= torch.equal(x, ref)
    say(f"misaligned bf16 view: {'identical' if torch.equal(x, ref) else 'DIFFERENT'}")
    # 3. CUDA graph: 96 all-reduces per replay, like one decode step
    src = torch.randn(96, 2560, device="cuda", dtype=torch.bfloat16, generator=gen)
    work = torch.empty_like(src)
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        work.copy_(src)
        for i in range(96):
            ar(work[i])
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        with torch.cuda.graph(g, stream=s):
            work.copy_(src)
            for i in range(96):
                ar(work[i])
    torch.cuda.synchronize()
    dist.barrier(group=cpu)
    ref = src.clone()
    dist.all_reduce(ref)
    good = 0
    for _ in range(20):
        g.replay()
        torch.cuda.synchronize()
        good += int(torch.equal(work, ref))
    ok_all &= good == 20
    say(f"graph (96 all-reduces per replay): {good}/20 replays identical to RCCL")
    # 4. time per call, 5 KB bf16
    x = torch.randn(2560, device="cuda", dtype=torch.bfloat16, generator=gen)
    for name, fn in (("RCCL", lambda: dist.all_reduce(x)), ("host", lambda: ar(x))):
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        say(f"{name:5s}: {a.elapsed_time(b) * 1000 / iters:6.1f} us per 5 KB all-reduce (eager, back to back)")
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    dist.barrier(group=cpu)
    t0.record()
    for _ in range(50):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    say(f"graph : {t0.elapsed_time(t1) * 1000 / (50 * 96):6.1f} us per all-reduce (96 per replay, incl. the copy)")
    say("ALL OK" if ok_all else "FAILURES (see above)")
    dist.barrier(group=cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    a = ap.parse_args()
    os.environ.setdefault("FREETOKEN_HOST_ALLREDUCE_TIMEOUT_S", "10")
    mp.spawn(worker, args=(a.iters, 29631), nprocs=2, join=True)
