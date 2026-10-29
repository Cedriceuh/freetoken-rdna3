"""GPU check of the copy-engine all-reduce of large messages (FREETOKEN_DMA_ALLREDUCE) against RCCL, on both cards.

  HIP_VISIBLE_DEVICES=0,1 python rdna3/tests/dma_allreduce_check.py

Two processes (rank r on HIP device r). Checks: bit-identical results to dist.all_reduce for bf16 / fp32 from the
smallest message taken to the slot size, sizes that end in a partial piece, two messages in flight at once (the
prefill blocks' async all-reduces), many in a row; then the time per call of both paths by size, and ten GEMMs with
four 40 MiB all-reduces beside them on another stream (the prefill overlap).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    cpu = dist.new_group(backend="gloo")
    from freetoken.kernel.dma_allreduce import DmaAllReduce

    ar = DmaAllReduce(rank, 2, cpu)
    say = print if rank == 0 else (lambda *a, **k: None)
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    ok_all = True
    # 1. bit-identical to RCCL, whole and partial pieces
    for dtype in (torch.bfloat16, torch.float32):
        el = torch.tensor([], dtype=dtype).element_size()
        for nbytes in (ar.min_bytes, ar.min_bytes + 4 * el, 5 * ar.piece + 2 * el, 20 << 20, ar.slot_bytes):
            bad = 0
            for _ in range(5):
                x = torch.randn(nbytes // el, device="cuda", dtype=dtype, generator=gen) * (rank + 1)
                ref = x.clone()
                dist.all_reduce(ref)
                assert ar.supports(x)
                ar(x)
                bad += int(not torch.equal(x, ref))
            ok_all &= bad == 0
            say(f"{str(dtype):15s} {nbytes / 2**20:7.2f} MiB: {5 - bad}/5 identical to RCCL")
    # 1b. inference tensors (the server's activations), updated in place by the issuing thread
    with torch.inference_mode():
        x = torch.randn(ar.min_bytes // 2, device="cuda", dtype=torch.bfloat16, generator=gen)
        ref = x.clone()
        dist.all_reduce(ref)
        ar(x)
        same = torch.equal(x, ref)
    ok_all &= same
    say(f"inference tensor: {'identical' if same else 'DIFFERENT'}")
    # 2. two in flight, then thirty in a row (the slot reused while the other rank may still read it)
    xs = [torch.randn(8192 * 2560, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in range(2)]
    refs = [x.clone() for x in xs]
    for r in refs:
        dist.all_reduce(r)
    works = [ar.all_reduce_async(x) for x in xs]
    for w in works:
        w.wait()
    same = all(torch.equal(x, r) for x, r in zip(xs, refs))
    ok_all &= same
    say(f"two 40 MiB in flight: {'identical' if same else 'DIFFERENT'}")
    bad = 0
    for i in range(30):
        x = torch.randn((i % 7 + 1) * 1024 * 1024, device="cuda", dtype=torch.bfloat16, generator=gen)
        ref = x.clone()
        dist.all_reduce(ref)
        ar(x)
        bad += int(not torch.equal(x, ref))
    ok_all &= bad == 0
    say(f"30 in a row, 2-14 MiB: {30 - bad}/30 identical")
    # 3. time per call
    for rows in (1024, 2048, 4096, 8192, 16384):
        x = torch.randn(rows * 2560, device="cuda", dtype=torch.bfloat16, generator=gen)
        if not ar.supports(x):
            continue
        t = {}
        for name, fn in (("RCCL", lambda: dist.all_reduce(x)), ("dma", lambda: ar(x))):
            ts = []
            for _ in range(12):
                dist.barrier(group=cpu)
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
            t[name] = sorted(ts)[6]
        say(f"{rows * 5120 / 2**20:5.1f} MiB: RCCL {t['RCCL']:.2f} ms, dma {t['dma']:.2f} ms")
    # 4. beside GEMMs on the compute stream
    A = torch.randn(8192, 2560, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(9270, 2560, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(8192 * 2560, device="cuda", dtype=torch.bfloat16, generator=gen)
    side = torch.cuda.Stream()
    for _ in range(3):  # BLAS warm-up
        torch.nn.functional.linear(A, W)
    for name in ("alone", "RCCL", "dma"):
        dist.barrier(group=cpu)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        if name != "alone":
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(4):
                    dist.all_reduce(x) if name == "RCCL" else ar(x)
        for _ in range(10):
            torch.nn.functional.linear(A, W)
        torch.cuda.current_stream().wait_stream(side)
        b.record()
        torch.cuda.synchronize()
        say(f"10 GEMMs {'+ 4 x 40 MiB ' + name if name != 'alone' else 'alone'}: {a.elapsed_time(b):.1f} ms")
    say("ALL OK" if ok_all else "FAILURES (see above)")
    dist.barrier(group=cpu)
    ar.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(29632,), nprocs=2, join=True)
