"""GPU check of the ROCm batch memcpy used by the prefill hit-D2D split (--moe-prefill-hit-d2d).

Copies the miss runs of one fake expert layer (pinned host -> device) the way _prefetch_split does, with the
hipMemcpyBatchAsync binding and with the hipMemcpyAsync loop, checks the bytes, and times both against the
full-layer copy the legacy prefill path makes. Needs one free GPU (~1 GiB):

    python rdna3/tests/batch_memcpy_check.py [--hit 0.33] [--experts 512] [--iters 20]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from freetoken.kernel import batch_memcpy as bm


def runs(miss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
    return miss[run_starts], np.diff(np.concatenate((run_starts, [miss.size])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hit", type=float, default=0.33)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    E = a.experts
    feats = [640 * 1024, 320 * 1024]  # two big banks per expert row, like gate_up / down packed
    srcs = [torch.randint(0, 255, (E, f), dtype=torch.uint8).pin_memory() for f in feats]
    dsts = [torch.zeros((E, f), dtype=torch.uint8, device="cuda") for f in feats]
    rng = np.random.default_rng(0)
    hit = rng.random(E) < a.hit
    miss = np.nonzero(~hit)[0]
    starts, lengths = runs(miss)
    dst, src, nbytes = [], [], []
    for s, d, f in zip(srcs, dsts, feats):
        dst.extend(d.data_ptr() + starts * f)
        src.extend(s.data_ptr() + starts * f)
        nbytes.extend(lengths * f)
    dst_t, src_t, n_t = (torch.tensor(x, dtype=torch.int64) for x in (dst, src, nbytes))
    miss_bytes = int(n_t.sum())
    print(f"{E} experts, hit {hit.mean():.2f}, {miss.size} misses in {starts.size} runs -> {len(dst)} entries, "
          f"{miss_bytes / 2**20:.0f} MiB of {sum(E * f for f in feats) / 2**20:.0f} MiB")
    module = bm._jit_batch_memcpy_rocm_module()
    stream = torch.cuda.Stream()

    def timed(fn) -> float:
        fn()
        stream.synchronize()
        t = time.perf_counter()
        for _ in range(a.iters):
            fn()
        stream.synchronize()
        return (time.perf_counter() - t) / a.iters

    def full() -> None:
        with torch.cuda.stream(stream):
            for s, d in zip(srcs, dsts):
                d.copy_(s, non_blocking=True)

    host_ms = {}

    def split(fn, key):
        def run() -> None:
            t = time.perf_counter()
            fn(dst_t, src_t, n_t, stream.cuda_stream)
            host_ms[key] = (time.perf_counter() - t) * 1e3
        return run

    t_full = timed(full)
    print(f"full layer copy_: {t_full * 1e3:.2f} ms ({sum(E * f for f in feats) / t_full / 1e9:.1f} GB/s)")
    for key, fn in (("batch", module.batch_memcpy), ("loop", module.batch_memcpy_loop)):
        for d in dsts:
            d.zero_()
        try:
            t = timed(split(fn, key))
        except Exception as exc:  # noqa: BLE001
            print(f"{key}: FAILED {exc}")
            continue
        ok = all(torch.equal(d[torch.from_numpy(miss).cuda()], s[torch.from_numpy(miss)].cuda())
                 for s, d in zip(srcs, dsts))
        untouched = all(int(d[torch.from_numpy(np.nonzero(hit)[0]).cuda()].count_nonzero()) == 0 for d in dsts)
        print(f"{key}: {t * 1e3:.2f} ms ({miss_bytes / t / 1e9:.1f} GB/s, {t / t_full:.2f}x full layer), "
              f"host enqueue {host_ms[key]:.2f} ms, miss rows exact={ok}, hit rows untouched={untouched}")
    print(f"loader picks: {bm.load_batch_memcpy()}")


if __name__ == "__main__":
    main()
