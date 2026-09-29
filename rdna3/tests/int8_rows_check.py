"""GPU check of the multi-row int8 GEMV (several decoding requests): for every tuned (N, K) shape, M rows at once must give
each row exactly what a batch of one gives (torch.equal, so -0 == +0), and the time of M rows vs M single calls inside
a CUDA graph.

  python rdna3/tests/int8_rows_check.py [--rows 2,4]
"""
from __future__ import annotations

import argparse

import torch

from freetoken.kernel.triton import int8_gemv as ig


def graph_us(fn, reps: int = 20) -> float:
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            fn()
    for _ in range(3):
        g.replay()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="2,4")
    a = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    shapes = sorted(set(ig._CONFIGS) | {(9270, 2560), (7210, 2560), (2560, 3456), (2560, 2688)})
    ok_all = True
    for N, K in shapes:
        w = torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
        sc = torch.rand(N, device="cuda", generator=g) * 1e-3
        for M in (int(r) for r in a.rows.split(",")):
            x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", generator=g)
            together = ig.gemv_int8(x, w, sc)
            alone = torch.cat([ig.gemv_int8(x[r:r + 1], w, sc) for r in range(M)])
            same = torch.equal(together, alone)
            ok_all &= same
            t_m = graph_us(lambda: ig.gemv_int8(x, w, sc))
            t_1 = graph_us(lambda: [ig.gemv_int8(x[r:r + 1], w, sc) for r in range(M)])
            cfg = ig._CONFIGS.get((N, K)) or ig._default_config(N, K)
            print(f"({N:6d}, {K:5d}) cfg {cfg} rows {M}: identical={same}  {t_m:8.1f} us together vs {t_1:8.1f} us "
                  f"one by one ({t_1 / t_m:.2f}x)", flush=True)
    print("ALL OK" if ok_all else "FAILURES")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
