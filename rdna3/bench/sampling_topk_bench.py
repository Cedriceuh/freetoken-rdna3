"""Top-k-first sampler (kernel/triton/sampling_topk.py) vs the full-vocab path (sample_impl: on ROCm a sorted threshold
+ the multi-CTA draw, see _sorted_threshold in kernel/triton/sampling.py). The "full" label below is that path.

1. Distribution: draw N tokens from fixed logits with both samplers and compare the empirical frequencies
   with the exact top-k -> renormalize -> top-p distribution computed in float64 (total variation distance,
   plus a chi-square style check that no token outside the support is ever drawn).
2. Rank agreement: two fresh module states produce the same token sequence (the TP ranks sample
   independently and must stay in step).
3. Speed: GPU time of one sampling step with the host kept ahead of the GPU (a long matmul is queued
   first, so launch overhead is hidden the way it is behind a decode graph replay), plus plain
   back-to-back wall time per call.
"""
from __future__ import annotations

import importlib
import math
import sys

import torch

from freetoken.engine import sample as S
from freetoken.kernel.triton import sampling_topk as T

V = 248320
dev = torch.device("cuda")


def exact_dist(logits_row: torch.Tensor, temp: float, k: int, p: float | None) -> torch.Tensor:
    x = logits_row.double() / temp
    vals, idx = torch.topk(x, k)
    probs = torch.softmax(vals, -1)
    if p is not None:
        keep = (probs.cumsum(-1) - probs) < p
        probs = probs * keep
        probs = probs / probs.sum()
    out = torch.zeros_like(x)
    out[idx] = probs
    return out


def make_logits(B: int, seed: int, dtype=torch.bfloat16) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(B, V, generator=g) * 2.0
    # a peaked head like a real LM: a few strong candidates over a long tail
    for b in range(B):
        hot = torch.randperm(V, generator=g)[:40]
        base[b, hot] += torch.linspace(9.0, 4.0, 40)
    return base.to(dev, dtype)


def tvd(counts: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    emp = counts.double() / counts.sum()
    outside = float(emp[ref == 0].sum())
    return 0.5 * float((emp - ref).abs().sum()), outside


def draw_many(fn, n: int) -> torch.Tensor:
    toks = [fn() for _ in range(n)]
    return torch.stack(toks).cpu()


def check_distribution() -> bool:
    ok = True
    cases = [  # (B, temps, ks, ps)
        (1, [0.6], [20], [0.95]),
        (1, [1.0], [20], None),
        (1, [0.3], [50], [0.8]),
        (3, [0.6, 1.0, 0.8], [20, 5, 40], [0.95, 1.0, 0.5]),
    ]
    n = 40000
    for ci, (B, temps, ks, ps) in enumerate(cases):
        logits = make_logits(B, 100 + ci)
        t = torch.tensor(temps, device=dev, dtype=torch.float32)
        k = torch.tensor(ks, device=dev, dtype=torch.int32)
        p = torch.tensor(ps, device=dev, dtype=torch.float32) if ps is not None else None
        kmax = max(ks)
        new = draw_many(lambda: T.top_k_top_p_sampling_from_logits(logits, t, k, p, kmax), n)
        old = draw_many(lambda: S.sample_impl(logits.float(), t, k, p), n)
        for b in range(B):
            ref = exact_dist(logits[b].float().cpu(), temps[b], ks[b], None if ps is None else ps[b])
            # expected sampling noise for n draws over the support
            support = int((ref > 0).sum())
            noise = math.sqrt(support / (2 * math.pi * n))
            for name, toks in (("new", new), ("full", old)):
                counts = torch.bincount(toks[:, b].long(), minlength=V)
                d, outside = tvd(counts, ref)
                good = outside == 0.0 and d < 3 * noise + 0.005
                if name == "new":
                    ok &= good
                # full is reported, not gated: it keeps every token tied with the k-th value (as flashinfer does),
                # which this float64 reference (torch.topk) cuts
                verdict = ("ok" if good else "FAIL") if name == "new" else ("exact" if good else "deviates")
                print(f"case {ci} row {b} {name:6s}: support={support:3d} tvd={d:.4f} (noise~{noise:.4f}) "
                      f"outside={outside:.5f} {verdict}")
    return ok


def check_rank_agreement() -> bool:
    logits = make_logits(2, 7)
    t = torch.tensor([0.6, 1.0], device=dev)
    k = torch.tensor([20, 20], device=dev, dtype=torch.int32)
    p = torch.tensor([0.95, 0.9], device=dev)
    seqs = []
    for _ in range(2):
        importlib.reload(T)  # fresh counter, like a second rank process
        seqs.append(torch.stack([T.top_k_top_p_sampling_from_logits(logits, t, k, p, 20) for _ in range(200)]).cpu())
    same = torch.equal(seqs[0], seqs[1])
    print(f"rank agreement over 200 steps: {'ok' if same else 'FAIL'}")
    return same


def bench() -> None:
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    for B in (1, 4):
        logits = make_logits(B, 3)
        t = torch.full((B,), 0.6, device=dev)
        k = torch.full((B,), 20, device=dev, dtype=torch.int32)
        p = torch.full((B,), 0.95, device=dev)
        runs = {
            "full": lambda: S.sample_impl(logits.float(), t, k, p),
            "topk-first": lambda: T.top_k_top_p_sampling_from_logits(logits, t, k, p, 20),
        }
        for name, fn in runs.items():
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            gpu = []
            for _ in range(50):
                for _ in range(8):
                    a @ a  # keep the GPU busy so every sampler launch is queued before it runs
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                fn()
                e1.record()
                torch.cuda.synchronize()
                gpu.append(e0.elapsed_time(e1) * 1000)
            gpu.sort()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(200):
                fn()
            e1.record()
            torch.cuda.synchronize()
            wall = e0.elapsed_time(e1) * 1000 / 200
            print(f"B={B} {name:10s}: gpu median {gpu[len(gpu) // 2]:7.1f} us  p90 {gpu[int(len(gpu) * 0.9)]:7.1f} us"
                  f"  back-to-back {wall:7.1f} us/call")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    good = check_rank_agreement()
    good &= check_distribution()
    bench()
    print("ALL OK" if good else "SOME CHECKS FAILED")
    sys.exit(0 if good else 1)
