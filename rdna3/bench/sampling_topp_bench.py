"""ROCm top-k / top-p draws: threshold from a sort (_sorted_threshold) + the multi-CTA inverse-CDF draw (_draw),
against float64 and against upstream's fused exact kernels run with one CTA per row.

With one CTA per row, the fused top-p kernel scatter-adds the whole row into 256 bins through atomics before it
narrows anything: ~10 ms per call on a peaked 248k-token row, ~60 ms on a flat one. The sort path costs the same
on any row.

1. Kept set: count(x >= thr) against the float64 definition (top-k: the k-th largest, ties kept; top-p after
   top-k: the largest t with mass(x >= t) >= p * mass(top-k)) on peaked and flat rows, the engine's batch form
   included (top-k-off rows with k = vocab, greedy rows with k = vocab and p = 1).
2. Same draws: for a fixed seed both paths draw from the same uniform, so they pick the same token unless it lands
   within fp32 rounding of a boundary (the two sum the kept mass in different orders). Only where the fused
   kernels are fast (peaked rows, top-k + top-p).
3. Distribution: 40k draws vs float64 (total variation distance), plus the renormalize API.
4. Speed per call (GPU time with the host kept ahead, and back to back), peaked and flat rows, B = 1 and 4.
"""
from __future__ import annotations

import math
import sys
import time

import torch

from freetoken.kernel.triton import sampling as M

V = 248320
dev = torch.device("cuda")


def logits_peaked(B: int, seed: int) -> torch.Tensor:
    # a real LM row: a long low tail and a few strong candidates
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, V, generator=g)
    for b in range(B):
        x[b, torch.randperm(V, generator=g)[:40]] += torch.linspace(14.0, 8.0, 40)
    return x.to(dev, torch.bfloat16)


def logits_flat(B: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(B, V, generator=g) * 0.5).to(dev, torch.bfloat16)


def ref_keep(row: torch.Tensor, k: int | None, p: float | None) -> torch.Tensor:
    """float64 kept set, the kernels' definitions."""
    x = row.double().cpu()
    keep = torch.ones_like(x, dtype=torch.bool)
    if k is not None and k < V:
        keep = x >= torch.topk(x, k).values[-1]
    if p is not None and p < 1.0:
        kept = torch.where(keep, x, torch.zeros_like(x))
        vals = torch.sort(kept, descending=True).values
        target = p * float(kept.sum())
        j = int(torch.searchsorted(vals.cumsum(0), torch.tensor([target], dtype=torch.float64)).clamp(max=V - 1))
        keep &= x >= vals[j]
    return keep


CASES = [  # name, logits, temps, ks (None = top-k off), ps (None = top-p off)
    ("peaked top-p 0.8 T0.7", lambda: logits_peaked(1, 1), [0.7], None, [0.8]),
    ("peaked top-p 0.95 T1.0", lambda: logits_peaked(1, 2), [1.0], None, [0.95]),
    ("flat top-p 0.9", lambda: logits_flat(1, 3), [1.0], None, [0.9]),
    ("peaked k1000 + p0.95", lambda: logits_peaked(1, 4), [0.6], [1000], [0.95]),
    ("flat k5000", lambda: logits_flat(1, 5), [1.0], [5000], None),
    ("engine batch", lambda: torch.cat([logits_peaked(3, 6), logits_flat(2, 7)]), [0.7, 1.0, 0.6, 0.6, 1.0],
     [V, V, 20, 1000, V], [0.8, 1.0, 0.95, 0.95, 0.9]),
]


def _args(ks, ps):
    tk = torch.tensor(ks, device=dev, dtype=torch.int32) if ks else None
    tp = torch.tensor(ps, device=dev) if ps else None
    return tk, tp


def check_kept_set() -> bool:
    ok = True
    for name, mk, temps, ks, ps in CASES:
        probs = M.softmax(mk().float(), torch.tensor(temps, device=dev))
        tk, tp = _args(ks, ps)
        thr = M._sorted_threshold(probs, tk, tp)
        for r in range(len(temps)):
            ref = ref_keep(probs[r], ks[r] if ks else None, ps[r] if ps else None)
            got = (probs[r] >= thr[r]).cpu()
            differ = int((got != ref).sum())
            # a nucleus of ~200k flat tokens puts float64 and fp32 cumulative sums a few tokens apart at the edge
            good = differ <= max(1, int(ref.sum()) // 20000)
            ok &= good
            print(f"{name:24s} row {r}: kept {int(got.sum()):6d} (float64 {int(ref.sum()):6d}, {differ} differ)"
                  f"  {'ok' if good else 'FAIL'}", flush=True)
    return ok


def fused_draw(probs, tk, tp, seed):
    # upstream's exact kernels (one CTA per row on ROCm), whatever the platform default
    if tp is None:
        return M._topk(probs, tk, True, seed, 0)
    return M._topp(probs, tp, tk, True, seed, 0)


def sorted_draw(probs, tk, tp, seed):
    return M._draw(probs, M._sorted_threshold(probs, tk, tp), seed, 0)


def check_same_draws() -> bool:
    ok = True
    n = 1000
    for name, mk, temps, ks, ps in CASES:
        if "flat" in name or name == "engine batch":
            continue  # fused top-p is ~60 ms per call on flat rows; their kept sets are checked above
        probs = M.softmax(mk().float(), torch.tensor(temps, device=dev))
        tk, tp = _args(ks, ps)
        a = torch.stack([sorted_draw(probs, tk, tp, i) for i in range(n)]).cpu()
        b = torch.stack([fused_draw(probs, tk, tp, i) for i in range(n)]).cpu()
        differ = int((a != b).sum())
        good = differ <= n // 200
        ok &= good
        print(f"{name:24s}: {n} seeded draws, sort path vs fused kernels: {differ} differ  {'ok' if good else 'FAIL'}",
              flush=True)
    return ok


def check_distribution() -> bool:
    ok = True
    n = 40000
    for name, mk, temps, ks, ps in (CASES[0], CASES[3]):
        probs = M.softmax(mk().float(), torch.tensor(temps, device=dev))
        tk, tp = _args(ks, ps)
        keep = ref_keep(probs[0], ks[0] if ks else None, ps[0] if ps else None)
        ref = probs[0].double().cpu() * keep
        ref = ref / ref.sum()
        if tk is None:
            toks = torch.stack([M.top_p_sampling_from_probs(probs, tp) for _ in range(n)])
        else:
            toks = torch.stack([M.top_k_top_p_sampling_from_probs(probs, tk, tp) for _ in range(n)])
        emp = torch.bincount(toks.cpu()[:, 0].long(), minlength=V).double() / n
        d = 0.5 * float((emp - ref).abs().sum())
        outside = float(emp[~keep].sum())
        noise = math.sqrt(int(keep.sum()) / (2 * math.pi * n))
        good = outside == 0.0 and d < 3 * noise + 0.005
        ok &= good
        print(f"draws, {name}: support={int(keep.sum())} tvd={d:.4f} (noise~{noise:.4f}) outside={outside:.5f} "
              f"{'ok' if good else 'FAIL'}", flush=True)
    return ok


def check_renorm() -> bool:
    probs = M.softmax(torch.cat([logits_peaked(1, 8), logits_flat(1, 9)]).float(), torch.tensor([0.7, 1.0], device=dev))
    out_p = M.top_p_renorm_probs(probs, torch.tensor([0.8, 0.9], device=dev))
    out_k = M.top_k_renorm_probs(probs, torch.tensor([20, 5000], device=dev, dtype=torch.int32))
    ok = True
    for r, (pp, kk) in enumerate(((0.8, 20), (0.9, 5000))):
        kp, kk_ref = ref_keep(probs[r], None, pp), ref_keep(probs[r], kk, None)
        dp = int(((out_p[r] > 0).cpu() != kp).sum())
        good = (dp <= max(1, int(kp.sum()) // 20000) and abs(float(out_p[r].sum()) - 1) < 1e-4
                and torch.equal((out_k[r] > 0).cpu(), kk_ref) and abs(float(out_k[r].sum()) - 1) < 1e-4)
        ok &= good
        print(f"renorm row {r}: top-p kept {int((out_p[r] > 0).sum())} (float64 {int(kp.sum())}) sum {float(out_p[r].sum()):.6f};"
              f" top-k kept {int((out_k[r] > 0).sum())} (k {kk}) sum {float(out_k[r].sum()):.6f}  {'ok' if good else 'FAIL'}",
              flush=True)
    return ok


def timeit(fn, reps=30):
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    gpu = []
    for _ in range(reps):
        for _ in range(8):
            a @ a  # keep the GPU busy so the launch is queued before it runs
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        gpu.append(e0.elapsed_time(e1) * 1000)
    gpu.sort()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return gpu[len(gpu) // 2], e0.elapsed_time(e1) * 1000 / reps


def bench() -> None:
    for B in (1, 4):
        for flat in (False, True):
            probs = M.softmax((logits_flat(B, 9) if flat else logits_peaked(B, 9)).float(), torch.full((B,), 0.7, device=dev))
            p = torch.full((B,), 0.9, device=dev)
            k = torch.full((B,), 1000, device=dev, dtype=torch.int32)
            for label, fn, reps in (("top-p sort", lambda: sorted_draw(probs, None, p, None), 30),
                                    ("top-p fused", lambda: fused_draw(probs, None, p, None), 10 if flat else 30),
                                    ("k1000+p sort", lambda: sorted_draw(probs, k, p, None), 30),
                                    ("k1000+p fused", lambda: fused_draw(probs, k, p, None), 30)):
                g, w = timeit(fn, reps)
                print(f"B={B} {'flat  ' if flat else 'peaked'} {label:14s}: gpu median {g:8.1f} us  back-to-back {w:8.1f} us/call",
                      flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print(torch.cuda.get_device_name(0), "sorted threshold:", M._SORTED_THRESHOLD, flush=True)
    good = check_kept_set()
    good &= check_same_draws()
    print(f"[{time.time() - t0:.0f} s] kept sets / same draws done", flush=True)
    good &= check_distribution()
    good &= check_renorm()
    print(f"[{time.time() - t0:.0f} s] distribution / renorm done", flush=True)
    bench()
    print(f"[{time.time() - t0:.0f} s]", "ALL OK" if good else "SOME CHECKS FAILED", flush=True)
    sys.exit(0 if good else 1)
