"""Decode (M=1) dense projection micro-benchmark on one RDNA3 GPU: bf16 (torch F.linear) vs 8-bit weight-only.

Shapes and per-token call counts are the per-rank TP=2 decode GEMMs of Qwen3.8-Flash-Next. Every measurement cycles
over enough weight copies to overflow the 96 MB Infinity Cache, so the numbers are DRAM-bound like real decode.
Prints per-shape microseconds and the projected per-token dense time of each variant.

Run (one GPU):
  docker run --rm --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0 \
    -v <repo>:/src:ro -e PYTHONPATH=/src/python freetoken-rdna3:latest python /src/rdna3/bench/dense_gemv_bench.py
"""
from __future__ import annotations

import math
import os
import sys

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_u8_to_f32

# (name, N out, K in, calls per token per rank)
SHAPES = [
    ("gdn.in_proj", 8240, 2560, 36),
    ("o_proj+gdn.out_proj", 2560, 3072, 48),
    ("attn.qkv", 6656, 2560, 12),
    ("hc.down_inject", 336, 10240, 96),
    ("hc.up", 10240, 320, 97),
    ("router", 512, 2560, 48),
    ("shared.gate_up+indexer", 640, 2560, 60),
    ("shared.down", 2560, 320, 48),
    ("ple.key", 10240, 2560, 1),
    ("ple.value", 2560, 2560, 1),
    ("lm_head", 124160, 2560, 1),
]
CACHE_BYTES = 384 << 20  # cycle at least this many weight bytes per measurement
MAX_SPLIT = 64

KIND_BF16, KIND_INT8, KIND_FP8 = 0, 1, 2


def _configs():
    out = []
    for bn in (16, 32, 64):
        for bk in (128, 256, 512):
            for split in (1, 2, 4, 8, 16, 32, 64):
                for w in (2, 4, 8):
                    if bn * bk // (32 * w) > 256:  # keep per-thread register tiles sane
                        continue
                    out.append(triton.Config({"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT": split}, num_warps=w))
    return out


@triton.autotune(configs=_configs(), key=["N", "K", "KIND"])
@triton.jit
def gemv_splitk(a_ptr, w_ptr, part_ptr, N, K, stride_wn, stride_pk,
                SPLIT, KIND: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    k_per = tl.cdiv(tl.cdiv(K, SPLIT), BLOCK_K) * BLOCK_K
    k0 = pid_k * k_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for kk in range(0, k_per, BLOCK_K):
        offs_k = k0 + kk + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(a_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]
        m = n_mask[:, None] & k_mask[None, :]
        if KIND == 1:
            w = tl.load(ptrs, mask=m, other=0).to(tl.float32)
        elif KIND == 2:
            w = e4m3_u8_to_f32(tl.load(ptrs, mask=m, other=0))
        else:
            w = tl.load(ptrs, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(w * a[None, :], axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n, acc, mask=n_mask)


@triton.jit
def reduce_scale(part_ptr, scale_ptr, out_ptr, N, stride_pk, SPLIT: tl.constexpr, HAS_SCALE: tl.constexpr,
                 BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + k * stride_pk + offs, mask=mask, other=0.0)
    if HAS_SCALE:
        acc = acc * tl.load(scale_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


def ours(a, w, scale, kind, part, out):
    N, K = w.shape
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["SPLIT"])  # noqa: E731
    gemv_splitk[grid](a, w, part, N, K, w.stride(0), part.stride(0), KIND=kind)
    split = gemv_splitk.best_config.kwargs["SPLIT"]
    reduce_scale[(triton.cdiv(N, 512),)](part, scale, out, N, part.stride(0), SPLIT=split,
                                          HAS_SCALE=kind != KIND_BF16, BLOCK=512)
    return out


def quant_int8(w):
    s = w.float().abs().amax(dim=1).clamp_min(1e-8) / 127.0
    return (w.float() / s[:, None]).round().clamp(-127, 127).to(torch.int8), s


def quant_fp8(w):
    s = w.float().abs().amax(dim=1).clamp_min(1e-8) / 448.0
    return (w.float() / s[:, None]).to(torch.float8_e4m3fn), s


def timed(fn, copies, iters):
    """GPU time per call from CUDA-graph replay (decode runs under graphs: no host launch cost)."""
    for i in range(3):  # warm-up; also runs the Triton autotune outside the capture
        fn(copies[i % len(copies)])
    torch.cuda.synchronize()
    calls = max(len(copies), 24)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(copies[i % len(copies)])
    g.replay()
    torch.cuda.synchronize()
    reps = max(3, iters // calls)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / (reps * calls)  # us


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"# {props.name}", flush=True)
    from freetoken.kernel.triton.fp8_pertensor_linear import _gemv as main_fp8_gemv

    totals = {}
    print(f"{'shape':24s} {'N':>7s} {'K':>6s} {'calls':>5s} | " + " ".join(f"{v:>11s}" for v in
          ("bf16_torch", "bf16_ours", "int8_ours", "fp8_ours", "fp8_main")) + "   (us per call; err int8/fp8)", flush=True)
    skip = set(filter(None, os.environ.get("SHAPES_SKIP", "").split(",")))
    for name, N, K, calls in SHAPES:
        if name in skip:
            continue
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        a = torch.randn(K, device=dev, dtype=torch.bfloat16)
        ref = (w.float() @ a.float())
        wi, si = quant_int8(w)
        wf, sf = quant_fp8(w)
        # a tensor bigger than the cycle budget cannot sit in the 96 MB Infinity Cache anyway
        n_bf16 = 1 if N * K * 2 >= CACHE_BYTES else min(64, math.ceil(CACHE_BYTES / (N * K * 2)))
        n_8 = 1 if N * K >= CACHE_BYTES else min(128, math.ceil(CACHE_BYTES / (N * K)))
        cb = [w] + [w.clone() for _ in range(n_bf16 - 1)]
        part = torch.empty(MAX_SPLIT, N, device=dev, dtype=torch.float32)
        out = torch.empty(N, device=dev, dtype=torch.bfloat16)
        row = {}
        iters = max(20, min(2000, int(4e9 / (N * K * 2))))
        row["bf16_torch"] = timed(lambda x: torch.nn.functional.linear(a, x), cb, iters)
        row["bf16_ours"] = timed(lambda x: ours(a, x, None, KIND_BF16, part, out), cb, iters)
        del cb
        ci = [wi] + [wi.clone() for _ in range(n_8 - 1)]
        row["int8_ours"] = timed(lambda x: ours(a, x, si, KIND_INT8, part, out), ci, iters)
        err_i = ((ours(a, wi, si, KIND_INT8, part, out).float() - ref).norm() / ref.norm()).item()
        del ci
        cf = [wf.view(torch.uint8)] + [wf.view(torch.uint8).clone() for _ in range(n_8 - 1)]
        row["fp8_ours"] = timed(lambda x: ours(a, x, sf, KIND_FP8, part, out), cf, iters)
        err_f = ((ours(a, wf.view(torch.uint8), sf, KIND_FP8, part, out).float() - ref).norm() / ref.norm()).item()
        try:
            row["fp8_main"] = timed(lambda x: main_fp8_gemv(a, x.view(torch.float8_e4m3fn), sf, torch.bfloat16), cf, iters)
        except Exception as e:  # noqa: BLE001
            print(f"  fp8_main failed: {e!r}"[:200], file=sys.stderr)
            row["fp8_main"] = float("nan")
        del cf
        for k, v in row.items():
            totals[k] = totals.get(k, 0.0) + v * calls
        print(f"{name:24s} {N:7d} {K:6d} {calls:5d} | " + " ".join(f"{row[v]:11.1f}" for v in
              ("bf16_torch", "bf16_ours", "int8_ours", "fp8_ours", "fp8_main")) +
              f"   ({err_i:.2e} / {err_f:.2e})", flush=True)
        del w, wi, wf, part, out
        torch.cuda.empty_cache()
    print("\nbest configs (N, K, kind) -> BLOCK_N, BLOCK_K, SPLIT, num_warps:")
    for key, cfg in sorted(gemv_splitk.cache.items(), key=lambda kv: str(kv[0])):
        kw = cfg.kwargs
        print(f"  {key[:3]}: ({kw['BLOCK_N']}, {kw['BLOCK_K']}, {kw['SPLIT']}, {cfg.num_warps})")
    bytes_bf16 = sum(N * K * 2 * c for _, N, K, c in SHAPES)
    print(f"\nprojected dense time per token per rank ({bytes_bf16 / 1e9:.2f} GB of bf16 weights):")
    for k, v in totals.items():
        print(f"  {k:11s} {v / 1000:7.2f} ms")


if __name__ == "__main__":
    main()
