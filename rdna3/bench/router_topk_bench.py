"""Micro-bench of the MoE router softmax + top-k kernel (kernel/triton/moe_router.py) on RDNA3.

fused_topk_softmax runs one warp per row for N <= 512 experts ("swept on H100/B200"); Qwen3.8-Flash has N = 512,
top-10, so every decode step runs 48 single-program launches of 10 sequential argmax passes on 32 lanes. This times
num_warps 1 / 2 / 4 / 8 (and the rows per program) inside a CUDA graph of 48 back-to-back launches, like a decode step,
and checks the chosen experts are identical and the weights equal to within fp32 rounding.

  python rdna3/bench/router_topk_bench.py [--rows 1] [--experts 512] [--topk 10]
"""
from __future__ import annotations

import argparse

import torch
import triton

from freetoken.kernel.triton import moe_router as mr


def launch(logits, w, i, block_m, warps, topk):
    M, N = logits.shape
    mr._router_triton_kernel[(triton.cdiv(M, block_m),)](
        logits, w, i, None, M, logits.stride(0), logits.stride(1), w.stride(0), w.stride(1), i.stride(0), i.stride(1),
        N=N, K=topk, BLOCK_M=block_m, BLOCK_N=triton.next_power_of_2(N), BLOCK_K=triton.next_power_of_2(topk),
        RENORMALIZE=True, HAS_TOKEN_LIMIT=False, launch_pdl=False, num_warps=warps,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--layers", type=int, default=48)
    a = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    L, M, N, K = a.layers, a.rows, a.experts, a.topk
    logits = [torch.randn(M, N, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(L)]
    ref_w, ref_i = zip(*(mr.fused_topk_softmax(x, K, True) for x in logits))
    shipped_bm = max(1, min(4, 256 // triton.next_power_of_2(N)))
    shipped_w = 1 if triton.next_power_of_2(N) <= 512 else 4
    print(f"rows {M}, experts {N}, top-{K}; shipped BLOCK_M={shipped_bm} num_warps={shipped_w}")
    for block_m in sorted({shipped_bm, 1, 2, 4}):
        for warps in (1, 2, 4, 8):
            ws = [torch.empty(M, K, device="cuda", dtype=torch.float32) for _ in range(L)]
            idx = [torch.empty(M, K, device="cuda", dtype=torch.int32) for _ in range(L)]
            for x, w, i in zip(logits, ws, idx):
                launch(x, w, i, block_m, warps, K)
            torch.cuda.synchronize()
            same_i = all(torch.equal(i, r) for i, r in zip(idx, ref_i))
            max_dw = max((w - r).abs().max().item() for w, r in zip(ws, ref_w))
            graph = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
                for x, w, i in zip(logits, ws, idx):
                    launch(x, w, i, block_m, warps, K)
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(200):
                graph.replay()
            t1.record()
            torch.cuda.synchronize()
            us = t0.elapsed_time(t1) * 1000 / (200 * L)
            tag = " (shipped)" if (block_m, warps) == (shipped_bm, shipped_w) else ""
            print(f"  BLOCK_M={block_m} num_warps={warps}: {us:6.2f} us per launch, experts identical={same_i}, "
                  f"max weight diff {max_dw:.1e}{tag}")


if __name__ == "__main__":
    main()
