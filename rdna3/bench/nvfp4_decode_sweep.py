"""Tile sweep of the NVFP4 decode expert GEMV (moe/fused_nvfp4.py `_decode_nvfp4_marlin_kernel`) on RDNA3.

The shipped decode tiles were swept on qwen35/qwen3moe shapes (plus a deep-K variant measured on qwen4_exp gate_up);
this times, for Qwen3.8-Flash per rank (gate_up N = 2 * inter, K = 2560; down N = 2560, K = inter), each
(BLOCK_N, BLOCK_KW, num_warps) inside a CUDA graph of 48 launches (one decode step), for 1 / 2 / 4 decoding requests
(top-10 routes each) on slot banks bigger than the Infinity Cache, and checks every output against the shipped tiles
bit for bit (`exact`): only exact configs are candidates, the decode must keep its numbers.

  python rdna3/bench/nvfp4_decode_sweep.py [--inter 352] [--slots 1024] [--rows 1,2,4]
"""
from __future__ import annotations

import argparse
import itertools

import torch
import triton

from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view
from freetoken.moe import fused_nvfp4 as fm

L, K_TOP = 48, 10


def launch(a, packed, scale, glob, c, tw, ids, mul_routed, a_row_is_route, block_n, block_kw, warps):
    M, top_k = ids.shape
    N = packed.shape[1]
    K = packed.shape[2] * 2
    p32 = packed.view(torch.int32)
    sc = e4m3_kernel_view(scale)
    total = M * top_k
    fm._decode_nvfp4_marlin_kernel[(total, triton.cdiv(N, block_n))](
        a, p32, sc, glob, c, tw, ids, fm._e2m1_lut(a.device.index), total, N, K,
        a.stride(0), a.stride(1), p32.stride(0), p32.stride(1), p32.stride(2),
        sc.stride(0), sc.stride(1), sc.stride(2), glob.stride(0), glob.stride(1),
        c.stride(0), c.stride(1), c.stride(2), tw.stride(0), tw.stride(1), ids.stride(0), ids.stride(1),
        BLOCK_SIZE_N=block_n, BLOCK_SIZE_KW=block_kw, TOP_K=top_k, A_ROW_IS_ROUTE=a_row_is_route,
        MUL_ROUTED_WEIGHT=mul_routed, compute_type=fm._tl_dtype(c.dtype), num_warps=warps,
    )


def shipped(K: int) -> tuple[int, int, int]:
    deep = K > fm._DECODE_MARLIN_DEEPK_THRESHOLD
    return ((fm._DECODE_MARLIN_DEEPK_BLOCK_N, fm._DECODE_MARLIN_DEEPK_BLOCK_KW) if deep else
            (fm._DECODE_MARLIN_BLOCK_N, fm._DECODE_MARLIN_BLOCK_KW)) + (fm._DECODE_MARLIN_WARPS,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", type=int, default=352)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--slots", type=int, default=1024)
    ap.add_argument("--rows", default="1,2,4")
    a = ap.parse_args()
    S, H, I = a.slots, a.hidden, a.inter
    g = torch.Generator(device="cuda").manual_seed(0)
    fp8 = torch.float8_e4m3fn

    def bank(n: int, k: int):
        return (torch.randint(0, 256, (S, n, k // 2), dtype=torch.uint8, device="cuda", generator=g),
                torch.randint(0x30, 0x40, (S, n, k // 16), dtype=torch.uint8, device="cuda", generator=g).view(fp8),
                torch.full((S, n), 0.01, dtype=torch.float16, device="cuda"))

    gu, dn = bank(2 * I, H), bank(H, I)
    print(f"inter {I}, slots {S}, banks {sum(t.numel() * t.element_size() for t in gu + dn) / 2**20:.0f} MiB", flush=True)
    for M in (int(r) for r in a.rows.split(",")):
        ids = [torch.stack([torch.randperm(S, device="cuda", generator=g)[:K_TOP] for _ in range(M)]).to(torch.int32)
               for _ in range(L)]
        tw = [torch.softmax(torch.randn(M, K_TOP, device="cuda", generator=g), 1) for _ in range(L)]
        x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda", generator=g)
        act = torch.randn(M * K_TOP, I, dtype=torch.bfloat16, device="cuda", generator=g)
        for name, bk, ain, n_out, mul, row_route in (("gate_up", gu, x, 2 * I, False, False),
                                                     ("down", dn, act, H, True, True)):
            K = bk[0].shape[2] * 2
            outs = [torch.empty((M, K_TOP, n_out), dtype=torch.bfloat16, device="cuda") for _ in range(L)]

            def step(cfg):
                for l in range(L):
                    launch(ain, *bk, outs[l], tw[l], ids[l], mul, row_route, *cfg)

            base = shipped(K)
            step(base)
            torch.cuda.synchronize()
            ref = [o.clone() for o in outs]
            results = []
            grid = itertools.product((4, 8, 16, 32), (16, 32, 64, 128), (2, 4, 8))
            for cfg in sorted(set(grid) | {base}, key=lambda c: c != base):
                try:
                    step(cfg)
                    torch.cuda.synchronize()
                    exact = all(torch.equal(o, r) for o, r in zip(outs, ref))
                    graph = torch.cuda.CUDAGraph()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
                        step(cfg)
                    for _ in range(5):
                        graph.replay()
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for _ in range(50):
                        graph.replay()
                    t1.record()
                    torch.cuda.synchronize()
                    us = t0.elapsed_time(t1) * 1000 / (50 * L)
                except Exception as exc:  # noqa: BLE001 -- invalid tile for this shape
                    continue
                results.append((us, cfg, exact))
            t_base = next(us for us, cfg, _ in results if cfg == base)
            best = sorted((r for r in results if r[2]), key=lambda r: r[0])[:4]
            fast_any = sorted(results, key=lambda r: r[0])[:2]
            print(f"rows {M} {name:7s} (N {n_out}, K {K}): shipped {base} {t_base:6.2f} us/launch; best exact: "
                  + ", ".join(f"{c} {us:.2f} ({t_base / us:.2f}x)" for us, c, _ in best)
                  + " | fastest any: " + ", ".join(f"{c} {us:.2f} exact={e}" for us, c, e in fast_any), flush=True)


if __name__ == "__main__":
    main()
