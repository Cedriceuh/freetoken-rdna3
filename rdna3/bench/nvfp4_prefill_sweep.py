"""Tile sweep of the NVFP4 prefill MoE GEMMs (moe/fused_nvfp4.py) on RDNA3, Qwen3.8-Flash shapes.

The shipped _prefill_config was picked by an offline sweep for MiniMax-M2 shapes (upstream, NVIDIA). This times the
two GEMMs of one prefill MoE block separately (gate_up: N = 2 * inter, K = hidden; down: N = hidden, K = inter) for
each (BLOCK_M, BLOCK_N, BLOCK_KB, num_warps, num_stages), on 512 experts / top-10 / hidden 2560 banks that are larger
than the Infinity Cache, and checks each result against the shipped config. A different BLOCK_KB reorders the fp32
sums, so the comparison is a relative error; the tiles kept for FREETOKEN_NVFP4_PREFILL_TUNED change only M / N tiles,
warps and stages, which keeps them bit-exact.

  python rdna3/bench/nvfp4_prefill_sweep.py [--inter 352] [--tokens 4096] [--quick]

--inter: this rank's expert intermediate (352 on the XTX, 288 on the XT at FREETOKEN_TP_SPLIT=0.55).
"""
from __future__ import annotations

import argparse
import time

import torch

from freetoken.kernel.triton.moe_align import moe_align_block_size
from freetoken.moe import fused_nvfp4 as fm


def timed(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", type=int, default=352)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="a small grid (smoke test)")
    ap.add_argument("--only", action="append", default=[],
                    help="time just these configs, 'gate_up|down:BM,BN,BKB,warps,stages' (repeatable), no descent")
    a = ap.parse_args()
    E, H, I, M, K = a.experts, a.hidden, a.inter, a.tokens, a.topk
    dev = "cuda"
    fp8 = torch.float8_e4m3fn
    g = torch.Generator(device=dev).manual_seed(0)

    def bank(n: int, k: int):
        packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g)
        scale = torch.randint(0x30, 0x40, (E, n, k // 16), dtype=torch.uint8, device=dev, generator=g).view(fp8)
        glob = torch.full((E, n), 0.01, dtype=torch.float16, device=dev)
        return packed, scale, glob

    gu = bank(2 * I, H)
    dn = bank(H, I)
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev, generator=g)
    topk_ids = torch.rand(M, E, device=dev, generator=g).topk(K, dim=1).indices.to(torch.int32).contiguous()
    topk_w = torch.softmax(torch.randn(M, K, device=dev, generator=g), dim=1)
    tw = topk_w.reshape(-1).contiguous()
    act = torch.randn(M * K, I, dtype=torch.bfloat16, device=dev, generator=g)
    print(f"shapes: tokens {M}, experts {E}, top-{K}, hidden {H}, inter {I}; banks "
          f"{(sum(t.numel() * t.element_size() for t in gu + dn)) / 2**20:.0f} MiB")

    def run_gemm(which: str, cfg: dict, align):
        sorted_ids, expert_ids, ntpp = align
        if which == "gate_up":
            out = torch.empty((M, K, 2 * I), device=dev, dtype=torch.bfloat16)
            fm._prefill_gemm(x, *gu, out, tw, sorted_ids, expert_ids, ntpp, M * K, K, False, cfg)
        else:
            out = torch.empty((M, K, H), device=dev, dtype=torch.bfloat16)
            fm._prefill_gemm(act, *dn, out, tw, sorted_ids, expert_ids, ntpp, M * K, 1, True, cfg)
        return out

    base = fm._prefill_config(M)
    aligns = {}
    for bm in (16, 32, 64, 128):
        aligns[bm] = moe_align_block_size(topk_ids, bm, E)
    ref = {w: run_gemm(w, base, aligns[base["BLOCK_SIZE_M"]]).float() for w in ("gate_up", "down")}
    t_base = {w: timed(lambda w=w: run_gemm(w, base, aligns[base["BLOCK_SIZE_M"]]), a.iters) for w in ("gate_up", "down")}
    print(f"shipped config {base}: gate_up {t_base['gate_up']:.2f} ms, down {t_base['down']:.2f} ms", flush=True)

    best = {"gate_up": (t_base["gate_up"], base), "down": (t_base["down"], base)}
    rows = []
    t_start = time.time()
    seen = set()

    def try_cfg(w: str, bm, bn, bkb, warps, stages):
        cfg = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_KB=bkb, GROUP_SIZE_M=8 if bm > 16 else 1,
                   num_warps=warps, num_stages=stages)
        key = (w, bm, bn, bkb, warps, stages)
        if key in seen:
            return
        seen.add(key)
        try:
            out = run_gemm(w, cfg, aligns[bm]).float()
            err = ((out - ref[w]).norm() / ref[w].norm()).item()
            t = timed(lambda: run_gemm(w, cfg, aligns[bm]), a.iters)
        except Exception as exc:  # noqa: BLE001 -- an invalid tile (LDS overflow, bad divisor): skip it
            rows.append((w, cfg, None, None, str(exc)[:60]))
            print(f"  {w:7s} {cfg}: skipped ({str(exc)[:60]})", flush=True)
            return
        rows.append((w, cfg, t, err, ""))
        print(f"  {w:7s} {t:7.2f} ms ({t_base[w] / t:4.2f}x) err {err:.1e} {cfg}  [{time.time() - t_start:.0f}s]",
              flush=True)
        if err < 1e-2 and t < best[w][0]:
            best[w] = (t, cfg)

    axes = {"BLOCK_SIZE_M": (16, 32, 64, 128), "BLOCK_SIZE_N": (32, 64, 128), "BLOCK_SIZE_KB": (32, 64, 128),
            "num_warps": (4, 8), "num_stages": (1, 2, 4)}
    order = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_KB", "num_warps", "num_stages")
    if a.quick:
        axes = {k: v[:2] for k, v in axes.items()}
    for spec in a.only:
        w, vals = spec.split(":")
        try_cfg(w, *(int(v) for v in vals.split(",")))
    for w in (("gate_up", "down") if not a.only else ()):
        # coordinate descent from the shipped config: one axis at a time, keep the best, two passes
        cur = {k: base[k] for k in order}
        for _ in range(2):
            for axis in order:
                for v in axes[axis]:
                    trial = {**cur, axis: v}
                    try_cfg(w, *(trial[k] for k in order))
                bt, bcfg = best[w]
                cur = {k: bcfg[k] for k in order}
    for w in ("gate_up", "down"):
        ok = sorted((r for r in rows if r[0] == w and r[2] is not None and r[3] < 1e-2), key=lambda r: r[2])
        print(f"\n{w}: top 5 of {len(ok)} valid configs (shipped {t_base[w]:.2f} ms)")
        for _, cfg, t, err, _ in ok[:5]:
            print(f"  {t:7.2f} ms ({t_base[w] / t:4.2f}x)  err {err:.1e}  {cfg}")
        bad = [r for r in rows if r[0] == w and (r[2] is None or r[3] >= 1e-2)]
        if bad:
            print(f"  {len(bad)} configs skipped (invalid or err >= 1e-2), e.g. {bad[0][1]} {bad[0][4] or bad[0][3]}")
    print(f"\nsweep took {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
