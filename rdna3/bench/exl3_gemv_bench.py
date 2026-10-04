"""EXL3 decode GEMV vs the NVFP4 marlin decode GEMV, Qwen3.8-Flash-Next per-rank shapes, one GPU.

Each call reads top_k random experts out of a slot bank larger than the Infinity Cache, as decode does.
Usage: exl3_gemv_bench.py [K ...]   (bits per weight of the EXL3 experts, default 3 4)
"""
import sys

import torch

from freetoken.kernel.triton.exl3 import exl3_gemv
from freetoken.layers.quantization import exl3_codec as ex
from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin

dev = "cuda"
S, TOP_K, ITERS = 512, 10, 200


def timed(fn):
    for _ in range(10):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(ITERS):
            fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    g.replay()
    torch.cuda.synchronize()
    start.record()
    g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / ITERS  # us per call


ks = [int(k) for k in sys.argv[1:]] or [3, 4]
slots = torch.randint(0, S, (TOP_K,), device=dev, dtype=torch.int32)
print(f"{torch.cuda.get_device_name()}  S={S} top_k={TOP_K}")
SHAPES = (("gate_up i=384", 2560, 768, 2), ("down i=384", 384, 2560, 1), ("gate_up i=256", 2560, 512, 2), ("down i=256", 256, 2560, 1))
only = __import__("os").environ.get("EXL3_BENCH_SHAPES")
for name, kin, n, halves in [x for x in SHAPES if not only or x[0].split(" i=")[1] in only.split(",")]:
    # NVFP4 reference: [S, N, K/2] codes, [S, N, K/16] e4m3 scales, [S, N] globals
    packed = torch.randint(0, 255, (S, n, kin // 2), device=dev, dtype=torch.uint8)
    scale = torch.rand(S, n, kin // 16, device=dev).to(torch.float8_e4m3fn)
    glob = torch.rand(S, n, device=dev, dtype=torch.float16)
    a = torch.randn(1 if halves == 2 else TOP_K, kin, device=dev, dtype=torch.bfloat16)
    c = torch.empty(1, TOP_K, n, device=dev, dtype=torch.bfloat16)
    tw = torch.rand(1, TOP_K, device=dev)
    ids = slots.view(1, TOP_K).long()
    t_nv = timed(lambda: _decode_gemm_marlin(a, packed, scale, glob, c, tw, ids, False, halves == 1))
    line = f"{name:14s} nvfp4 {t_nv:6.1f} us ({S and packed[0].numel() + scale[0].numel() + 2 * n} B/expert)"
    del packed, scale
    for K in ks:
        bank = torch.randint(-32768, 32767, (S, kin // 16, n // 16, 16 * K), device=dev, dtype=torch.int16)
        ar = torch.randn(TOP_K, halves, kin, device=dev)
        if True:
            best = None
            for kt in (1, 2, 4, 8):
                for nt in (1, 2, 4):
                    for sk in (1, 2, 4, 8):
                        for w in (2, 4, 8):
                            if (kin // 16) % (kt * sk) or (n // 16 // halves) % nt:
                                continue
                            out = torch.empty(sk, TOP_K, n, device=dev)
                            try:
                                t = timed(lambda: exl3_gemv(ar, bank, slots, ex.CB_MUL1, out=out, kt=kt, nt=nt,
                                                            split_k=sk, num_warps=w))
                            except Exception as e:  # noqa: BLE001
                                print(f"   K={K} kt={kt} nt={nt} sk={sk} w={w}: {type(e).__name__}")
                                continue
                            if best is None or t < best[0]:
                                best = (t, kt, nt, sk, w)
            line += f" | K={K} {best[0]:5.1f} us (kt, nt, split, warps) = {best[1:]}"
        del bank
    print(line)
