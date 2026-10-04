"""Decode MoE step, EXL3 chain (rotate-in, gate|up GEMV, mid, down GEMV, rotate-out sum) vs the NVFP4 marlin decode,
48 layers in one CUDA graph like a decode step, one GPU. Per-rank shapes of Qwen3.8-Flash-Next (hidden 2560, top-10).

Usage: exl3_moe_bench.py [I ...]   (this rank's intermediate slice, default 384 256)
"""
import sys

import torch

from freetoken.layers.quantization import exl3_codec as ex
from freetoken.moe.fused_exl3 import fused_experts_exl3
from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

dev = "cuda"
S, H, TOP_K, LAYERS, K = 512, 2560, 10, 48, 3


def timed(fn, iters=20):
    fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters  # ms per graph


print(torch.cuda.get_device_name())
for inter in [int(x) for x in sys.argv[1:]] or [384, 256]:
    x = torch.randn(1, H, device=dev).to(torch.bfloat16)
    ids = [torch.randint(0, S, (1, TOP_K), device=dev, dtype=torch.int32) for _ in range(LAYERS)]
    w = torch.softmax(torch.randn(1, TOP_K, device=dev), -1)
    gu = torch.randint(-32768, 32767, (S, H // 16, 2 * inter // 16, 16 * K), device=dev, dtype=torch.int16)
    gus = (torch.rand(S, 2, H, device=dev) * 0.02).half()
    guv = torch.rand(S, 2 * inter, device=dev).half()
    dn = torch.randint(-32768, 32767, (S, inter // 16, H // 16, 16 * K), device=dev, dtype=torch.int16)
    dns = (torch.rand(S, inter, device=dev) * 0.02).half()
    dnv = torch.rand(S, H, device=dev).half()

    def exl3_step():
        for layer in range(LAYERS):
            fused_experts_exl3(x, gu, gus, guv, dn, dns, dnv, w, ids[layer], codebook=ex.CB_MUL1, is_prefill=False)

    t_exl3 = timed(exl3_step)
    del gu, dn
    p_gu = torch.randint(0, 255, (S, 2 * inter, H // 2), device=dev, dtype=torch.uint8)
    s_gu = torch.rand(S, 2 * inter, H // 16, device=dev).to(torch.float8_e4m3fn)
    g_gu = torch.rand(S, 2 * inter, device=dev).half()
    p_dn = torch.randint(0, 255, (S, H, inter // 2), device=dev, dtype=torch.uint8)
    s_dn = torch.rand(S, H, inter // 16, device=dev).to(torch.float8_e4m3fn)
    g_dn = torch.rand(S, H, device=dev).half()
    ids64 = [i.long() for i in ids]

    def nvfp4_step():
        for layer in range(LAYERS):
            fused_experts_decode_nvfp4_marlin(x, p_gu, s_gu, g_gu, p_dn, s_dn, g_dn, w, ids64[layer], "silu", False)

    t_nv = timed(nvfp4_step)
    print(f"I={inter}: 48-layer MoE decode, EXL3 K={K} {t_exl3:.2f} ms, NVFP4 {t_nv:.2f} ms")
    del p_gu, p_dn, s_gu, s_dn
