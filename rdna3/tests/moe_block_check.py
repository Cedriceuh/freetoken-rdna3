"""GPU check: the token-blocked NVFP4 prefill MoE (NVFP4_PREFILL_BLOCK) returns exactly what the one-shot call returns.

Run it with and without FREETOKEN_NVFP4_PREFILL_TUNED=1: chunks of 17000 and 16400 tokens leave last blocks of 616 / 16
tokens at 4096, 8192 and 16384 (4712 / 4112 at 12288), so the blocks take other tiles than the one-shot call."""
import sys

import torch

import freetoken.moe.fused_nvfp4 as F

dev = torch.device("cuda")
torch.manual_seed(0)
E, H, I, K = 32, 512, 128, 10
fp8 = torch.float8_e4m3fn
gate_up = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
gate_up_scale = (torch.rand(E, 2 * I, H // 16, device=dev) * 0.5 + 0.25).to(fp8)
gate_up_global = torch.rand(E, 2 * I, device=dev, dtype=torch.float32) * 0.01
down = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
down_scale = (torch.rand(E, H, I // 16, device=dev) * 0.5 + 0.25).to(fp8)
down_global = torch.rand(E, H, device=dev, dtype=torch.float32) * 0.01
banks = (gate_up, gate_up_scale, gate_up_global, down, down_scale, down_global)

bad = 0
for M in (17000, 16400):
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(M)]).to(torch.int32)
    w = torch.softmax(torch.randn(M, K, device=dev), -1)
    F.NVFP4_PREFILL_BLOCK = 0
    ref = F.fused_experts_nvfp4(x, *banks, w, ids, E)
    for block in (4096, 8192, 12288, 16384):
        F.NVFP4_PREFILL_BLOCK = block
        got = F.fused_experts_nvfp4(x, *banks, w, ids, E)
        same = torch.equal(got, ref)
        bad += not same
        print(f"{M} tokens, {block}-token blocks vs one-shot: "
              f"{'bit-identical' if same else 'DIFFERENT, max abs %.3e' % (got.float() - ref.float()).abs().max()}"
              f" (finite {torch.isfinite(ref).all().item()}, tuned tiles {F.NVFP4_PREFILL_TUNED})")
sys.exit(1 if bad else 0)
