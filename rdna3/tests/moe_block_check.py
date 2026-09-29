"""GPU check: the token-blocked NVFP4 prefill MoE (NVFP4_PREFILL_BLOCK) returns exactly what the one-shot call returns."""
import sys

import torch

import freetoken.moe.fused_nvfp4 as F

dev = torch.device("cuda")
torch.manual_seed(0)
E, H, I, M, K = 32, 512, 128, 9001, 10
fp8 = torch.float8_e4m3fn
gate_up = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev)
gate_up_scale = (torch.rand(E, 2 * I, H // 16, device=dev) * 0.5 + 0.25).to(fp8)
gate_up_global = torch.rand(E, 2 * I, device=dev, dtype=torch.float16) * 0.01
down = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev)
down_scale = (torch.rand(E, H, I // 16, device=dev) * 0.5 + 0.25).to(fp8)
down_global = torch.rand(E, H, device=dev, dtype=torch.float16) * 0.01
x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(M)]).to(torch.int32)
w = torch.softmax(torch.randn(M, K, device=dev), -1)
banks = (gate_up, gate_up_scale, gate_up_global, down, down_scale, down_global)

F.NVFP4_PREFILL_BLOCK = 0
ref = F.fused_experts_nvfp4(x, *banks, w, ids, E)
F.NVFP4_PREFILL_BLOCK = 4096
got = F.fused_experts_nvfp4(x, *banks, w, ids, E)
same = torch.equal(got, ref)
print(f"blocked vs one-shot: {'bit-identical' if same else 'DIFFERENT, max abs %.3e' % (got.float() - ref.float()).abs().max()}"
      f" (finite {torch.isfinite(ref).all().item()})")
sys.exit(0 if same else 1)
