"""Check the INT8 GEMV against a dequantized reference, eager and under CUDA-graph replay, on one GPU."""
import torch
from freetoken.kernel.triton.int8_gemv import _CONFIGS, gemv_int8
from freetoken.layers.quantization.int8_weight_only import quantize_rows

torch.manual_seed(0)
for (N, K) in list(_CONFIGS) + [(320, 10240), (1000, 2560)]:
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    q, s = quantize_rows(w)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    deq = q.float() * s[:, None]
    got = gemv_int8(x, q, s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = gemv_int8(x, q, s)
    x.copy_(torch.randn_like(x)); g.replay(); torch.cuda.synchronize()
    e_kernel = ((out.float() - x.float() @ deq.T).norm() / (x.float() @ deq.T).norm()).item()
    e_quant = ((x.float() @ deq.T - x.float() @ w.float().T).norm() / (x.float() @ w.float().T).norm()).item()
    print(f"{N}x{K}: kernel vs dequant {e_kernel:.1e} (bf16 rounding), int8 vs bf16 weights {e_quant:.1e}")
