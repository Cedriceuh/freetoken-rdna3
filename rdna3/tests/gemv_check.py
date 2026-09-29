"""Check the batch-1 Triton GEMV path against F.linear (eager and under CUDA-graph capture) on one GPU."""
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from freetoken.kernel.triton.dense_gemv import _BF16_CONFIGS
from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel

k = TorchLinearKernel.__new__(TorchLinearKernel)
torch.manual_seed(0)
for (N, K) in _BF16_CONFIGS:
    layer = SimpleNamespace(weight=torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02, bias=None)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x.float(), layer.weight.float())
    got = k.apply(layer, x)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = k.apply(layer, x)
    x.copy_(torch.randn_like(x)); g.replay(); torch.cuda.synchronize()
    ref2 = F.linear(x.float(), layer.weight.float())
    e1 = ((got.float() - ref).norm() / ref.norm()).item()
    e2 = ((out.float() - ref2).norm() / ref2.norm()).item()
    print(f"{N}x{K}: eager rel err {e1:.1e}, graph-replay rel err {e2:.1e}, shape {tuple(got.shape)}")
