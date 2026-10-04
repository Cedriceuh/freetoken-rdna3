"""Check the EXL3 routed-expert path (fused_experts_exl3: rotations + products) against a dense torch reference built
from exl3_codec.reconstruct, on one GPU: decode (GEMV) and prefill (grouped GEMM), the banks packed by the MoE kernel
for a whole intermediate and for a 384/256 two-rank split (the slices' outputs must add up), prefill token blocks
(1024-token blocks bit-identical to the whole 3000-token chunk), and a CUDA-graph replay."""
import torch
import torch.nn.functional as F

from freetoken.layers.quantization import exl3_codec as ex
from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.layers.quantization.moe.exl3 import TritonExl3MoEKernel
from freetoken.layers.quantization.scheme import exl3_scheme
from freetoken.moe.fused_exl3 import fused_experts_exl3

torch.manual_seed(0)
dev = "cuda"
E, H, I, K, TOP_K = 12, 2560, 640, 3, 10
cb = ex.CB_MUL1


def trellis(k, n):
    return ex.pack_trellis(torch.randint(0, 1 << K, (E, k // 16, n // 16, 256)), K)


def scales(n, mag):
    return ((torch.rand(E, n) + 0.5) * mag * torch.randn(E, n).sign()).half()


pieces = {
    "gate_trellis": trellis(H, I), "gate_suh": scales(H, 0.015), "gate_svh": scales(I, 0.05),
    "up_trellis": trellis(H, I), "up_suh": scales(H, 0.015), "up_svh": scales(I, 0.05),
    "down_trellis": trellis(I, H), "down_suh": scales(I, 0.015), "down_svh": scales(H, 0.05),
}
W = {p: torch.stack([ex.reconstruct(pieces[f"{p}_trellis"][e], pieces[f"{p}_suh"][e], pieces[f"{p}_svh"][e], cb,
                                    dtype=torch.float64) for e in range(E)]) for p in ("gate", "up", "down")}


def reference(x, topk_w, topk_ids):
    x = x.double()
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(topk_ids[t, k])
            h = F.silu(x[t] @ W["gate"][e]) * (x[t] @ W["up"][e])
            out[t] += topk_w[t, k].double() * (h @ W["down"][e])
    return out


def banks(tp_rank, tp_size, split):
    import os

    os.environ["FREETOKEN_TP_SPLIT"] = split
    from freetoken.distributed import split as sp
    sp.tp_shares.cache_clear() if hasattr(sp.tp_shares, "cache_clear") else None
    sp._parse_shares.cache_clear()
    sp.set_intermediate_unit(128)
    cfg = MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=TOP_K, tp_rank=tp_rank, tp_size=tp_size,
                    scheme=exl3_scheme(K, cb), strategy="offload")
    kern = TritonExl3MoEKernel()
    assert kern.unusable_reason(cfg) is None, kern.unusable_reason(cfg)
    out = {role: torch.empty((E, *spec.shape), dtype=spec.dtype) for role, spec in kern.layout(cfg).items()}
    kern.pack(pieces, cfg, out)
    return {role: t.to(dev) for role, t in out.items()}, cfg


def run(b, x, w, ids, prefill):
    return fused_experts_exl3(x, b["gate_up"], b["gate_up_suh"], b["gate_up_svh"], b["down"], b["down_suh"],
                              b["down_svh"], w, ids, codebook=cb, is_prefill=prefill, num_experts=E)


bad = 0
full, _ = banks(0, 1, "")
rank_banks = [banks(r, 2, "0.6")[0] for r in range(2)]
print("TP 0.6 slices:", [tuple(b["down"].shape) for b in rank_banks])
for M, prefill in ((1, False), (4, False), (37, True), (300, True)):
    x = torch.randn(M, H).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:TOP_K] for _ in range(M)]).to(torch.int32)
    w = torch.softmax(torch.randn(M, TOP_K), -1)
    ref = reference(x, w, ids)
    xg, wg, ig = x.to(dev), w.to(dev), ids.to(dev)
    got = run(full, xg, wg, ig, prefill).double().cpu()
    tp = sum(run(b, xg, wg, ig, prefill).double().cpu() for b in rank_banks)
    tol = 2e-2 if prefill else 1e-2  # bf16 output; prefill rounds its activations to fp16
    for name, y in (("whole", got), ("tp 384+256", tp)):
        err = ((y - ref).norm() / ref.norm()).item()
        ok = err < tol
        bad += not ok
        print(f"{'prefill' if prefill else 'decode '} M={M:3d} {name:10s}: rel err {err:.2e} {'ok' if ok else 'FAIL'}")

# prefill token blocks: 1024-token blocks (the last one 952 tokens) give the same bits as the whole chunk in one pass
import freetoken.moe.fused_exl3 as fused

M = 3000
x = torch.randn(M, H, device=dev).to(torch.bfloat16)
ids = torch.stack([torch.randperm(E)[:TOP_K] for _ in range(M)]).to(torch.int32).to(dev)
w = torch.softmax(torch.randn(M, TOP_K, device=dev), -1)
blocks = {}
for block in (1024, 0):
    fused.EXL3_PREFILL_BLOCK = block
    blocks[block] = run(full, x, w, ids, True)
fused.EXL3_PREFILL_BLOCK = 4096
same = torch.equal(blocks[1024], blocks[0])
bad += not same
print(f"prefill M={M} in 1024-token blocks vs one pass: {'bit-identical' if same else 'DIFFER (FAIL)'}")

# decode under CUDA-graph replay, new inputs copied into the captured buffers
x = torch.randn(2, H, device=dev).to(torch.bfloat16)
ids = torch.stack([torch.randperm(E)[:TOP_K] for _ in range(2)]).to(torch.int32).to(dev)
w = torch.softmax(torch.randn(2, TOP_K, device=dev), -1)
run(full, x, w, ids, False)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = run(full, x, w, ids, False)
x.copy_(torch.randn_like(x))
ids.copy_(torch.stack([torch.randperm(E)[:TOP_K] for _ in range(2)]).to(torch.int32))
g.replay()
torch.cuda.synchronize()
err = ((y.double().cpu() - reference(x.cpu(), w.cpu(), ids.cpu())).norm() / reference(x.cpu(), w.cpu(), ids.cpu()).norm()).item()
bad += err >= 1e-2
print(f"graph replay: rel err {err:.2e} {'ok' if err < 1e-2 else 'FAIL'}")
print("PASS" if bad == 0 else f"FAIL ({bad})")
