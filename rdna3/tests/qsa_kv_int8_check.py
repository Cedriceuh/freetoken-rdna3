"""GPU check of the int8 QSA K/V path: quantizing store round trip, then the sparse attend kernel on int8 K/V
against the same kernel on bf16 K/V (and an fp32 torch reference), on the Qwen3.8-Flash-Next per-rank geometry
(24 q heads / 2 kv heads at TP=1 is 12 per kv head; TP=2 keeps 12 q heads on 1 kv head), head_dim 256.
Random K/V plus a realistic variant with outlier channels (a few dims 10x larger, as RMS-normed keys often have).
"""
from __future__ import annotations

import sys

import torch

from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention
from freetoken.kernel.triton.qsa.kv_int8 import store_kv_int8

dev = torch.device("cuda")
torch.manual_seed(0)
PAGE, D, KVH, HQ = 64, 256, 1, 12


def make_kv(tokens: int, outliers: bool) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(tokens, KVH, D, device=dev)
    v = torch.randn(tokens, KVH, D, device=dev) * 0.5
    if outliers:
        k[..., :4] *= 10.0
        v[..., 7] *= 20.0
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def reference(q, k, v, idx):
    # q [T, HQ, D]; k/v [tokens, KVH, D]; idx [T, topk] token ids
    kk = k[idx.long()].float()  # [T, topk, KVH, D]
    vv = v[idx.long()].float()
    kk = kk.repeat_interleave(HQ // KVH, dim=2)
    vv = vv.repeat_interleave(HQ // KVH, dim=2)
    s = torch.einsum("thd,tkhd->thk", q.float(), kk) * D ** -0.5
    return torch.einsum("thk,tkhd->thd", s.softmax(-1), vv)


def run(rows: int, topk: int, tokens: int, outliers: bool) -> bool:
    pages = tokens // PAGE
    k, v = make_kv(tokens, outliers)
    loc = torch.randperm(tokens, device=dev).to(torch.int32)  # scattered physical slots
    kc = torch.zeros(tokens, KVH, D, dtype=torch.bfloat16, device=dev)
    vc = torch.zeros_like(kc)
    kc[loc.long()] = k
    vc[loc.long()] = v
    kq = torch.zeros(tokens, KVH, D, dtype=torch.int8, device=dev)
    vq = torch.zeros_like(kq)
    ks = torch.zeros(tokens, KVH, dtype=torch.float32, device=dev)
    vs = torch.zeros_like(ks)
    store_kv_int8(k, v, loc, kq, vq, ks, vs)
    k_rt = (kq.float() * ks[..., None])[loc.long()]
    rt_err = ((k_rt - k.float()).norm() / k.float().norm()).item()
    exp_scale = k.float().abs().amax(-1) / 127.0
    scale_ok = torch.allclose(ks[loc.long()], exp_scale, rtol=1e-6, atol=0)

    # identity block table: logical page p -> physical page p (loc already scrambles the token slots)
    block_table = torch.arange(pages, dtype=torch.int32, device=dev)[None, :].contiguous()
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=dev)
    q = torch.randn(rows, HQ, D, device=dev).to(torch.bfloat16)
    logical = torch.stack([torch.randperm(tokens, device=dev)[:topk] for _ in range(rows)]).to(torch.int32)
    phys = loc.long()[logical.long()]  # physical slot of each selected logical token
    out_bf16 = qsa_sparse_paged_attention(q, kc.view(pages, PAGE, KVH, D), vc.view(pages, PAGE, KVH, D),
                                          phys.to(torch.int32), block_table, token_to_req)
    out_int8 = qsa_sparse_paged_attention(q, kq.view(pages, PAGE, KVH, D), vq.view(pages, PAGE, KVH, D),
                                          phys.to(torch.int32), block_table, token_to_req,
                                          k_scale=ks.view(pages, PAGE, KVH), v_scale=vs.view(pages, PAGE, KVH))
    ref = reference(q, kc, vc, phys)
    err_bf16 = ((out_bf16.float() - ref).norm() / ref.norm()).item()
    err_int8 = ((out_int8.float() - ref).norm() / ref.norm()).item()
    # int8 per-row absmax: ~0.8 % round trip on gaussian rows, ~1 % on the attention output; outlier channels inflate
    # the row's scale (reported, judged by the model-quality runs rather than gated here)
    limit = 0.10 if outliers else 0.02
    ok = scale_ok and rt_err < limit and err_int8 < limit and torch.isfinite(out_int8).all().item()
    print(f"rows={rows:4d} topk={topk:4d} tokens={tokens:6d} outliers={outliers!s:5s}: store rel err {rt_err:.4f}, "
          f"attend rel err bf16 {err_bf16:.4f} int8 {err_int8:.4f} {'ok' if ok else 'FAIL'}")
    return ok


def bench(tokens: int = 65536, topk: int = 2048) -> None:
    pages = tokens // PAGE
    k, v = make_kv(tokens, False)
    kc, vc = k.contiguous(), v.contiguous()
    kq = torch.zeros(tokens, KVH, D, dtype=torch.int8, device=dev)
    vq = torch.zeros_like(kq)
    ks = torch.zeros(tokens, KVH, dtype=torch.float32, device=dev)
    vs = torch.zeros_like(ks)
    loc = torch.arange(tokens, dtype=torch.int32, device=dev)
    store_kv_int8(k, v, loc, kq, vq, ks, vs)
    block_table = torch.arange(pages, dtype=torch.int32, device=dev)[None, :].contiguous()
    token_to_req = torch.zeros(1, dtype=torch.int32, device=dev)
    q = torch.randn(1, HQ, D, device=dev).to(torch.bfloat16)
    idx = torch.randperm(tokens, device=dev)[:topk].to(torch.int32)[None, :].contiguous()
    runs = {
        "bf16": lambda: qsa_sparse_paged_attention(q, kc.view(pages, PAGE, KVH, D), vc.view(pages, PAGE, KVH, D), idx,
                                                   block_table, token_to_req),
        "int8": lambda: qsa_sparse_paged_attention(q, kq.view(pages, PAGE, KVH, D), vq.view(pages, PAGE, KVH, D), idx,
                                                   block_table, token_to_req, k_scale=ks.view(pages, PAGE, KVH),
                                                   v_scale=vs.view(pages, PAGE, KVH)),
        "store1": lambda: store_kv_int8(k[:1], v[:1], loc[:1], kq, vq, ks, vs),
    }
    for name, fn in runs.items():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        print(f"decode {name:6s}: {e0.elapsed_time(e1) * 1000 / 200:6.1f} us/call (graph replay)")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0))
    good = True
    for rows, topk, tokens in ((1, 2048, 65536), (4, 512, 4096), (256, 2048, 16384)):
        for outliers in (False, True):
            good &= run(rows, topk, tokens, outliers)
    bench()
    print("ALL OK" if good else "SOME CHECKS FAILED")
    sys.exit(0 if good else 1)
