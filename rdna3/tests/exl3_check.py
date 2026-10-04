"""Check the EXL3 Triton kernels against the torch reference codec, on one GPU.

dequant: bit-exact with exl3_codec.decode_tiles for every bitrate and codebook.
gemv: per-route products against the decoded reference (fp32 accumulation order only).
"""
import torch

from freetoken.kernel.triton.exl3 import exl3_dequant, exl3_gemv
from freetoken.layers.quantization import exl3_codec as ex

torch.manual_seed(0)
dev = "cuda"


def random_trellis(*lead, K):
    return ex.pack_trellis(torch.randint(0, 1 << K, (*lead, 256)), K)


bad = 0
for K in range(1, 9):
    for cb in (ex.CB_MUL1, ex.CB_MCG, ex.CB_3INST):
        t = random_trellis(12, 20, K=K)
        ref = ex.decode_tiles(t, cb)
        got = exl3_dequant(t.to(dev), cb).cpu()
        same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
        bad += not same
        if not same:
            diff = (got.float() - ref.float()).abs()
            print(f"dequant K={K} cb={cb}: MISMATCH {int((diff > 0).sum())} of {diff.numel()}, max {diff.max():.3g}")
print(f"dequant: {'all bit-exact' if bad == 0 else f'{bad} mismatching cases'}")

for K, (kin, n, halves) in ((3, (2560, 768, 2)), (3, (384, 2560, 1)), (4, (2560, 512, 2)), (5, (256, 2560, 1)), (6, (512, 256, 2))):
    S, R = 6, 10
    bank = random_trellis(S, kin // 16, n // 16, K=K)
    slots = torch.randint(0, S, (R,), dtype=torch.int32)
    a = torch.randn(R, halves, kin)
    hn = n // halves

    def reference(decode):
        deq = [decode(bank[s]).double() for s in range(S)]
        ref = torch.empty(R, n, dtype=torch.float64)
        for r in range(R):
            w = deq[int(slots[r])]
            for hf in range(halves):
                ref[r, hf * hn:(hf + 1) * hn] = a[r, hf].double() @ w[:, hf * hn:(hf + 1) * hn]
        return ref

    ref = reference(lambda t: ex.decode_tiles(t, ex.CB_MUL1))
    for split in (1, 2):
        got = exl3_gemv(a.to(dev), bank.to(dev), slots.to(dev), ex.CB_MUL1, split_k=split).sum(0).cpu().double()
        err = ((got - ref).norm() / ref.norm()).item()
        ok = err < 1e-5
        bad += not ok
        print(f"gemv K={K} kin={kin} n={n} halves={halves} split_k={split}: rel err {err:.2e} {'ok' if ok else 'FAIL'}")

# n-gram table rows: random rings, row scales and head biases
from freetoken.kernel.triton.exl3 import exl3_ngram_dequant  # noqa: E402

for K in (4, 5, 6):
    N, dim, heads = 333, 160, 16
    packed = torch.randint(-32768, 32767, (N, 1 + dim * K // 16), dtype=torch.int16)
    packed[:, 0] = (torch.rand(N) * 0.05 + 0.01).half().view(torch.int16)
    bias = (torch.randn(heads, dim) * 0.01).half()
    ref = ex.decode_ngram_rows(packed, K, bias[torch.arange(N) % heads])
    got = exl3_ngram_dequant(packed.to(dev), K, bias.to(dev), torch.empty(N, dim, device=dev, dtype=torch.float32)).cpu()
    err = ((got - ref).abs().max() / ref.abs().max()).item()
    ok = err < 1e-6
    bad += not ok
    print(f"ngram K={K}: max rel err {err:.1e} {'ok' if ok else 'FAIL'}")

print("PASS" if bad == 0 else f"FAIL ({bad})")
