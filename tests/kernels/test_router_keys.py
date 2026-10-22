"""FREETOKEN_ROUTER_KEYS: the router's one-reduction top-k returns the experts, order and weight bits of the
three-reduction loop, over rows full of equal values (bf16 router logits), signed zeros, infinities, NaN, rows with
fewer finite values than picks, and widths that take 4 warps or are not powers of two."""

import pytest
import torch

from freetoken.kernel.triton.moe_router import fused_topk_softmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None,
                                reason="needs a ROCm GPU (the bits checked are ROCm's)")


def _rows(kind: str, m: int, n: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(m, n, generator=gen)
    if kind == "coarse":  # few distinct values: ties everywhere, among the winners too
        x = (x * 4).round() / 4
    elif kind == "zeros":  # signed zeros at the top
        x = -x.abs()
        x[:, ::3] = 0.0
        x[:, 1::3] = -0.0
    elif kind == "inf":
        x[:, ::7] = -float("inf")
        x[:, 5] = float("inf") if m > 1 else x[:, 5]
    elif kind == "nan":
        x[:, ::5] = float("nan")
    elif kind == "sparse":  # fewer finite values than picks: the rest of the picks among the -inf lanes
        x[:, 3:] = -float("inf")
        x[0, :] = -float("inf")  # and a row of -inf only
    return x


@pytest.mark.parametrize("kind", ["normal", "coarse", "zeros", "inf", "nan", "sparse"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m,n,k", [(1, 512, 10), (3, 512, 10), (4, 512, 10), (16, 512, 10), (61, 512, 10),
                                   (5, 256, 8), (9, 128, 4), (2, 64, 6), (300, 512, 10),
                                   (3, 768, 8), (2, 1024, 10), (5, 384, 8), (4, 160, 6)])
@pytest.mark.parametrize("renormalize", [True, False])
def test_router_keys_bit_exact(kind, dtype, m, n, k, renormalize):
    gen = torch.Generator().manual_seed(m * 1000 + n + k)
    x = _rows(kind, m, n, gen).to(dtype).cuda()
    limit = torch.tensor([max(1, m - 1)], dtype=torch.int64, device="cuda")
    for lim in (None, limit):
        w0, i0 = fused_topk_softmax(x, k, renormalize, lim, keys=False)
        w1, i1 = fused_topk_softmax(x, k, renormalize, lim, keys=True)
        assert torch.equal(i0, i1)
        assert torch.equal(w0.view(torch.int32), w1.view(torch.int32))


def test_router_keys_strided_rows():
    gen = torch.Generator().manual_seed(7)
    base = torch.randn(6, 1024, generator=gen).to(torch.bfloat16).cuda()
    x = base[:, :512]  # a row stride of 1024
    w0, i0 = fused_topk_softmax(x, 10, True, keys=False)
    w1, i1 = fused_topk_softmax(x, 10, True, keys=True)
    assert torch.equal(i0, i1)
    assert torch.equal(w0.view(torch.int32), w1.view(torch.int32))
