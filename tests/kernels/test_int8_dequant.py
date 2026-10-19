"""FREETOKEN_INT8_PREFILL_DEQUANT: the Triton dequantization must give torch.mul's bits (int8 * fp32 row scale, rounded
once to bf16)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("shape", [(704, 2560), (9270, 2560), (336, 10240), (10240, 336), (37, 4099), (1, 7),
                                   (70000, 64)])
def test_triton_dequant_matches_torch_mul(shape):
    from freetoken.kernel.triton.int8_dequant import dequant_int8_rows

    N, K = shape
    g = torch.Generator(device="cuda").manual_seed(N * 31 + K)
    q = torch.randint(-127, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
    # row scales over several decades, so products land on many bf16 exponents and rounding ties
    scale = torch.rand(N, device="cuda", generator=g) * 10.0 ** torch.randint(-6, 1, (N,), device="cuda", generator=g)
    ref = torch.empty(N, K, dtype=torch.bfloat16, device="cuda")
    torch.mul(q, scale[:, None], out=ref)
    out = torch.empty_like(ref)
    dequant_int8_rows(q, scale, out)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
