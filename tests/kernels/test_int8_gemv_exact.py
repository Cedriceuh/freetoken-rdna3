"""FREETOKEN_INT8_EXACT_GEMV: the HIP int8 GEMVs give the Triton kernels' bits on the model's decode shapes at 1-8 rows
(GPU, ROCm)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="needs a ROCm GPU")


def _triton(fn, *args, **kwargs):
    """``fn`` with the routing to the HIP kernels off: the Triton kernels' result."""
    from freetoken.kernel.triton import int8_gemv as tg

    saved, tg._EXACT_HIP = tg._EXACT_HIP, False
    try:
        return fn(*args, **kwargs)
    finally:
        tg._EXACT_HIP = saved


def _inputs(n, k, rows, seed, row_stride=None):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda", generator=g)
    s = torch.rand(n, device="cuda", generator=g) * 0.02 + 1e-4
    # a wide spread of magnitudes, so that another order of the additions shows in the rounded sums
    x = torch.randn(rows, row_stride or k, device="cuda", generator=g)
    x = (x * torch.exp(torch.randn(rows, row_stride or k, device="cuda", generator=g) * 1.5)).to(torch.bfloat16)
    return w, s, x[:, :k]


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("shape", [(7210, 2560), (9270, 2560), (2560, 2688), (2560, 3456), (2560, 3072), (64615, 2560)])
def test_one_pass_tile_same_bits(shape, rows):
    from freetoken.kernel import int8_gemv_exact as ex
    from freetoken.kernel.triton import int8_gemv as tg

    w, s, x = _inputs(*shape, rows, rows)
    assert tg._config(*shape) == (2, 4096, 1, 4)
    ref = _triton(tg.gemv_int8, x, w, s)
    assert torch.equal(ex.gemv_int8_exact(x, w, s).view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_qkv_tile_same_bits(rows):
    from freetoken.kernel import int8_gemv_exact as ex
    from freetoken.kernel.triton import int8_gemv as tg

    w, s, x = _inputs(6656, 2560, rows, 10 + rows)
    assert tg._config(6656, 2560) == (4, 4096, 1, 8)
    ref = _triton(tg.gemv_int8, x, w, s)
    assert torch.equal(ex.gemv_int8_exact_n4w8(x, w, s).view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("silu", [(0, 0), (4, 320)])
@pytest.mark.parametrize("n", [336, 320])
def test_hc_down_tile_same_bits(n, silu, rows):
    from freetoken.kernel import int8_gemv_exact as ex
    from freetoken.kernel.triton import int8_gemv as tg

    w, s, x = _inputs(n, 10240, rows, 20 + rows)
    s = s * 0.1
    assert tg._config(n, 10240) == (1, 2048, 1, 8)
    ref = _triton(tg.gemv_int8, x, w, s, silu_hc=silu[0], silu_cols=silu[1])
    got = ex.gemv_int8_exact_k2048(x, w, s, silu[0], silu[1])
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 6, 7, 8])
def test_hc_mix_same_bits(rows):
    from freetoken.kernel import int8_gemv_exact as ex
    from freetoken.kernel.triton import int8_gemv as tg

    hd = 2560
    w, s, lora = _inputs(4 * hd, 320, rows, 30 + rows, row_stride=336)  # the lora columns of the down output's rows
    rn = torch.randn(rows, 4 * hd, device="cuda").to(torch.bfloat16)
    ref = _triton(tg.gemv_int8_hc_mix, lora, w, s, rn, 4)
    got = ex.gemv_int8_hc_mix_exact(lora, w, s, rn)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_routing_takes_the_hip_kernels(monkeypatch):
    """gemv_int8 and gemv_int8_hc_mix route the four tiles to the HIP kernels."""
    from freetoken.kernel import int8_gemv_exact as ex
    from freetoken.kernel.triton import int8_gemv as tg

    calls = []
    for name in ("gemv_int8_exact", "gemv_int8_exact_n4w8", "gemv_int8_exact_k2048", "gemv_int8_hc_mix_exact"):
        fn = getattr(ex, name)
        monkeypatch.setattr(ex, name, lambda *a, _f=fn, _n=name, **k: (calls.append(_n), _f(*a, **k))[1])
    monkeypatch.setattr(tg, "_EXACT_HIP", True)
    w, s, x = _inputs(7210, 2560, 4, 40)
    tg.gemv_int8(x, w, s)
    w, s, x = _inputs(7210, 2560, 1, 44)
    tg.gemv_int8(x, w, s)  # one row of a small shape stays on Triton
    w, s, x = _inputs(64615, 2560, 1, 45)
    tg.gemv_int8(x, w, s)  # the draft head's rows take the HIP kernel at one row too
    w, s, x = _inputs(6656, 2560, 1, 41)
    tg.gemv_int8(x, w, s)
    w, s, x = _inputs(336, 10240, 2, 42)
    tg.gemv_int8(x, w, s, silu_hc=4, silu_cols=320)
    w, s, lora = _inputs(4 * 2560, 320, 3, 43, row_stride=336)
    tg.gemv_int8_hc_mix(lora, w, s, torch.randn(3, 4 * 2560, device="cuda").to(torch.bfloat16), 4)
    assert calls == ["gemv_int8_exact", "gemv_int8_exact", "gemv_int8_exact_n4w8", "gemv_int8_exact_k2048",
                     "gemv_int8_hc_mix_exact"]
