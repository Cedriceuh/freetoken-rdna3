"""FREETOKEN_GEMV_SHARED_TILES: the bf16 split-K GEMV loads each weight tile once for the 2-4 rows of a verify step;
every row keeps the bits it gets with the tile loaded once per row (the router and the 640-row shapes, with and
without the last-program reduce)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("shape", [(512, 2560), (640, 2560), (8240, 2560), (10240, 320)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
@pytest.mark.parametrize("last", [False, True])
def test_shared_tiles_keep_each_rows_bits(monkeypatch, shape, rows, last):
    import freetoken.kernel.triton.dense_gemv as G

    N, K = shape
    g = torch.Generator(device="cuda").manual_seed(N + K + rows)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    x = torch.randn(rows, K, device="cuda", generator=g).to(torch.bfloat16)
    cfg = G._BF16_CONFIGS[shape]
    counters = lambda: G.gemv_counters(w, cfg) if last else None
    monkeypatch.setattr(G, "_SHARED_TILES", False)
    want = G.gemv_bf16(x, w, cfg, counters())
    monkeypatch.setattr(G, "_SHARED_TILES", True)
    got = G.gemv_bf16(x, w, cfg, counters())
    alone = torch.cat([G.gemv_bf16(x[r : r + 1], w, cfg, counters()) for r in range(rows)])
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert torch.equal(got.view(torch.int16), alone.view(torch.int16))
