"""The EXL3 mid step's column-sliced kernel (prefill, fp16 output) gives the bits of the 64-row tile kernel it replaces:
each output is one dot of a row with one column of the Hadamard, so taking 32 columns at a time (the default) changes nothing but
the registers (the tile kernel spilled ~860 VGPRs, 2.7 KB of scratch per lane)."""

from __future__ import annotations

import pytest
import torch
import triton

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
@pytest.mark.parametrize("R", [257, 2220, 9999])
@pytest.mark.parametrize("inter", [256, 384])
def test_mid_cols_matches_tile_kernel(R, inter):
    from freetoken.kernel.triton import exl3

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(R + inter)
    S, splits = 700, 2
    c = torch.randn(splits, R, 2 * inter, device=dev, generator=g) * 3
    slots = torch.randint(-1, S, (R,), device=dev, generator=g, dtype=torch.int32)  # -1: a CPU route
    svh = (torch.rand(S, 2 * inter, device=dev, generator=g) * 0.05).half()
    suh = torch.randn(S, inter, device=dev, generator=g).sign().half()
    ref = torch.empty((R, 1, inter), dtype=torch.float16, device=dev)
    exl3._exl3_mid_kernel[(triton.cdiv(R, 64), inter // 128)](
        c, slots, svh, suh, exl3._hadamard_pm1(dev), ref, R, inter, splits, c.stride(0), c.stride(1), svh.stride(0),
        suh.stride(0), inter, BR=64, CB=1, PRECISE=False, HOIST=False, num_warps=4, num_stages=1,
    )
    assert exl3._MID_COLS, "FREETOKEN_EXL3_MID_COLS=0: the column kernel is off"
    out = exl3.exl3_mid(c, slots, svh, suh, dtype=torch.float16)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
