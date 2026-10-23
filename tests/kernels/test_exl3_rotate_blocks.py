"""FREETOKEN_EXL3_ROTATE_BLOCKS and FREETOKEN_EXL3_VEC_WARPS: the decode rotate-in with several 128-blocks a program,
and the rotate-in and mid steps at 1 or 2 warps a program, give the bits of one block a program at 4 warps (the same
_had128_vec on the same tile; its reduction order holds up to 4 warps), at the verify step's row counts."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 12])
@pytest.mark.parametrize("blocks", [2, 4, 5])
def test_grouped_rotate_in_matches_one_block_a_program(monkeypatch, tokens, blocks):
    import freetoken.kernel.triton.exl3 as E

    dev, H, top_k, S = "cuda", 2560, 10, 64
    g = torch.Generator(device=dev).manual_seed(tokens * 10 + blocks)
    x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16, generator=g) * 3
    suh = (torch.randn(S, 2, H, device=dev, generator=g) * 2).to(torch.float16)
    slots = torch.randint(-1, S, (tokens * top_k,), device=dev, generator=g, dtype=torch.int32)  # -1: a CPU route
    monkeypatch.setattr(E, "_ROTATE_BLOCKS", 1)
    monkeypatch.setattr(E, "_VEC_WARPS", 4)
    want = E.exl3_rotate_in(x, suh, slots, top_k)
    monkeypatch.setattr(E, "_ROTATE_BLOCKS", blocks)
    monkeypatch.setattr(E, "_ROTATE_GROUP_ROWS", 1)  # group at every row count
    for warps in (1, 2, 4):
        monkeypatch.setattr(E, "_VEC_WARPS", warps)
        got = E.exl3_rotate_in(x, suh, slots, top_k)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), warps


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 12])
@pytest.mark.parametrize("splits", [1, 2])
def test_mid_step_at_fewer_warps_keeps_the_bits(monkeypatch, tokens, splits):
    import freetoken.kernel.triton.exl3 as E

    dev, top_k, S, inter = "cuda", 10, 64, 640
    g = torch.Generator(device=dev).manual_seed(tokens * 7 + splits)
    R = tokens * top_k
    c = torch.randn(splits, R, 2 * inter, device=dev, generator=g) * 4
    svh = (torch.randn(S, 2 * inter, device=dev, generator=g) * 2).to(torch.float16)
    dsuh = torch.randn(S, inter, device=dev, generator=g).sign().to(torch.float16)
    slots = torch.randint(-1, S, (R,), device=dev, generator=g, dtype=torch.int32)
    monkeypatch.setattr(E, "_VEC_WARPS", 4)
    want = E.exl3_mid(c, slots, svh, dsuh)
    for warps in (1, 2):
        monkeypatch.setattr(E, "_VEC_WARPS", warps)
        got = E.exl3_mid(c, slots, svh, dsuh)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), warps
