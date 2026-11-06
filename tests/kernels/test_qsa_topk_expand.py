"""FREETOKEN_QSA_TOPK_EXPAND: the block top-k kernels' expansion epilogue gives expand_qsa_block_indices' token
indices (and the same blocks), on the one-program path (a short history) and the split path (up to 255k tokens), for
decode rows of one or several requests, rows near a request's start, and fewer visible blocks than the budget."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

RATIO, TOKEN_TOPK = 4, 2048


@pytest.mark.parametrize("ctx", [700, 3000, 16_000, 65_000, 120_000, 255_000])
@pytest.mark.parametrize("rows,reqs", [(1, 1), (3, 1), (4, 1), (8, 2), (16, 4)])
@pytest.mark.parametrize("seed", range(3))
def test_expand_epilogue_matches_the_expand_kernel(ctx, rows, reqs, seed):
    from freetoken.kernel.triton.qsa import expand_qsa_block_indices
    from freetoken.kernel.triton.qsa.topk import qsa_block_topk, qsa_block_topk_scratch_width

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 100 + rows + ctx)
    block_topk = TOKEN_TOPK // RATIO
    columns = (ctx + 64 * 8) // RATIO  # the table's width a little past the history
    m = rows // reqs
    seq_lens = torch.tensor([max(5, ctx - 997 * i) for i in range(reqs)], dtype=torch.int32, device=dev)
    if seed == 2:
        seq_lens[0] = min(int(seq_lens[0]), 37)  # fewer complete blocks than the budget
    token_to_req = torch.arange(reqs, dtype=torch.int32, device=dev).repeat_interleave(m)
    positions = torch.cat([seq_lens[i] - m + torch.arange(m, device=dev, dtype=torch.int32) for i in range(reqs)])
    visible = torch.div(positions + 1, RATIO, rounding_mode="floor").clamp(max=columns).to(torch.int32)
    logits = torch.rand(rows, columns, generator=g, device=dev)
    if seed == 1:
        logits = torch.floor(logits * 64) / 64  # many ties
    width = qsa_block_topk_scratch_width(columns, block_topk)
    scratch = torch.empty((rows, max(width, 1)), dtype=torch.int32, device=dev) if width else None

    blocks_ref = torch.empty((rows, block_topk), dtype=torch.int32, device=dev)
    qsa_block_topk(logits.clone(), visible, blocks_ref, scratch)
    idx_ref = torch.empty((rows, TOKEN_TOPK + RATIO - 1), dtype=torch.int32, device=dev)
    expand_qsa_block_indices(blocks_ref, positions, seq_lens, token_to_req, RATIO, TOKEN_TOPK, idx_ref)

    blocks = torch.empty_like(blocks_ref)
    idx = torch.full_like(idx_ref, -7)
    done = qsa_block_topk(logits.clone(), visible, blocks, scratch,
                          (positions, seq_lens, token_to_req, RATIO, TOKEN_TOPK, idx))
    assert done
    assert torch.equal(blocks, blocks_ref)
    assert torch.equal(idx, idx_ref)
