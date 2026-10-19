"""FREETOKEN_QSA_SCORE_ROWS: scoring 16 query rows of one request a program gives the one-row kernel's bits."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("depth,rows,holes", [(0, 70, False), (100, 333, False), (5000, 700, False),
                                              (20000, 1500, False), (60000, 4097, False), (20000, 1500, True)])
def test_rows_kernel_matches_one_row_kernel(depth, rows, holes):
    from freetoken.kernel.triton.qsa.score import qsa_mqa_paged

    heads, dim, ratio, page = 4, 128, 4, 64
    total = depth + rows
    cols = total // ratio
    pages = (cols + page - 1) // page
    g = torch.Generator(device="cuda").manual_seed(depth + rows)
    q = torch.randn(rows, heads, dim, device="cuda", generator=g).bfloat16()
    cache = torch.randn(pages, page, 1, dim, device="cuda", generator=g).bfloat16()
    table = torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(1, -1)
    if holes:  # unmapped pages score -inf
        table[0, ::3] = -1
    to_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    positions = torch.arange(depth, total, device="cuda", dtype=torch.int64)
    lens = torch.tensor([total], dtype=torch.int32, device="cuda")
    out = []
    for one in (False, True):
        logits = torch.full((rows, cols), float("nan"), device="cuda")
        visible = torch.empty(rows, dtype=torch.int32, device="cuda")
        qsa_mqa_paged(q, cache, table, to_req, positions, lens, ratio, logits, visible, one_request=one)
        out.append((logits, visible))
    (ref, vis_ref), (got, vis) = out
    assert torch.equal(vis, vis_ref)
    live = torch.arange(cols, device="cuda")[None, :] < vis_ref[:, None]
    assert torch.equal(got[live].view(torch.int32), ref[live].view(torch.int32))
    assert not math.isnan(float(ref[live].sum()))
    assert holes == bool(torch.isinf(ref[live]).any())
