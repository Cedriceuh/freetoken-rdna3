"""causal_conv1d_decode_rows: several tokens per request in one launch equal that many single-token updates, bit for
bit, and the snapshots are the state after each row."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_rows_equal_sequential_single_token_updates(rows):
    from freetoken.kernel.triton.causal_conv1d_triton import causal_conv1d_decode, causal_conv1d_decode_rows

    torch.manual_seed(rows)
    slots, dim, width, reqs = 6, 384, 4, 3
    state = torch.randn(slots, dim, width - 1, device="cuda").to(torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda").to(torch.bfloat16)
    x = torch.randn(reqs, rows, dim, device="cuda").to(torch.bfloat16)
    idx = torch.tensor([4, 1, 2], dtype=torch.int32, device="cuda")

    ref_state = state.clone()
    ref_out, ref_snaps = [], []
    for j in range(rows):
        ref_out.append(causal_conv1d_decode(x[:, j].contiguous(), ref_state, weight, idx))
        ref_snaps.append(ref_state.index_select(0, idx.long()).clone())

    got_state = state.clone()
    snaps = torch.zeros(rows - 1, reqs, dim, width - 1, dtype=torch.bfloat16, device="cuda")
    out = causal_conv1d_decode_rows(x, got_state, weight, idx, snapshots=snaps)
    assert torch.equal(out, torch.stack(ref_out, dim=1))
    assert torch.equal(got_state, ref_state)
    for j in range(rows - 1):
        assert torch.equal(snaps[j], ref_snaps[j])
