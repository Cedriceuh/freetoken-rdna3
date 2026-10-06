"""``gemv_bf16`` with several rows (one reduce launch for all of them) gives each row exactly its batch-of-one bits."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows", [2, 3, 4, 8])
def test_rows_match_one_at_a_time(rows):
    from freetoken.kernel.triton.dense_gemv import gemv_bf16

    gen = torch.Generator(device="cuda").manual_seed(rows)
    w = (torch.randn(512, 2560, device="cuda", generator=gen) * 0.02).bfloat16()
    x = torch.randn(rows, 2560, device="cuda", generator=gen).bfloat16()
    for cfg in [(4, 512, 4, 4), (2, 1024, 2, 4), (8, 256, 8, 8)]:
        together = gemv_bf16(x, w, cfg)
        alone = torch.cat([gemv_bf16(x[r : r + 1], w, cfg) for r in range(rows)])
        assert torch.equal(together, alone)
