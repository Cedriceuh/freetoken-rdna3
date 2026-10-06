"""The Triton ``moe_align_block_size`` (small single-CTA path and large four-kernel path) against a torch reference:
the padded block count, the expert of every block, and each expert's tokens inside its own region. Triton 3.8's AMD
range analysis folded comparisons on ``tl.histogram`` counts to false there, which left the expert ids unwritten
(small path) or every count at zero (large path)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    eff = num_experts + 1  # the sentinel expert slot of fused.py's convention
    counts = torch.bincount(topk_ids.flatten().cpu().long(), minlength=eff)[:eff]
    nblk = (counts + block_size - 1) // block_size
    starts = (torch.cumsum(nblk, 0) - nblk) * block_size
    return counts, starts, int(nblk.sum()) * block_size, torch.repeat_interleave(torch.arange(eff), nblk)


@pytest.mark.parametrize(
    "tokens,num_experts,block_size",
    [(1, 512, 16), (4, 512, 32), (37, 12, 32), (100, 512, 32),  # small path: numel <= 1024
     (103, 512, 16), (300, 12, 32), (4096, 512, 32), (16384, 512, 64)],  # large path
)
def test_moe_align_matches_reference(tokens, num_experts, block_size):
    from freetoken.kernel.triton.moe_align import moe_align_block_size

    gen = torch.Generator().manual_seed(tokens * 7 + num_experts)
    topk_ids = torch.stack([torch.randperm(num_experts, generator=gen)[:10] for _ in range(tokens)]).int().cuda()
    sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, block_size, num_experts)
    counts, starts, ref_ntpp, ref_experts = _reference(topk_ids, block_size, num_experts)

    assert int(ntpp) == ref_ntpp
    assert torch.equal(expert_ids[: ref_ntpp // block_size].cpu(), ref_experts.int())
    numel = topk_ids.numel()
    flat = topk_ids.flatten().cpu()
    regions = sorted_ids[:ref_ntpp].cpu()
    for e in range(num_experts + 1):
        size = int((counts[e] + block_size - 1) // block_size * block_size)
        region = regions[int(starts[e]) : int(starts[e]) + size]
        assert sorted(region[region < numel].tolist()) == torch.nonzero(flat == e).flatten().tolist()
        assert int((region == numel).sum()) == size - int(counts[e])  # the rest holds the sentinel
