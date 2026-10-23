"""FREETOKEN_QSA_FUSED_WRITES: qsa_index_writes is the three launches after the QSA group compression
(qsa_index_norm_rope of the index keys into the slab, of the indexer queries, qsa_store_rows into the pending ring),
bit for bit."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required")


def _case(tokens: int, heads: int, by_head: bool, seed: int, head_dim: int = 128, rotary: int = 64):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16
    rand = lambda *s: torch.randn(*s, generator=g, device=dev).to(bf)
    k_rows = tokens
    pooled = rand(k_rows, head_dim)
    positions = torch.randint(0, 4000, (tokens,), generator=g, device=dev, dtype=torch.int32)
    cos_sin = torch.randn(4096, rotary, generator=g, device=dev)
    k_w, q_w = rand(head_dim), rand(head_dim)
    slab_rows = 3 * k_rows + 8
    dest = torch.randint(-1, slab_rows, (k_rows,), generator=g, device=dev, dtype=torch.int32)
    dest = torch.where(dest >= 0, torch.randperm(slab_rows, generator=g, device=dev)[:k_rows].to(torch.int32), dest)
    if by_head:  # [tokens, heads, head_dim] read in place from a wider projection row
        q = rand(tokens, heads * head_dim + 96)[:, : heads * head_dim].view(tokens, heads, head_dim)
    else:
        q = rand(tokens * heads, head_dim)
    blocks, page = 6, 64
    ring = rand(blocks, page, head_dim)
    raw = rand(tokens, head_dim)
    slots = torch.randint(-2, blocks * page, (tokens,), generator=g, device=dev, dtype=torch.int32)
    slots = torch.where(slots >= 0, torch.randperm(blocks * page, generator=g, device=dev)[:tokens].to(torch.int32), slots)
    return pooled, positions, cos_sin, k_w, dest, slab_rows, q, q_w, ring, slots, raw


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 13, 64, 300])
@pytest.mark.parametrize("heads,by_head", [(4, False), (4, True), (1, False)])
def test_fused_writes_match_the_three_launches(tokens, heads, by_head):
    from freetoken.kernel.triton.qsa import qsa_index_norm_rope, qsa_index_writes, qsa_store_rows

    pooled, positions, cos_sin, k_w, dest, slab_rows, q, q_w, ring, slots, raw = _case(tokens, heads, by_head, tokens)
    eps = 1e-6
    slab_a = torch.zeros(slab_rows, pooled.shape[1], dtype=pooled.dtype, device="cuda")
    slab_b = slab_a.clone()
    qa = torch.zeros(tokens * heads, pooled.shape[1], dtype=pooled.dtype, device="cuda")
    qb = qa.clone()
    ring_a, ring_b = ring.clone(), ring.clone()

    qsa_index_norm_rope(pooled, positions, cos_sin, k_w, eps, slab_a, dest_rows=dest)
    qsa_index_norm_rope(q, positions, cos_sin, q_w, eps, qa, heads=heads)
    qsa_store_rows(ring_a, slots, raw)
    qsa_index_writes(pooled, positions, cos_sin, k_w, slab_b, dest, q, positions, cos_sin, q_w, qb, heads,
                     ring_b, slots, raw, eps)
    torch.cuda.synchronize()
    assert torch.equal(slab_a.view(torch.int16), slab_b.view(torch.int16))
    assert torch.equal(qa.view(torch.int16), qb.view(torch.int16))
    assert torch.equal(ring_a.view(torch.int16), ring_b.view(torch.int16))
