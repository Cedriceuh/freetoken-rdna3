"""FREETOKEN_QSA_FUSED_GATE: qsa_sparse_paged_attention(..., gate=g) is sigmoid_gate_mul(qsa_sparse_paged_attention(...),
g) bit for bit, the gate applied in the split merge (decode) or by its own kernel (one split)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows,decode", [(1, True), (2, True), (4, True), (8, True), (100, False), (2048, False)])
def test_gated_attention_matches_the_separate_gate(rows, decode):
    from freetoken.kernel.triton.attn_gate import sigmoid_gate_mul
    from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention

    dev = "cuda"
    ctx, page, heads, dim, topk = 16384, 64, 12, 256, 2048
    g = torch.Generator(device=dev).manual_seed(rows)
    q = torch.randn(rows, heads, dim, device=dev, dtype=torch.bfloat16, generator=g)
    k = torch.randn(ctx // page, page, 1, dim, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn(ctx // page, page, 1, dim, device=dev, dtype=torch.bfloat16, generator=g)
    idx = torch.sort(torch.randint(0, ctx, (rows, topk), device=dev, generator=g, dtype=torch.int32), dim=1).values
    valid = torch.randint(1, topk + 1, (rows, 1), device=dev, generator=g)
    idx = torch.where(torch.arange(topk, device=dev)[None, :] < valid, idx, -1).to(torch.int32).contiguous()
    table = torch.arange(ctx // page, device=dev, dtype=torch.int32).unsqueeze(0).contiguous()
    req = torch.zeros(rows, device=dev, dtype=torch.int32)
    # the gate is the second half of each head of the fused q|gate projection: strided rows and heads
    qg = torch.randn(rows, heads, 2 * dim, device=dev, dtype=torch.bfloat16, generator=g) * 4
    gate = qg[..., dim:]
    want = sigmoid_gate_mul(qsa_sparse_paged_attention(q, k, v, idx, table, req, decode=decode), gate)
    got = qsa_sparse_paged_attention(q, k, v, idx, table, req, decode=decode, gate=gate)
    assert got.shape == (rows, heads * dim) and got.is_contiguous()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
