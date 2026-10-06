"""The fused MoE epilogue (``shared_gate_sum_mul_add``) gives exactly what ``moe_sum_reduce`` + ``shared_gate_sigmoid``
+ ``shared_gate_mul_add`` give: per-expert outputs (decode) or an already summed one (prefill), one to many tokens."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 37, 1500])
@pytest.mark.parametrize("topk", [1, 10])
def test_fused_epilogue_is_bit_exact(tokens, topk):
    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.kernel.triton.moe_shared_gate import (
        shared_gate_mul_add,
        shared_gate_sigmoid,
        shared_gate_sum_mul_add,
    )

    gen = torch.Generator(device="cuda").manual_seed(tokens * 31 + topk)
    hidden = 2560
    x = torch.randn(tokens, hidden, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(hidden, device="cuda", generator=gen) * 0.02).bfloat16()
    per_expert = (torch.randn(tokens, topk, hidden, device="cuda", generator=gen) * 0.5).bfloat16()
    shared = torch.randn(tokens, hidden, device="cuda", generator=gen).bfloat16()

    routed = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    moe_sum_reduce_triton(per_expert, routed)
    gate = shared_gate_sigmoid(x, w)
    expected = shared_gate_mul_add(routed, shared, gate)

    got = shared_gate_sum_mul_add(per_expert, shared, x, w)
    assert torch.equal(got, expected)
    # an output the experts already summed, seen as one expert
    assert torch.equal(shared_gate_sum_mul_add(routed.view(tokens, 1, hidden), shared, x, w), expected)
