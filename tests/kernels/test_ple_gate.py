"""FREETOKEN_PLE_GATE_FUSED: the fused PLE gate gives torch's eight launches' bits for every bf16 dot product (all
65,536 values, NaN excepted), and the gated rows of the decode shapes."""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _torch_gate(s: torch.Tensor, value: torch.Tensor, hidden: int) -> torch.Tensor:
    gate = s / math.sqrt(hidden)
    gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
    return (gate * value.unsqueeze(-2)).flatten(-2)


def test_every_bf16_value():
    from freetoken.kernel.triton.ple_gate import ple_gate

    dev = torch.device("cuda")
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    vals = bits[~torch.isnan(bits.float())]
    vals = torch.cat([vals, vals[: (-vals.numel()) % 4]])  # a whole number of rows of 4 gates
    s = vals.to(dev).view(-1, 4, 1)  # [T, hc, 1] as the layer's sum leaves it
    value = torch.ones(s.shape[0], 8, dtype=torch.bfloat16, device=dev)
    ref = _torch_gate(s, value, 2560)
    got = ple_gate(s, value, 2560)
    diff = (got.view(torch.int16) != ref.view(torch.int16))
    assert not diff.any(), f"{int(diff.sum())} of {diff.numel()} differ"


@pytest.mark.parametrize("rows", [1, 3, 4, 16])
@pytest.mark.parametrize("seed", range(3))
def test_decode_shapes(rows, seed):
    from freetoken.kernel.triton.ple_gate import ple_gate

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed * 10 + rows)
    s = (torch.randn(rows, 4, 1, generator=g, device=dev) * 30).to(torch.bfloat16)
    value = (torch.randn(rows, 2560, generator=g, device=dev) * 2).to(torch.bfloat16)
    assert torch.equal(ple_gate(s, value, 2560).view(torch.int16), _torch_gate(s, value, 2560).view(torch.int16))
