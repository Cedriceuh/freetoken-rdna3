"""The MTP head's NVFP4 quantizer writes the layout the NVFP4 dequant kernel reads."""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen4_exp.mtp import quantize_nvfp4

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _torch_dequant(packed, scale, g):
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                       device=packed.device)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).flatten(-2).long()
    vals = lut[codes].view(packed.shape[0], -1, 16)
    return (vals * scale.float().unsqueeze(-1) * g.float()).flatten(-2)


def test_round_trip_error_is_fp4_sized():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    packed, scale, g = quantize_nvfp4(w)
    assert packed.dtype == torch.uint8 and packed.shape == (64, 128)
    assert scale.dtype == torch.float8_e4m3fn and scale.shape == (64, 16)
    back = _torch_dequant(packed, scale, g)
    # half the widest e2m1 step (4 -> 6) is a block max / 6; the e4m3 scale's rounding (<= 1/16) and the clip at 6
    # add up to another block max / 16
    block_max = w.view(64, 16, 16).abs().amax(-1, keepdim=True)
    assert ((back - w).view(64, 16, 16).abs() <= block_max * (1 / 6 + 1 / 16) + 1e-6).all()
    assert torch.nn.functional.cosine_similarity(back.flatten(), w.flatten(), dim=0) > 0.99


def test_exact_values_survive():
    w = torch.tensor([[0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, 6.0] * 2])
    packed, scale, g = quantize_nvfp4(w)
    assert torch.allclose(_torch_dequant(packed, scale, g), w, atol=1e-6)


@requires_cuda
def test_matches_the_dequant_kernel():
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4

    torch.manual_seed(1)
    w = (torch.randn(128, 512) * 0.05).cuda()
    packed, scale, g = quantize_nvfp4(w)
    glob = g.to(torch.float16).expand(128).contiguous()
    got = dequant_nvfp4(packed.unsqueeze(0), scale.unsqueeze(0), glob.unsqueeze(0),
                        torch.zeros(1, dtype=torch.int32, device="cuda"), dtype=torch.float32)[0]
    want = _torch_dequant(packed, scale, g.to(torch.float16))
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-6)
