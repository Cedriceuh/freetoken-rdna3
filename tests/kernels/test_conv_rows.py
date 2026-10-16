"""FREETOKEN_GDN_CONV_ROWS: the token-major prefill conv (causal_conv1d_varlen_rows) gives the channel-major conv's
values and conv states bit for bit, q / k / v written apart, the input read in place from a wider projection."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("lens", [[4096], [1000, 37, 3000], [5, 2, 70]])
def test_rows_conv_matches_channel_major(lens):
    from freetoken.kernel.triton.causal_conv1d_triton import causal_conv1d_varlen, causal_conv1d_varlen_rows

    g = torch.Generator(device="cuda").manual_seed(sum(lens))
    key, value, width, slots = 1152, 3456, 4, 6
    dim = 2 * key + value
    total = sum(lens)
    proj = torch.randn((total, dim + 3200), device="cuda", generator=g).to(torch.bfloat16)  # + z, b, a columns
    conv_in = proj[:, :dim]
    weight = torch.randn((dim, width), device="cuda", generator=g).to(torch.bfloat16)
    states = torch.randn((slots, dim, width - 1), device="cuda", generator=g).to(torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    idx = torch.tensor([4, 1, 3][: len(lens)], dtype=torch.int32, device="cuda")
    init = torch.tensor([True, False, True][: len(lens)], device="cuda")

    ref_states = states.clone()
    ref = causal_conv1d_varlen(conv_in.transpose(0, 1).contiguous(), weight, ref_states, cu, idx, init).transpose(0, 1)
    new_states = states.clone()
    outs, start = [], 0
    for d in (key, key, value):
        out = torch.empty((total, d), dtype=torch.bfloat16, device="cuda")
        causal_conv1d_varlen_rows(conv_in[:, start:start + d], weight[start:start + d], new_states[:, start:start + d],
                                  cu, idx, init, out)
        outs.append(out)
        start += d
    assert torch.equal(torch.cat(outs, dim=1), ref)
    assert torch.equal(new_states, ref_states)
