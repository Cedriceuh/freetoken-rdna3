"""FREETOKEN_FEWER_COPIES / FREETOKEN_GEMV_LAST_REDUCES: the in-place and in-kernel forms give the bits of the forms they replace (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("T", [1, 2, 3, 4, 7, 300])
@pytest.mark.parametrize("H", [4, 12])
def test_sigmoid_gate_mul_matches_torch(T, H):
    from freetoken.kernel.triton.attn_gate import sigmoid_gate_mul

    D = 256
    torch.manual_seed(T * 100 + H)
    qg = (torch.randn(T, H, 2 * D, device="cuda") * 3).to(torch.bfloat16)  # q | gate per head, like qkv_proj's
    gate = qg[..., D:]
    o = torch.randn(T, H, D, device="cuda").to(torch.bfloat16)
    ref = o.reshape(-1, H * D) * torch.sigmoid(gate.reshape(-1, H * D))
    assert torch.equal(sigmoid_gate_mul(o, gate), ref)


def test_sigmoid_gate_mul_extremes():
    from freetoken.kernel.triton.attn_gate import sigmoid_gate_mul

    bits = torch.arange(0, 65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    g = bits[torch.isfinite(bits)]
    n = g.numel() // 256 * 256
    g = g[:n].view(-1, 1, 256)
    o = torch.randn_like(g, dtype=torch.float32).to(torch.bfloat16)
    ref = o.reshape(-1, 256) * torch.sigmoid(g.reshape(-1, 256))
    assert torch.equal(sigmoid_gate_mul(o, g), ref)


@pytest.mark.parametrize("T", [1, 2, 4, 5, 2048])
def test_gemma_norm_strided_matches_copy_then_inplace(T):
    from freetoken.layers.norm import GemmaPlusOneRMSNorm

    nq, nkv, D = 12, 1, 256
    torch.manual_seed(T)
    norm = GemmaPlusOneRMSNorm(D, eps=1e-6)
    norm.weight = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    W = nq * 2 * D + 2 * nkv * D
    qkv = torch.randn(T, W, device="cuda").to(torch.bfloat16)
    qg, k, _ = qkv.split([nq * 2 * D, nkv * D, nkv * D], dim=-1)
    qg = qg.view(-1, nq, 2 * D)
    # the strided forms first: at one row k is already contiguous and the old form normalizes it in place
    new_q = norm.forward_strided(qg[..., :D])
    new_k = norm.forward_strided(k.view(-1, nkv, D))
    ref_q = qg[..., :D].contiguous()
    norm.forward_inplace(ref_q)
    ref_k = k.contiguous().view(-1, nkv, D)
    norm.forward_inplace(ref_k)
    assert torch.equal(new_q, ref_q)
    assert torch.equal(new_k, ref_k)


@pytest.mark.parametrize("T", [1, 2, 4, 3, 300])
@pytest.mark.parametrize("rows_per_block", [1, None])
def test_gated_norm_reads_z_in_place(T, rows_per_block):
    from freetoken.kernel.fla import rms_norm_gated

    H, D = 27, 128
    torch.manual_seed(T)
    w = torch.randn(D, device="cuda").to(torch.bfloat16)
    proj = torch.randn(T, 5000 + H * D + 2 * H, device="cuda").to(torch.bfloat16)
    _, z, _ = torch.split(proj, [5000, H * D, 2 * H], dim=-1)
    z3 = z.reshape(T, H, D)
    x = torch.randn(T * H, D, device="cuda").to(torch.bfloat16)
    kw = dict(weight=w, bias=None, eps=1e-6, is_rms_norm=True, norm_before_gate=True, activation="silu",
              rows_per_block=rows_per_block)
    ref = rms_norm_gated(x=x, z=z3.reshape(-1, D), **kw)
    assert torch.equal(rms_norm_gated(x=x, z=z3, **kw), ref)


@pytest.mark.parametrize("M", [1, 2, 3, 4, 8])
def test_gemv_last_program_reduce_is_bit_exact(M):
    from freetoken.kernel.triton.dense_gemv import gemv_bf16, gemv_config, gemv_counters

    torch.manual_seed(M)
    for N in (512, 640):
        w = (torch.randn(N, 2560, device="cuda") * 0.02).to(torch.bfloat16)
        cfg = gemv_config(w)
        cnt = gemv_counters(w, cfg)
        assert cnt is not None
        for trial in range(30):  # the counters must come back to zero after every call
            x = (torch.randn(M, 2560, device="cuda") * (1 + trial % 5)).to(torch.bfloat16)
            assert torch.equal(gemv_bf16(x, w, cfg, counters=cnt), gemv_bf16(x, w, cfg))
        assert int(cnt.abs().sum()) == 0
        # under CUDA-graph replay, with new inputs each time
        x = torch.randn(M, 2560, device="cuda").to(torch.bfloat16)
        gemv_bf16(x, w, cfg, counters=cnt)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = gemv_bf16(x, w, cfg, counters=cnt)
        for _ in range(20):
            x.copy_(torch.randn(M, 2560, device="cuda").to(torch.bfloat16))
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, gemv_bf16(x, w, cfg))


@pytest.mark.parametrize("T", [1, 2, 4, 9])
def test_qsa_index_norm_rope_reads_heads_in_place(T):
    from freetoken.kernel.triton.qsa import qsa_index_norm_rope

    H, D, R = 4, 128, 64
    torch.manual_seed(T)
    proj = torch.randn(T, H * D + D, device="cuda").to(torch.bfloat16)
    q, _ = proj.split([H * D, D], dim=-1)
    q3 = q.reshape(-1, H, D)
    pos = torch.randint(0, 1000, (T,), device="cuda", dtype=torch.int32)
    cs = torch.randn(1000, R, device="cuda")
    w = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    out_new = torch.empty(T * H, D, device="cuda", dtype=torch.bfloat16)
    out_ref = torch.empty_like(out_new)
    qsa_index_norm_rope(q3, pos, cs, w, 1e-6, out_new, heads=H)
    qsa_index_norm_rope(q3.contiguous().view(-1, D), pos, cs, w, 1e-6, out_ref, heads=H)
    assert torch.equal(out_new, out_ref)


@pytest.mark.parametrize("N", [1, 3, 4, 300])
def test_hc_combine_inplace_matches_out_of_place(N):
    from freetoken.kernel.triton.hc import hc_combine, hc_combine_rmsnorm

    hc, D = 4, 2560
    torch.manual_seed(N)
    R = torch.randn(N, hc * D, device="cuda").to(torch.bfloat16)
    y = torch.randn(N, D, device="cuda").to(torch.bfloat16)
    s = torch.randn(N, hc, device="cuda").to(torch.bfloat16)
    w = (torch.randn(D, device="cuda") * 0.1).to(torch.bfloat16)
    ref = hc_combine(R, y, s, hc)
    R2 = R.clone()
    got = hc_combine(R2, y, s, hc, inplace=True)
    assert got.data_ptr() == R2.data_ptr() and torch.equal(got, ref)
    ref_c, ref_n = hc_combine_rmsnorm(R, y, s, w, 1e-6, hc)
    R3 = R.clone()
    got_c, got_n = hc_combine_rmsnorm(R3, y, s, w, 1e-6, hc, inplace=True)
    assert got_c.data_ptr() == R3.data_ptr() and torch.equal(got_c, ref_c) and torch.equal(got_n, ref_n)
