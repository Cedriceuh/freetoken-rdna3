"""FREETOKEN_QSA_MIN_WARPS, FREETOKEN_QSA_PREFILL_STAGES / _WARPS and FREETOKEN_GDN_PREFILL_WARPS: the prefill kernels
launched with more warps (no register spills on RDNA3) or fewer pipeline stages give the bits of upstream's launch
(GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows", [100, 2048])
def test_qsa_prefill_attention_more_warps_same_bits(monkeypatch, rows):
    import freetoken.kernel.triton.qsa.attend as attend

    dev = "cuda"
    ctx, page, heads, dim, topk = 16384, 64, 12, 256, 2048
    g = torch.Generator(device=dev).manual_seed(rows)
    q = torch.randn(rows, heads, dim, device=dev, dtype=torch.bfloat16, generator=g)
    k = torch.randn(ctx // page, page, 1, dim, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn(ctx // page, page, 1, dim, device=dev, dtype=torch.bfloat16, generator=g)
    idx = torch.sort(torch.randint(0, ctx, (rows, topk), device=dev, generator=g, dtype=torch.int32), dim=1).values
    valid = torch.randint(1, topk + 1, (rows, 1), device=dev, generator=g)  # early rows see fewer tokens
    idx = torch.where(torch.arange(topk, device=dev)[None, :] < valid, idx, -1).to(torch.int32).contiguous()
    table = torch.arange(ctx // page, device=dev, dtype=torch.int32).unsqueeze(0).contiguous()
    req = torch.zeros(rows, device=dev, dtype=torch.int32)
    outs = []
    for warps in (0, 4):
        monkeypatch.setattr(attend, "_MIN_WARPS", warps)
        outs.append(attend.qsa_sparse_paged_attention(q, k, v, idx, table, req))
    assert torch.equal(outs[0], outs[1])


def _qsa_inputs(rows, seed):
    dev = "cuda"
    ctx, page, heads, dim, topk = 16384, 64, 12, 256, 2048
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(rows, heads, dim, device=dev, dtype=torch.bfloat16, generator=g)
    k = torch.randn(ctx // page, page, 1, dim, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn(ctx // page, page, 1, dim, device=dev, dtype=torch.bfloat16, generator=g)
    idx = torch.sort(torch.randint(0, ctx, (rows, topk), device=dev, generator=g, dtype=torch.int32), dim=1).values
    valid = torch.randint(1, topk + 1, (rows, 1), device=dev, generator=g)
    idx = torch.where(torch.arange(topk, device=dev)[None, :] < valid, idx, -1).to(torch.int32).contiguous()
    table = torch.arange(ctx // page, device=dev, dtype=torch.int32).unsqueeze(0).contiguous()
    req = torch.zeros(rows, device=dev, dtype=torch.int32)
    return q, k, v, idx, table, req


@pytest.mark.parametrize("rows", [513, 2048])
def test_qsa_prefill_attention_one_stage_prefetch_same_bits(monkeypatch, rows):
    """The one-split prefill profile at 1 stage / 4 warps with the next tile's rows looked up ahead (the ROCm defaults),
    at 1 stage / 8 warps without and at 2 stages / 4 warps with, against upstream's 2 stages / 4 warps."""
    import freetoken.kernel.triton.qsa.attend as attend

    args = _qsa_inputs(rows, rows + 7)
    outs = []
    for stages, warps, ahead in ((2, 0, False), (1, 4, True), (1, 8, False), (2, 4, True)):
        monkeypatch.setattr(attend, "_MIN_WARPS", 4)
        monkeypatch.setattr(attend, "_PREFILL_STAGES", stages)
        monkeypatch.setattr(attend, "_PREFILL_WARPS", warps)
        monkeypatch.setattr(attend, "_PREFILL_PREFETCH", ahead)
        outs.append(attend.qsa_sparse_paged_attention(*args))
    assert all(torch.equal(outs[0], o) for o in outs[1:])


@pytest.mark.parametrize("rows", [1, 4])
def test_qsa_decode_host_kv_one_stage_same_bits(monkeypatch, rows):
    """The decode profile over host-mapped K/V pages (FREETOKEN_QSA_KV_HOST) at one pipeline stage against two."""
    import freetoken.kernel.triton.qsa.attend as attend
    from freetoken.kernel.host_mapped import host_mapped_empty

    q, k, v, idx, table, req = _qsa_inputs(rows, rows + 11)
    hk, hv = (host_mapped_empty(t.shape, t.dtype, t.device) for t in (k, v))
    hk.copy_(k)
    hv.copy_(v)
    outs = []
    for stages in (2, 1):
        monkeypatch.setattr(attend, "_DECODE_HOST_STAGES", stages)
        outs.append(attend.qsa_sparse_paged_attention(q, hk, hv, idx, table, req, decode=True, kv_host=True))
    outs.append(attend.qsa_sparse_paged_attention(q, k, v, idx, table, req, decode=True))  # VRAM, the default launch
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


def test_gdn_prefill_more_warps_same_bits(monkeypatch):
    import freetoken.kernel.fla.chunk_delta_h as delta_h
    import freetoken.kernel.fla.wy_fast as wy
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla

    autotuned = delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64

    class Launch:
        """The state-recurrence kernel at a given launch, bypassing its one-config autotuner."""

        def __init__(self, warps):
            self.warps = warps

        def __getitem__(self, grid):
            def run(*args, **kwargs):
                return autotuned.fn[grid](*args, **kwargs, BV=32, num_warps=self.warps, num_stages=2)

            return run

    torch.manual_seed(0)
    dev = "cuda"
    lens = [3000, 1100]
    T, HK, HV, D = sum(lens), 7, 21, 128  # one rank's heads at the uneven split
    q = torch.randn(1, T, HK, D, device=dev).to(torch.bfloat16)
    k = torch.randn(1, T, HK, D, device=dev).to(torch.bfloat16)
    v = torch.randn(1, T, HV, D, device=dev).to(torch.bfloat16)
    g = -torch.rand(1, T, HV, device=dev) * 0.1
    beta = torch.rand(1, T, HV, device=dev)
    state0 = torch.randn(4, HV, D, D, device=dev) * 0.01
    idx = torch.tensor([1, 3], device=dev, dtype=torch.int32)
    cu = torch.tensor([0, lens[0], T], device=dev, dtype=torch.int64)
    results = []
    for warps in (0, 8):  # 0: upstream's launches (4 warps; recompute_w_u at 3 stages)
        monkeypatch.setattr(delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", Launch(warps or 4))
        monkeypatch.setattr(wy, "GDN_PREFILL_WARPS", warps)
        state = state0.clone()
        out, h = gdn_prefill_chunk_fla(q, k, v, g, beta, state_source=state, indices=idx, cu_seqlens=cu,
                                       scale=D ** -0.5, return_h=True)
        results.append((out, h, state))
    for a, b in zip(*results):
        assert torch.equal(a, b)
