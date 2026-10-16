"""FREETOKEN_QSA_KV_HOST: host-mapped device tensors and the prefill page staging give the bits of VRAM (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_host_mapped_tensor_reads_and_writes_in_place() -> None:
    from freetoken.kernel.host_mapped import host_mapped_empty

    h = host_mapped_empty((4096, 256), torch.bfloat16, torch.device("cuda", torch.cuda.current_device()))
    assert h.is_cuda
    ref = torch.randn(4096, 256, device="cuda").bfloat16()
    h.copy_(ref)
    idx = torch.randint(0, 4096, (777,), device="cuda")
    assert torch.equal(h[idx], ref[idx])  # a GPU gather across PCIe
    h[idx] = -ref[idx]  # a GPU scatter into host memory
    ref[idx] = -ref[idx]
    assert torch.equal(h.cpu(), ref.cpu())


def test_stage_pages_copies_exactly_the_pages_asked() -> None:
    from freetoken.kernel.host_mapped import host_mapped_empty
    from freetoken.kernel.triton.qsa.stage import stage_pages

    pages, shape = 300, (64, 1, 256)
    h = host_mapped_empty((pages, *shape), torch.bfloat16, torch.device("cuda", torch.cuda.current_device()))
    h.copy_(torch.randn(pages, *shape, device="cuda").bfloat16())
    dst = torch.zeros(pages, *shape, device="cuda", dtype=torch.bfloat16)
    want = torch.tensor([5, 0, 299, 17, 17, 120], device="cuda", dtype=torch.int32)  # repeats are harmless
    stage_pages(h, dst, want)
    hit = torch.zeros(pages, dtype=torch.bool, device="cuda")
    hit[want.long()] = True
    assert torch.equal(dst[hit], h[hit])
    assert not dst[~hit].any()


def _selected_case(seed: int):
    """Two requests over a shuffled page pool, a prefill's selected tokens with -1 holes, a block table with unused
    (-1) entries past each request's pages, and K/V pages in host-mapped memory."""
    from freetoken.kernel.host_mapped import host_mapped_empty

    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device("cuda", torch.cuda.current_device())
    pages, ps, width, topk, heads = 400, 64, 160, 512, 12
    lens = [150 * ps + 17, 40 * ps]
    perm = torch.randperm(pages, generator=g).to(torch.int32)
    table = torch.full((2, width), -1, dtype=torch.int32)
    table[0, :151] = perm[:151]
    table[1, :40] = perm[151:191]
    k, v = (host_mapped_empty((pages, ps, 1, 256), torch.bfloat16, dev) for _ in range(2))
    k.copy_(torch.randn(pages, ps, 1, 256, generator=g).bfloat16())
    v.copy_(torch.randn(pages, ps, 1, 256, generator=g).bfloat16())
    rows = 96
    token_to_req = (torch.arange(rows) >= 70).to(torch.int32)
    idx = torch.stack([torch.randint(0, lens[r], (topk,), generator=g) for r in token_to_req.tolist()]).to(torch.int32)
    idx[torch.rand(rows, topk, generator=g) < 0.2] = -1
    in_use = torch.cat([table[0, :151], table[1, :40]])
    q = torch.randn(rows, heads, 256, generator=g).bfloat16()
    return (q.to(dev), k, v, idx.to(dev), table.to(dev), token_to_req.to(dev), in_use.to(dev))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stage_selected_pages_stages_the_rows_read(seed: int) -> None:
    from freetoken.kernel.triton.qsa.stage import stage_selected_pages

    q, k, v, idx, table, t2r, in_use = _selected_case(seed)
    ps = k.shape[1]
    dk, dv = torch.zeros_like(k, device="cuda"), torch.zeros_like(v, device="cuda")
    mark = torch.zeros(k.shape[0] * ps, dtype=torch.int32, device="cuda")
    stage_selected_pages(k, dk, v, dv, in_use, idx, t2r, table, mark, 7)
    want = set()
    for r, row in enumerate(idx.tolist()):
        for t in row:
            if t >= 0:
                want.add(int(table[int(t2r[r]), t // ps]) * ps + t % ps)
    assert set((mark == 7).nonzero().view(-1).tolist()) == want
    hit = torch.zeros(k.shape[0] * ps, dtype=torch.bool, device="cuda")
    hit[list(want)] = True
    flat = lambda t: t.reshape(k.shape[0] * ps, -1)
    assert torch.equal(flat(dk)[hit], flat(k)[hit]) and torch.equal(flat(dv)[hit], flat(v)[hit])
    assert not flat(dk)[~hit].any() and not flat(dv)[~hit].any()


@pytest.mark.parametrize("seed", [3, 4])
def test_attention_on_selected_pages_same_bits(seed: int) -> None:
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention
    from freetoken.kernel.triton.qsa.stage import stage_pages, stage_selected_pages

    q, k, v, idx, table, t2r, in_use = _selected_case(seed)
    fk, fv = torch.empty_like(k, device="cuda"), torch.empty_like(v, device="cuda")
    stage_pages(k, fk, in_use)
    stage_pages(v, fv, in_use)
    # the rows the selection skips hold NaN: a read of any of them would show
    sk, sv = (torch.full(k.shape, float("nan"), dtype=k.dtype, device="cuda") for _ in range(2))
    stage_selected_pages(k, sk, v, sv, in_use, idx, t2r, table,
                         torch.zeros(k.shape[0] * k.shape[1], dtype=torch.int32, device="cuda"), 1)
    full = qsa_sparse_paged_attention(q, fk, fv, idx, table, t2r)
    sel = qsa_sparse_paged_attention(q, sk, sv, idx, table, t2r)
    assert torch.equal(full, sel)
    assert not full.isnan().any()


def test_stage_selected_falls_back_to_whole_pages() -> None:
    """Rows that are not a power of two wide: every page asked, as stage_pages does (no crash)."""
    from freetoken.kernel.triton.qsa.stage import stage_selected_pages

    pages, shape = 50, (64, 3, 128)  # 3 KV heads: 384-element rows
    k = torch.randn(pages, *shape, device="cuda").bfloat16()
    v = torch.randn(pages, *shape, device="cuda").bfloat16()
    dk, dv = torch.zeros_like(k), torch.zeros_like(v)
    want = torch.tensor([3, 7, 7, 41], device="cuda", dtype=torch.int32)
    idx = torch.full((2, 8), -1, dtype=torch.int32, device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, -1)
    stage_selected_pages(k, dk, v, dv, want, idx, torch.zeros(2, dtype=torch.int32, device="cuda"), table,
                         torch.zeros(pages * 64, dtype=torch.int32, device="cuda"), 1)
    hit = torch.zeros(pages, dtype=torch.bool, device="cuda")
    hit[want.long()] = True
    assert torch.equal(dk[hit], k[hit]) and torch.equal(dv[hit], v[hit]) and not dk[~hit].any()
