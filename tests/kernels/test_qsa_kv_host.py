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


def _decode_cache_case(seed: int, requests: int = 3, cap: int = 8192):
    """Host-mapped K/V over a shuffled page pool, two decoding requests in table rows 0 and 2 (the attention's block
    table: the batch's rows of ``table``), and one layer's decode cache (attend.py DecodeCache) for ``requests`` table
    rows, its ``cap`` slots shared by them."""
    from freetoken.kernel.host_mapped import host_mapped_empty
    from freetoken.kernel.triton.qsa.attend import DecodeCache

    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device("cuda", torch.cuda.current_device())
    pages, ps, width = 900, 64, 400
    k, v = (host_mapped_empty((pages, ps, 1, 256), torch.bfloat16, dev) for _ in range(2))
    k.copy_(torch.randn(pages, ps, 1, 256, generator=g).bfloat16())
    v.copy_(torch.randn(pages, ps, 1, 256, generator=g).bfloat16())
    perm = torch.randperm(pages, generator=g).to(torch.int32)
    table = torch.full((requests + 1, width), -1, dtype=torch.int32)
    table[0, :400] = perm[:400]
    table[2, :400] = perm[400:800]
    tokens = width * ps
    dc = DecodeCache(
        k=torch.zeros(cap, 256, dtype=torch.bfloat16, device=dev),
        v=torch.zeros(cap, 256, dtype=torch.bfloat16, device=dev),
        tag=torch.full((cap,), -1, dtype=torch.int64, device=dev),
        stamp=torch.randint(-9, 9, (requests, tokens), dtype=torch.int32, device=dev),  # a prefill's leftovers
        slot=torch.randint(-9, cap + 9, (requests, tokens), dtype=torch.int32, device=dev),
        hand=torch.zeros(1, dtype=torch.int32, device=dev), cnt=torch.zeros(1, dtype=torch.int32, device=dev),
        ctl=torch.zeros(2, dtype=torch.int32, device=dev),
        seq_lens=torch.zeros(2, dtype=torch.int32, device=dev),
        ring=torch.tensor([0, 2], dtype=torch.int32, device=dev))
    return g, dev, k, v, table.to(dev), perm, dc


def _decode_selection(g, lens, base, rows):
    """Each request's rows: its groups of 4 tokens (some replaced row to row) sorted, then its last 3 tokens."""
    out = []
    for r, L in enumerate(lens):
        for i in range(rows):
            grp = base[r].clone()
            k = torch.rand(grp.shape[0], generator=g) < 0.25 * (i > 0)
            grp[k] = torch.randint(0, (L - 3) // 4, (int(k.sum()),), generator=g)
            grp = torch.sort(grp).values
            out.append(torch.cat([(grp[:, None] * 4 + torch.arange(4)).view(-1), torch.arange(L - 3, L)]))
    return torch.stack(out).to(torch.int32)


@pytest.mark.parametrize("seed", [0, 1])
def test_decode_cache_same_bits_over_calls(seed: int) -> None:
    """FREETOKEN_QSA_DECODE_CACHE: 24 decode calls of two requests (3 rows each, overlapping selections), the newest
    tokens' K/V rewritten every call (rejected drafts), request slot 2 and its pages handed to another request at
    call 12 with the prefill's invalidation: every output has the uncached kernel's bits, and without the
    invalidation it would not."""
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    g, dev, k, v, table, perm, dc = _decode_cache_case(seed)
    rows, ps = 3, k.shape[1]
    lens = [20000, 23000]
    base = [torch.randperm(4900, generator=g)[:512] for _ in lens]
    t2r = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), rows).to(dev)
    stale = None
    filled = 0
    for call in range(24):
        if call == 12:  # another request takes table row 2 and the same pages (freed, given again): its prefill
            # writes its own K/V there and invalidates the cache
            pages = perm[400:800].long()
            k[pages] = torch.randn(400, *k.shape[1:], generator=g).bfloat16().to(dev)
            v[pages] = torch.randn(400, *v.shape[1:], generator=g).bfloat16().to(dev)
            lens[1] = 21000
            stale = (dc.tag.clone(),)
            dc.tag.fill_(-1)
        for r, L in enumerate(lens):  # this call's newest rows: rewritten, as a rejected draft's would be
            for t in range(L - 6, L):
                page = int(table[int(dc.ring[r]), t // ps])
                k[page, t % ps] = torch.randn(1, 256, generator=g).bfloat16().to(dev)
                v[page, t % ps] = torch.randn(1, 256, generator=g).bfloat16().to(dev)
        for r in range(2):
            swap = torch.rand(512, generator=g) < 0.3
            base[r][swap] = torch.randint(0, (lens[r] - 3) // 4, (int(swap.sum()),), generator=g)
        idx = _decode_selection(g, lens, base, rows).to(dev)
        q = torch.randn(idx.shape[0], 12, 256, generator=g).bfloat16().to(dev)
        dc.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        bt = table[dc.ring.long()]
        want = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r, decode=True, kv_host=True)
        if call == 12:  # the same call without the invalidation reads the old owner's rows
            saved = [t.clone() for t in (dc.k, dc.v, dc.stamp, dc.slot, dc.hand, dc.cnt, dc.ctl)]
            dc.tag.copy_(stale[0])
            bad = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r, decode=True, kv_host=True, dcache=dc)
            assert not torch.equal(bad, want)
            for t, s in zip((dc.k, dc.v, dc.stamp, dc.slot, dc.hand, dc.cnt, dc.ctl), saved):
                t.copy_(s)
            dc.tag.fill_(-1)
        got = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r, decode=True, kv_host=True, dcache=dc)
        assert torch.equal(got, want), f"call {call}"
        assert int(dc.ctl[0]) == call + 1 and int(dc.ctl[1]) == 0 and not dc.cnt.any()
        filled = max(filled, int((dc.tag >= 0).sum()))
        lens = [L + int(torch.randint(1, 4, (1,), generator=g)) for L in lens]
    assert filled > 6000  # the cache held more than both requests' last selections


def test_decode_cache_in_a_graph_same_bits() -> None:
    """The fill and the cached attention captured once and replayed over changing selections and lengths (as a decode
    graph is): the uncached kernel's bits every replay; a capacity close to a call's tokens, so slots are reused
    within a few calls and some tokens find none."""
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    g, dev, k, v, table, perm, dc = _decode_cache_case(5, cap=4000)
    table = table[dc.ring.long()]
    rows = 2
    lens = [15000, 17000]
    base = [torch.randperm(3700, generator=g)[:512] for _ in lens]
    t2r = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), rows).to(dev)
    idx = _decode_selection(g, lens, base, rows).to(dev)
    q = torch.randn(idx.shape[0], 12, 256, generator=g).bfloat16().to(dev)
    out = torch.empty_like(q)
    dc.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))

    def step():
        out.copy_(qsa_sparse_paged_attention(q, k, v, idx, table, t2r, decode=True, kv_host=True, dcache=dc))

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    for call in range(12):
        for r in range(2):
            swap = torch.rand(512, generator=g) < 0.3
            base[r][swap] = torch.randint(0, (lens[r] - 3) // 4, (int(swap.sum()),), generator=g)
        lens = [L + 2 for L in lens]
        idx.copy_(_decode_selection(g, lens, base, rows).to(dev))
        q.copy_(torch.randn(q.shape, generator=g).bfloat16().to(dev))
        dc.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        graph.replay()
        want = qsa_sparse_paged_attention(q, k, v, idx, table, t2r, decode=True, kv_host=True)
        assert torch.equal(out, want), f"replay {call}"


def test_decode_cache_with_a_slab_past_2gb() -> None:
    """K/V views of one storage past 2 GB (as the host slab of every layer is) next to the cache's own buffers: the
    kernels compile (no pointer-range hint on the slab, one on the cache) and give the uncached bits."""
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    g, dev, k0, v0, table, perm, dc = _decode_cache_case(7)
    n = k0.numel()
    slab = torch.empty((1 << 31) // 2 + 2 * n, dtype=torch.bfloat16, device=dev)
    k, v = slab[:n].view(k0.shape), slab[n: 2 * n].view(v0.shape)
    k.copy_(k0)
    v.copy_(v0)
    table = table[dc.ring.long()]
    lens = [15000, 17000]
    base = [torch.randperm(3700, generator=g)[:512] for _ in lens]
    t2r = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), 2).to(dev)
    dc.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
    for call in range(3):
        idx = _decode_selection(g, lens, base, 2).to(dev)
        q = torch.randn(idx.shape[0], 12, 256, generator=g).bfloat16().to(dev)
        want = qsa_sparse_paged_attention(q, k, v, idx, table, t2r, decode=True, kv_host=True)
        got = qsa_sparse_paged_attention(q, k, v, idx, table, t2r, decode=True, kv_host=True, dcache=dc)
        assert torch.equal(got, want), f"call {call}"


def test_decode_cache_after_a_page_repointed() -> None:
    """The prefix cache can point a decoding request's pages at the tree's copy of the same tokens (no prefill in
    between): the cache must then read the new pages, not keep the rows it read from the old ones."""
    from freetoken.kernel.triton.qsa import qsa_sparse_paged_attention

    g, dev, k, v, table, perm, dc = _decode_cache_case(11)
    rows = 2
    lens = [15000, 17000]
    base = [torch.randperm(3700, generator=g)[:512] for _ in lens]
    t2r = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), rows).to(dev)
    dc.seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
    idx = _decode_selection(g, lens, base, rows).to(dev)
    q = torch.randn(idx.shape[0], 12, 256, generator=g).bfloat16().to(dev)
    bt = table[dc.ring.long()]
    qsa_sparse_paged_attention(q, k, v, idx, bt, t2r, decode=True, kv_host=True, dcache=dc)  # fills the cache
    # request 0's first 100 logical pages now point at unused pages holding other K/V (perm[800:] are free)
    spare = perm[800:900]
    table[0, :100] = spare
    k[spare.long()] = torch.randn(100, *k.shape[1:], generator=g).bfloat16().to(dev)
    v[spare.long()] = torch.randn(100, *v.shape[1:], generator=g).bfloat16().to(dev)
    bt = table[dc.ring.long()]
    want = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r, decode=True, kv_host=True)
    got = qsa_sparse_paged_attention(q, k, v, idx, bt, t2r, decode=True, kv_host=True, dcache=dc)
    assert torch.equal(got, want)
