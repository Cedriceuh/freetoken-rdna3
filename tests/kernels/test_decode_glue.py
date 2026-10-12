"""Decode glue in kernels: each one gives the bits of the torch ops it replaces (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _plan_reference(out_loc, positions, token_to_req, ring_slots, cu_seqlens, ratio, cap, scratch_base, rope, table,
                    section, cache):
    """QSASparseAttnBackend._plan_index_writes as torch ops."""
    from freetoken.layers.rotary import mrope_cos_sin_rows

    out_loc, pos = out_loc.to(torch.int64), positions.to(torch.int64)
    rp = (rope if rope.dim() == 2 else rope.unsqueeze(0).expand(3, -1)).to(torch.int32)
    table.index_copy_(0, out_loc, rp.t().contiguous())
    first = table.index_select(0, out_loc - out_loc % ratio).t()
    q, k = mrope_cos_sin_rows(cache, rp, section), mrope_cos_sin_rows(cache, first, section)
    rows = torch.arange(out_loc.numel(), device=out_loc.device)
    req = token_to_req.to(torch.int64)
    slots = ring_slots.to(torch.int64).index_select(0, req)
    cmp = torch.where(out_loc % ratio == ratio - 1, out_loc // ratio, scratch_base + slots).to(torch.int32)
    keep = rows >= cu_seqlens.to(torch.int64).index_select(0, req + 1) - cap
    ring = torch.where(keep, slots * cap + pos % cap, torch.full_like(slots, -1)).to(torch.int32)
    return cmp, ring, torch.arange(out_loc.numel(), dtype=torch.int32, device=out_loc.device), q, k


@pytest.mark.parametrize("seed", range(24))
def test_qsa_plan_kernel_matches_torch(seed):
    from freetoken.kernel.triton.qsa.plan import qsa_plan_index_writes
    from freetoken.layers.rotary import build_section_table, rope_cos_sin_table

    dev = torch.device("cuda")
    section = build_section_table((11, 11, 10), "interleaved").to(dev)
    cache = rope_cos_sin_table(128, 64, 300000, 10000000.0, None).to(dev)
    g = torch.Generator().manual_seed(seed)
    bs, m = 1 + seed % 4, 1 + seed % 4
    lens = torch.randint(1, 300, (bs,), generator=g) if seed % 3 == 0 else torch.full((bs,), m)  # a prefill or a decode
    T = int(lens.sum())
    cu = torch.cat([torch.zeros(1, dtype=torch.int64), lens.cumsum(0)]).to(torch.int32)
    t2r = torch.repeat_interleave(torch.arange(bs), lens).to(torch.int32)
    slots = torch.randperm(64, generator=g)[:bs].to(torch.int32)
    out = torch.randperm(200000, generator=g)[:T].to(torch.int32)
    pos = torch.randint(0, 250000, (T,), generator=g).to(torch.int32)
    rope = torch.randint(0, 250000, (3, T), generator=g).to(torch.int32) if seed % 2 else pos.clone()
    args = [x.to(dev) for x in (out, pos, t2r, slots, cu)]
    t1 = torch.randint(0, 250000, (200064, 3), generator=g).to(torch.int32).to(dev)
    t2 = t1.clone()
    ref = _plan_reference(*args, 8, 64, 200000, rope.to(dev), t1, section, cache)
    got = qsa_plan_index_writes(*args, 8, 64, 200000, rope.to(dev), t2, section, cache)
    assert all(torch.equal(a, b) for a, b in zip(ref, got)) and torch.equal(t1, t2)


@pytest.mark.parametrize("B,m", [(1, 2), (1, 3), (1, 4), (2, 4), (4, 2)])
@pytest.mark.parametrize("seed", range(6))
def test_ple_spec_conv_kernel_matches_torch(B, m, seed):
    from freetoken.models.qwen4_exp.ple_conv_kernel import ple_spec_conv

    dev = torch.device("cuda")
    C, L, K, dil = 10240, 9, 4, 3
    g = torch.Generator(device=dev).manual_seed(seed)
    state = torch.randn(B, C, L, generator=g, device=dev).to(torch.bfloat16)
    rows = torch.randn(B, m, C, generator=g, device=dev).to(torch.bfloat16)
    weight = torch.randn(C, K, generator=g, device=dev)
    slots = torch.randperm(8, device=dev)[:B]
    states0 = torch.randn(8, C, L, generator=g, device=dev).to(torch.bfloat16)
    s1, n1 = states0.clone(), torch.zeros(3, 4, C, L, dtype=torch.bfloat16, device=dev)
    st, outs = state, []
    for j in range(m):  # PLELayer._spec_decode_conv
        column = rows[:, j].unsqueeze(-1)
        outs.append((torch.cat([st[..., ::dil], column], dim=-1).float() * weight).sum(-1))
        st = torch.cat([st[..., 1:], column], dim=-1)
        if j < m - 1:
            n1[j, :B].copy_(st)
    s1.index_copy_(0, slots, st)
    ref = torch.stack(outs, dim=1).reshape(B * m, -1).to(torch.bfloat16)
    s2, n2 = states0.clone(), torch.zeros(3, 4, C, L, dtype=torch.bfloat16, device=dev)
    got = ple_spec_conv(state, rows, weight, n2, s2, slots, dil)
    assert torch.equal(ref, got) and torch.equal(s1, s2) and torch.equal(n1[: m - 1, :B], n2[: m - 1, :B])


@pytest.mark.parametrize("seed", range(16))
def test_rollback_copy_matches_apply_rollback(seed, monkeypatch):
    import freetoken.spec_decode as sd
    from freetoken.spec_decode import RollbackTarget

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    B, m, S, L, n_slots, pad = 1 + seed % 4, 2 + seed % 3, 4, 3, 9, 8
    inter = torch.randn(L, B, S - 1, 5, 6, 7, generator=g, device=dev)
    snaps = torch.randn(L, S - 1, B, 11, 3, generator=g, device=dev).to(torch.bfloat16)
    buf = torch.randint(0, 1000, (B, 2 + S), generator=g, device=dev)
    psn = torch.randn(S - 1, B, 13, 9, generator=g, device=dev).to(torch.bfloat16)
    states0 = (torch.randn(L, n_slots, 5, 6, 7, generator=g, device=dev),
               torch.randn(L, n_slots, 11, 3, generator=g, device=dev).to(torch.bfloat16),
               torch.randint(0, 1000, (n_slots, 2), generator=g, device=dev),
               torch.randn(n_slots, 13, 9, generator=g, device=dev).to(torch.bfloat16))
    cols = torch.arange(2, device=dev)

    def targets(rec, conv, ctx, pst):  # the four of a Qwen3.8 verify step, as gdn.py and ple.py register them
        return {
            "r": RollbackTarget(rec, lambda step, b: inter[:, b, step], dim=1, copy=(
                inter, inter.stride(0), inter.stride(1), inter.stride(2), 0, rec.stride(0), rec.stride(1), L, 210)),
            "c": RollbackTarget(conv, lambda step, b: snaps[:, step, b], dim=1, copy=(
                snaps, snaps.stride(0), snaps.stride(2), snaps.stride(1), 0, conv.stride(0), conv.stride(1), L, 33)),
            "x": RollbackTarget(ctx, lambda step, b: buf[b].gather(1, (step + 1).unsqueeze(1) + cols), copy=(
                buf, 0, buf.stride(0), buf.stride(1), 1, 0, ctx.stride(0), 1, 2)),
            "p": RollbackTarget(pst, lambda step, b: psn[step, b], copy=(
                psn, 0, psn.stride(1), psn.stride(0), 0, 0, pst.stride(0), 1, 117)),
        }

    slots = torch.randperm(n_slots - 1, device=dev)[:B].to(torch.int32) + 1
    n_acc = torch.randint(1, m + 1, (B,), generator=g, device=dev)
    ref, got = [t.clone() for t in states0], [t.clone() for t in states0]
    monkeypatch.setattr(sd, "_TARGETS", targets(*ref))
    monkeypatch.setattr(sd, "SPEC_GLUE_KERNELS", False)
    sd.apply_rollback(slots, n_acc, pad, m)
    monkeypatch.setattr(sd, "_TARGETS", targets(*got))
    monkeypatch.setattr(sd, "SPEC_GLUE_KERNELS", True)
    sd.apply_rollback(slots, n_acc, pad, m)
    # the padding slot takes the writes of every request that kept all its rows (in no set order): skip it
    other = torch.tensor([i for i in range(n_slots) if i != pad], device=dev)
    for a, b, dim in zip(ref, got, (1, 1, 0, 0)):
        assert torch.equal(a.index_select(dim, other), b.index_select(dim, other))


@pytest.mark.parametrize("seed", range(16))
def test_spec_glue_kernels_match_torch(seed):
    from freetoken.kernel.triton.spec_glue import chain_meta, draft_rows, greedy_n_acc, write_step_tokens

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    R, m, k, W, depth = 1 + seed % 4, 1 + seed % 4, seed % 4, 50, 1 + seed % 3
    # token-pool writes (the scheduler after a verify step); padded rows (pos < 0) write the junk last column
    pool = torch.randint(0, 9, (8, W), generator=g, device=dev, dtype=torch.int32)
    rows = torch.randperm(8, device=dev)[:R]
    pos = torch.randint(-1, 40, (R,), generator=g, device=dev)
    toks = torch.randint(0, 100000, (R, m), generator=g, device=dev, dtype=torch.int32)
    drafts = torch.randint(0, 100000, (R, k), generator=g, device=dev, dtype=torch.int32) if k else None
    n_acc = torch.randint(1, m + 1, (R,), generator=g, device=dev)
    p1, p2 = pool.clone(), pool.clone()
    cols = torch.where(pos.unsqueeze(1) >= 0, pos.unsqueeze(1) + torch.arange(m, device=dev), -1)
    p1[(rows.unsqueeze(1).expand(-1, m), cols)] = toks
    if drafts is not None:
        dcols = (pos + n_acc).unsqueeze(1) + torch.arange(k, device=dev)
        p1[(rows.unsqueeze(1).expand(-1, k), torch.where(pos.unsqueeze(1) >= 0, dcols.clamp(max=W - 1), -1))] = drafts
    write_step_tokens(p2, rows, pos, toks, drafts, n_acc)
    assert torch.equal(p1[:, : W - 1], p2[:, : W - 1])
    # kept counts of greedy rows
    tokens = torch.randint(0, 4, (R, m), generator=g, device=dev)
    ids = torch.randint(0, 4, (R * m,), generator=g, device=dev, dtype=torch.int32)
    ref = 1 + (tokens[:, :-1] == ids.view(R, m)[:, 1:].to(tokens.dtype)).to(torch.int64).cumprod(-1).sum(-1)
    assert torch.equal(greedy_n_acc(tokens, ids, m), ref)
    # the MTP head's rows and the chained guesses' positions
    positions = torch.randint(0, 100000, (R * m,), generator=g, device=dev, dtype=torch.int32)
    pick, chain = draft_rows(positions, n_acc, m, depth)
    assert torch.equal(pick, torch.arange(R, device=dev) * m + n_acc - 1)
    last = positions.view(R, m)[:, 0].to(torch.int64) + n_acc - 1
    assert all(torch.equal(chain[:, j], last + j) for j in range(1, depth))
    # a chained guess's addressing
    pt = torch.randint(0, 10 ** 6, (6, 300), generator=g, device=dev, dtype=torch.int32)
    tidx = torch.randint(0, 6, (R,), generator=g, device=dev, dtype=torch.int32)
    cpos = torch.randint(0, 300, (R,), generator=g, device=dev)
    delta = torch.randint(-50, 50, (R,), generator=g, device=dev, dtype=torch.int32) if seed % 2 else None
    p32, seq_lens, out_loc, mrope = chain_meta(cpos, tidx, pt, delta)
    assert torch.equal(p32, cpos.to(torch.int32)) and torch.equal(seq_lens, cpos.to(torch.int32) + 1)
    assert torch.equal(out_loc, pt[tidx.to(torch.int64), cpos])
    if delta is not None:
        assert torch.equal(mrope, (cpos.to(torch.int32) + delta).unsqueeze(0).expand(3, -1).contiguous())
