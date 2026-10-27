"""FREETOKEN_SPARSE_LOGITS: the k best of every rank's vocabulary shard, merged, are what torch.topk and torch.argmax
return for the whole row (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None,
                                reason="needs a ROCm GPU (the bits checked are ROCm's)")


def _merged(full: torch.Tensor, tp: int, k: int):
    """What ParallelLMHead.candidates computes, the ranks' shards on one GPU and the all-gather a concatenation."""
    from freetoken.kernel.triton.vocab_topk import merge_candidates, pack_candidates

    vocab = full.shape[1]
    shard = -(-vocab // tp)
    packed = []
    for r in range(tp):
        width = min(shard, vocab - r * shard)
        local = torch.zeros(full.shape[0], shard, dtype=full.dtype, device=full.device)  # padded columns past width
        local[:, :width] = full[:, r * shard: r * shard + width]
        packed.append(pack_candidates(local, k, r * shard, width))
    return merge_candidates(torch.cat(packed), tp, k, full.shape[0])


@pytest.mark.parametrize("tp,k,vocab,rows", [(2, 1, 248320, 4), (2, 20, 248320, 8), (2, 256, 248320, 3),
                                             (2, 1, 1001, 5), (3, 40, 30011, 6), (4, 7, 4099, 2)])
def test_candidates_are_the_rows_top_k(tp, k, vocab, rows):
    g = torch.Generator(device="cuda").manual_seed(tp * 1000 + k)
    full = torch.randn(rows, vocab, generator=g, device="cuda") * 4 - 20  # all negative: padded zeros must not win
    vals, ids, amax = _merged(full, tp, k)
    ref_vals, ref_ids = torch.topk(full, k, dim=-1)
    assert torch.equal(vals, ref_vals) and torch.equal(ids, ref_ids)
    assert torch.equal(amax, full.argmax(-1))


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("k", [1, 20])
def test_uneven_shards_with_equal_values(rows, k):
    """FREETOKEN_LMHEAD_SPLIT's shards (136,576 / 111,744 rows at 0.55), bf16 values (dozens equal at the top, within
    and across the shards) in fp32 as the LM head's logits reach the sampling."""
    from freetoken.kernel.triton.vocab_topk import merge_candidates, pack_candidates

    g = torch.Generator(device="cuda").manual_seed(rows * 100 + k)
    full = (torch.randn(rows, 248320, generator=g, device="cuda") * 0.3 + 20).to(torch.bfloat16).float()
    widths, start, packed = (136576, 111744), 0, []
    for width in widths:
        local = torch.zeros(rows, max(widths), dtype=full.dtype, device="cuda")  # gathered at the widest shard
        local[:, :width] = full[:, start:start + width]
        packed.append(pack_candidates(local, k, start, width))
        start += width
    vals, ids, amax = merge_candidates(torch.cat(packed), 2, k, rows)
    ref_vals, ref_ids = torch.topk(full, k, dim=-1)
    assert torch.equal(vals, ref_vals) and torch.equal(ids, ref_ids)
    assert torch.equal(amax, full.argmax(-1))


@pytest.mark.parametrize("k", [1, 2, 7, 20, 64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 5, 16])
def test_equal_bf16_logits_come_in_topks_order(k, dtype, rows):
    """The int8 LM head returns bf16 logits: many candidates share a value. The same values and ids in the same order
    as torch.topk of the whole rows (the order sampling draws in; one row orders them otherwise than several), and the
    lowest id among the maxima as argmax."""
    g = torch.Generator(device="cuda").manual_seed(k * 31 + rows)
    for trial in range(8):
        spread = (3.0, 1.0, 0.3)[trial % 3]  # 0.3: dozens of equal values at the top
        full = (torch.randn(rows, 248320, generator=g, device="cuda") * spread + 20).to(torch.bfloat16).to(dtype)
        vals, ids, amax = _merged(full, 2, k)
        ref_vals, ref_ids = torch.topk(full.float(), k, dim=-1)
        assert torch.equal(vals, ref_vals) and torch.equal(ids, ref_ids)
        assert torch.equal(amax, full.argmax(-1))


@pytest.mark.parametrize("k", [1, 3, 20])
def test_equal_maxima_resolve_to_the_lowest_id_like_argmax(k):
    """k = 1 takes each shard's argmax: any number of equal maxima; k > 1 each shard's torch.topk: exact while no
    shard holds more than k values equal to its largest (rows below hold at most 3)."""
    vocab, tp = 248320, 2
    full = torch.randn(6, vocab, device="cuda", generator=torch.Generator(device="cuda").manual_seed(k))
    full[0, [5, 200000]] = 50.0  # across the two shards
    full[1, [130000, 124160]] = 50.0  # both in the second shard
    full[2, [7, 3, 124159]] = 50.0  # three in the first shard, its last column among them
    full[3] = -full[3].abs() - 1.0
    full[3, 9], full[3, 10] = -0.0, 0.0  # -0.0 and 0.0 compare equal: the lower id
    full[4, [100, 101, 124160 + 100]] = float("inf")
    full[5, :] = 1.0  # every value equal
    vals, ids, amax = _merged(full, tp, k)
    check = [0, 1, 3, 4] + ([2] if k >= 3 else []) + ([5] if k == 1 else [])
    assert torch.equal(amax[check], full.argmax(-1)[check])
    assert torch.equal(vals[check, 0], full.max(-1).values[check])


def test_sampling_from_candidates_draws_what_the_whole_rows_draw():
    """spec_sample and draft_distribution on the candidates: the same tokens, kept counts and distributions as on the
    whole rows, for the same generator state, with bf16 logits (equal values among the candidates)."""
    from freetoken.engine.spec_sample import draft_distribution, spec_sample

    vocab, tp, reqs, m = 248320, 2, 3, 4
    gen = torch.Generator(device="cuda").manual_seed(7)
    temps_all = torch.tensor([0.6, 1.0, 0.3], device="cuda")
    top_k_all = torch.tensor([20, 7, 20], dtype=torch.int32, device="cuda")
    top_p_all = torch.tensor([0.95, 1.0, 0.8], device="cuda")
    kmax = 20
    bf = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
    for trial in range(6):
        head = bf(torch.randn(reqs, vocab, generator=gen, device="cuda") * 0.5 + 12)
        vals, ids, _ = _merged(head, tp, kmax)
        if trial % 2:  # one request: one row
            head, vals, ids = head[:1], *_merged(head[:1], tp, kmax)[:2]
            temps, top_k, top_p = temps_all[:1], top_k_all[:1], top_p_all[:1]
        else:
            temps, top_k, top_p = temps_all, top_k_all, top_p_all
        g1 = torch.Generator(device="cuda").manual_seed(trial)
        g2 = torch.Generator(device="cuda").manual_seed(trial)
        t_full, q_full = draft_distribution(head, temps, top_k, top_p, kmax, gen=g1)
        t_c, (q_ids, q_probs) = draft_distribution(None, temps, top_k, top_p, kmax, gen=g2, cands=(vals, ids, vocab),
                                                  fused=False)
        assert torch.equal(t_full, t_c)
        assert torch.equal(torch.zeros_like(q_full).scatter_(-1, q_ids, q_probs), q_full)

        temps, top_k, top_p = temps_all, top_k_all, top_p_all
        rows = bf(torch.randn(reqs, m, vocab, generator=gen, device="cuda") * 0.5 + 10)
        rows[:, :, :50] += 3  # a peaked head so drafts get accepted and rejected
        drafts = torch.randint(0, 50, (reqs, m - 1), generator=gen, device="cuda")
        qd = torch.softmax(torch.randn(reqs, m - 1, vocab, generator=gen, device="cuda"), -1)
        q_top = torch.topk(qd, kmax, dim=-1)
        q_dense = torch.zeros_like(qd).scatter_(-1, q_top.indices, q_top.values)
        rv, ri, _ = _merged(rows.reshape(reqs * m, vocab), tp, kmax)
        for q_arg, q_sparse in ((None, None), (q_dense, (q_top.indices, q_top.values))):
            g1 = torch.Generator(device="cuda").manual_seed(100 + trial)
            g2 = torch.Generator(device="cuda").manual_seed(100 + trial)
            ref = spec_sample(rows, drafts, temps, top_k, top_p, kmax, gen=g1, q=q_arg)
            got = spec_sample(None, drafts, temps, top_k, top_p, kmax, gen=g2, q=q_sparse,
                              cands=(rv.view(reqs, m, kmax), ri.view(reqs, m, kmax), vocab), fused=False)
            assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


@pytest.mark.parametrize("tp,k,rows", [(1, 1, 4), (1, 20, 3), (2, 20, 8), (2, 1, 1), (2, 256, 2)])
def test_mapped_columns_give_the_reduced_rows_top_k(tp, k, rows):
    """FREETOKEN_DRAFT_VOCAB: columns of ascending, not contiguous, ids (``ids_map``) merge to torch.topk of the row
    restricted to those ids, ties toward the lowest id (bf16 values: many equal), the ids the global ones."""
    from freetoken.kernel.triton.vocab_topk import merge_candidates, pack_candidates

    g = torch.Generator(device="cuda").manual_seed(tp * 100 + k)
    vocab = 248320
    keep = torch.cat([torch.arange(98304, device="cuda"), torch.arange(248044, 248077, device="cuda")])
    full = (torch.randn(rows, vocab, generator=g, device="cuda") * 3).to(torch.bfloat16).float()
    sub = full[:, keep]
    bounds = [0, keep.numel()] if tp == 1 else [0, 54000, keep.numel()]
    packed = [pack_candidates(sub[:, a:b].contiguous(), k, 0, b - a, ids_map=keep[a:b].to(torch.int32))
              for a, b in zip(bounds, bounds[1:])]
    vals, ids, amax = merge_candidates(torch.cat(packed), tp, k, rows)
    ref_vals, ref_pos = torch.topk(sub, k, dim=-1)
    assert torch.equal(vals, ref_vals)
    assert torch.equal(ids, keep[ref_pos])
    assert torch.equal(amax, keep[sub.argmax(-1)])


def test_draft_head_scores_its_rows_with_the_lm_heads_int8_gemv(tmp_path):
    """build_draft_head on one rank: the first n rows and the tokenizer's added ones, scored by the LM head's own int8
    GEMV on those rows; its candidates are the global ids of the reduced row's top k."""
    import json
    from types import SimpleNamespace

    from freetoken.distributed import set_tp_info
    from freetoken.kernel.triton.int8_gemv import gemv_int8
    from freetoken.layers.draft_vocab import build_draft_head

    set_tp_info(0, 1)
    g = torch.Generator(device="cuda").manual_seed(7)
    vocab, dim, n = 4096, 256, 1000
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 4090}, {"id": 4093}]}))
    weight = torch.randint(-127, 128, (vocab, dim), generator=g, device="cuda", dtype=torch.int8)
    scale = torch.rand(vocab, generator=g, device="cuda") * 0.01
    lm_head = SimpleNamespace(weight=weight, weight_scale=scale, num_embeddings=vocab, tp_size=1, vocab_shards=None,
                              vocab_range=(0, vocab), _comm=None)
    head = build_draft_head(lm_head, str(tmp_path), None, n=n)
    keep = torch.tensor([*range(n), 4090, 4093], device="cuda")
    assert torch.equal(head.ids.long(), keep)
    x = (torch.randn(2, dim, generator=g, device="cuda")).to(torch.bfloat16)
    logits = head.forward(x)
    assert torch.equal(logits, gemv_int8(x, weight[keep], scale[keep]))
    vals, ids, amax = head.candidates(logits, 5, 2)
    ref_vals, ref_pos = torch.topk(logits.float(), 5, dim=-1)
    assert torch.equal(ids, keep[ref_pos]) and torch.equal(amax, keep[logits.float().argmax(-1)])
