"""Speculative sampling with a deterministic draft emits exactly what plain sampling would (CPU, statistical)."""
from __future__ import annotations

import torch

from freetoken.engine.spec_sample import _distribution, spec_sample

N = 200_000


def _empirical(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.bincount(tokens, minlength=vocab).float() / tokens.numel()


def _target(logits, temp, top_k, top_p, k_max):
    ids, probs = _distribution(logits.unsqueeze(0), temp, top_k, top_p, k_max)
    return torch.zeros(logits.shape[-1]).scatter_add_(0, ids[0], probs[0])


def test_first_token_follows_the_target_whatever_the_draft():
    gen = torch.Generator().manual_seed(0)
    vocab = 8
    row0 = torch.tensor([2.0, 1.5, 0.3, -1.0, 0.8, -2.0, 0.0, 1.0])
    row1 = torch.tensor([-1.0, 0.5, 2.0, 0.1, 0.0, 1.0, -0.5, 0.3])
    for top_k, top_p, k_max in ((None, None, None), (torch.tensor([4]), None, 4), (torch.tensor([5]), torch.tensor([0.8]), 5)):
        for draft in (0, 4, 5):  # the target's favourite, a plausible one, one outside the support
            logits = torch.stack([row0, row1]).expand(N, 2, vocab)
            temp = torch.full((N,), 0.7)
            tk = None if top_k is None else top_k.expand(N)
            tp = None if top_p is None else top_p.expand(N)
            tokens, n_acc = spec_sample(logits, torch.full((N, 1), draft), temp, tk, tp, k_max, gen)
            p0 = _target(row0, torch.tensor([0.7]), top_k, top_p, k_max)
            got = _empirical(tokens[:, 0], vocab)
            assert (got - p0).abs().max() < 0.006, (top_k, top_p, draft, got, p0)
            # the draft is kept exactly when it was emitted first, and then the second token follows p1
            assert torch.equal(n_acc == 2, tokens[:, 0] == draft)
            kept = n_acc == 2
            if kept.sum() > 20_000:
                p1 = _target(row1, torch.tensor([0.7]), top_k, top_p, k_max)
                assert (_empirical(tokens[kept, 1], vocab) - p1).abs().max() < 0.01


def test_chained_drafts_keep_the_marginals():
    gen = torch.Generator().manual_seed(1)
    vocab = 5
    rows = torch.tensor([[1.0, 0.2, -0.5, 0.0, 0.7], [0.1, 1.2, 0.0, -1.0, 0.4], [0.0, 0.0, 1.5, 0.3, -0.2]])
    logits = rows.expand(N, 3, vocab)
    tokens, n_acc = spec_sample(logits, torch.tensor([[0, 1]]).expand(N, 2), torch.ones(N), gen=gen)
    p0 = rows[0].softmax(-1)
    assert (_empirical(tokens[:, 0], vocab) - p0).abs().max() < 0.006
    # given the first draft kept, the second emitted token follows p1
    first_kept = n_acc >= 2
    assert (_empirical(tokens[first_kept, 1], vocab) - rows[1].softmax(-1)).abs().max() < 0.01
    assert int(n_acc.max()) == 3 and int(n_acc.min()) == 1


def test_sampled_drafts_keep_the_target_marginals():
    """Drafts drawn from another distribution q (the head's, under the same top-k / top-p): the emitted tokens still
    follow p, and a draft close to p is accepted far more often than its argmax would be."""
    from freetoken.engine.spec_sample import draft_distribution

    gen = torch.Generator().manual_seed(2)
    vocab = 8
    target = torch.tensor([[1.0, 0.9, 0.8, -1.0, 0.2, -2.0, 0.0, 0.5], [0.3, -0.2, 1.4, 0.1, 0.0, 0.9, -0.5, 0.3]])
    head = torch.tensor([[0.9, 1.1, 0.4, -0.5, 0.3, -1.0, 0.2, 0.1]])  # q != p, overlapping supports
    for top_k, top_p, k_max in ((None, None, None), (torch.tensor([4]), None, 4), (torch.tensor([5]), torch.tensor([0.85]), 5)):
        temp = torch.full((N,), 0.8)
        tk = None if top_k is None else top_k.expand(N)
        tp = None if top_p is None else top_p.expand(N)
        drafts, q = draft_distribution(head.expand(N, vocab), temp, tk, tp, k_max, gen)
        logits = target.expand(N, 2, vocab)
        tokens, n_acc = spec_sample(logits, drafts.unsqueeze(-1), temp, tk, tp, k_max, gen, q=q.unsqueeze(1))
        p0 = _target(target[0], torch.tensor([0.8]), top_k, top_p, k_max)
        assert (_empirical(tokens[:, 0], vocab) - p0).abs().max() < 0.006, (top_k, top_p)
        kept = n_acc == 2
        assert torch.equal(kept, tokens[:, 0] == drafts)
        p1 = _target(target[1], torch.tensor([0.8]), top_k, top_p, k_max)
        assert (_empirical(tokens[kept, 1], vocab) - p1).abs().max() < 0.01
        # acceptance = sum min(p, q) > p(argmax q), the deterministic draft's acceptance
        q0 = q[0]
        assert kept.float().mean() > p0[int(q0.argmax())] + 0.1


def test_support_is_the_samplers_own():
    """With ties the support is the one of the sampler that draws such a request: the top-k-first kernel for a small
    top_k (exactly k candidates, kept while the mass ahead of them is below top_p), the sorted threshold of the
    full-vocabulary path otherwise (ties at the k-th value and at the top-p cut kept)."""
    from freetoken.kernel.triton.sampling import _sorted_threshold

    probs = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1]])
    logits, temp = probs.log(), torch.ones(1)
    for top_k, top_p in ((None, 0.5), (None, 0.7), (2, None), (2, 0.7), (4, 0.5), (None, 1.0)):
        tk = None if top_k is None else torch.tensor([top_k])
        tp = None if top_p is None else torch.tensor([top_p])
        ids, got = _distribution(logits, temp, tk, tp, None)  # full vocabulary
        thr = _sorted_threshold(probs, tk, tp)
        want = set(torch.nonzero(probs[0] >= thr[0]).flatten().tolist())
        assert set(ids[0][got[0] > 0].tolist()) == want, (top_k, top_p)
        if top_k is not None:  # top-k-first: the kernel's rule over its k candidates, in their order
            ids, got = _distribution(logits, temp, tk, tp, top_k)
            p = torch.softmax(logits[0, ids[0]], -1)
            keep = [i == 0 or top_p is None or float(p[:i].sum()) < top_p for i in range(top_k)]
            assert [bool(x) for x in got[0] > 0] == keep, (top_k, top_p)
