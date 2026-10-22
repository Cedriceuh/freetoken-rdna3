"""FREETOKEN_SPARSE_LOGITS: a verify step's sparse draft distributions hold the same probabilities as the dense ones
(the drafts' q kept as (ids, probs), point masses for argmax drafts, narrower q padded)."""

from types import SimpleNamespace

import torch

from freetoken.engine.engine import Engine
from freetoken.engine.spec_sample import _q_at


def test_sparse_draft_q_matches_dense():
    vocab, m1 = 50, 3
    g = torch.Generator().manual_seed(0)
    drafts = torch.randint(0, vocab, (3, m1), generator=g)

    def sparse_q(width):
        ids = torch.stack([torch.randperm(vocab, generator=g)[:width] for _ in range(m1)])
        probs = torch.softmax(torch.randn(m1, width, generator=g), -1)
        return ids, probs

    reqs = [SimpleNamespace(spec_q=sparse_q(5)), SimpleNamespace(spec_q=None), SimpleNamespace(spec_q=sparse_q(3))]
    batch = SimpleNamespace(reqs=reqs)
    q_sparse = Engine._draft_q(batch, drafts, vocab, sparse=True)
    assert isinstance(q_sparse, tuple) and q_sparse[0].shape == (3, m1, 5)
    q_dense = Engine._draft_q(batch, drafts, vocab, sparse=False)
    assert q_dense.shape == (3, m1, vocab)
    at = torch.arange(vocab).expand(3, m1, vocab)
    assert torch.equal(_q_at(q_sparse, at), q_dense)
    assert torch.equal(_q_at(q_sparse, drafts.unsqueeze(-1)), q_dense.gather(-1, drafts.unsqueeze(-1)))
