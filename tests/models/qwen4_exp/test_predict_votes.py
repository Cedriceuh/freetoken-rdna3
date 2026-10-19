"""FREETOKEN_PREFILL_PREDICT_MIN_VOTES: the experts copied ahead are those at least N rows of the prediction name."""

import torch

from freetoken.models.qwen4_exp.moe import _voted


def test_voted_keeps_the_experts_named_often_enough():
    predicted = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 5, 6], [2, 7, 3]])
    for k, want in ((1, {1, 2, 3, 4, 5, 6, 7}), (2, {2, 3}), (3, {2, 3}), (4, {3})):
        got = _voted(predicted, 8, k)
        assert got.shape == (8, 1)
        assert set(got.view(-1).tolist()) == want, k


def test_voted_names_one_expert_when_none_is_named_enough():
    got = _voted(torch.tensor([[0, 1], [2, 3]]), 4, 2)
    assert len(set(got.view(-1).tolist())) == 1
