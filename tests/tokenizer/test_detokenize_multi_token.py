"""A speculative verify step sends several tokens of one request in one batch: the streamed pieces must add up to
the text, with no piece repeated (each repeated uid is decoded from the state the previous message left)."""
from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager

VOCAB = {1: "Ce", 2: " doc", 3: "ument", 4: " est", 5: " la", 6: " ré", 7: "férence"}


class _Tok:
    eos_token_id = 0

    def batch_decode(self, ids):
        return ["".join(VOCAB[i] for i in row) for row in ids]


def _stream(batches):
    det = DetokenizeManager(_Tok(), frozenset({0}))
    pieces = {}
    for batch in batches:
        msgs = [DetokenizeMsg(uid=u, next_token=t, finished=False) for u, t in batch]
        for (u, _), text in zip(batch, det.detokenize(msgs)):
            pieces.setdefault(u, []).append(text)
    return {u: "".join(p) for u, p in pieces.items()}


def test_two_tokens_of_a_request_in_one_batch():
    one_by_one = _stream([[(7, t)] for t in (1, 2, 3, 4, 5, 6, 7)])
    paired = _stream([[(7, 1), (7, 2)], [(7, 3)], [(7, 4), (7, 5)], [(7, 6), (7, 7)]])
    assert paired == one_by_one == {7: "Ce document est la référence"}


def test_interleaved_requests_keep_their_own_order():
    out = _stream([[(1, 1), (2, 4), (1, 2), (2, 5)], [(1, 3), (2, 6), (2, 7)]])
    assert out == {1: "Ce document", 2: " est la référence"}
