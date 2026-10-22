"""FREETOKEN_LMHEAD_SPLIT: the LM head's logits come out with the same bits whatever the ranks' row shares, the
int8 GEMV taking the even head's K tiles for any shard height (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_vocab_shards_follow_the_tp_split(monkeypatch):
    import freetoken.distributed.split as split

    monkeypatch.setenv("FREETOKEN_TP_SPLIT", "0.55")
    monkeypatch.setattr(split, "LMHEAD_SPLIT", True)
    assert split.vocab_shards(248320, 2) == [(0, 136576), (136576, 111744)]
    monkeypatch.setattr(split, "LMHEAD_SPLIT", False)
    assert split.vocab_shards(248320, 2) is None


def test_only_registered_shards_share_the_heads_tiles():
    """A TP=1 head (248,320 rows) keeps its own tiles: only the shapes an uneven head registers take the even head's."""
    from freetoken.kernel.triton import int8_gemv as ig

    full = ig._config(248320, 2560)
    assert full == ig._default_config(248320, 2560)
    ig.share_config((136576, 2560), (124160, 2560))
    assert ig._config(136576, 2560) == ig._CONFIGS[(124160, 2560)]
    assert ig._config(248320, 2560) == full


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_lm_head_logits_do_not_depend_on_the_split(rows):
    from freetoken.kernel.triton.int8_gemv import gemv_int8, share_config

    share_config((136576, 2560), (124160, 2560))
    share_config((111744, 2560), (124160, 2560))

    g = torch.Generator(device="cuda").manual_seed(rows)
    vocab, hidden = 248320, 2560
    w = torch.randint(-127, 128, (vocab, hidden), generator=g, device="cuda", dtype=torch.int8)
    scale = torch.rand(vocab, generator=g, device="cuda") * 1e-3
    x = torch.randn(rows, hidden, generator=g, device="cuda").to(torch.bfloat16)
    def logits(cut):
        return torch.cat([gemv_int8(x, w[a:b].contiguous(), scale[a:b].contiguous()) for a, b in cut], dim=1)
    even = logits([(0, 124160), (124160, vocab)])
    assert torch.equal(logits([(0, 136576), (136576, vocab)]), even)
    assert torch.equal(logits([(0, 111744), (111744, vocab)]), even)
