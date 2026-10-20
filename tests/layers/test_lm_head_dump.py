"""FREETOKEN_DUMP_PROMPT_TOPK: the prompt top-k dump of a tied LM head saves the log-probabilities of the logits
forward computes (CPU)."""

from types import SimpleNamespace

import torch


def test_tied_lm_head_dumps_the_logits_forward_computes(monkeypatch, tmp_path):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers import ParallelLMHead, VocabParallelEmbedding
    from freetoken.layers import embedding as E

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    tied = VocabParallelEmbedding(1000, 64)
    tied.weight = torch.randn(1000, 64, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    head = ParallelLMHead(1000, 64, tie_word_embeddings=True, tied_embedding=tied)
    x = torch.randn(70, 64, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)  # two 64-row parts
    monkeypatch.setattr(E, "_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(E, "_DUMP_COUNT", 0)
    monkeypatch.setattr(E, "get_global_ctx", lambda: SimpleNamespace(batch=SimpleNamespace(input_ids=torch.arange(70))))
    head._dump_prompt_topk(x, k=5)
    saved = torch.load(tmp_path / "prompt-0.pt")
    want = torch.log_softmax(tied.quant_method.logits(tied, x, None).float(), dim=-1).topk(5, dim=-1)
    assert torch.equal(saved["ids"], want.indices) and torch.equal(saved["lp"], want.values)
