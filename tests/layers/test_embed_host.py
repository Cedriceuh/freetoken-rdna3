"""FREETOKEN_EMBED_HOST: the indexing kernel reads an input table kept in mapped host memory with the bits of the VRAM
table (whole table and a TP shard's vocab range), and only untied input tables move (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("vocab_range", [None, (600, 400)])
def test_indexing_a_mapped_host_table_same_bits(vocab_range):
    from freetoken.kernel import indexing
    from freetoken.kernel.host_mapped import host_mapped_empty

    g = torch.Generator(device="cuda").manual_seed(7)
    rows = 1000 if vocab_range is None else vocab_range[1]
    table = torch.randn(rows, 2560, device="cuda", generator=g).to(torch.bfloat16)
    table[3] = -0.0  # signed zeros cross unchanged
    host = host_mapped_empty(tuple(table.shape), table.dtype, table.device)
    host.copy_(table)
    ids = torch.randint(0, 1000, (4099,), device="cuda", generator=g, dtype=torch.int32)
    ids[:5] = torch.tensor([3, 603, 0, 999, 600], dtype=torch.int32)
    a = indexing(weights=table, indices=ids, vocab_range=vocab_range)
    b = indexing(weights=host, indices=ids, vocab_range=vocab_range)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_only_untied_input_tables_move(monkeypatch):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers import BaseOP, ParallelLMHead, VocabParallelEmbedding
    from freetoken.layers import embedding as E

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(E, "_EMBED_HOST", True)

    class _Root(BaseOP):
        def __init__(self):
            self.embed_tokens = VocabParallelEmbedding(1000, 256)
            self.tied = VocabParallelEmbedding(1000, 256)
            self.lm_head = ParallelLMHead(1000, 256, tie_word_embeddings=True, tied_embedding=self.tied)

    root = _Root()
    g = torch.Generator(device="cuda").manual_seed(1)
    for op in (root.embed_tokens, root.tied):
        op.weight = torch.randn(1000, 256, device="cuda", generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 1000, (513,), device="cuda", generator=g, dtype=torch.int32)
    before = root.embed_tokens.forward(ids)
    tied_weight = root.tied.weight
    allocated = torch.cuda.memory_allocated()
    assert E.place_embeddings_on_host(root) == 1000 * 256 * 2
    assert root.tied.weight is tied_weight
    assert torch.cuda.memory_allocated() == allocated - 1000 * 256 * 2  # the table left the torch allocator
    assert root.embed_tokens.weight.is_cuda
    assert torch.equal(root.embed_tokens.forward(ids), before)
