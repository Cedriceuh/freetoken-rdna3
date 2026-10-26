"""FREETOKEN_STEP_INPUTS: the one kernel that writes a graph decode step's inputs (kernel/triton/step_inputs.py) gives
the values of the torch ops it replaces (the scheduler's gathers, GraphCaptureBuffer.copy_from, QSA's _stage_decode)."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("bs,m", [(1, 1), (1, 4), (2, 3), (3, 2), (4, 4)])
@pytest.mark.parametrize("mrope", [False, True])
def test_decode_inputs_match_the_torch_ops(bs, m, mrope):
    from freetoken.kernel.triton.step_inputs import decode_inputs

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(bs * 10 + m)
    slots, width, page = 6, 4160, 64
    token_pool = torch.randint(0, 150000, (slots, width), generator=g, device=dev, dtype=torch.int32)
    page_table = torch.randint(-1, 10 ** 6, (slots, width), generator=g, device=dev, dtype=torch.int32)
    rows = torch.randperm(slots, device=dev)[:bs]
    R = bs * m
    pos = (torch.randint(0, width - m, (bs, 1), generator=g, device=dev) + torch.arange(m, device=dev)).view(-1)
    pos = pos.to(torch.int32)
    lin = torch.randint(0, 50, (bs,), generator=g, device=dev, dtype=torch.int32)
    kv = torch.randint(1, width, (bs,), generator=g, device=dev, dtype=torch.int32)
    rope = torch.randint(0, 10 ** 5, (3, R), generator=g, device=dev, dtype=torch.int32)
    # the arena: positions, mrope, slots, K/V lengths (int32 words), then the rows (int64 words, 8-byte aligned)
    off_pos, off_mrope, off_lin, off_kv = 0, R, 4 * R, 4 * R + bs
    off_map = (4 * R + 2 * bs + 1) // 2
    arena = torch.zeros(2 * off_map + 2 * bs, dtype=torch.int32, device=dev)
    arena[off_pos: off_pos + R] = pos
    arena[off_mrope: off_mrope + 3 * R] = rope.view(-1)
    arena[off_lin: off_lin + bs] = lin
    arena[off_kv: off_kv + bs] = kv
    arena.view(torch.int64)[off_map: off_map + bs] = rows
    big = 8  # the graph buffers are sized for more requests than this step's
    input_ids = torch.full((big * 4,), -7, dtype=torch.int32, device=dev)
    out_loc, positions = input_ids.clone(), input_ids.clone()
    mrope_buf = torch.full((3 * big * 4,), -7, dtype=torch.int32, device=dev) if mrope else None
    lin_buf = torch.full((big,), -7, dtype=torch.int32, device=dev)
    pages = -(-width // page)
    block_table = torch.full((big, pages), -7, dtype=torch.int32, device=dev)
    kvlen, ring = torch.full((big,), -7, dtype=torch.int32, device=dev), torch.full((big,), -7, dtype=torch.int32,
                                                                                    device=dev)
    decode_inputs(arena, token_pool, page_table, input_ids, out_loc, positions, mrope_buf, lin_buf,
                  (block_table, kvlen, ring, page), (off_pos, off_mrope, off_lin, off_kv, off_map), bs, m)
    mapping = rows.repeat_interleave(m)
    assert torch.equal(input_ids[:R], token_pool[mapping, pos.long()])
    assert torch.equal(out_loc[:R], page_table[mapping, pos.long()])
    assert torch.equal(positions[:R], pos)
    if mrope:
        assert torch.equal(mrope_buf[: 3 * R].view(3, R), rope)
    assert torch.equal(lin_buf[:bs], lin)
    assert torch.equal(kvlen[:bs], kv) and torch.equal(ring[:bs], rows.to(torch.int32))
    assert torch.equal(block_table[:bs], page_table[:, ::page].index_select(0, rows) // page)
    assert (input_ids[R:] == -7).all() and (block_table[bs:] == -7).all()
