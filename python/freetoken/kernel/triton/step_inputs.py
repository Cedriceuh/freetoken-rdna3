"""A graph-replayed decode step's inputs in one launch (engine/step_inputs.py): from the values the host staged (each
row's position and mrope positions, each request's page-table row, GatedDeltaNet slot and K/V length), the token and
K/V slot each row reads (``token_pool`` / ``page_table`` at its row and position) and the QSA block table, written
straight into the captured graph's input buffers. Integer copies and gathers only."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _floor_div(v, d):
    # torch's // on integers floors (a negative page-table entry, if any, keeps the torch bits)
    q = v // d
    return tl.where((v % d != 0) & (v < 0), q - 1, q)


@triton.jit
def _decode_inputs_kernel(arena, arena64, token_pool, stride_tp, page_table, stride_pt, input_ids, out_loc, positions,
                          mrope, lin_idx, block_table, stride_bt, kvlen, ring, off_pos, off_map, off_mrope, off_lin,
                          off_kv, R, page, pages, M: tl.constexpr, HAS_MROPE: tl.constexpr, HAS_LIN: tl.constexpr,
                          HAS_QSA: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.load(arena64 + off_map + b)
    for j in tl.static_range(M):
        r = b * M + j
        p = tl.load(arena + off_pos + r)
        tl.store(positions + r, p)
        tl.store(input_ids + r, tl.load(token_pool + row * stride_tp + p))
        tl.store(out_loc + r, tl.load(page_table + row * stride_pt + p))
        if HAS_MROPE:
            for t in tl.static_range(3):
                tl.store(mrope + t * R + r, tl.load(arena + off_mrope + t * R + r))
    if HAS_LIN:
        tl.store(lin_idx + b, tl.load(arena + off_lin + b))
    if HAS_QSA:
        tl.store(kvlen + b, tl.load(arena + off_kv + b))
        tl.store(ring + b, row.to(tl.int32))
        for c0 in range(0, pages, BLOCK):
            c = c0 + tl.arange(0, BLOCK)
            m = c < pages
            v = tl.load(page_table + row * stride_pt + c * page, mask=m)
            tl.store(block_table + b * stride_bt + c, _floor_div(v, page), mask=m)


def decode_inputs(arena, token_pool, page_table, input_ids, out_loc, positions, mrope, lin_idx, qsa, offsets, bs: int,
                  m: int) -> None:
    """``offsets``: (positions, mrope, GatedDeltaNet slots, K/V lengths) in int32 words of ``arena``, then the
    requests' page-table rows in int64 words of the same buffer; ``qsa``: (block_table, kvlen, ring slots, page size)
    static buffers, or None."""
    off_pos, off_mrope, off_lin, off_kv, off_map = offsets
    assert token_pool.stride(1) == 1 and page_table.stride(1) == 1
    block_table, kvlen, ring, page = qsa if qsa is not None else (positions, positions, positions, 1)
    pages = block_table.shape[1] if qsa is not None else 0
    _decode_inputs_kernel[(bs,)](
        arena, arena.view(torch.int64), token_pool, token_pool.stride(0), page_table, page_table.stride(0), input_ids,
        out_loc, positions, mrope if mrope is not None else positions, lin_idx if lin_idx is not None else positions,
        block_table, block_table.stride(0), kvlen, ring, off_pos, off_map, off_mrope, off_lin, off_kv, bs * m, page, pages,
        M=m, HAS_MROPE=mrope is not None, HAS_LIN=lin_idx is not None, HAS_QSA=qsa is not None, BLOCK=1024, num_warps=4)


__all__ = ["decode_inputs"]
