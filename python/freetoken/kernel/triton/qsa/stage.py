"""Host K/V (FREETOKEN_QSA_KV_HOST): copy whole pages of one layer from host-mapped memory into a VRAM buffer, all the
pages in use or only the rows of a prefill's selected tokens."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_pages_kernel(src, dst, pages, PAGE_ELEMS: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.load(pages + tl.program_id(0)).to(tl.int64)
    base = p * PAGE_ELEMS
    for off in tl.static_range(0, PAGE_ELEMS, BLOCK):
        o = base + off + tl.arange(0, BLOCK)
        tl.store(dst + o, tl.load(src + o))


@triton.jit(do_not_specialize=["stamp"])  # a new stamp every call: no recompiles on its value
def _mark_rows_kernel(indices, token_to_req, block_table, mark, stamp, stride_ir, stride_tr, num_requests, num_pages,
                      TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    # the K/V row of every token the attention reads, by the mapping and validity tests of
    # _qsa_sparse_paged_gqa_splitk_kernel (attend.py): mark[page * PAGE_SIZE + offset] = stamp
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(token_to_req + row)
    safe_req = tl.minimum(tl.maximum(req, 0), num_requests - 1)
    tok = tl.load(indices + row * stride_ir + cols, mask=cols < TOPK, other=-1)
    safe = tl.maximum(tok, 0)
    lp = safe // PAGE_SIZE
    valid = (req >= 0) & (req < num_requests) & (tok >= 0) & (lp < WIDTH)
    pp = tl.load(block_table + safe_req.to(tl.int64) * stride_tr + tl.minimum(lp, WIDTH - 1), mask=valid, other=-1)
    valid &= (pp >= 0) & (pp < num_pages)
    slot = tl.maximum(pp, 0).to(tl.int64) * PAGE_SIZE + safe % PAGE_SIZE
    tl.store(mark + slot, stamp, mask=valid)


@triton.jit(do_not_specialize=["stamp"])
def _stage_marked_kernel(src_k, dst_k, src_v, dst_v, pages, mark, stamp, PAGE_SIZE: tl.constexpr, ROW: tl.constexpr,
                         ROWS: tl.constexpr):
    # one page: its marked rows only (a masked-off lane reads nothing across PCIe)
    p = tl.load(pages + tl.program_id(0)).to(tl.int64)
    r = tl.arange(0, PAGE_SIZE)
    hit = tl.load(mark + p * PAGE_SIZE + r) == stamp
    if tl.max(hit.to(tl.int32), axis=0) > 0:
        for r0 in tl.static_range(0, PAGE_SIZE, ROWS):
            rr = r0 + tl.arange(0, ROWS)
            m = (tl.load(mark + p * PAGE_SIZE + rr) == stamp)[:, None]
            o = (p * PAGE_SIZE + rr)[:, None] * ROW + tl.arange(0, ROW)[None, :]
            tl.store(dst_k + o, tl.load(src_k + o, mask=m), mask=m)
            tl.store(dst_v + o, tl.load(src_v + o, mask=m), mask=m)


def stage_selected_pages(src_k: torch.Tensor, dst_k: torch.Tensor, src_v: torch.Tensor, dst_v: torch.Tensor,
                         pages: torch.Tensor, indices: torch.Tensor, token_to_req: torch.Tensor,
                         block_table: torch.Tensor, mark: torch.Tensor, stamp: int) -> None:
    """:func:`stage_pages` of K and V restricted to the rows (tokens) of ``pages`` that ``indices`` select (the
    attention's ``[rows, topk]`` logical token ids, -1 = none): every K/V row the attention then reads is staged, so
    the same bits. ``mark``: int32 per physical token slot (pages x page size), holding no ``stamp`` yet (each call
    takes a new one)."""
    for t in (src_k, dst_k, src_v, dst_v):
        assert t.shape == src_k.shape and t.is_contiguous() and t.dtype == src_k.dtype
    rows, topk = indices.shape
    n = pages.numel()
    if n == 0 or rows == 0:
        return
    num_pages, page_size = src_k.shape[0], src_k.shape[1]
    row_elems = src_k[0, 0].numel()
    assert indices.stride(1) == 1 and block_table.stride(1) == 1 and mark.numel() >= num_pages * page_size
    if page_size & (page_size - 1) or row_elems & (row_elems - 1):  # the row tiles need powers of two: whole pages
        stage_pages(src_k, dst_k, pages)
        stage_pages(src_v, dst_v, pages)
        return
    block = 256
    _mark_rows_kernel[(rows, triton.cdiv(topk, block))](
        indices, token_to_req, block_table, mark, stamp, indices.stride(0), block_table.stride(0),
        block_table.shape[0], num_pages, TOPK=topk, PAGE_SIZE=page_size, WIDTH=block_table.shape[1],
        BLOCK=block, num_warps=4)
    _stage_marked_kernel[(n,)](src_k, dst_k, src_v, dst_v, pages, mark, stamp, PAGE_SIZE=page_size, ROW=row_elems,
                               ROWS=max(1, min(page_size, 4096 // row_elems)), num_warps=4)


def stage_pages(src: torch.Tensor, dst: torch.Tensor, pages: torch.Tensor) -> None:
    """``dst[pages] = src[pages]`` for [num_pages, ...] contiguous page tensors of the same shape (pages may repeat)."""
    assert src.shape == dst.shape and src.is_contiguous() and dst.is_contiguous() and src.dtype == dst.dtype
    n = pages.numel()
    if n == 0:
        return
    page_elems = src[0].numel()
    assert page_elems % 2048 == 0, f"a page of {page_elems} elements is not a whole number of 2048-element blocks"
    _stage_pages_kernel[(n,)](src, dst, pages, PAGE_ELEMS=page_elems, BLOCK=2048, num_warps=4)
