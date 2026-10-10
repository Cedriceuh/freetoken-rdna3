"""Host K/V (FREETOKEN_QSA_KV_HOST): copy whole pages of one layer from host-mapped memory into a VRAM buffer."""

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


def stage_pages(src: torch.Tensor, dst: torch.Tensor, pages: torch.Tensor) -> None:
    """``dst[pages] = src[pages]`` for [num_pages, ...] contiguous page tensors of the same shape (pages may repeat)."""
    assert src.shape == dst.shape and src.is_contiguous() and dst.is_contiguous() and src.dtype == dst.dtype
    n = pages.numel()
    if n == 0:
        return
    page_elems = src[0].numel()
    assert page_elems % 2048 == 0, f"a page of {page_elems} elements is not a whole number of 2048-element blocks"
    _stage_pages_kernel[(n,)](src, dst, pages, PAGE_ELEMS=page_elems, BLOCK=2048, num_warps=4)
