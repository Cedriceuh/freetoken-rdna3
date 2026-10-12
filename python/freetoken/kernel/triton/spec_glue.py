"""A verify step's glue as one kernel per piece (FREETOKEN_SPEC_GLUE_KERNELS, default on), bit-exact by construction
(integer arithmetic and copies):

- ``rollback_copy``: one rollback target of ``spec_decode.apply_rollback`` -- each request's state row put back to
  where its last kept row left it (``step = clamp(n_acc - 1, 0, m - 2)``), or written to the padding slot when the
  request kept every row -- instead of an arange, where, gather, cast and index_copy per target;
- ``write_step_tokens``: the scheduler's token-pool writes after a verify step (the target's tokens at columns
  pos .. pos + m - 1, then the drafts from pos + n_acc on, clamped to the last column; padded rows (pos < 0) write the
  junk last column) instead of two aranges, wheres, clamps and index_puts;
- ``greedy_n_acc``: the kept counts of greedy rows (the matching prefix of drafts) instead of a compare, cast, cumprod,
  sum and add;
- ``draft_rows``: the MTP head's row per request and every chained guess's position;
- ``chain_meta``: a chained guess's addressing (positions, sequence lengths, the page-table slot, mrope positions).

Like the torch ops they replace, they take the indices as given (the scheduler keeps them in range); unlike them,
they do not assert it on the device.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rollback_copy_kernel(src, dst, slots, n_acc, pad, m, so, sb, sst, step_add, do, dslot, S,
                          BLOCK: tl.constexpr):
    o = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1)
    blk = tl.program_id(2)
    n = tl.load(n_acc + b).to(tl.int64)
    slot = tl.load(slots + b).to(tl.int64)
    step = tl.minimum(tl.maximum(n - 1, 0), tl.maximum(m - 2, 0))
    dst_slot = tl.where(n >= m, pad, slot).to(tl.int64)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    v = tl.load(src + o * so + b.to(tl.int64) * sb + (step + step_add) * sst + offs, mask=mask)
    tl.store(dst + o * do + dst_slot * dslot + offs, v, mask=mask)


def rollback_copy(src: torch.Tensor, so: int, sb: int, sst: int, step_add: int, dst: torch.Tensor, do: int,
                  dslot: int, outer: int, inner: int, slots: torch.Tensor, n_acc: torch.Tensor, pad: int,
                  m: int) -> None:
    B = slots.numel()
    BLOCK = 2048 if inner >= 2048 else triton.next_power_of_2(inner)
    _rollback_copy_kernel[(outer, B, triton.cdiv(inner, BLOCK))](
        src, dst, slots, n_acc, pad, m, so, sb, sst, step_add, do, dslot, inner, BLOCK=BLOCK, num_warps=4)


@triton.jit
def _write_step_tokens_kernel(pool, stride_p, W, rows, pos, tokens, stride_t, M: tl.constexpr, drafts, stride_d,
                              n_acc, K: tl.constexpr, HAS_DRAFTS: tl.constexpr):
    r = tl.program_id(0)
    row = tl.load(rows + r).to(tl.int64)
    p = tl.load(pos + r).to(tl.int64)
    base = pool + row * stride_p
    for j in tl.static_range(M):
        c = tl.where(p >= 0, p + j, W - 1)
        tl.store(base + c, tl.load(tokens + r * stride_t + j).to(pool.dtype.element_ty))
    if HAS_DRAFTS:
        start = p + tl.load(n_acc + r).to(tl.int64)
        for j in tl.static_range(K):
            c = tl.where(p >= 0, tl.minimum(start + j, W - 1), W - 1)
            tl.store(base + c, tl.load(drafts + r * stride_d + j).to(pool.dtype.element_ty))


def write_step_tokens(pool: torch.Tensor, rows: torch.Tensor, pos: torch.Tensor, tokens: torch.Tensor,
                      drafts: torch.Tensor | None, n_acc: torch.Tensor) -> None:
    """``tokens [R, m]``; ``drafts [R, k]`` or None; ``pool [rows, W]`` (unit column stride)."""
    R, M = tokens.shape
    assert pool.stride(1) == 1 and tokens.stride(1) == 1
    K = drafts.shape[1] if drafts is not None else 1
    if drafts is not None:
        assert drafts.stride(1) == 1
    _write_step_tokens_kernel[(R,)](
        pool, pool.stride(0), pool.shape[1], rows, pos, tokens, tokens.stride(0), M,
        drafts if drafts is not None else tokens, drafts.stride(0) if drafts is not None else 0, n_acc, K,
        HAS_DRAFTS=drafts is not None, num_warps=1)


@triton.jit
def _greedy_n_acc_kernel(tokens, stride_t, input_ids, n_acc, M: tl.constexpr):
    """n_acc = 1 + the length of the prefix of rows j whose target token matches the draft at row j + 1."""
    r = tl.program_id(0).to(tl.int64)
    n = 1
    run = 1
    for j in tl.static_range(M - 1):
        t = tl.load(tokens + r * stride_t + j).to(tl.int64)
        d = tl.load(input_ids + r * M + j + 1).to(tl.int64)
        run = run * (t == d).to(tl.int32)
        n += run
    tl.store(n_acc + r, n.to(tl.int64))


def greedy_n_acc(tokens: torch.Tensor, input_ids: torch.Tensor, m: int) -> torch.Tensor:
    """``1 + (tokens[:, :-1] == input_ids.view(-1, m)[:, 1:]).cumprod(-1).sum(-1)`` (int64)."""
    reqs = tokens.shape[0]
    out = torch.empty(reqs, dtype=torch.int64, device=tokens.device)
    assert tokens.stride(1) == 1 and input_ids.is_contiguous()
    _greedy_n_acc_kernel[(reqs,)](tokens, tokens.stride(0), input_ids, out, M=m, num_warps=1)
    return out


@triton.jit
def _draft_rows_kernel(positions, n_acc, pick, chain_pos, M: tl.constexpr, DEPTH: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    n = tl.load(n_acc + r).to(tl.int64)
    tl.store(pick + r, r * M + n - 1)
    last = tl.load(positions + r * M).to(tl.int64) + n - 1
    for j in tl.static_range(1, DEPTH):
        tl.store(chain_pos + r * DEPTH + j, last + j)


def draft_rows(positions: torch.Tensor, n_acc: torch.Tensor, m: int, depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``pick = arange(reqs) * m + n_acc - 1`` and ``chain_pos[:, j] = positions.view(reqs, m)[:, 0] + n_acc - 1 + j``
    for 1 <= j < depth (column 0 unused), int64."""
    reqs = n_acc.numel()
    pick = torch.empty(reqs, dtype=torch.int64, device=n_acc.device)
    chain = torch.empty(reqs, max(depth, 1), dtype=torch.int64, device=n_acc.device)
    _draft_rows_kernel[(reqs,)](positions, n_acc, pick, chain, M=m, DEPTH=max(depth, 1), num_warps=1)
    return pick, chain


@triton.jit
def _chain_meta_kernel(pos, stride_pos, table_idx, page_table, stride_pt, pos32, seq_lens, out_loc, delta, mrope,
                       reqs, HAS_MROPE: tl.constexpr):
    r = tl.program_id(0)
    p = tl.load(pos + r * stride_pos)
    p32 = p.to(tl.int32)
    tl.store(pos32 + r, p32)
    tl.store(seq_lens + r, p32 + 1)
    row = tl.load(table_idx + r).to(tl.int64)
    tl.store(out_loc + r, tl.load(page_table + row * stride_pt + p))
    if HAS_MROPE:
        v = p32 + tl.load(delta + r)
        for s in tl.static_range(3):
            tl.store(mrope + s * reqs + r, v)


def chain_meta(pos: torch.Tensor, table_idx: torch.Tensor, page_table: torch.Tensor, delta: torch.Tensor | None):
    """``(pos32, seq_lens = pos32 + 1, out_loc = page_table[table_idx, pos], mrope [3, reqs] = pos32 + delta)``."""
    reqs = pos.numel()
    dev = pos.device
    pos32 = torch.empty(reqs, dtype=torch.int32, device=dev)
    seq_lens = torch.empty(reqs, dtype=torch.int32, device=dev)
    out_loc = torch.empty(reqs, dtype=page_table.dtype, device=dev)
    mrope = torch.empty(3, reqs, dtype=torch.int32, device=dev) if delta is not None else None
    assert page_table.stride(1) == 1
    _chain_meta_kernel[(reqs,)](pos, pos.stride(0), table_idx, page_table, page_table.stride(0), pos32, seq_lens,
                                out_loc, delta if delta is not None else pos32, mrope if mrope is not None else pos32,
                                reqs, HAS_MROPE=delta is not None, num_warps=1)
    return pos32, seq_lens, out_loc, mrope


__all__ = ["chain_meta", "draft_rows", "greedy_n_acc", "rollback_copy", "write_step_tokens"]
