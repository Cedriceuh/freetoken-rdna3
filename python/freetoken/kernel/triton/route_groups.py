"""Which program of a decode MoE GEMV takes which routes, when the routes that read one expert share a program.

A decode step of several rows (a verify step, several requests) routes a third of its rows' top-k to an expert another
row also reads. The EXL3 decode GEMV then decodes each trellis tile once for up to 4 routes of one slot
(FREETOKEN_EXL3_SHARED_ROUTES); this table says which: ``grp[r] = (n, r1, r2, r3)``, route r taking itself and r1..
(n of them) when it is the first of a run of up to GROUP routes of its slot in index order, n = 0 when another route's
program takes it. The EXL3 input rotation writes it (_route_group_row) without a launch of its own.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _route_group_row(slot_ptr, grp_ptr, r, R, RP: tl.constexpr, GROUP: tl.constexpr):
    """Route r's row of the table, from the R slots at ``slot_ptr`` (RP: a power of two >= R)."""
    offs = tl.arange(0, RP)
    slot = tl.load(slot_ptr + r)
    same = tl.load(slot_ptr + offs, mask=offs < R, other=-2) == slot
    cum = tl.cumsum(same.to(tl.int32), axis=0)  # routes of this slot up to each index
    rank = tl.sum(tl.where(offs == r, cum, 0), axis=0) - 1
    j = tl.arange(0, 4)
    member = same[None, :] & (cum[None, :] == rank + 1 + j[:, None]) & (j[:, None] < GROUP)  # [4, RP]
    n = tl.sum(tl.sum(member.to(tl.int32), axis=1), axis=0)
    idx = tl.sum(tl.where(member, offs[None, :], 0), axis=1)
    out = tl.where(j == 0, tl.where(rank % GROUP == 0, n, 0), idx)
    tl.store(grp_ptr + r * 4 + j, out)


@triton.jit
def _route_groups_kernel(slot_ptr, grp_ptr, R, RP: tl.constexpr, GROUP: tl.constexpr):
    _route_group_row(slot_ptr, grp_ptr, tl.program_id(0), R, RP, GROUP)


def route_groups(slots: torch.Tensor, group: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """The ``[R, 4]`` int32 table of the R routes' ``slots`` (any shape, contiguous int32), runs of up to ``group``."""
    R = slots.numel()
    grp = torch.empty((R, 4), dtype=torch.int32, device=slots.device) if out is None else out
    _route_groups_kernel[(R,)](slots, grp, R, RP=triton.next_power_of_2(R), GROUP=group, num_warps=1)
    return grp


__all__ = ["route_groups"]
