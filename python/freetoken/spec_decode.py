"""Speculative decoding scaffolding (experimental; qwen4_exp).

``FREETOKEN_SPEC_VERIFY_M=m`` (default 1 = off): every decode step runs ``m`` rows per request -- its last token at
position L and ``m - 1`` draft rows at L+1.. -- through the normal decode path, captured graphs included. The
step keeps the first ``n_acc`` rows of each request (1 <= n_acc <= m) and rolls every per-request recurrent state
back to "after row n_acc - 1" with :func:`apply_rollback`. Without a draft model the extra rows are placeholders
and n_acc is 1, so decoding is unchanged while the step costs what a verify pass costs.

Attention KV needs no rollback: rejected rows wrote at positions >= L + n_acc, which the next steps overwrite
before any query can see them (causal by position). The recurrent states do: each layer that updates one in place
registers a :class:`RollbackTarget` once (static snapshot buffers, so the captured graphs write into them and the
copy back runs after each replay on the same stream, without a host sync).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict

import torch

SPEC_M = max(1, int(os.environ.get("FREETOKEN_SPEC_VERIFY_M", "1")))
# FREETOKEN_MTP=1: build and load the model's MTP draft head (qwen4_exp: models/qwen4_exp/mtp.py) as one extra
# layer -- its attention a QSA layer of the KV pool, its routed experts an NVFP4 bank layer of the offload cache.
# Only with verify rows (SPEC_M > 1): without them nothing checks its drafts and the head would be pure cost.
MTP_REQUESTED = os.environ.get("FREETOKEN_MTP", "0") == "1"
MTP_ENABLED = MTP_REQUESTED and SPEC_M > 1


def _rows_set() -> tuple[int, ...]:
    env = os.environ.get("FREETOKEN_SPEC_VERIFY_MS", "")
    if env:
        ms = tuple(sorted({max(1, int(x)) for x in env.split(",") if x.strip()}))
    elif MTP_ENABLED and SPEC_M > 1:
        ms = tuple(range(1, SPEC_M + 1))
    else:
        ms = (SPEC_M,)
    assert max(ms) == SPEC_M, f"FREETOKEN_SPEC_VERIFY_MS {ms} must top out at FREETOKEN_SPEC_VERIFY_M={SPEC_M}"
    return ms


# Rows per request a decode step may run, one captured graph set each (FREETOKEN_SPEC_VERIFY_MS, default 1..SPEC_M
# with the MTP head): a verify pass costs per row, and with several requests the extra rows touch so many
# more experts that plain decode wins (measured: 4 requests at m=3 make 69 tok/s in all, 94 plain), so a step
# runs SPEC_M rows only while at most FREETOKEN_SPEC_BS_MAX requests decode (default 1).
SPEC_MS = _rows_set()
SPEC_BS_MAX = max(1, int(os.environ.get("FREETOKEN_SPEC_BS_MAX", "1")))
# FREETOKEN_SPEC_SAMPLED_DRAFTS=1: a sampled request's drafts are draws from the MTP head's distribution under its
# own sampling params (exact verify against that distribution) instead of the head's argmax (a point mass). Measured:
# 2-7 % more tokens kept per step, the same speed (120 items at the model's T 1.0: 434 s argmax, 439 s drawn), so off
SPEC_SAMPLED_DRAFTS = os.environ.get("FREETOKEN_SPEC_SAMPLED_DRAFTS", "0") == "1"


# Decode step time per rows per request at one request, measured (FREETOKEN_SPEC_TIMING, GPU time per step: forward +
# verify + MTP head + chain; 1 row is the plain step without the head), by tensor-parallel size: two cards (xtx-xt,
# 2026-10-01), one card (xt, 2026-10-03: each extra row brings ~3 more missing experts per layer over PCIe, so 2 rows
# is the count that pays)
_MEASURED_COSTS = {2: {1: 17.7, 2: 25.3, 3: 30.7, 4: 35.5}, 1: {1: 31.0, 2: 53.0, 3: 78.0, 4: 102.0}}


def _costs(tp_size: int) -> dict[int, float]:
    """The step costs for ``tp_size`` cards (only the ratios matter; more than two cards use the two-card table).
    FREETOKEN_SPEC_COSTS="2:26,3:31" overrides entries; a row count missing from the table costs the last entry's
    per-row time."""
    costs = dict(_MEASURED_COSTS.get(tp_size, _MEASURED_COSTS[2]))
    for item in os.environ.get("FREETOKEN_SPEC_COSTS", "").split(","):
        if ":" in item:
            m, ms = item.split(":")
            costs[int(m)] = float(ms)
    return costs


SPEC_COSTS: dict[int, float] | None = None  # resolved on first use, once the TP size is known
# FREETOKEN_SPEC_DYNAMIC=1 (the default when two or more row counts above 1 are captured): each request's rows per step
# follow its measured draft acceptance -- the row count that keeps the most tokens per unit of step time
SPEC_DYNAMIC = os.environ.get("FREETOKEN_SPEC_DYNAMIC", "1") == "1" and len([m for m in SPEC_MS if m > 1]) > 1
_PRIOR = 0.8   # acceptance assumed for a draft position before a request has tried it
_DECAY = 0.9   # per-step memory of a position's acceptance average
_RELAX = 0.02  # per step, an untried position drifts toward the deepest tried one (so a short depth gets retried)


def _cost(m: int) -> float:
    global SPEC_COSTS
    if SPEC_COSTS is None:
        from freetoken.distributed import try_get_tp_info

        tp = try_get_tp_info()
        SPEC_COSTS = _costs(tp.size if tp is not None else 2)
    if m in SPEC_COSTS:
        return SPEC_COSTS[m]
    top = max(SPEC_COSTS)
    return SPEC_COSTS[top] * m / top


class DepthStats:
    """One request's running acceptance per draft position: ``rate[j]``, the chance draft j is kept when drafts < j
    were (the MTP head's chained guesses get worse with depth)."""

    def __init__(self) -> None:
        self.rate = [_PRIOR] * max(SPEC_M - 1, 0)

    def update(self, m: int, n_acc: int) -> None:
        """A verify step of ``m`` rows kept ``n_acc``: drafts 0..min(n_acc, m-1)-1 were tried, the first n_acc-1 kept."""
        tried = min(n_acc, m - 1)
        for j in range(tried):
            kept = 1.0 if j < n_acc - 1 else 0.0
            self.rate[j] = _DECAY * self.rate[j] + (1 - _DECAY) * kept
        if 0 < m - 1 < len(self.rate):
            ref = self.rate[m - 2]
            for j in range(m - 1, len(self.rate)):
                self.rate[j] += _RELAX * (ref - self.rate[j])

    def best_rows(self) -> int:
        """The row count (> 1) with the most expected kept tokens per unit of step time."""
        best, best_m = -1.0, SPEC_M
        for m in SPEC_MS:
            if m < 2:
                continue
            expect, run = 1.0, 1.0
            for j in range(m - 1):
                run *= self.rate[j]
                expect += run
            if expect / _cost(m) > best:
                best, best_m = expect / _cost(m), m
        return best_m


def _wanted_rows(req) -> int:
    stats = getattr(req, "spec_stats", None)
    return stats.best_rows() if SPEC_DYNAMIC and stats is not None else SPEC_M


def _room(req) -> int:
    """Tokens ``req`` may still emit (its ``max_device_len`` is at most the page table's width)."""
    return getattr(req, "remain_len", SPEC_M)


def draft_depth(reqs, m: int = 1) -> int:
    """Drafts the MTP head chains after a decode step of ``reqs`` with ``m`` rows each: as many as their next step may
    verify, none (the head only writes its KV, which later drafts attend to) while more than SPEC_BS_MAX requests
    decode. The chain writes its KV up to the step's last row + depth - 1, kept below ``max_device_len``."""
    if len(reqs) > SPEC_BS_MAX or SPEC_M < 2:
        return 0
    return min(max(_wanted_rows(r) for r in reqs) - 1, min(_room(r) for r in reqs) - m + 2)


def rows_for(reqs) -> int:
    """Rows per request for a decode step of ``reqs``: SPEC_MS[0] while more than SPEC_BS_MAX requests decode, else
    the rows they want (SPEC_M, or with SPEC_DYNAMIC what their acceptance favours) capped by the drafts each one
    holds (``Req.spec_drafts``, set by the engine when it drafts; without a draft model every row is a placeholder).
    Never more rows than a request may still emit: past ``max_device_len`` they would index beyond the page table when
    the request may fill the context (such a step runs the most rows that fit, eagerly if they have no graph)."""
    if len(reqs) > SPEC_BS_MAX or SPEC_MS[-1] == 1:
        m = SPEC_MS[0]
    else:
        cap = min(max(_wanted_rows(r) for r in reqs), 1 + min(getattr(r, "spec_drafts", SPEC_M - 1) for r in reqs))
        m = max((k for k in SPEC_MS if k <= cap), default=SPEC_MS[0])
    room = min(_room(r) for r in reqs)
    return m if m <= room else max((k for k in SPEC_MS if k <= room), default=1)


def rows_of(batch) -> int:
    """Rows per request of ``batch``: its ``spec_m`` for a decode step, 1 for a prefill."""
    return getattr(batch, "spec_m", 1) if batch.is_decode else 1


@dataclass
class RollbackTarget:
    """The state of request ``b`` (state slot ``slots[b]`` along ``dim``) after its row j is ``snapshot(j, b)`` for
    j < m-1; after row m-1 it is already in place.

    ``snapshot(step [B] int64, b [B] int64)`` reads the layer's static buffers and returns the state rows laid out like
    ``state.index_select(dim, slots)``. ``dim`` 1 stacks every layer of a pool (``[layers, slots, ...]``), so one copy
    serves them all. ``padding_slot`` takes the writes of the requests that keep every row.
    """

    state: torch.Tensor
    snapshot: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    dim: int = 0
    padding_slot: int = 0


_TARGETS: Dict[str, RollbackTarget] = {}


def register(name: str, target: RollbackTarget) -> None:
    _TARGETS[name] = target


def apply_rollback(slots: torch.Tensor, n_acc: torch.Tensor, padding_slot: int | None = None,
                   m: int | None = None) -> None:
    """Put every registered state of the ``B`` requests (state rows ``slots``) where row ``n_acc - 1`` left it.
    ``padding_slot``: the slot padded rows use (the engine's dummy request), overriding the targets' own."""
    if not _TARGETS:
        return
    slots = slots.to(torch.int64)
    b = torch.arange(slots.numel(), device=slots.device)
    m = SPEC_M if m is None else m  # the step's rows per request
    keep = (n_acc >= m).view(-1)
    step = (n_acc.to(torch.int64) - 1).clamp(min=0, max=max(m - 2, 0)).view(-1)
    for target in _TARGETS.values():
        # a request that kept every row is already right: its copy goes to the padding slot instead
        pad = target.padding_slot if padding_slot is None else padding_slot
        dst = torch.where(keep, torch.full_like(slots, pad), slots)
        target.state.index_copy_(target.dim, dst, target.snapshot(step, b).to(target.state.dtype))


class SpecTimer:
    """FREETOKEN_SPEC_TIMING=N (bench runs): each decode step's phases -- the time between consecutive marks, on the
    GPU (events on the step's stream) and on the host (launch time) -- averaged per (requests, rows) and logged
    every N steps, with one stream sync then."""

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.every = int(os.environ.get("FREETOKEN_SPEC_TIMING", "0"))
        # a step whose GPU or host time, or whose distance to the previous step, exceeds this many ms is logged on its
        # own (every rank, with the wall-clock time): the occasional multi-second decode stall
        self.stall_ms = float(os.environ.get("FREETOKEN_SPEC_STALL_MS", "500"))
        self.stream, self._now, self._wall = stream, time.perf_counter, time.time
        self.marks: list = []
        self.pending: list = []
        self._prev_start: float | None = None

    def prefill(self) -> None:
        """A prefill ran: the next decode step's distance to the previous one is not a stall."""
        self._prev_start = None

    def mark(self, name: str) -> None:
        if self.every:
            e = torch.cuda.Event(enable_timing=True)
            e.record(self.stream)
            self.marks.append((name, e, self._now()))

    def end(self, key: tuple) -> None:
        if not self.every or not self.marks:
            return
        self.mark("end")
        start = self.marks[0][2]
        gap = (start - self._prev_start) * 1e3 if self._prev_start is not None else 0.0
        self._prev_start = start
        self.pending.append((key, self.marks, self._wall(), gap))
        self.marks = []
        if len(self.pending) < self.every:
            return
        self.stream.synchronize()
        from freetoken.distributed import get_tp_info
        from freetoken.utils import init_logger

        log = init_logger(__name__)
        acc: dict = {}
        for k, marks, wall, gap in self.pending:
            phases = acc.setdefault(k, {})
            step = []
            for (_, e0, t0), (name, e1, t1) in zip(marks, marks[1:]):
                g, h, n = phases.get(name, (0.0, 0.0, 0))
                dg, dh = e0.elapsed_time(e1), (t1 - t0) * 1e3
                phases[name] = (g + dg, h + dh, n + 1)
                step.append((name, dg, dh))
            gpu, host = marks[0][1].elapsed_time(marks[-1][1]), (marks[-1][2] - marks[0][2]) * 1e3
            if max(gpu, host, gap) > self.stall_ms:
                parts = " | ".join(f"{name} {g:.1f}/{h:.1f}" for name, g, h in step)
                log.warning(f"[spec-stall] rank {get_tp_info().rank} {time.strftime('%H:%M:%S', time.localtime(wall))}: "
                            f"{k[0]} req x {k[1]} rows, step gpu {gpu:.1f} host {host:.1f} ms, {gap:.1f} ms after the "
                            f"previous step's start | ms gpu/host: {parts}")
        self.pending = []
        for (reqs, m), phases in sorted(acc.items()):
            parts = " | ".join(f"{name} {g / n:.2f}/{h / n:.2f}" for name, (g, h, n) in phases.items())
            steps = next(iter(phases.values()))[2]
            log.info_rank0(f"[spec-time] {reqs} req x {m} rows, {steps} steps, ms gpu/host: {parts}")


__all__ = ["SPEC_BS_MAX", "SPEC_DYNAMIC", "SPEC_M", "SPEC_MS", "DepthStats", "RollbackTarget", "SpecTimer",
           "apply_rollback", "draft_depth", "register", "rows_for", "rows_of"]
