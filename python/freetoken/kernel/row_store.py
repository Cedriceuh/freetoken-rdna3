"""Python face of the disk row store extension (``kernel/csrc/row_store``) and its CUDA-graph sync.

``RowStore`` reads fixed-width rows straight out of checkpoint shard files (io_uring or a pread pool,
O_DIRECT) into pinned staging by row id; ``PleStore`` adds PLE's n-gram hashing.

Graph sync: a captured graph cannot wait for host I/O, so a captured ``lookup`` WAITs on a pinned
flag through a stream memop (``wait_reset``) that the host ``signal``s once the fill landed; where
the driver rejects memops in capture, or captures a wait that does not hold, the caller falls back to launch-gating
(fill before replay). On ROCm the memops are HIP's own (hipStreamWaitValue64 / WriteValue64).
``probe_wait_sync`` decides which, once, at startup.
"""

from __future__ import annotations

import torch

from freetoken.kernel import _row_store
from freetoken.kernel.pinned import alloc_pinned_tensor

RowStore = _row_store.RowStore
PleStore = _row_store.PleStore


def probe_wait_sync(mode: str, device: torch.device) -> bool:
    """``mode``: ``auto`` (probe), ``wait`` (require memops) or ``gate`` (never use them)."""
    if mode not in ("auto", "wait", "gate"):
        raise ValueError(f"unknown row-store sync mode {mode!r}; expected auto, wait or gate")
    if mode == "gate":
        return False
    scratch = alloc_pinned_tensor(1, dtype=torch.int64)
    scratch.zero_()
    stream = torch.cuda.current_stream(device)
    ok = (
        _row_store.memop_write(stream.cuda_stream, scratch.data_ptr(), 7) == 0
        and _row_store.memop_wait_geq(stream.cuda_stream, scratch.data_ptr(), 7) == 0
    )
    if ok:
        stream.synchronize()
        ok = int(scratch[0]) == 7
    ok = ok and _captured_wait_holds(device)
    if mode == "wait" and not ok:
        raise RuntimeError("wait-sync requested but stream memops are unavailable")
    return ok


def _captured_wait_holds(device: torch.device, replays: int = 3) -> bool:
    """A WAIT/RESET captured in a graph must hold EVERY replay until the host signals, the copy behind it included.
    Some HIP runtimes capture it but let the second replay run straight through (the GPU would read the PLE rows
    before the fill); ROCm 10 holds on gfx1100 (its release notes add stream-capture support for hipStreamWaitValue /
    WriteValue)."""
    import time

    flag = alloc_pinned_tensor(1, dtype=torch.int64)
    flag.zero_()
    # the PLE lookup's own pattern: the captured wait, then a copy of host bytes the host writes only after the launch
    src = alloc_pinned_tensor(64, dtype=torch.uint8)
    src.zero_()
    dst = torch.zeros(64, dtype=torch.uint8, device=device)
    out = torch.zeros(1, device=device)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
            wait_reset(side, flag)
            dst.copy_(src, non_blocking=True)
            out.add_(dst[:1].float())
    except Exception:  # capture rejected the memops
        return False
    held_all, total = True, 0
    for i in range(1, replays + 1):
        graph.replay()
        time.sleep(0.01)
        held_all &= not torch.cuda.current_stream(device).query()  # still waiting on the flag
        src.fill_(i)  # the host's fill lands after the launch: the captured copy must read it
        signal(flag)  # always release it before syncing
        torch.cuda.synchronize(device)
        held_all &= bool((dst == i).all())
        total += i
    return held_all and int(out.item()) == total and int(flag[0]) == 0


def wait_reset(stream: torch.cuda.Stream, flag: torch.Tensor) -> None:
    """Enqueue WAIT(flag >= 1) then flag := 0 on ``stream`` (capturable)."""
    _row_store.memop_wait_reset(stream.cuda_stream, flag.data_ptr())


def signal(flag: torch.Tensor) -> None:
    """Host side: release a pending WAIT (a release-store of 1)."""
    _row_store.signal_flag(flag.data_ptr())


__all__ = ["RowStore", "PleStore", "probe_wait_sync", "wait_reset", "signal"]
