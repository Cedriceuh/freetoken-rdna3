"""The rank-0 -> rank-N request relay must not lose the first request.

Two real processes, a real gloo group, real ZeroMQ ipc sockets -- the served shape of
`SchedulerIOMixin.__init__` at TP=2, minus the engine. The failing case is deterministic when rank 1
builds its SUB after rank 0 has already published: a PUB drops what nobody subscribes to.
"""
from __future__ import annotations

import socket
import time
from types import SimpleNamespace

import pytest
import torch.multiprocessing as mp

PAYLOAD = b"first-request-bytes"
JOIN_TIMEOUT_S = 60


def _config(tmp: str, rank: int, size: int) -> SimpleNamespace:
    from freetoken.distributed.info import DistributedInfo

    return SimpleNamespace(
        tp_info=DistributedInfo(rank=rank, size=size),
        offline_mode=False,
        zmq_backend_addr=f"ipc://{tmp}/backend",
        zmq_detokenizer_addr=f"ipc://{tmp}/detok",
        backend_create_detokenizer_link=True,
        zmq_scheduler_broadcast_addr=f"ipc://{tmp}/broadcast",
    )


def _close(io) -> None:
    """The engine's shutdown does this through each queue's stop(); a context with an open socket
    blocks its term() at interpreter exit, so the test closes explicitly and with linger 0."""
    for name in ("_recv_from_tokenizer", "_send_into_tokenizer", "_send_into_ranks", "_recv_from_rank0"):
        q = getattr(io, name, None)
        if q is not None:
            q.socket.close(linger=0)
            q.context.term()


def _rank_main(rank: int, size: int, tmp: str, port: int, delay: dict, handshake: bool, result):
    import faulthandler
    import sys

    import torch.distributed as dist

    from freetoken.scheduler.io import SchedulerIOMixin

    faulthandler.dump_traceback_later(30, exit=True, file=sys.stderr)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size)
    group = dist.group.WORLD
    io = None
    try:
        time.sleep(delay.get(rank, 0.0))
        io = SchedulerIOMixin.__new__(SchedulerIOMixin)
        if not handshake:
            io._handshake_rank_relay = lambda tp_info: None  # the pre-fix engine
        SchedulerIOMixin.__init__(io, _config(tmp, rank, size), group)
        if handshake:
            io.sync_all_ranks()  # launch.py: the barrier that precedes "Scheduler is ready"
        if rank == 0:
            io._send_into_ranks.put_raw(PAYLOAD)  # the first request, relayed at ready+0 s
            if not handshake:
                io.sync_all_ranks()
            result.put((rank, "sent"))
        else:
            sub = io._recv_from_rank0.socket
            if not handshake:
                io.sync_all_ranks()
            got = sub.recv() if sub.poll(timeout=5000) else None
            result.put((rank, got))
    finally:
        result.close()
        result.join_thread()
        dist.barrier()
        dist.destroy_process_group()
        if io is not None:
            _close(io)


def _run(tmp_path, delay: dict, handshake: bool):
    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    size = 2
    procs = [
        ctx.Process(target=_rank_main, args=(r, size, str(tmp_path), port, delay, handshake, result))
        for r in range(size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(JOIN_TIMEOUT_S)
    alive = [p.pid for p in procs if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.kill()
    assert not alive, f"ranks still running after {JOIN_TIMEOUT_S} s (a wedge): {alive}"
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    out = {}
    while not result.empty():
        r, v = result.get()
        out[r] = v
    return out


@pytest.mark.parametrize("delay", [{}, {1: 1.0}, {0: 1.0}], ids=["together", "rank1-late", "rank0-late"])
def test_first_request_reaches_rank1_with_handshake(tmp_path, delay):
    out = _run(tmp_path, delay, handshake=True)
    assert out[0] == "sent"
    assert out[1] == PAYLOAD, out[1]


def test_without_handshake_a_late_subscriber_loses_the_first_request(tmp_path):
    """The mechanism the handshake exists for, made deterministic: rank 0 publishes before rank 1's
    SUB exists (rank 1 is 1 s late), so the PUB has no subscription to match and drops the frame.
    In the served engine the same loss happens on a subscription that is created but not yet
    registered at the PUB (measured: 5 hellos / 203 ms on ipc with both ranks together)."""
    out = _run(tmp_path, {1: 1.0}, handshake=False)
    assert out[0] == "sent"
    assert out[1] is None, "the pre-fix relay delivered the first request; the race did not reproduce"
