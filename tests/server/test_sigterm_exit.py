"""A SIGTERM'd ``ft serve`` leaves no multiprocessing semaphore behind: uvicorn re-raises the SIGTERM it stopped on
once its graceful stop is done, and a process dying of it ran no exit handler, so the ack queue's named semaphores
(and a killed worker's) stayed in /dev/shm -- the host's under ``docker run --ipc=host``, ~5 a start."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import textwrap
import time

import pytest

_CHILD = textwrap.dedent(
    """
    import multiprocessing as mp
    import signal
    import sys
    import time
    from types import SimpleNamespace

    import uvicorn
    from fastapi import FastAPI

    import freetoken.server.api_server as api


    def worker(q):  # a TP rank: its own named semaphore, killed by the stop's SIGTERM before it could unlink it
        lock = mp.Lock()
        q.put(lock._semlock.name)
        time.sleep(600)


    if __name__ == "__main__":
        mp.set_start_method("spawn", force=True)
        q = mp.Queue()
        p = mp.Process(target=worker, args=(q,), daemon=False)
        p.start()
        names = [q._rlock._semlock.name, q._wlock._semlock.name, q._sem._semlock.name, q.get(timeout=60)]
        api._GLOBAL_STATE = SimpleNamespace(backend_processes=[p], shutdown=lambda: api._terminate_backend_workers([p]))
        from multiprocessing import resource_tracker

        print(resource_tracker._resource_tracker._pid, " ".join(names), flush=True)
        app = FastAPI(lifespan=api.lifespan)
        signal.signal(signal.SIGTERM, api._exit_on_sigterm)  # what run_api_server installs before uvicorn.run
        uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
    """
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _listening(port: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.05)
    return False


def _gone(pid: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            with open(f"/proc/{pid}/stat") as f:
                if f.read().split()[2] == "Z":
                    return True
        except FileNotFoundError:
            return True
        time.sleep(0.02)
    return False


def _stop(tmp_path, kill_tracker: bool) -> tuple[int, list[str], list[str]]:
    """Start the child, SIGTERM it once it serves; ``kill_tracker``: SIGKILL its resource tracker first, as the
    container's teardown does when the main process exits (the tracker then cleans nothing up)."""
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    port = _free_port()
    child = subprocess.Popen([sys.executable, str(script), str(port)], stdout=subprocess.PIPE, text=True,
                             env=dict(os.environ), cwd=tmp_path)
    try:
        tracker, *names = child.stdout.readline().split()
        assert len(names) == 4 and all(os.path.exists(_path(n)) for n in names)
        assert _listening(port, time.monotonic() + 60)
        if kill_tracker:
            os.kill(int(tracker), signal.SIGKILL)
            assert _gone(int(tracker), time.monotonic() + 10)
        child.send_signal(signal.SIGTERM)
        code = child.wait(timeout=60)
        return code, names, [n for n in names if os.path.exists(_path(n))]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        for n in names if "names" in locals() else []:
            if os.path.exists(_path(n)):
                os.unlink(_path(n))


def _path(name: str) -> str:
    return f"/dev/shm/sem.{name.lstrip('/')}"


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="POSIX semaphores live in /dev/shm on Linux")
def test_sigterm_unlinks_its_own_semaphores_before_dying(tmp_path):
    """The ack queue's semaphores go even when no tracker is left to clean up after the process (docker --init: the
    main process exits, the namespace's other processes are killed)."""
    code, names, left = _stop(tmp_path, kill_tracker=True)
    assert code == -signal.SIGTERM  # it still dies of the signal: a supervisor sees the same stop
    assert not set(left) & set(names[:3]), left


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="POSIX semaphores live in /dev/shm on Linux")
def test_sigterm_leaves_no_semaphore(tmp_path):
    """With its tracker: the killed worker's semaphore goes too."""
    code, _, left = _stop(tmp_path, kill_tracker=False)
    assert code == -signal.SIGTERM
    assert not left, left


_STREAM_CHILD = textwrap.dedent(
    """
    import asyncio
    import sys

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    import freetoken.server.api_server as api

    app = FastAPI()


    @app.get("/stream")
    async def stream():
        async def forever():  # an answer longer than any stop timeout
            while True:
                yield b"data: x\\n\\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(forever(), media_type="text/event-stream")


    if __name__ == "__main__":
        uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning", **api._stop_grace())
    """
)


@pytest.mark.parametrize("grace, stops", [("1", True), ("0", False)])
def test_sigterm_during_a_stream_stops_within_the_grace(tmp_path, grace, stops):
    """FREETOKEN_STOP_GRACE_S: a SIGTERM while a stream is open stops the server once the grace is over; with no limit
    (0, uvicorn's own default) it waits for the stream, here forever (a supervisor's stop timeout then kills it)."""
    script = tmp_path / "stream_child.py"
    script.write_text(_STREAM_CHILD)
    port = _free_port()
    child = subprocess.Popen([sys.executable, str(script), str(port)], env=dict(os.environ, FREETOKEN_STOP_GRACE_S=grace),
                             cwd=tmp_path)
    client = socket.socket()
    try:
        assert _listening(port, time.monotonic() + 60)
        client.connect(("127.0.0.1", port))
        client.sendall(f"GET /stream HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        client.settimeout(10)
        got = b""
        while b"data: x" not in got:  # the headers, then the stream's first chunk: it is open
            got += client.recv(4096)
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=8)
            stopped = True
        except subprocess.TimeoutExpired:
            stopped = False
        assert stopped == stops
    finally:
        client.close()
        if child.poll() is None:
            child.kill()
            child.wait()
