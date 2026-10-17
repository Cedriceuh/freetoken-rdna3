"""A backend worker that dies after ready must take the whole ``ft serve`` down, even while a
client holds a stream open: uvicorn's graceful stop waits for open streams without a timeout,
and a stream to a dead backend never ends (keepalives only), so the surviving workers kept their
GPUs and the process never exited (seen when the host OOM killer took one TP rank)."""

from __future__ import annotations

import http.client
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time

_CHILD = textwrap.dedent(
    """
    import multiprocessing as mp
    import signal
    import sys
    import time
    from types import SimpleNamespace

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    import freetoken.server.api_server as api


    def stubborn_worker():  # a surviving TP rank stuck waiting on its dead peer
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(1)


    worker = mp.get_context("fork").Process(target=stubborn_worker, daemon=False)
    worker.start()
    api._GLOBAL_STATE = SimpleNamespace(backend_processes=[worker])
    print(worker.pid, flush=True)

    app = FastAPI()


    @app.get("/stream")
    def stream():
        api._exit_after_backend_death(0.2)  # what _on_failure arms when a worker dies

        def keepalives():
            while True:
                yield b": keepalive\\n\\n"
                time.sleep(0.1)

        return StreamingResponse(keepalives(), media_type="text/event-stream")


    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
    """
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _hold_stream(port: int, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            conn.request("GET", "/stream")
        except OSError:  # server not listening yet
            time.sleep(0.2)
            continue
        try:
            resp = conn.getresponse()
            while resp.read(64):
                pass
        except (OSError, http.client.HTTPException):  # the server went away mid-stream
            pass
        return


def _gone_or_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _dies_within(pid: int, timeout: float) -> bool:
    # the server SIGKILLs the stubborn worker just before it exits, without waiting for its death
    deadline = time.monotonic() + timeout
    while not _gone_or_zombie(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_backend_death_exits_nonzero_with_an_open_stream_and_a_stubborn_worker():
    port, worker_pid = _free_port(), None
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(port)], stdout=subprocess.PIPE, text=True, env=dict(os.environ)
    )
    try:
        worker_pid = int(child.stdout.readline())
        threading.Thread(target=_hold_stream, args=(port, time.monotonic() + 60), daemon=True).start()
        # the old path (SIGTERM to itself) never returns here: uvicorn waits on the stream forever
        assert child.wait(timeout=60) == 1
        assert _dies_within(worker_pid, 5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        if worker_pid is not None and not _gone_or_zombie(worker_pid):
            os.kill(worker_pid, 9)
