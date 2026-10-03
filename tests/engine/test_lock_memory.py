"""The engine locks its pages only when asked (default) and only with an unlimited RLIMIT_MEMLOCK: locking future pages
under a finite limit would make allocations fail once it is reached."""
from __future__ import annotations

import ctypes
import resource

from freetoken.engine import engine


def _no_libc(*args, **kwargs):
    raise AssertionError("mlockall must not be called")


def test_off_by_env(monkeypatch):
    monkeypatch.setenv("FREETOKEN_MLOCK", "0")
    monkeypatch.setattr(ctypes, "CDLL", _no_libc)
    engine._lock_memory()


def test_finite_limit_leaves_memory_unlocked(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MLOCK", raising=False)
    monkeypatch.setattr(resource, "getrlimit", lambda what: (8 << 20, 8 << 20))
    monkeypatch.setattr(ctypes, "CDLL", _no_libc)
    engine._lock_memory()


def test_unlimited_limit_locks_current_and_future_pages_on_fault(monkeypatch):
    calls = []

    class _Libc:
        def mlockall(self, flags):
            calls.append(flags)
            return 0

    monkeypatch.delenv("FREETOKEN_MLOCK", raising=False)
    monkeypatch.setattr(resource, "getrlimit", lambda what: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: _Libc())
    engine._lock_memory()
    assert calls == [1 | 2 | 4]  # MCL_CURRENT | MCL_FUTURE | MCL_ONFAULT
