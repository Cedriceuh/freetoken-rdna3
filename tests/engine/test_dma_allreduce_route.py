"""FREETOKEN_DMA_ALLREDUCE: which all-reduces the host-staged communicator gives the copy-engine path (CPU, fakes)."""

import torch

from freetoken.distributed.impl import HostStagedDistributedImpl
from freetoken.moe import prefill_hooks


class _Path:
    def __init__(self, name, limit_lo, limit_hi, calls):
        self.name, self.lo, self.hi, self.calls = name, limit_lo, limit_hi, calls

    def supports(self, x):
        return self.lo <= x.numel() <= self.hi

    def __call__(self, x):
        self.calls.append((self.name, "sync"))
        return x

    def all_reduce(self, x):
        return self(x)

    def all_reduce_async(self, x):
        self.calls.append((self.name, "async"))
        return self.name


def test_sizes_pick_host_then_dma_then_fallback():
    calls = []
    impl = HostStagedDistributedImpl(_Path("host", 0, 10, calls), _Path("rccl", 0, 1 << 30, calls),
                                     _Path("dma", 100, 1000, calls))
    prefill_hooks.ASYNC_AR[0] = False
    for n in (5, 50, 500, 5000):
        impl.all_reduce(torch.zeros(n))
    assert [c[0] for c in calls] == ["host", "rccl", "dma", "rccl"]
    assert prefill_hooks.ASYNC_AR[0]  # the engine synchronizes before the next decode step
    calls.clear()
    assert impl.all_reduce_async(torch.zeros(500)) == "dma"
    assert impl.all_reduce_async(torch.zeros(5)) is None  # the host path runs in place
    assert impl.all_reduce_async(torch.zeros(50)) == "rccl"
    assert calls == [("dma", "async"), ("host", "sync"), ("rccl", "async")]
    prefill_hooks.ASYNC_AR[0] = False


def test_without_dma_nothing_changes():
    calls = []
    impl = HostStagedDistributedImpl(_Path("host", 0, 10, calls), _Path("rccl", 0, 1 << 30, calls))
    prefill_hooks.ASYNC_AR[0] = False
    impl.all_reduce(torch.zeros(500))
    assert calls == [("rccl", "sync")] and not prefill_hooks.ASYNC_AR[0]
