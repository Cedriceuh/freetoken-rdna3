from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    def all_reduce_async(self, x: torch.Tensor):
        """Start an in-place all-reduce of ``x``; returns a handle whose ``wait()`` makes the current stream wait for
        it, or None when it already ran on the current stream."""
        self.all_reduce(x)
        return None

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_reduce_async(self, x: torch.Tensor):
        if dist.get_world_size() == 1:
            return None
        return dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


@dataclass
class HostStagedDistributedImpl(DistributedImpl):
    """Small all-reduces through shared host memory (kernel/host_allreduce.py), large ones through it by the copy
    engines when ``big`` is set (kernel/dma_allreduce.py), the rest on ``fallback``. The choice depends only on dtype /
    size / layout, identical on both ranks."""

    ar: object
    fallback: DistributedImpl
    big: object | None = None

    def _big(self, x: torch.Tensor) -> bool:
        if self.big is None or not self.big.supports(x):
            return False
        from freetoken.moe import prefill_hooks

        prefill_hooks.ASYNC_AR[0] = True  # the compute stream waits for the copy streams on the GPU
        return True

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.ar.supports(x):
            return self.ar(x)
        if self._big(x):
            return self.big(x)
        return self.fallback.all_reduce(x)

    def all_reduce_async(self, x: torch.Tensor):
        if self.ar.supports(x):
            self.ar(x)
            return None
        if self._big(x):
            return self.big.all_reduce_async(x)
        return self.fallback.all_reduce_async(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.fallback.all_gather(x)


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_reduce_async(self, x: torch.Tensor):
        return self.plugins[-1].all_reduce_async(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from freetoken.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def enable_host_allreduce(tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup) -> bool:
    """FREETOKEN_HOST_ALLREDUCE=1 with two ranks: route small all-reduces through host memory."""
    from freetoken.kernel import host_allreduce

    if tp_info.size != 2 or not host_allreduce.enabled():
        return False
    if getattr(torch.version, "hip", None) is None:
        from freetoken.utils import init_logger

        init_logger(__name__).warning("FREETOKEN_HOST_ALLREDUCE=1 ignored: ROCm (hipHostRegister) only")
        return False
    ar = host_allreduce.HostAllReduce(tp_info.rank, tp_info.size, tp_cpu_group)
    from freetoken.kernel import dma_allreduce

    big = dma_allreduce.DmaAllReduce(tp_info.rank, tp_info.size, tp_cpu_group) if dma_allreduce.enabled() else None
    fallback = DistributedCommunicator.plugins[-1]
    DistributedCommunicator.plugins.append(HostStagedDistributedImpl(ar, fallback, big))
    return True


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    for plugin in DistributedCommunicator.plugins:
        for ar in (getattr(plugin, "ar", None), getattr(plugin, "big", None)):
            if ar is not None and hasattr(ar, "close"):
                ar.close()  # unregister before the mapping can be garbage-collected
    DistributedCommunicator.plugins = []
