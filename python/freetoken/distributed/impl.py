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
    """Small all-reduces through shared host memory (kernel/host_allreduce.py), the rest on
    ``fallback``. The choice depends only on dtype / size / layout, identical on both ranks."""

    ar: object
    fallback: DistributedImpl

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.ar.supports(x):
            return self.ar(x)
        return self.fallback.all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.fallback.all_gather(x)


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

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
    fallback = DistributedCommunicator.plugins[-1]
    DistributedCommunicator.plugins.append(HostStagedDistributedImpl(ar, fallback))
    return True


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    for plugin in DistributedCommunicator.plugins:
        ar = getattr(plugin, "ar", None)
        if ar is not None and hasattr(ar, "close"):
            ar.close()  # unregister before the mapping can be garbage-collected
    DistributedCommunicator.plugins = []
