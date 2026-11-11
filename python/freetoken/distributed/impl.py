from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from freetoken.distributed import DistributedInfo
    from freetoken.kernel import PyNCCLCommunicator


# FREETOKEN_NO_RCCL=1 (two ranks, with FREETOKEN_HOST_ALLREDUCE; default 0): the engine never creates the RCCL
# communicator, which takes 274 MiB of VRAM a card: the all-reduces above the host slots go through the mid slots
# (FREETOKEN_HOST_ALLREDUCE_MID_MB, 20 by default then), the broadcasts too, and the rare rest (a first-use agreement
# between the ranks) through the CPU process group
NO_RCCL = os.environ.get("FREETOKEN_NO_RCCL", "0") == "1"
_CPU_GROUP: list = [None]  # the CPU process group, set by the engine under NO_RCCL


def agree(flag: bool) -> bool:
    """Whether ``flag`` holds on every rank (a first-use agreement: the ranks must issue the same collectives)."""
    if dist.get_world_size() == 1:
        return bool(flag)
    if _CPU_GROUP[0] is not None:
        ok = torch.tensor([int(flag)], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=_CPU_GROUP[0])
    else:
        ok = torch.tensor([int(flag)], dtype=torch.int32, device=torch.cuda.current_device())
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    return bool(ok.item())


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

    def broadcast(self, x: torch.Tensor) -> torch.Tensor:
        """Rank 0's ``x`` into every rank's ``x`` (in place), as ``torch.distributed.broadcast(x, src=0)``."""
        if dist.get_world_size() > 1:
            dist.broadcast(x, src=0)
        return x


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
class CpuStagedDistributedImpl(DistributedImpl):
    """FREETOKEN_NO_RCCL's last resort, for what no host path takes: the tensor to the host, the collective on the CPU
    process group, back (a host sync each time). The same values: a sum of two ranks in fp32, bytes otherwise."""

    group: object

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        c = x.detach().float().cpu() if x.is_floating_point() else x.detach().cpu()
        dist.all_reduce(c, op=dist.ReduceOp.SUM, group=self.group)
        x.copy_(c.to(x.dtype))
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        world = dist.get_world_size(self.group)
        raw = x.detach().contiguous().view(-1).view(torch.uint8).cpu()
        out = torch.empty(world * raw.numel(), dtype=torch.uint8)
        dist.all_gather_into_tensor(out, raw, group=self.group)
        return out.to(x.device).view(x.dtype).view(world * x.shape[0], *x.shape[1:])

    def broadcast(self, x: torch.Tensor) -> torch.Tensor:
        raw = x.detach().contiguous().view(-1).view(torch.uint8).cpu()
        dist.broadcast(raw, src=0, group=self.group)
        x.copy_(raw.to(x.device).view(x.dtype).view(x.shape))
        return x


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


# FREETOKEN_HOST_AR_HC_COMBINE (default 1): a decoder layer sets HC_COMBINE[0] = {"res", "inj", "w", "eps", "rows",
# "hidden"} before its attention / mixer / MoE block; the block's closing all-reduce (the first matching one) takes it,
# runs the hyper-connection's combine + RMSNorm in the all-reduce's launch and leaves {"x": the reduced tensor, "rn":
# the normed streams} in it (kernel/host_allreduce.py HostAllReduce.hc_combine)
HC_COMBINE: list = [None]


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
        ctx = HC_COMBINE[0]
        if (ctx is not None and "rn" not in ctx and x.dim() == 2 and tuple(x.shape) == (ctx["rows"], ctx["hidden"])
                and x.dtype is torch.bfloat16 and x.is_contiguous() and self.ar.supports(x)):
            ctx["rn"] = self.ar.hc_combine(x, ctx["res"], ctx["inj"], ctx["w"], ctx["eps"])
            ctx["x"] = x
            return x
        if self.ar.supports(x):
            return self.ar(x)
        if self._big(x):
            return self.big(x)
        if self.ar.supports_mid(x):
            return self.ar.mid(x)
        return self.fallback.all_reduce(x)

    def all_reduce_async(self, x: torch.Tensor):
        if self.ar.supports(x):
            self.ar(x)
            return None
        if self._big(x):
            return self.big.all_reduce_async(x)
        if self.ar.supports_mid(x):
            self.ar.mid(x)
            return None
        return self.fallback.all_reduce_async(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        if self.ar.supports_gather(x):
            return self.ar.gather(x)
        return self.fallback.all_gather(x)

    def broadcast(self, x: torch.Tensor) -> torch.Tensor:
        # FREETOKEN_HOST_ALLGATHER: a small broadcast (a sampled step's tokens) through the gather's slots, rank 1
        # keeping rank 0's half: the same bytes, without RCCL's launch (~45 us of host time against ~13)
        if x.is_contiguous() and x.numel() and self.ar.supports_gather(x.view(-1)):
            both = self.ar.gather(x.view(-1))
            if self.ar.rank != 0:
                x.view(-1).copy_(both[: x.numel()])
            return x
        if self.ar.supports_mid_broadcast(x):
            return self.ar.mid_broadcast(x)
        return self.fallback.broadcast(x)


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_reduce_async(self, x: torch.Tensor):
        return self.plugins[-1].all_reduce_async(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)

    def broadcast(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].broadcast(x)


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
    if NO_RCCL:
        os.environ.setdefault(host_allreduce.MID_ENV, "20")  # every all-reduce below the copy engines' 20 MiB
        _CPU_GROUP[0] = tp_cpu_group
        DistributedCommunicator.plugins = [CpuStagedDistributedImpl(tp_cpu_group)]
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
