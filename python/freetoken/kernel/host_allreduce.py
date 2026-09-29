"""Two-rank all-reduce through shared host memory (FREETOKEN_HOST_ALLREDUCE=1).

For tensor parallelism on two GPUs without peer-to-peer access: the collective library stages
every small decode all-reduce through host memory with its own proxy machinery (~20-25 us per
5 KB message on the 7900 XTX + XT eagerly, ~12 us inside a CUDA graph). Here both ranks map the same shared-memory pages into
their GPU and one kernel per rank copies, signals, waits and sums (csrc/jit/host_allreduce.cuh).
Fixed launch arguments and a device-side sequence counter keep it CUDA-graph capturable.

Env:
  FREETOKEN_HOST_ALLREDUCE=1          enable (two ranks only; float32 / bfloat16, up to the slot size)
  FREETOKEN_HOST_ALLREDUCE_MAX        largest message in bytes (default 262144); bigger ones stay on RCCL
  FREETOKEN_HOST_ALLREDUCE_UNCACHED   1 (default): register the pages as uncached fine-grained memory;
                                      0: default registration (fine-grained)
  FREETOKEN_HOST_ALLREDUCE_TIMEOUT_S  a peer missing for this long traps the kernel (default 300; 0 = never).
                                      Generous on purpose: on a cold kernel cache the two ranks (different
                                      shapes under an uneven split) JIT-compile at different times.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import secrets
from functools import lru_cache

import torch

from .utils import load_jit

ENABLE_ENV = "FREETOKEN_HOST_ALLREDUCE"
MAX_ENV = "FREETOKEN_HOST_ALLREDUCE_MAX"
_DATA_OFFSET = 4096
_HIP_REGISTER_MAPPED = 0x2
_HIP_REGISTER_UNCACHED = 0x80000000


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def max_bytes() -> int:
    return int(os.environ.get(MAX_ENV, str(256 * 1024)))


@lru_cache(maxsize=None)
def _module():
    return load_jit(
        "host_allreduce",
        "v2",  # v2: alignas(16) out buffer
        cuda_files=["host_allreduce.cuh"],
        cuda_wrappers=[("run", "&HostAllReduce::run")],
    )


def _hip_library_path() -> str:
    """The HIP runtime torch already loaded (the pip ROCm SDK keeps it off the default loader path)."""
    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if "libamdhip64.so" in path:
                return path
    return "libamdhip64.so"


@lru_cache(maxsize=None)
def _hip():
    lib = ctypes.CDLL(_hip_library_path(), mode=ctypes.RTLD_GLOBAL)
    lib.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    lib.hipHostRegister.restype = ctypes.c_int
    lib.hipHostUnregister.argtypes = [ctypes.c_void_p]
    lib.hipHostUnregister.restype = ctypes.c_int
    lib.hipHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    return lib


class HostAllReduce:
    """One per rank; ``cpu_group`` (gloo) only exchanges the shared-memory name at setup."""

    def __init__(self, rank: int, world: int, cpu_group, slot_bytes: int | None = None) -> None:
        if world != 2:
            raise ValueError("host all-reduce supports exactly two ranks")
        if getattr(torch.version, "hip", None) is None:
            raise RuntimeError("host all-reduce is implemented for ROCm (hipHostRegister) only")
        import torch.distributed as dist

        self.rank = rank
        self.slot_bytes = int(slot_bytes or max_bytes())
        self.slot_bytes = (self.slot_bytes + 4095) // 4096 * 4096
        size = _DATA_OFFSET + 4 * self.slot_bytes
        name = [f"/dev/shm/freetoken-hostar-{os.getpid()}-{secrets.token_hex(6)}" if rank == 0 else None]
        dist.broadcast_object_list(name, src=0, group=cpu_group)
        path = name[0]
        if rank == 0:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
        dist.barrier(group=cpu_group)
        if rank != 0:
            fd = os.open(path, os.O_RDWR)
        self._map = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        dist.barrier(group=cpu_group)
        if rank == 0:
            os.unlink(path)  # both ranks hold the mapping; nothing is left in /dev/shm
        self._host = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        flags = _HIP_REGISTER_MAPPED
        if os.environ.get("FREETOKEN_HOST_ALLREDUCE_UNCACHED", "1") == "1":
            flags |= _HIP_REGISTER_UNCACHED
        hip = _hip()
        err = hip.hipHostRegister(ctypes.c_void_p(self._host), size, flags)
        if err != 0:
            raise RuntimeError(f"hipHostRegister failed ({err}) for the host all-reduce region")
        dev = ctypes.c_void_p()
        err = hip.hipHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(self._host), 0)
        if err != 0:
            raise RuntimeError(f"hipHostGetDevicePointer failed ({err})")
        self.base = int(dev.value)
        self.counter = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())
        timeout_s = float(os.environ.get("FREETOKEN_HOST_ALLREDUCE_TIMEOUT_S", "300"))
        # wall_clock64 ticks at 100 MHz on gfx9 / gfx11; 0 or less: effectively never
        self.timeout_ticks = int(timeout_s * 1e8) if timeout_s > 0 else 1 << 62
        self._size = size
        self._run = _module().run
        dist.barrier(group=cpu_group)

    def close(self) -> None:
        """Unregister and unmap the shared region (idempotent). The kernel must no longer run."""
        if getattr(self, "_map", None) is None:
            return
        torch.cuda.synchronize()
        _hip().hipHostUnregister(ctypes.c_void_p(self._host))
        self._map.close()
        self._map = None

    def supports(self, x: torch.Tensor) -> bool:
        """Rank-invariant choice (dtype, size, layout only): both ranks must take the same path."""
        return (x.dtype in (torch.bfloat16, torch.float32) and x.is_contiguous()
                and x.numel() * x.element_size() <= self.slot_bytes)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._run(x.view(-1), self.counter, self.base, self.rank, self.slot_bytes, self.timeout_ticks)
        return x
