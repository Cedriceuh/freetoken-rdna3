"""Two-rank all-reduce of large messages through shared host memory, moved by the copy engines (FREETOKEN_DMA_ALLREDUCE=1).

Without peer access between the two GPUs, RCCL stages a prefill's all-reduce through host memory at ~10-11 GB/s, with
kernels that hold CUs while the bytes cross PCIe (7.8 ms for 80 MiB on the 7900 XTX + XT). Here each rank copies its
message to its slot of a shared pinned region with the copy engines (D2H), piece by piece, and flags each piece; the
other rank waits for the flag on a stream (hipStreamWaitValue64), copies the piece into a small VRAM ring (H2D) and adds
it to its own. Both PCIe directions work at once (28 GB/s each on these cards), no CU waits on the bus, and a rank adds
the same two values RCCL adds: the same bits (one addition per element, commutative).

Every runtime call is made on the caller's thread, in program order: a stream that waits on the other rank's flag can
share a hardware queue with RCCL's spinning kernels, and both ranks issuing everything in the same order is what keeps
those waits from forming a cycle (issued from a thread, a 96k-token read hung on an RCCL all-reduce behind them).

Env:
  FREETOKEN_DMA_ALLREDUCE=1            enable (with FREETOKEN_HOST_ALLREDUCE, two ranks, ROCm)
  FREETOKEN_DMA_ALLREDUCE_MIN_KB       smallest message taken (default 20480: a 4096-token block); smaller ones stay on
                                       RCCL / the host path (a short prefill streams its experts over PCIe meanwhile)
  FREETOKEN_DMA_ALLREDUCE_SLOT_MB      largest message (default 80: a 16k-token chunk of hidden 2560 in bf16)
  FREETOKEN_DMA_ALLREDUCE_PIECE_KB     piece size (default 8192); the VRAM ring holds 4 pieces
"""
from __future__ import annotations

import ctypes
import mmap
import os
import secrets

import torch

from .host_allreduce import _hip_library_path

ENABLE_ENV = "FREETOKEN_DMA_ALLREDUCE"
_HDR = 4096
_RING = 4
_ALL = 0xFFFFFFFFFFFFFFFF
_D2H, _H2D = 2, 1
_GTE = 0  # hipStreamWaitValueGte


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


_lib = None


def _hip():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_hip_library_path(), mode=ctypes.RTLD_GLOBAL)
        vp, sz = ctypes.c_void_p, ctypes.c_size_t
        lib.hipHostRegister.argtypes = [vp, sz, ctypes.c_uint]
        lib.hipHostUnregister.argtypes = [vp]
        lib.hipHostGetDevicePointer.argtypes = [ctypes.POINTER(vp), vp, ctypes.c_uint]
        lib.hipMemcpyAsync.argtypes = [vp, vp, sz, ctypes.c_int, vp]
        lib.hipStreamWaitValue64.argtypes = [vp, vp, ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint64]
        lib.hipStreamWriteValue64.argtypes = [vp, vp, ctypes.c_uint64, ctypes.c_uint]
        for f in ("hipHostRegister", "hipHostUnregister", "hipHostGetDevicePointer", "hipMemcpyAsync",
                  "hipStreamWaitValue64", "hipStreamWriteValue64"):
            getattr(lib, f).restype = ctypes.c_int
        _lib = lib
    return _lib


def _check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"DMA all-reduce: {what} failed ({err})")


class _Done:
    """``wait()``: the current stream waits for the sum (the all_reduce_async handle)."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self.event = event

    def wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.event)


class DmaAllReduce:
    """One per rank; ``cpu_group`` (gloo) only exchanges the shared-memory name at setup. Both ranks must call it with
    the same sizes in the same order (the choice in ``supports`` depends on dtype, size and layout only)."""

    def __init__(self, rank: int, world: int, cpu_group) -> None:
        if world != 2:
            raise ValueError("DMA all-reduce supports exactly two ranks")
        import torch.distributed as dist

        self.rank, self.peer = rank, 1 - rank
        self.min_bytes = int(os.environ.get("FREETOKEN_DMA_ALLREDUCE_MIN_KB", "20480")) << 10
        self.slot_bytes = int(os.environ.get("FREETOKEN_DMA_ALLREDUCE_SLOT_MB", "80")) << 20
        self.piece = int(os.environ.get("FREETOKEN_DMA_ALLREDUCE_PIECE_KB", "8192")) << 10
        size = _HDR + 2 * self.slot_bytes
        name = [f"/dev/shm/freetoken-dmaar-{os.getpid()}-{secrets.token_hex(6)}" if rank == 0 else None]
        dist.broadcast_object_list(name, src=0, group=cpu_group)
        if rank == 0:
            fd = os.open(name[0], os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
        dist.barrier(group=cpu_group)
        if rank != 0:
            fd = os.open(name[0], os.O_RDWR)
        self._map = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        dist.barrier(group=cpu_group)
        if rank == 0:
            os.unlink(name[0])  # both ranks hold the mapping; nothing is left in /dev/shm
        self._host = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        hip = _hip()
        _check(hip.hipHostRegister(ctypes.c_void_p(self._host), size, 0x2), "hipHostRegister")  # mapped
        dev = ctypes.c_void_p()
        _check(hip.hipHostGetDevicePointer(ctypes.byref(dev), ctypes.c_void_p(self._host), 0), "hipHostGetDevicePointer")
        flags = int(dev.value)
        # pieces of rank r's slot in host memory, and pieces of the other slot rank r has copied in (cumulative)
        self._ready = [ctypes.c_void_p(flags + r * 64) for r in range(2)]
        self._read = [ctypes.c_void_p(flags + 256 + r * 64) for r in range(2)]
        self._slot = [self._host + _HDR + r * self.slot_bytes for r in range(2)]
        self._out, self._in = torch.cuda.Stream(), torch.cuda.Stream()  # D2H; H2D and the sums
        self._ring = torch.empty(_RING * self.piece, dtype=torch.uint8, device=torch.cuda.current_device())
        self._pieces = 0  # pieces each rank has sent so far: the same count on both ranks
        dist.barrier(group=cpu_group)

    def close(self) -> None:
        if getattr(self, "_map", None) is None:
            return
        torch.cuda.synchronize()
        _hip().hipHostUnregister(ctypes.c_void_p(self._host))
        self._map.close()
        self._map = None

    def supports(self, x: torch.Tensor) -> bool:
        n = x.numel() * x.element_size()
        return (x.dtype in (torch.bfloat16, torch.float32) and x.is_contiguous() and self.min_bytes <= n <= self.slot_bytes
                and not torch.cuda.is_current_stream_capturing())

    def all_reduce_async(self, x: torch.Tensor) -> _Done:
        """Queue the in-place sum of ``x`` on both ranks; ``x`` must stay alive and untouched until ``wait()``."""
        hip = _hip()
        start = torch.cuda.Event()
        start.record()
        self._out.wait_event(start)
        so, si = ctypes.c_void_p(self._out.cuda_stream), ctypes.c_void_p(self._in.cuda_stream)
        n, el = x.numel() * x.element_size(), x.element_size()
        base = self._pieces
        mine, theirs = self._slot[self.rank], self._slot[self.peer]
        # the other rank has copied in every piece of this rank's earlier messages: the slot can be written again
        _check(hip.hipStreamWaitValue64(so, self._read[self.peer], base, _GTE, _ALL), "hipStreamWaitValue64")
        flat, ring, src, dst = x.view(-1), self._ring.view(x.dtype), x.data_ptr(), self._ring.data_ptr()
        for i, off in enumerate(range(0, n, self.piece)):
            ln, g, k = min(self.piece, n - off), base + i + 1, i % _RING
            _check(hip.hipMemcpyAsync(ctypes.c_void_p(mine + off), ctypes.c_void_p(src + off), ln, _D2H, so), "D2H")
            _check(hip.hipStreamWriteValue64(so, self._ready[self.rank], g, 0), "hipStreamWriteValue64")
            # the other rank's piece is in host memory, and this rank's too (the sum overwrites it); the ring slot was
            # last read by a sum earlier on this stream
            _check(hip.hipStreamWaitValue64(si, self._ready[self.peer], g, _GTE, _ALL), "hipStreamWaitValue64")
            _check(hip.hipStreamWaitValue64(si, self._ready[self.rank], g, _GTE, _ALL), "hipStreamWaitValue64")
            _check(hip.hipMemcpyAsync(ctypes.c_void_p(dst + k * self.piece), ctypes.c_void_p(theirs + off), ln, _H2D, si),
                   "H2D")
            _check(hip.hipStreamWriteValue64(si, self._read[self.rank], g, 0), "hipStreamWriteValue64")
            with torch.cuda.stream(self._in):
                p0 = k * self.piece // el
                flat[off // el:(off + ln) // el].add_(ring[p0:p0 + ln // el])
        self._pieces = base + (n + self.piece - 1) // self.piece
        done = torch.cuda.Event()
        done.record(self._in)
        return _Done(done)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.all_reduce_async(x).wait()
        return x
