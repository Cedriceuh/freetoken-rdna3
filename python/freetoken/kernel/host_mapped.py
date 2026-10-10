"""Device tensors whose storage is pinned, device-mapped host memory (zero-copy over PCIe).

A pinned+mapped host allocation (``alloc_pinned_tensor``) is device-visible at its host address (UVA), but torch sees a
CPU tensor, so kernels and allocator-aware ops refuse it. ``host_mapped_empty`` hands the same memory to torch as a
GPU tensor through DLPack: every kernel then reads and writes it in place across PCIe, and no VRAM is used.

On ROCm the memory is allocated non-coherent (``hipHostMallocNonCoherent``): the GPU's L2 may then keep the lines it
reads, so the verify rows of one request, which select mostly the same tokens, cross PCIe once instead of once each
(QSA decode attention, 4 rows of 2051 tokens: 129 us instead of 313, 122 from VRAM; the coherent default of a mapped
allocation is not cached). Copies and host reads made after the kernels see what they wrote
(tests/kernels/test_qsa_kv_host.py); while kernels work on it, only the owning GPU may touch it.
"""

from __future__ import annotations

import ctypes
import functools
import math

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", _DLDevice), ("ndim", ctypes.c_int), ("dtype", _DLDataType),
                ("shape", ctypes.POINTER(ctypes.c_int64)), ("strides", ctypes.POINTER(ctypes.c_int64)),
                ("byte_offset", ctypes.c_uint64)]


_KDL_CUDA, _KDL_ROCM = 2, 10
_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_get_pointer.restype = ctypes.c_void_p
_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


_HIP_HOST_MALLOC_FLAGS = 0x1 | 0x2 | 0x80000000  # Portable | Mapped | NonCoherent


@functools.cache
def _hip() -> ctypes.CDLL:
    # the HIP runtime torch already loaded (same process state as torch's allocations)
    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        path = next(line.split()[-1] for line in maps if "libamdhip64" in line)
    lib = ctypes.CDLL(path)
    lib.hipHostMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.hipHostFree.argtypes = [ctypes.c_void_p]
    return lib


class _HostBlock:
    """One hipHostMalloc block, freed when the last tensor over it goes."""

    def __init__(self, nbytes: int) -> None:
        self.ptr = ctypes.c_void_p()
        self._free = _hip().hipHostFree  # bound now: module globals may be gone when the last tensor dies at exit
        err = _hip().hipHostMalloc(ctypes.byref(self.ptr), nbytes, _HIP_HOST_MALLOC_FLAGS)
        if err:
            raise RuntimeError(f"hipHostMalloc({nbytes} B, non-coherent) failed with HIP error {err}")

    def __del__(self) -> None:
        if self.ptr.value:
            self._free(self.ptr)


def _alloc_noncoherent(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    nbytes = math.prod(shape) * dtype.itemsize
    block = _HostBlock(nbytes)
    buf = (ctypes.c_char * nbytes).from_address(block.ptr.value)
    buf._block = block  # the tensor holds the buffer, the buffer the block
    return torch.frombuffer(buf, dtype=dtype).view(shape)


def host_mapped_empty(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An uninitialized ``device`` tensor of ``shape`` backed by pinned, mapped host memory."""
    if torch.version.hip is not None:
        host = _alloc_noncoherent(shape, dtype)
    else:
        host = alloc_pinned_tensor(*shape, dtype=dtype)
    capsule = torch.utils.dlpack.to_dlpack(host)  # the capsule's deleter keeps ``host`` alive with the new tensor
    managed = ctypes.cast(_get_pointer(capsule, b"dltensor"), ctypes.POINTER(_DLTensor)).contents
    managed.data = device_ptr(host)
    managed.device.device_type = _KDL_ROCM if torch.version.hip is not None else _KDL_CUDA
    managed.device.device_id = torch.device(device).index if torch.device(device).index is not None else \
        torch.cuda.current_device()
    out = torch.utils.dlpack.from_dlpack(capsule)
    assert out.is_cuda and out.data_ptr() == device_ptr(host), (out.device, out.data_ptr())
    return out
