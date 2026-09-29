from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from freetoken.utils import init_logger

from .utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi import Module

logger = init_logger(__name__)

# ROCm: "batch" = hipMemcpyBatchAsync (HIP >= 7.1), "loop" = one hipMemcpyAsync per entry;
# unset tries batch, then loop. The ROCm module has its own name because the AOT cache
# may hold a "batch_memcpy" built before ROCm support (its panic stub).
BATCH_MEMCPY_ROCM_ENV = "FREETOKEN_BATCH_MEMCPY_ROCM"


@lru_cache(maxsize=None)
def _jit_batch_memcpy_module() -> Module:
    return load_jit(
        "batch_memcpy",
        cuda_files=["batch_memcpy.cuh"],
        cuda_wrappers=[("batch_memcpy", "&BatchMemcpy::run")],
    )


@lru_cache(maxsize=None)
def _jit_batch_memcpy_rocm_module() -> Module:
    return load_jit(
        "batch_memcpy_rocm",
        "v2",  # v2: HIP_VERSION guard, sticky-error clear
        cuda_files=["batch_memcpy.cuh"],
        cuda_wrappers=[
            ("batch_memcpy", "&BatchMemcpy::run"),
            ("batch_memcpy_loop", "&BatchMemcpy::run_loop"),
        ],
    )


def _hip_version() -> tuple[int, int] | None:
    hip = getattr(torch.version, "hip", None)
    if not hip:
        return None
    return tuple(int(x) for x in hip.split(".")[:2])


def _load_rocm():
    hip = _hip_version()
    mode = os.getenv(BATCH_MEMCPY_ROCM_ENV, "").strip().lower()
    if mode not in ("", "batch", "loop"):
        raise RuntimeError(f"{BATCH_MEMCPY_ROCM_ENV}={mode!r}: expected batch or loop")
    module = _jit_batch_memcpy_rocm_module()
    modes = [mode] if mode else ["batch", "loop"]
    errors = []
    for m in modes:
        if m == "batch" and hip < (7, 1):
            errors.append(f"batch: hipMemcpyBatchAsync needs HIP >= 7.1 (torch built with {hip})")
            continue
        fn = module.batch_memcpy if m == "batch" else module.batch_memcpy_loop
        try:
            _probe(fn)
        except Exception as exc:  # noqa: BLE001 -- try the next mode
            errors.append(f"{m}: {exc}")
            continue
        logger.info(f"batch memcpy on ROCm: {m}")
        return fn
    raise RuntimeError("; ".join(errors))


def _probe(fn) -> None:
    """One real 16-byte H2D through the binding. A version-gated build (panic
    branch) or a driver without batch-memcpy support loads cleanly and only fails
    at call time; probing here turns every such mode into a load_batch_memcpy
    exception the caller's fallback path can catch."""
    src = torch.arange(16, dtype=torch.uint8).pin_memory()
    dst = torch.zeros(16, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    fn(
        torch.tensor([dst.data_ptr()]),
        torch.tensor([src.data_ptr()]),
        torch.tensor([16]),
        stream.cuda_stream,
    )
    stream.synchronize()
    if not torch.equal(dst.cpu(), src):
        raise RuntimeError("cudaMemcpyBatchAsync probe copied wrong bytes")


def load_batch_memcpy():
    """Build (once), probe, and return the batch-memcpy entry point, or raise.

    The 8-argument cudaMemcpyBatchAsync signature this binding uses is CUDA 13.0's
    (12.8/12.9 had an extra failIdx parameter); gate on the torch runtime version
    before paying for the JIT build, then verify with a real copy.
    """
    if _hip_version() is not None:
        return _load_rocm()
    cuda = torch.version.cuda
    if cuda is None or tuple(int(x) for x in cuda.split(".")[:2]) < (13, 0):
        raise RuntimeError(f"cudaMemcpyBatchAsync binding requires CUDA >= 13.0 (torch built with {cuda})")
    fn = _jit_batch_memcpy_module().batch_memcpy
    _probe(fn)
    return fn


def batch_memcpy_jit(
    dst_ptrs: torch.Tensor,
    src_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    stream: int,
) -> None:
    """Enqueue one cudaMemcpyBatchAsync of ``len(sizes)`` independent copies.

    ``dst_ptrs``/``src_ptrs``/``sizes`` are same-length CPU int64 tensors of raw
    addresses and byte counts; ``stream`` is a raw cudaStream_t handle
    (``torch.cuda.Stream.cuda_stream``), which must not be the legacy NULL stream.
    """
    load_batch_memcpy()(dst_ptrs, src_ptrs, sizes, stream)
