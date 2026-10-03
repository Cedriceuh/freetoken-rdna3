"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp; their torch
wrappers moved to a host-only ``gguf_bind.cpp`` so that they also build on ROCm. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil

import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc

def _rocm_sdk_paths() -> tuple[list[str], list[str]]:
    """Include and link flags the ROCm pip SDK needs for this extension: torch's ROCM_HOME may not hold the HIP headers
    (7.14 wheels point it at the venv), and the SDK ships only libamdhip64.so.N, which the shared JIT link flags
    (``kernel.utils._rocm_link_flags``) expose under the -lamdhip64 name torch links with."""
    import importlib.util
    from torch.utils.cpp_extension import ROCM_HOME

    from freetoken.kernel.utils import _rocm_link_flags

    roots = [ROCM_HOME] if ROCM_HOME else []
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is not None and spec.submodule_search_locations:
        roots += list(spec.submodule_search_locations)
    root = next((r for r in roots if os.path.isfile(os.path.join(r, "include", "hip", "hip_runtime.h"))), None)
    try:
        ldflags = _rocm_link_flags()
    except RuntimeError:  # no libamdhip64 found: leave the link to torch's defaults
        ldflags = []
    return ([os.path.join(root, "include")] if root is not None else []), ldflags


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    extra_cuda_cflags = ["-O3"]
    # nvcc-only flags: on ROCm the device compiler is clang (called directly by torch's ROCm wheels, which reject them),
    # its own host compiler, and it treats constexpr functions as host + device already
    is_hip = torch.version.hip is not None
    if not is_hip:
        extra_cuda_cflags.append("--expt-relaxed-constexpr")
    host_cxx = _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        if not is_hip:
            extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu holds the kernels and their launchers; gguf_bind.cpp the torch wrappers and the PYBIND11_MODULE,
    # compiled as host code (torch's headers under HIP need rocThrust, which the ROCm pip SDK lacks).
    include_paths, ldflags = [str(_CSRC)], []
    if is_hip:  # the host compile of gguf_bind.cpp and the link need the SDK's own paths
        sdk_includes, ldflags = _rocm_sdk_paths()
        include_paths += sdk_includes
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu"), str(_CSRC / "gguf_bind.cpp")],
        extra_include_paths=include_paths,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=ldflags,
        verbose=True,
    )


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
