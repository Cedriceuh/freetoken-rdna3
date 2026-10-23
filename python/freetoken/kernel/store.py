from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)
# its own name since k and v take their own row strides: the JIT cache keys on the wrapper, not the .cu's text (the
# AOT build, kernel/aot.py, uses the same)
STORE_MODULE = "store_kv_strides"


@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, *config)
    return load_jit(
        STORE_MODULE,
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    # the warp copy moves 16-byte vectors: a row slice that is not 16-byte aligned goes through a contiguous copy
    if (k.data_ptr() | k.stride(0) * k.element_size()) % 16:
        k = k.contiguous()
    if (v.data_ptr() | v.stride(0) * v.element_size()) % 16:
        v = v.contiguous()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
