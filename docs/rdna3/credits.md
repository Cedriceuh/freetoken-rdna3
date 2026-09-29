# Credits

- **FreeToken** by FlashML ([FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken), Apache License 2.0):
  the engine this repository modifies (expert offload, radix and hybrid caches, FTW format, APIs, the Qwen3.8-Flash-Next
  model code, the exact sampling kernels whose barrier-free draw this build uses on ROCm).
- **zihaomu**: FreeToken's ROCm support pull requests
  [#133](https://github.com/FlashML-org/FreeToken/pull/133), [#134](https://github.com/FlashML-org/FreeToken/pull/134)
  and [#135](https://github.com/FlashML-org/FreeToken/pull/135) (HIP-portable JIT kernels, CUDA-only backends gated
  off, RCCL for tensor parallelism), included as they were proposed.
- **lukascechovic** ([lukascechovic/FreeToken](https://github.com/lukascechovic/FreeToken), branch `rocm-gfx1201`, a
  community ROCm port of FreeToken): the LDS budget clamp for RDNA's 64 KiB, the TP>1 relay
  handshake, the GDN boundary carry across prefill chunks, the first qwen4_exp tensor-parallel port that this one was
  rewritten from, and the `Dockerfile.gfx1201` (in that port) that `Dockerfile.rdna3` derives from.
- **flash-linear-attention**, vendored under `python/freetoken/kernel/fla` as upstream does.
- **Qwen3.8-Flash-Next** by the Qwen team and its NVFP4 checkpoint by RadixArk; their licenses apply to the weights.

Original to this repository: the uneven tensor-parallel split and unequal-card planning, weight-only int8 dense layers
and their GEMVs, the split-K bf16 GEMVs, the host-memory all-reduce, the top-k-first sampler and the sorted-threshold
sampling path for ROCm, rank-0 token broadcast, the tuned NVFP4 tiles, the blocked PLE and prefill MoE paths, expert
reuse in prefill on ROCm, batch-invariant multi-request decoding, the RAM tier for conversations, the profiles and
tooling, and the measurements in these docs.
