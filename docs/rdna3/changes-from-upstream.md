# Changes from upstream

This repository is a modified version of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) (Apache
License 2.0), taken at `0d652e7` (2026-09-26). Every file changed or added relative to that commit is listed here with
the reason, as the notice of changes the Apache License 2.0 asks for (section 4(b)). Line counts are `added/removed`. Who wrote what:
[credits.md](credits.md).

## ROCm portability (upstream PRs #133-#135 by zihaomu, and fixes on top)

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/backend.py` | 10/13 | CUDA-only backends (and the NVFP4 CUDA kernels) gated off on ROCm; Triton on ROCm |
| `python/freetoken/kernel/csrc/include/freetoken/utils.cuh` | 11/1 | TVM-FFI JIT kernels portable to HIP |
| `python/freetoken/kernel/csrc/jit/index.cu`, `store.cu` | 10/8, 5/5 | same |
| `python/freetoken/kernel/pynccl.py` | 7/1 | tensor parallelism routed to RCCL on ROCm |
| `python/freetoken/kernel/triton/qsa/attend.py` | 51/3 | BLOCK_N clamped to RDNA's 64 KiB LDS; int8 K/V path (experimental) |
| `tests/engine/test_rocm_communication.py`, `tests/kernels/test_backend.py`, `tests/kernels/test_jit_index_store.py` | new | tests of the above |

## Qwen3.8-Flash-Next (qwen4_exp) on several GPUs

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/models/qwen4_exp/weight.py` | 205/11 | per-rank sharding of every weight (even or uneven split); explicit error for the vision tower at TP>1 |
| `python/freetoken/models/qwen4_exp/attention.py` | 20/7 | attention heads and output projection sharded |
| `python/freetoken/models/qwen4_exp/gdn.py` | 41/14 | GatedDeltaNet heads sharded, unevenly when a split is set |
| `python/freetoken/models/qwen4_exp/moe.py` | 30/2 | routed / shared experts sharded; one all-reduce per MoE block |
| `python/freetoken/models/qwen4_exp/ple.py` | 106/3 | PLE heads sharded; the PLE convolution processed in time blocks for long and batched prefills |
| `python/freetoken/models/qwen4_exp/ple_disk.py` | 4/0 | per-rank PLE rows from disk |
| `python/freetoken/layers/quantization/moe/nvfp4.py`, `layers/quantization/moe/base.py` | 60/3, 14/1 | NVFP4 expert banks sliced per rank, unevenly when a split is set |
| `python/freetoken/models/qwen3_5_moe/moe.py` | 7/3 | `_SharedExpert` accepts a rank-local width (used by qwen4_exp's uneven split) |
| `python/freetoken/layers/linear.py`, `layers/moe.py` | 14/4, 4/1 | uneven shard sizes; the fused MoE all-reduce |
| `python/freetoken/kvcache/linear_state_pool.py` | 10/3 | GDN state pool sized per rank |
| `tests/models/qwen4_exp/test_tp_shard.py`, `test_tp_uneven.py`, `test_ple_conv_blocks.py`, `test_ple.py` | new / 2/1 | sharding, uneven split and blocked PLE tests; the snapshot test compares with a tolerance |
| `tests/models/qwen4_exp/common.py` | 14/0 | `as_rank()`: run a test block as one rank of a TP group |

## Two unequal cards

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/distributed/split.py` | 134/0 | the uneven split (`FREETOKEN_TP_SPLIT`): which dimensions split, rounding to NVFP4 scale blocks |
| `python/freetoken/engine/engine.py` | 95/6 | memory planned on the smaller card (`FREETOKEN_TP_ALLOW_IMBALANCE`); runtime cache rebuild refused under an uneven split; rank 0's sampled tokens broadcast (`FREETOKEN_TP_SYNC_TOKENS`); int8 conversion; profiling hooks |
| `python/freetoken/distributed/impl.py`, `distributed/__init__.py` | 38/0, 2/1 | host-memory all-reduce wiring |
| `python/freetoken/kernel/host_allreduce.py`, `kernel/csrc/jit/host_allreduce.cuh` | 143/0, 179/0 | the host-memory all-reduce (`FREETOKEN_HOST_ALLREDUCE`) |
| `python/freetoken/scheduler/io.py` | 73/0 | the TP>1 relay handshakes before the first request |
| `tests/scheduler/test_io_relay_handshake.py` | new | |

## Dense layers

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/layers/quantization/int8_weight_only.py` | 171/0 | weight-only int8 conversion of the dense layers (`FREETOKEN_INT8_DENSE`), allocator compaction |
| `python/freetoken/kernel/triton/int8_gemv.py` | 132/0 | int8 decode GEMV for RDNA3, row-looped for several requests |
| `python/freetoken/kernel/triton/dense_gemv.py` | 92/0 | split-K bf16 decode GEMVs for RDNA3, row-looped (batch-invariant) |
| `python/freetoken/layers/quantization/linear/unquantized.py` | 33/0 | route bf16 decode projections to those GEMVs on gfx11 |
| `tests/layers/test_int8_weight_only.py` | new | |

## Experts and prefill

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/moe/fused_nvfp4.py` | 73/8 | tuned NVFP4 prefill / decode tiles for RDNA3 (opt-in, bit-exact); token-blocked prefill MoE |
| `python/freetoken/moe/offload_cache.py` | 16/2 | expert reuse during prefill on ROCm |
| `python/freetoken/kernel/batch_memcpy.py`, `kernel/csrc/jit/batch_memcpy.cuh` | 55/0, 86/1 | `hipMemcpyBatchAsync` (HIP >= 7.1) with a per-copy fallback |

## Sampling

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/triton/sampling_topk.py` | 71/0 | top-k-first sampler for small `top_k` |
| `python/freetoken/kernel/triton/sampling.py` | 55/4 | on ROCm, top-k / top-p from an exact sorted threshold + the barrier-free draw kernels (upstream's cooperative kernels hang on RDNA3); renormalize API from the same threshold |
| `python/freetoken/engine/sample.py` | 38/2 | routing between the two paths |

## Several requests and scheduling

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/engine/graph.py` | 5/0 | CUDA graphs captured for every batch size up to an explicit `--cuda-graph-max-bs` <= 8 |
| `python/freetoken/scheduler/scheduler.py` | 16/5 | optional decode interleaving between prefill chunks (`FREETOKEN_DECODE_INTERLEAVE`, off); RAM-tier hooks |
| `python/freetoken/scheduler/prefill.py`, `scheduler/cache.py` | 19/1, 93/3 | the GDN boundary carried across prefill chunk seams; RAM-tier matching and promotion |
| `tests/scheduler/test_gdn_boundary_carry.py` | new | |

## Conversations kept in RAM

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kvcache/host_kv_pool.py` | 334/0 | the RAM tier's buffers and copies (`FREETOKEN_HOST_KV`) |
| `python/freetoken/kvcache/hybrid_radix_cache.py` | 296/27 | demotion on eviction (with node split), promotion on match, pins |
| `python/freetoken/kvcache/radix_cache.py`, `kvcache/mha_pool.py`, `kvcache/qsa_pool.py` | 13/0, 3/0, 74/1 | node state for the tier; pools opt in explicitly; experimental int8 QSA K/V |
| `python/freetoken/attention/qsa_sparse.py`, `kernel/triton/qsa/kv_int8.py` | 3/0, 67/0 | experimental int8 QSA K/V (not recommended) |
| `tests/kvcache/test_host_kv_tier.py`, `tests/kvcache/test_qsa_kv_int8.py` | new | |

## Tooling and documentation

| File | Change |
|---|---|
| `python/freetoken/engine/ftprof.py` (93/0) | env-gated decode profiling (`FT_STATS_EVERY`, `FT_PROF_*`) |
| `python/freetoken/server/args.py` (3/1) | `--cuda-graph-max-bs` help |
| `Dockerfile.rdna3`, `.dockerignore` | the ROCm image |
| `rdna3/` | profiles, `serve.sh`, GPU checks, micro-benchmarks, TunableOp files, maintenance tools |
| `README.md`, `NOTICE`, `CHANGELOG.md`, `llms.txt`, `llms-full.txt`, `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `docs/rdna3/` | this repository's documentation; upstream's README moved to `docs/FREETOKEN_UPSTREAM_README.md` |
| `.github/` | upstream's release workflows and issue templates replaced by this repository's |
