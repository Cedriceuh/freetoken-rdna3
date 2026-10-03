# Changes from upstream

This repository is a modified version of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) (Apache
License 2.0), taken at `0d652e7` (2026-09-26). Every file changed or added relative to that commit is listed here with
the reason. Line counts are `added/removed`. Who wrote what: [credits.md](credits.md).

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
| `python/freetoken/models/qwen4_exp/weight.py` | 203/11 | per-rank sharding of the split weights (even or uneven split; the rest, the vision tower included, replicated) |
| `python/freetoken/models/qwen3_vl/vision.py` | 18/23 | the Qwen VL vision tower never split (whole on a rank) instead of tensor-parallel modules whose weights nothing sharded; a rank without a tower places nothing |
| `python/freetoken/models/qwen4_exp/model.py` | 3/1 | the vision tower built on rank 0 only (rank 0 encodes for every rank) |
| `python/freetoken/models/qwen4_exp/attention.py` | 20/7 | attention heads and output projection sharded |
| `python/freetoken/models/qwen4_exp/gdn.py` | 41/14 | GatedDeltaNet heads sharded, unevenly when a split is set |
| `python/freetoken/models/qwen4_exp/moe.py` | 30/2 | routed / shared experts sharded; one all-reduce per MoE block |
| `python/freetoken/models/qwen4_exp/ple.py` | 106/3 | PLE heads sharded (except with the disk source, which serves every head to every rank); the PLE convolution processed in time blocks for long and batched prefills |
| `python/freetoken/models/qwen4_exp/ple_disk.py` | 4/0 | the disk source declares that it serves every PLE head, so each rank reads them all |
| `python/freetoken/layers/quantization/moe/nvfp4.py`, `layers/quantization/moe/base.py` | 60/3, 14/1 | NVFP4 expert banks sliced per rank, unevenly when a split is set |
| `python/freetoken/models/qwen3_5_moe/moe.py` | 7/3 | `_SharedExpert` accepts a rank-local width (used by qwen4_exp's uneven split) |
| `python/freetoken/layers/linear.py`, `layers/moe.py` | 14/4, 4/1 | uneven shard sizes; the fused MoE all-reduce |
| `python/freetoken/kvcache/linear_state_pool.py` | 10/3 | GDN state pool sized per rank |
| `tests/models/qwen4_exp/test_tp_shard.py`, `test_tp_uneven.py`, `test_ple_conv_blocks.py`, `test_ple.py` | new / 2/1 | sharding, uneven split and blocked PLE tests; the snapshot test compares with a tolerance |
| `tests/models/qwen4_exp/common.py` | 16/2 | `as_rank()`: run a test block as one rank of a TP group; `meta_state_dict(vision=True)` |

## Two unequal cards

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/distributed/split.py` | 134/0 | the uneven split (`FREETOKEN_TP_SPLIT`): which dimensions split, rounding to NVFP4 scale blocks |
| `python/freetoken/engine/engine.py` | 124/8 | memory planned on the smaller card (`FREETOKEN_TP_ALLOW_IMBALANCE`); runtime cache rebuild refused under an uneven split; rank 0's sampled tokens broadcast (`FREETOKEN_TP_SYNC_TOKENS`); rank 0 encodes images and broadcasts the embeddings (the other ranks load no tower); int8 conversion; profiling hooks |
| `python/freetoken/distributed/impl.py`, `distributed/__init__.py` | 38/0, 2/1 | host-memory all-reduce wiring |
| `python/freetoken/kernel/host_allreduce.py`, `kernel/csrc/jit/host_allreduce.cuh` | 143/0, 179/0 | the host-memory all-reduce (`FREETOKEN_HOST_ALLREDUCE`) |
| `python/freetoken/scheduler/io.py` | 73/0 | the TP>1 relay handshakes before the first request |
| `tests/scheduler/test_io_relay_handshake.py` | new | |
| `tests/engine/test_mm_encoder.py` | 59/4 | rank 0's image embeddings reach rank 1, which never encodes |

## Dense layers

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/layers/quantization/int8_weight_only.py` | 173/0 | weight-only int8 conversion of the dense layers (`FREETOKEN_INT8_DENSE`; the vision tower stays bf16), allocator compaction |
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

## Speculative decoding with the MTP head

Line counts in this section and the next are relative to `d72e5cb` (0.1.0 plus image input), not to upstream.

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/spec_decode.py` | 274/0 | new: rows per step (capped by the tokens a request may still emit), the rollback targets, the adaptive depth with step costs per card count, the phase timing and stall probe |
| `python/freetoken/models/qwen4_exp/mtp.py` | 106/0 | new: the MTP head, and the NVFP4 quantizer for its bf16 experts |
| `python/freetoken/models/qwen4_exp/model.py`, `config.py`, `weight.py`, `__init__.py`, `models/config.py` | 30/0, 11/0, 74/3, 3/3, 4/1 | the head built and loaded as layer 48 (not without verify rows; unsupported head layouts refused); the streams it reads kept in buffers stable after capture |
| `python/freetoken/engine/engine.py` | 253/1 | the verify step, the drafts (chained), the rollback; the first token streamed before the head; image pad ids kept out of the head's lookup; the runtime cache rebuild refused while verify rows are on (the rollback holds the state pools); `FREETOKEN_MTP=1` on a model without the head refused at start |
| `python/freetoken/engine/spec_sample.py` | 106/0 | new: exact speculative sampling, with each sampler's own truncation |
| `python/freetoken/engine/graph.py` | 48/30 | a graph set per row count; `FREETOKEN_SPEC_BS_MAX` checked against them |
| `python/freetoken/scheduler/scheduler.py`, `scheduler/cache.py`, `scheduler/prefill.py`, `scheduler/status.py`, `core.py` | 134/40, 15/2, 2/0, 5/1, 6/0 | rows per step, pages allocated ahead, several kept tokens per step, the overlap loop with the head, no donated state from a verify step in flight; decode throughput counted in tokens |
| `python/freetoken/models/qwen4_exp/gdn.py`, `ple.py`, `ple_disk.py`, `models/qwen3_5_moe/gdn_kernels.py` | 48/2, 57/3, 37/6, 3/0 | m rows per request, with the states after each row saved; one row per norm program in decode; the PLE fill probe |
| `python/freetoken/kernel/triton/causal_conv1d_triton.py`, `kernel/fla/fused_sigmoid_gating_recurrent.py`, `kernel/fla/layernorm_gated.py`, `layers/norm.py` | 58/0, 2/1, 7/1, 4/2 | the conv over m rows in one launch; intermediate states; rows per norm program settable |
| `python/freetoken/kernel/triton/int8_gemv.py` | 53/1 | up to 4 rows share each weight tile (`FREETOKEN_INT8_ROW_GROUPS`) |
| `python/freetoken/attention/qsa_sparse.py`, `kernel/triton/qsa/attend.py`, `attention/linear.py`, `kvcache/qsa_pool.py`, `kvcache/__init__.py` | 22/13, 7/2, 4/1, 6/2, 2/1 | m rows per request in the attention metadata; decode keeps the one-row tile profile; the index ring sized for the drafts; the head's KV layer |
| `python/freetoken/layers/embedding.py` | 2/2 | the LM head's one-row shortcut keyed on rows; no second last-row gather |
| `python/freetoken/tokenizer/detokenize.py` | 20/0 | several tokens of one request in a step |
| `tests/test_spec_decode.py`, `tests/engine/test_spec_sample.py`, `tests/kernels/test_conv1d_decode_rows.py`, `tests/models/qwen4_exp/test_mtp.py`, `test_qsa_spec_rows.py`, `test_gdn.py`, `tests/scheduler/test_spec_alloc_ahead.py`, `test_scheduler_status.py`, `tests/tokenizer/test_detokenize_multi_token.py`, `tests/layers/test_int8_weight_only.py`, `tests/kvcache/test_qsa_pool.py` | new, except `test_gdn.py` 88/0, `test_scheduler_status.py` 11/0, `test_int8_weight_only.py` 18/0, `test_qsa_pool.py` 1/1 | tests of the above (`test_qsa_spec_rows.py` computes its toy projections a row at a time: ROCm 10's `F.linear` changes its sums with the row count for such shapes) |

## ROCm 10 image, PLE wait-sync, GGUF on ROCm, memory lock

| File | Lines | Change |
|---|---:|---|
| `Dockerfile.rdna3` | 23/5 | ROCm 10.0 / PyTorch 2.13 base; installs against its Triton 3.8, then copies the 7.14 image's Triton 3.7.1 over it (3.8 miscompiles these kernels; before the install, pip would replace torch 2.13, which requires its 3.8, with a CUDA one); pyproject's CUDA ceilings lifted at install |
| `python/freetoken/kernel/gguf.py`, `kernel/csrc/gguf/gguf_kernel.cu`, `dispatch.h`, `gguf_bind.cpp` | 41/8, 249/328, 37/9, new | the GGUF kernels build on ROCm: the torch wrappers in a host-only file (under HIP torch's headers include rocThrust, which the pip SDK lacks), nvcc-only flags for CUDA only, HIP's shuffles for the full-mask ones, the SDK's HIP headers, the shared ROCm link flags |
| `python/freetoken/kernel/csrc/row_store/row_store_ext.cpp`, `kernel/row_store.py` | 32/4, 40/1 | HIP stream memops for the PLE wait-sync (the probe used to look for the CUDA driver's only); a captured wait must hold three replays with the PLE copy behind it (`FREETOKEN_PLE_SYNC`) |
| `python/freetoken/kernel/host_allreduce.py`, `kernel/csrc/jit/host_allreduce.cuh` | 20/5, 11/5 | the optional wait log (`FREETOKEN_HOST_ALLREDUCE_WAITLOG`), removed at a clean shutdown |
| `python/freetoken/layers/moe.py`, `moe/offload_cache.py`, `models/qwen4_exp/moe.py` | 35/0, 20/0, 13/1 | the optional side-stream copy of the missing experts (`FREETOKEN_MOE_COPY_OVERLAP`) |
| `python/freetoken/engine/engine.py` | 30/0 | the process's memory locked once loaded (`FREETOKEN_MLOCK`) so that, with `vm.compact_unevictable_allowed=0`, memory compaction leaves the pages the GPU driver maps alone; a warning while it is 1 |
| `tests/kernels/test_row_store.py`, `tests/engine/test_lock_memory.py` | 32/0, new | the captured-wait probe; when the memory lock applies |

## Tooling and documentation

| File | Change |
|---|---|
| `python/freetoken/engine/ftprof.py` (93/0) | env-gated decode profiling (`FT_STATS_EVERY`, `FT_PROF_*`) |
| `python/freetoken/server/args.py` (3/1) | `--cuda-graph-max-bs` help |
| `Dockerfile.rdna3`, `.dockerignore` | the ROCm image |
| `rdna3/` | profiles (the MTP head on, TunableOp off), `serve.sh` (`--no-mtp`), GPU checks, micro-benchmarks and `bench/depth_sweep.py`, the ROCm 7.14 TunableOp files, maintenance tools |
| `README.md`, `NOTICE`, `CHANGELOG.md`, `llms.txt`, `llms-full.txt`, `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `docs/rdna3/` | this repository's documentation (the banner rendered from `docs/rdna3/assets/social-preview.html`); upstream's README moved to `docs/FREETOKEN_UPSTREAM_README.md` |
| `.github/` | upstream's release workflows and issue templates replaced by this repository's |
