# Changes from upstream

This repository is a modified version of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) (Apache
License 2.0), taken at `0d652e7` and merged with upstream up to `ee9789f`, then up to `9b68e00` (five of its commits
cherry-picked first, then the rest merged), then up to `cc56b6b`. Every file changed or added relative to upstream is
listed here with the reason. Line counts are `added/removed`, against the upstream commit a section or row names,
`0d652e7` otherwise; `git diff cc56b6b` gives the current ones. Files the merge made
identical to upstream again are gone from the tables ([maintaining.md](maintaining.md#the-stack-on-top-of-upstream-oldest-first)).
Who wrote what: [credits.md](credits.md).

## ROCm portability (upstream PRs #133-#135 by zihaomu, and fixes on top)

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/backend.py` | 2/2 against `ee9789f` | the module docstring (the gating itself is upstream's #134, `9b585b7`) |
| `python/freetoken/kernel/csrc/include/freetoken/utils.cuh` | 1/1 against `ee9789f` | on ROCm, `with_attr` refuses `use_pdl=True` (upstream ignores it); the HIP-portable JIT kernels are upstream's (`803a3b4`) |
| `python/freetoken/kernel/pynccl.py` | 7/1 | tensor parallelism routed to RCCL on ROCm |
| `python/freetoken/kernel/triton/qsa/attend.py` | 51/3 | BLOCK_N clamped to RDNA's 64 KiB LDS, sized with q's itemsize (upstream's own clamp, `7d6490c`, sized with the K/V cache's, runs after it and changes no tile; alone it would keep 64 with 1-byte K/V); int8 K/V path (off, not recommended) |
| `tests/engine/test_rocm_communication.py`, `tests/kernels/test_backend.py`, `tests/kernels/test_jit_index_store.py` | new | tests of the above |

## Qwen3.8-Flash-Next (qwen4_exp) on several GPUs

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/models/qwen4_exp/weight.py` | 203/11 | per-rank sharding of the split weights (even or uneven split; the rest, the vision tower included, replicated), ahead of upstream's scheme-driven dense reader (`163d1de`) |
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
| `python/freetoken/moe/fused_nvfp4.py`, `kernel/nvfp4_moe.py` | 73/8, 23/1 against `cc56b6b` | tuned NVFP4 prefill / decode tiles for RDNA3 (opt-in, bit-exact); token-blocked prefill MoE. The decode tiles were swept again for upstream's marlin decode kernel (`7644984`) and live in `kernel/nvfp4_moe.py`, where upstream moved the launches (`ac6a15d`); the prefill tiles were swept again for upstream's rewritten prefill GEMM (`03c2c9c`) |
| `python/freetoken/moe/offload_cache.py` | 16/2 | expert reuse during prefill on ROCm |
| `python/freetoken/kernel/batch_memcpy.py`, `kernel/csrc/jit/batch_memcpy.cuh` | 55/0, 86/1 | `hipMemcpyBatchAsync` (HIP >= 7.1) with a per-copy fallback |

## Sampling

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/triton/sampling_topk.py` | 71/0 | top-k-first sampler for small `top_k` |
| `python/freetoken/kernel/triton/sampling.py` | 55/4 | on ROCm, top-k / top-p from an exact sorted threshold + the barrier-free draw kernels (upstream's cooperative kernels hang on RDNA3); renormalize API from the same threshold |
| `python/freetoken/engine/sample.py` | 38/4 against `9b68e00` | routing between the two paths, under upstream's presence / frequency penalties (`c818639`), which `engine.py` also applies on MTP verify rows |

## Several requests and scheduling

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/engine/graph.py` | 5/0 | CUDA graphs captured for every batch size up to an explicit `--cuda-graph-max-bs` <= 8 |
| `python/freetoken/scheduler/scheduler.py` | 16/5 | optional decode interleaving between prefill chunks (`FREETOKEN_DECODE_INTERLEAVE`, off); RAM-tier hooks |
| `python/freetoken/scheduler/prefill.py`, `scheduler/cache.py` | 19/1, 93/3 | the GDN boundary carried across prefill chunk seams; RAM-tier matching and promotion |
| `tests/scheduler/test_gdn_boundary_carry.py` | new | |
| `python/freetoken/server/api_server.py`, `tests/server/test_backend_death_exit.py` | 12/4, new | a worker dead after start ends the API server with exit code 1 after a 10 s grace, streams still open included: it stops the surviving workers itself (SIGKILL after 5 s); upstream's SIGTERM to itself went through uvicorn's graceful stop, which waits for open streams without a timeout, and a stream to a dead backend never ends |

## Conversations kept in RAM

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kvcache/host_kv_pool.py` | 334/0 | the RAM tier's buffers and copies (`FREETOKEN_HOST_KV`) |
| `python/freetoken/kvcache/hybrid_radix_cache.py` | 296/27 | demotion on eviction (with node split), promotion on match, pins |
| `python/freetoken/kvcache/radix_cache.py`, `kvcache/mha_pool.py`, `kvcache/qsa_pool.py` | 13/0, 3/0, 74/1 | node state for the tier; pools opt in explicitly; int8 QSA K/V (off, not recommended) |
| `python/freetoken/attention/qsa_sparse.py`, `kernel/triton/qsa/kv_int8.py` | 3/0, 67/0 | int8 QSA K/V (off, not recommended) |
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

## ROCm 10 image, PLE wait-sync, memory lock

| File | Lines | Change |
|---|---:|---|
| `Dockerfile.rdna3` | 13/5 | ROCm 10.1 / PyTorch 2.14 base with its own Triton 3.8; pyproject's torchvision floor lowered to ROCm's pre-release (0.29.0a0) at install |
| `python/freetoken/kernel/triton/moe_align.py`, `tests/kernels/test_moe_align.py`, `moe/fused.py` | 13/9, new, 10/12 against `9b68e00` | ROCm keeps this align over upstream's staged one (`0bca88f`, upstream's fix for the same bug); no comparison on `tl.histogram` counts: Triton 3.8's AMD range analysis gives them an empty range and folds such comparisons to false, which left the expert ids unwritten (up to 1024 routes) or every count at zero (more), and the expert GEMMs then read wild addresses |
| `python/freetoken/kernel/csrc/row_store/row_store_ext.cpp`, `kernel/row_store.py`, `models/qwen4_exp/ple_disk.py` | 32/4, 40/1, 18/1 | HIP stream memops for the PLE wait-sync (upstream's probe looks for the CUDA driver's only); a captured wait must hold three replays with the PLE copy behind it (`FREETOKEN_PLE_SYNC`); on one GPU without the MTP head (overlapped decode steps) `auto` takes launch-gating, 44 % faster there |
| `tests/models/qwen4_exp/test_ple_disk.py` | 27/0 | `FREETOKEN_PLE_SYNC=auto` takes launch-gating on one GPU when the decode steps overlap (no MTP head) |
| `python/freetoken/kernel/host_allreduce.py`, `kernel/csrc/jit/host_allreduce.cuh` | 20/5, 11/5 | the optional wait log (`FREETOKEN_HOST_ALLREDUCE_WAITLOG`), removed at a clean shutdown |
| `python/freetoken/layers/moe.py`, `moe/offload_cache.py`, `models/qwen4_exp/moe.py` | 35/0, 20/0, 13/1 | the optional side-stream copy of the missing experts (`FREETOKEN_MOE_COPY_OVERLAP`) |
| `python/freetoken/engine/engine.py` | 30/0 | the process's memory locked once loaded (`FREETOKEN_MLOCK`) so that, with `vm.compact_unevictable_allowed=0`, memory compaction leaves the pages the GPU driver maps alone; a warning while it is 1 |
| `tests/kernels/test_row_store.py`, `tests/engine/test_lock_memory.py` | 32/0, new | the captured-wait probe; when the memory lock applies |

## EXL3 checkpoints

Line counts here are on top of the lines listed in the sections above.

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/layers/quantization/exl3_codec.py` | new | the EXL3 format in plain torch (trellis rings, codebooks, tile order, Hadamard reconstruction, n-gram rings): the reference the kernels are checked against |
| `python/freetoken/kernel/triton/exl3.py` | new | gfx1100 Triton kernels: tile dequant, split-K decode GEMV, grouped prefill GEMM, Hadamard rotations, n-gram row decode |
| `python/freetoken/layers/quantization/configs/exl3.py`, `configs/__init__.py`, `configs/base.py` | new, 4/3, 6/1 | the `exl3` dialect: routed experts stay EXL3 (bitrate read from the checkpoint), every other linear unquantized; dialects can see the checkpoint path |
| `python/freetoken/layers/quantization/scheme.py`, `moe/exl3.py`, `moe/__init__.py` | 17/0, new, 4/3 | the EXL3 kind and expert method: bank layout, per-rank slicing on 128-blocks, forward |
| `python/freetoken/moe/fused_exl3.py`, `moe/legacy_format.py` | new, 1/0 | the expert forward (decode GEMVs, blocked prefill GEMMs) and its bank tag |
| `python/freetoken/models/exl3_weights.py` | new | dense EXL3 linears decoded to bf16 at load; the expert reader |
| `python/freetoken/models/qwen4_exp/weight.py`, `config.py`, `ple_disk.py` | 108/7, 6/2, 69/5 | qwen4_exp reads EXL3 checkpoints: hash constants, expert and MTP-expert pieces, the trellis n-gram table in the disk PLE backend |
| `python/freetoken/distributed/split.py`, `models/register.py` | 10/1, 1/0 | a format can widen the intermediate split unit (EXL3: 128); the checkpoint path reaches the dialect |
| `python/freetoken/layers/quantization/moe/base.py`, `engine/engine.py`, `tests/engine/test_cache_budget.py` | 8/0, 17/2, 13/0 | an expert kernel can declare the temporaries of a prefill chunk (EXL3: 0.6 GiB per card at 16k tokens); the cache planner keeps them out of the expert cache and the KV pages on every rank |
| `tests/layers/test_exl3_codec.py`, `rdna3/tests/exl3_check.py`, `rdna3/tests/exl3_moe_check.py`, `rdna3/bench/exl3_gemv_bench.py`, `rdna3/bench/exl3_moe_bench.py` | new | CPU tests, GPU checks, micro-benchmarks |

## Fewer decode kernels (bit-exact fusions)

Line counts here are on top of the lines listed in the sections above.

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/triton/moe_shared_gate.py`, `moe/fused_nvfp4.py`, `models/qwen4_exp/moe.py` | 64/1, 5/0, 32/6 | the MoE epilogue in one kernel (`FREETOKEN_FUSED_MOE_EPILOGUE`): the NVFP4 decode can hand over its per-expert outputs |
| `python/freetoken/kernel/triton/hc.py`, `models/qwen4_exp/hc.py`, `models/qwen4_exp/model.py` | 67/0, 9/1, 59/4 | each hyper-connection combine fused with the next grouped RMSNorm (`FREETOKEN_FUSED_HC_NORM`) |
| `python/freetoken/kernel/triton/dense_gemv.py` | 7/6 | one split-K reduce launch for all the rows |
| `tests/kernels/test_moe_shared_gate.py`, `test_hc_combine_rmsnorm.py`, `test_gemv_bf16_rows.py`, `tests/models/qwen4_exp/test_moe_epilogue.py` | new | bit-exactness on the GPU; which expert paths take the fused epilogue |

## Fewer copies (bit-exact)

Line counts here are on top of the lines listed in the sections above.

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/models/qwen4_exp/attention.py`, `models/qwen4_exp/gdn.py`, `layers/norm.py`, `kernel/triton/attn_gate.py` | 29/6, 7/1, 7/0, new | q / k normalized where the projection wrote them, the output gate in one kernel, the GDN gate read in place (`FREETOKEN_FEWER_COPIES`) |
| `python/freetoken/kernel/fla/layernorm_gated.py`, `kernel/triton/qsa/compress.py`, `attention/qsa_sparse.py` | 24/4, 23/5, 1/1 | the gated norm and the QSA indexer's norm + rope take token / head strides |
| `python/freetoken/kernel/triton/dense_gemv.py`, `layers/quantization/linear/unquantized.py` | 38/7, 10/1 | the split-K bf16 GEMV's last program reduces (`FREETOKEN_GEMV_LAST_REDUCES`), counters allocated at layer finalize |
| `tests/kernels/test_fewer_copies.py` | new | bit-exactness on the GPU |

## Lower prefill peaks, kernels that no longer spill (bit-exact)

Line counts here are on top of the lines listed in the sections above.

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/triton/hc.py`, `models/qwen4_exp/hc.py`, `models/qwen4_exp/model.py`, `models/qwen4_exp/ple.py` | 7/3, 7/2, 9/3, 50/2 | the hc combines write over their input, the PLE output is added to the streams in place, block by block for one request (`FREETOKEN_HC_INPLACE`) |
| `python/freetoken/models/qwen4_exp/gdn.py`, `attention/linear.py` | 66/12, 10/2 | the GDN delta rule of a prefill 4096 tokens of a request at a time (`FREETOKEN_GDN_PREFILL_BLOCK`); the host bounds of each request and track entry in the metadata |
| `python/freetoken/models/qwen4_exp/mtp.py`, `engine/engine.py` | 16/7, 44/4 | the MTP head frees the prompt's streams and their normed copy once read and adds the embedding in place; the torch allocator capped short of the card on ROCm (`FREETOKEN_VRAM_MARGIN_MIB`) |
| `python/freetoken/kernel/triton/qsa/attend.py` | 9/0 | at least 4 warps on ROCm (`FREETOKEN_QSA_MIN_WARPS`) |
| `python/freetoken/kernel/fla/utils.py`, `chunk_delta_h.py`, `wy_fast.py` | 6/0, 2/1, 3/2 | two GDN prefill kernels at 8 warps on ROCm (`FREETOKEN_GDN_PREFILL_WARPS`) |
| `tests/kernels/test_gdn_prefill_blocks.py`, `test_prefill_warps.py`, `test_fewer_copies.py`, `tests/models/qwen4_exp/test_ple_conv_blocks.py` | new, new, 20/0, 34/0 | bit-exactness (GPU; the PLE one on the CPU) |

## Long contexts with the K/V in host RAM

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/models/qwen4_exp/config.py` | 6/1 | `FREETOKEN_ROPE_MAX_POSITION` positions of the plain rope (YaRN's `original x factor` table is upstream's `RotaryConfig.table_positions`, `235e201`, as is YaRN under the multi-axis rope) |
| `python/freetoken/kvcache/qsa_pool.py`, `kvcache/mha_pool.py`, `kernel/host_mapped.py`, `kernel/triton/qsa/stage.py`, `attention/qsa_sparse.py`, `kvcache/host_kv_pool.py` | 48/3, 7/8, new, new, 11/2, 4/0 | `FREETOKEN_QSA_KV_HOST`: the K/V slab in pinned, device-mapped host memory seen by torch as a GPU tensor (DLPack), allocated non-coherent on ROCm so that L2 keeps what the verify rows of one request share, a VRAM buffer that a prefill fills with the pages of one layer at a time, the VRAM budget without the slab; refused with `FREETOKEN_QSA_KV_INT8`, the RAM tier off with it |
| `python/freetoken/engine/engine.py` | 8/1 | with `FREETOKEN_QSA_KV_HOST` the planned K/V pool stays at `--kv-reserve-tokens` (a page past it pins ~0.9 MB of RAM); the rope-length error names `FREETOKEN_ROPE_MAX_POSITION` |
| `python/freetoken/layers/embedding.py` | 32/0 | `FREETOKEN_DUMP_PROMPT_TOPK` (tests only): every prefill row's top-40 log-probabilities and the input ids, saved by rank 0 |
| `rdna3/serve.sh` | 22/10 | `KV_TOKENS` in a profile: the K/V pool shared by the requests, apart from one request's `CTX`; `--list` prints a table (GPUs, context, pool, status) |
| `tests/models/qwen4_exp/test_qsa_kernels.py` | 19/4 against `ee9789f` | upstream's split top-k (`5031e3d`, which replaced this tree's own) also on the 1M profile's 250,000-column table, its boundary row cut where a reduction level divides the candidates; the plan's levels |
| `rdna3/bench/qsa_topk_bench.py` | new | the block top-k on the profiles' page tables |
| `tests/layers/test_rotary_yarn.py`, `tests/kernels/test_qsa_kv_host.py`, `rdna3/profiles/xtx-xt-1m.env`, `rdna3/profiles/xtx-xt-4x262k.env` | new | YaRN and the longer table against HF's rotary embedding (CPU); the host-mapped tensor and the page staging (GPU); one conversation of 1M, four of 262,144 at once |

## Expert misses on the CPU (`--hybrid`, off by default)

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`, `format.h`, `kernels/exl3.cpp` | 126/9, 19/0, new against `cc56b6b` | on upstream's executor split into formats, a format per MoE layer (`e8c6bf1`), and GPU sync backends (`2010271`, which also brought HIP's own memops): EXL3 experts (mul1, 2-4 bits, AVX2 + F16C, software prefetch, one decode for the rows sharing an expert, column blocks `FREETOKEN_CPU_MOE_EXL3_COLS`) as a format that runs its tasks whole (`MoeFormat::run_task`); the routes of rows sharing an expert run back to back (`FREETOKEN_CPU_MOE_GROUP`); two pinned flag words polled beside the doorbell for the hybrid kernels, and the task word exported; `FREETOKEN_CPU_MOE_ROUTE_STATS`; `kernels/exl3.cpp` keeps its own fp16 decode (upstream dropped the shared one, `0db5023`) |
| `python/freetoken/moe/cpu_executor.py` | 144/4 against `cc56b6b` | each tensor-parallel rank's pool on its own half of the cores; the EXL3 banks read in place; the hybrid publish / merge path (`FREETOKEN_HYBRID_FLAG_IN_KERNEL`) over device views of the pinned IO rows and flag words; the pools' cores by slice size (`FREETOKEN_CPU_MOE_CORES_BY_SLICE`, off) or with SMT siblings (`FREETOKEN_CPU_MOE_SMT`, off) |
| `python/freetoken/kernel/triton/hybrid_split.py`, `kernel/host_mapped.py` | new, 7/0 | the hybrid layer's two GPU ends as one kernel each, carrying the handshake (system-scope atomics on the pinned flag words); a device view of an existing pinned tensor |
| `python/freetoken/moe/offload_kernels.py`, `moe/offload_cache.py` | 35/21, 2/2 | upstream's hybrid ensure kernel (which splits the routes, `f02fbb1`) reads the routes as vectors and tells this step's slots by their usage stamp (same decisions; 1.1-2.2x faster at 4 rows, 1.5-8.5x at 16, synthetic routing), with barriers between its phases |
| `python/freetoken/layers/moe.py` | 35/3 | the publish / merge path; steps of fewer than `FREETOKEN_HYBRID_MIN_REQS` requests take the offload path |
| `python/freetoken/engine/engine.py` | 13/5 | the executor's row bound counts the MTP verify rows (sized up front, a larger task refused); `FREETOKEN_HYBRID_FETCH_FRACTION` (one share for every layer, in upstream's per-layer fractions); the TP CPU group handed to the executor |
| `python/freetoken/layers/quantization/moe/nvfp4.py`, `layers/quantization/moe/exl3.py` | 8/2, 14/3 | CPU decode allowed under tensor parallelism with NVFP4; an EXL3 CPU format (mul1, 2-4 bits); both kernels take inactive slots (`FREETOKEN_HYBRID_SKIP_CPU_ROUTES`) |
| `python/freetoken/kernel/triton/nvfp4_fused_moe.py`, `kernel/nvfp4_moe.py`, `kernel/triton/exl3.py`, `moe/fused_exl3.py` | 7/0, 4/0, 17/7, 6/3 | the decode GEMVs write zeros for a slot of -1 without a read; the EXL3 rotation and scale kernels read slot 0's scales for it |
| `tests/moe/test_cpu_moe_exl3.py`, `tests/moe/test_cpu_moe.py`, `tests/moe/test_nvfp4_backends.py` | new, 56/0 against `cc56b6b`, 24/0 | the EXL3 path against the reference codec, tensor-parallel slices, skipped routes, rows sharing an expert, the size check; the publish / merge path eager and replayed in a graph, a step without a CPU route among them; the GPU skip of the CPU's routes bit-identical (NVFP4, EXL3) |

## Decode on two cards, bit-exact

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/triton/exl3.py` | 6/2 | the output kernel's split count left unspecialized (Triton's constant-1 variant ran ~2x slower: rank 1's single K split); 4 N tiles a program for rank 1's gate\|up GEMV (the same sums) |
| `python/freetoken/kernel/triton/int8_gemv.py` | 225/15 | `hc_silu` in the store of the int8 GEMV (`silu_hc`); the up projection and `hc_gate_mix` in one kernel (`gemv_int8_hc_mix`, each weight row loaded once for up to 8 rows); a gate \| up GEMV storing SiLU-and-mul (`gemv_int8_act_mul`, tiles shared across rows); K blocks of 512 for the uneven split's shapes, opt-in and not bit-exact (`FREETOKEN_INT8_TP_TILES=1`) |
| `python/freetoken/models/qwen4_exp/hc.py`, `models/qwen3_5_moe/moe.py` | 37/2, 25/1 | the hyper-connection mix and the shared expert call them on int8 decode rows (`FREETOKEN_HC_SILU_EPILOGUE`, `FREETOKEN_HC_MIX_IN_GEMV`, `FREETOKEN_SHARED_ACT_IN_GEMV`) |
| `tests/layers/test_int8_weight_only.py`, `tests/moe/test_cpu_moe_exl3.py` | 68/0, 39/0 | each fused kernel bit-identical to the separate path over several random draws, 1-8 rows; one split against two exact halves in the output kernel, rank 1's 4 N tiles against 1 (GPU) |

## Decode glue in kernels, bit-exact

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/triton/qsa/plan.py`, `attention/qsa_sparse.py` | 84/0, 22/1 | the QSA layers' per-forward index-write plan in two kernels instead of ~30 torch ops (`FREETOKEN_QSA_PLAN_KERNEL`); `qsa_forward(kv_only=True)` stops once the K/V and the index keys are written |
| `python/freetoken/models/qwen4_exp/ple_conv_kernel.py`, `models/qwen4_exp/ple.py` | 78/0, 16/3 | the PLE's m-row decode conv in one kernel on gfx11 (`FREETOKEN_PLE_CONV_KERNEL`; no FMA contraction, torch's sum order there); its rollback target described for the fused copy |
| `python/freetoken/models/qwen4_exp/attention.py`, `models/qwen4_exp/mtp.py` | 16/6, 9/0 | the MTP head's KV-only pass attends to nothing (`FREETOKEN_MTP_KV_ONLY_STOP`; the QSA backend protocol and the dense reference take `kv_only` too) |
| `python/freetoken/kernel/triton/spec_glue.py`, `spec_decode.py`, `models/qwen4_exp/gdn.py`, `engine/engine.py`, `scheduler/scheduler.py` | 156/0, 27/2, 11/3, 65/11, 11/1 | a verify step's glue in kernels (`FREETOKEN_SPEC_GLUE_KERNELS`): the rollback copies (one kernel per state), the token-pool writes, the kept counts of greedy rows, the MTP head's rows and the chained guesses' addressing; the acceptance tally summed on the host |
| `tests/kernels/test_decode_glue.py` | 169/0 | each kernel bit-identical to the torch ops it replaces over random batches (GPU) |

## Agent turns and cold reads, bit-exact: short prefills through the expert cache, the prompt tokens reused, the K/V pages read, the GDN conv token-major; `--default-temperature`

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/moe/offload_cache.py`, `moe/offload_kernels.py` | 134/1, 115/0 | with one slot pool (upstream's default for a model whose MoE layers share one layout, `b416f85`), `ensure_prefill_routed`: a short prefill chunk's routed experts made resident in the slot cache (LRU by (usage, slot), the routed hits and the predicted ones protected), the misses copied by decode's copy, the next layer's predicted experts by the copy engine on the copy stream; the bookkeeping in three Triton kernels and one top-k a layer; the experts brought in demoted after their GEMMs |
| `python/freetoken/layers/moe.py`, `layers/quantization/moe/base.py`, `exl3.py`, `nvfp4.py` | 56/5, 5/0, 2/0, 3/1 | `FREETOKEN_PREFILL_CACHED_MAX` (default 1280 for the layers of a model that predicts); `ExpertView.slot_map` for the EXL3 / NVFP4 Triton prefill |
| `python/freetoken/moe/fused.py`, `moe/fused_exl3.py`, `moe/fused_nvfp4.py` | 7/0, 8/3, 9/3 | `blocks_to_slots`: the tiles of the expert-id alignment read their weights from the slots (the same tiles, the same bits) |
| `python/freetoken/models/qwen4_exp/moe.py`, `models/qwen4_exp/mtp.py` | 18/0, 1/0 | layer L+1's router on layer L's MoE input, each token's top-10, as the prediction (`FREETOKEN_PREFILL_PREDICT`) |
| `python/freetoken/kernel/batch_memcpy.py` | 4/0 | the probe's copy waits for its fill; a failed mode is logged |
| `python/freetoken/kernel/triton/exl3.py`, `tests/kernels/test_exl3_mid_cols.py`, `tests/kernels/test_exl3_gemm_stages.py` | 79/7, 34/0, new | the prefill's mid step 32 Hadamard columns at a time (`FREETOKEN_EXL3_MID_COLS`): the tile kernel's bits without its 2.7 KB of scratch per lane, 1.7-3.3x faster; the prefill GEMMs with two pipeline stages (the same bits, checked against one stage) |
| `Dockerfile.rdna3`, `rdna3/serve.sh` | 5/2, 2/0 | `TVM_FFI_CACHE_DIR` in the kernel-cache volume |
| `python/freetoken/kernel/triton/qsa/stage.py`, `kvcache/qsa_pool.py`, `attention/qsa_sparse.py`, `tests/kernels/test_qsa_kv_host.py` | 66/1, 26/5, 2/1, 84/0 | host K/V: a prefill stages only the K/V rows of the tokens it selects (`FREETOKEN_QSA_STAGE_SELECTED`), marked by the attention kernel's mapping |
| `python/freetoken/kernel/triton/causal_conv1d_triton.py`, `models/qwen4_exp/gdn.py`, `tests/kernels/test_conv_rows.py` | 46/0, 38/3, new | the GatedDeltaNet prefill conv token-major (`FREETOKEN_GDN_CONV_ROWS`): the projection's q / k / v columns read in place, each group written to its own contiguous buffer, the same kernel and taps |
| `python/freetoken/server/args.py`, `server/api_server.py`, `docs/cli.md`, `tests/server/test_default_temperature.py` | 13/0, 16/3, 1/0, 43/0 | `--default-temperature`: the temperature of requests that send none, in place of the checkpoint's |
| `python/freetoken/tokenizer/prefix_cache.py`, `tokenizer/tokenize.py`, `tests/tokenizer/test_prefix_cache.py` | 167/0, 11/5, 133/0 | `PromptPrefixCache`: a templated prompt reuses an earlier prompt's tokens up to the last `<\|im_start\|>` of their shared prefix and encodes the rest, the same tokens (`FREETOKEN_TOKENIZE_PREFIX_CACHE`) |
| `tests/README.md` | 1/0 | `FREETOKEN_TOKENIZE_CACHE_MODEL`, the checkpoint the prefix-cache test tokenizes with |
| `tests/moe/test_prefill_cached.py` | 273/0 | prefill from slots bit-identical to the full-layer buffers (EXL3, NVFP4), the LRU claims and demotion, the predicted prefetch then the fix-up copy, the split prefetch by DMA or shader copy (GPU) |
| `python/freetoken/models/qwen4_exp/gdn.py`, `moe/fused_nvfp4.py`, `moe/fused_exl3.py`, `tests/models/qwen4_exp/test_gdn_host_bounds.py`, `tests/moe/test_nvfp4_prefill_tiles.py`, `tests/moe/test_short_prefill_tiles.py` | 6/3, 17/1, 7/3, new, new, new | short prefills (agent turns): the GDN prefill conv takes its longest request from the host bounds (a host sync in every GDN layer before); NVFP4 prefills of up to 1536 tokens take 4-warp tiles (`FREETOKEN_NVFP4_PREFILL_TUNED`), EXL3 ones of up to 512 tokens 16-row tiles (the same K order, the same bits) |
| `python/freetoken/layers/moe.py`, `moe/offload_cache.py`, `models/qwen4_exp/moe.py`, `tests/moe/test_prefill_cached.py` | 36/0, 23/2, 21/1, 13/6 | a short prefill's fix-up copy (the routed experts the prediction missed) on the prefill copy stream while the shared expert runs (`FREETOKEN_PREFILL_COPY_OVERLAP`, the same copies and GEMMs); the test runs the predicted prefetch then the fix-up copy both ways |
| `python/freetoken/kernel/triton/fast_launch.py`, `server/launch.py`, `tests/kernels/test_triton_fast_launch.py` | new, 4/0, new | Triton launches through a shorter host path (`FREETOKEN_TRITON_FAST_LAUNCH`): the compiled kernel kept under a key computed from what Triton's specialization reads, its C launcher called with the same arguments; the test checks that equal keys specialize alike and that launches through it run the kernel Triton would |
| `python/freetoken/moe/offload_cache.py`, `tests/moe/test_prefill_cached.py` | 81/9, 9/2 | a slot-cache prefill reads its predicted experts' copy plan (the copy engine's host addresses) once the layer's GEMMs are queued, from pinned memory behind an event, instead of a sync right after the plan (`FREETOKEN_PREFILL_PREFETCH_DEFER`); the prefetch test runs both ways |
| `python/freetoken/kernel/triton/int8_dequant.py`, `layers/quantization/int8_weight_only.py`, `tests/kernels/test_int8_dequant.py` | new, 9/1, new | a prefill's int8 dense weights dequantized in one Triton pass with `torch.mul`'s bits (`FREETOKEN_INT8_PREFILL_DEQUANT`) |
| `python/freetoken/models/qwen4_exp/moe.py`, `moe/prefill_hooks.py`, `moe/fused_nvfp4.py`, `moe/fused_exl3.py`, `kernel/triton/moe_shared_gate.py`, `distributed/impl.py`, `tests/moe/test_prefill_block_hook.py` | 91/7, new, 4/0, 4/0, 7/4, 20/0, new | each prefill block of routed experts finishes its rows and starts their all-reduce while the next block computes (`FREETOKEN_PREFILL_AR_OVERLAP`, two ranks, the layer's path agreed across them once; the epilogue keeps the whole chunk's warps, an async all-reduce on the torch.distributed path); `FREETOKEN_PREFILL_COPY_OVERLAP=auto` with EXL3 experts too |
| `python/freetoken/models/qwen4_exp/gdn.py`, `tests/models/qwen4_exp/test_gdn_ar_overlap.py` | 82/2, new | a one-request prefill chunk of whole GDN blocks: each block's gated norm, output projection and all-reduce started as its delta rule finishes (`FREETOKEN_GDN_AR_OVERLAP`), after a check, agreed across the ranks, that hipBLASLt's block rows equal the whole call's |
| `python/freetoken/kernel/triton/qsa/score.py`, `attention/qsa_sparse.py`, `tests/kernels/test_qsa_score_rows.py` | 128/1, 1/0, new | the QSA indexer's scoring of a one-request prefill 16 query rows a program, the heads unpadded and summed in the padded reduction's order (`FREETOKEN_QSA_SCORE_ROWS`) |
| `tests/models/qwen4_exp/test_predict_votes.py` | new | with NVFP4 experts (`auto`), a short prefill copies ahead only the experts at least two tokens predict (`FREETOKEN_PREFILL_PREDICT_MIN_VOTES`, in `models/qwen4_exp/moe.py`) |
| `python/freetoken/models/qwen4_exp/ple_disk.py`, `scheduler/prefill.py`, `tests/models/qwen4_exp/test_ple_disk.py` | 140/18, 1/0, 36/0 | a prefill's PLE rows read on a thread with a store of its own (the lookup waits for them), and the next chunk's rows read ahead while a chunk of a longer prompt computes, taken when its tokens match (`FREETOKEN_PLE_PREFILL_PREFETCH`; a chunked request carries the whole prompt); at TP>1 only rank 0 reads them and broadcasts the bytes at the lookup (`FREETOKEN_PLE_PREFILL_RANK0`) |
| `python/freetoken/engine/engine.py` | 6/1 | one device synchronize before the decode step that follows a prefill whose compute stream waited for RCCL's on the GPU (`moe/prefill_hooks.py`'s flag): the decode graphs replayed after such a wait ran a third slower at four requests |
## Disk reads on four rings, tail tiles, the input embedding in RAM (bit-exact)

Line counts here are on top of the lines listed in the sections above.

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/kernel/csrc/row_store/row_store_ext.cpp`, `models/qwen4_exp/ple_disk.py`, `tests/kernels/test_row_store.py` | 333/49, 12/3, 57/0 | the disk row store splits a fill of 256 rows or more over four io_uring rings, one helper thread each (`FREETOKEN_ROW_STORE_RINGS`), reads a row across a 4 KiB page as its two pages, the second after another row's read (on the reference NVMe, reads spanning two 4 KiB units, even 2.5 % of a batch, halved the batch's rate), and can keep the rows it read in a RAM cache, which the store reading the prefills does (`FREETOKEN_PLE_ROW_CACHE_ROWS`) |
| `python/freetoken/kernel/triton/moe_align.py`, `moe/fused_nvfp4.py`, `moe/fused_exl3.py`, `tests/moe/test_prefill_tail_tiles.py` | 107/1, 35/17, 15/1, new | `moe_align_split`: each expert's full tiles in one launch, its remaining rows in smaller tiles in a second; NVFP4 prefill blocks of more than 6144 tokens, 128 + 32-row tiles in both GEMMs (`FREETOKEN_NVFP4_PREFILL_TAIL_TILES`), EXL3's gate-up GEMM 64 + 32-row tiles from 3072 tokens (`FREETOKEN_EXL3_PREFILL_TAIL_TILES`); the same K order, the same bits |
| `python/freetoken/layers/embedding.py`, `engine/engine.py`, `tests/layers/test_embed_host.py` | 49/0, 7/0, new | an input embedding not tied to the LM head kept in pinned, device-mapped host memory, read by the same lookup kernel (`FREETOKEN_EMBED_HOST`); the engine moves it after the int8 compaction, before the memory reading that sizes the expert cache and the K/V pool |

## On upstream up to `cc56b6b`

Line counts here are against the merge of `cc56b6b`, on top of the lines listed in the sections above.

| File | Lines | Change |
|---|---:|---|
| `python/freetoken/moe/fused_nvfp4.py`, `tests/moe/test_nvfp4_prefill_tiles.py`, `tests/moe/test_short_prefill_tiles.py`, `tests/moe/test_prefill_tail_tiles.py` | 14/13, 20/16, 3/2, 3/3 | the RDNA3 prefill tiles swept again on upstream's rewritten prefill GEMM (`03c2c9c`): 4 warps and 2 stages on both GEMMs from 1537 to 6144 tokens and on the down GEMM up to 64 and from 641 to 1536, a 64-row tile up to 6144 tokens, the 128-row one with tail tiles above (the same K step, the same bits); untuned prefills take upstream's tiles |
| `python/freetoken/models/qwen4_exp/weight.py`, `layers/quantization/moe/nvfp4.py`, `tests/models/qwen4_exp/test_weight.py`, `tests/models/qwen4_exp/test_tp_shard.py` | 6/3, 2/0, 11/0, 11/0 | the expert-piece hook takes the `layers` upstream's reader passes; the tensor-parallel slicing keeps the experts' static activation scales (`e4f69d9`) whole on every rank |
| `python/freetoken/layers/embedding.py`, `tests/layers/test_lm_head_dump.py` | 2/2, new | the prompt top-k dump of a tied LM head (`FREETOKEN_DUMP_PROMPT_TOPK`) computes its logits as the forward does (it called an `F.linear` that was never imported) |
| `rdna3/tests/exl3_moe_check.py` | 17/0 | the torch reference on as many CPU threads as the container's quota: oversubscribed, it decoded the same tiles differently now and then and the check failed |
| `tests/moe/test_cpu_moe_exl3.py`, `tests/moe/test_prefill_cached.py`, `tests/models/qwen4_exp/test_mtp.py`, `rdna3/bench/nvfp4_*_sweep.py`, `rdna3/bench/exl3_*_bench.py`, `rdna3/tests/moe_block_check.py` | 3/3, 4/4, 1/1, 9/7, 3/3, 2/2 | the executor's per-layer formats, fp32 NVFP4 globals, the launches in `kernel/nvfp4_moe.py` |
| `python/freetoken/spec_decode.py`, `engine/graph.py`, `tests/test_spec_decode.py`, `tests/engine/test_graph_sizing.py` | 27/10, 44/6, 20/0, new | `FREETOKEN_SPEC_BS_DEEP`: past it (up to `FREETOKEN_SPEC_BS_MAX`) a verify step runs at most 2 rows per request and the head drafts one guess, the graphs of more rows only up to it; the spec_decode buffers sized by eager forwards of the most rows, then the most requests, before any capture (they grow with the requests and freed the old ones under graphs captured before); `FREETOKEN_SPEC_SAMPLED_DRAFTS` on by default |

## Tooling and documentation

| File | Change |
|---|---|
| `python/freetoken/engine/ftprof.py` (106/0) | env-gated decode profiling (`FT_STATS_EVERY`, `FT_PROF_*`) |
| `python/freetoken/server/args.py` (3/1) | `--cuda-graph-max-bs` help |
| `Dockerfile.rdna3`, `.dockerignore` | the ROCm image |
| `rdna3/` | profiles (the MTP head on), `serve.sh` (`--no-mtp`; `--hybrid`; a lower `--memory-ratio` for EXL3 on two cards; `--check` and the RAM warning at start; `--oom-score-adj -999`), `download.sh` (the checkpoints), GPU checks, micro-benchmarks and `bench/depth_sweep.py`, maintenance tools |
| `tests/rdna3/test_serve_sh.py` (new) | `serve.sh` on a fake machine (topology, RAM, docker stub): the RAM arithmetic and the commands it suggests or prints |
| `README.md`, `NOTICE`, `CHANGELOG.md`, `llms.txt`, `llms-full.txt`, `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`, `docs/rdna3/` | this repository's documentation (the banner rendered from `docs/rdna3/assets/social-preview.html`); upstream's README moved to `docs/FREETOKEN_UPSTREAM_README.md` |
| `.github/` | upstream's release workflows and issue templates replaced by this repository's |
