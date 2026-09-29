# Options

Everything this build adds on top of upstream FreeToken is switched by an environment variable (set it in a profile,
or with `-e` on `docker run`). Unless stated otherwise the default is upstream's behaviour. Measurements are from the
reference machine (RX 7900 XTX + RX 7900 XT, Qwen3.8-Flash-Next NVFP4, TP=2): see [benchmarks.md](benchmarks.md).

"Bit-exact" means the same greedy answers as without the option (checked on 120 prompts); "reorders sums" means the
math is the same but floating-point additions happen in another order, so answers can differ at rounding level
without being worse (checked on a 671-item scored benchmark and, for the options marked *agent-validated*, on a
29-bug agentic coding benchmark).

For every `ft serve` flag, see upstream's [CLI reference](../cli.md); the flags the profiles use are explained in
[profiles.md](profiles.md).

## Two or more GPUs

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_TP_SPLIT` | unset (even) | Uneven tensor parallelism: rank 0's share at TP=2 (`0.55`) or one weight per rank (`11:9`, `3:3:2:2`). Splits the experts' intermediate dimension (routed experts on the NVFP4 Triton expert kernels; the shared expert on qwen4_exp) and the GatedDeltaNet heads (qwen4_exp); tested on qwen4_exp only; attention, the vocabulary and the PLE heads stay even. XTX + XT: `0.55` = +4 % decode; on the release build the XTX caches 8,287 experts instead of the XT's 7,228. Reorders sums (other shard boundaries), no measurable effect on the scored benchmark, *agent-validated*. |
| `FREETOKEN_TP_ALLOW_IMBALANCE` | unset | `1`: accept ranks with unequal free memory and plan every pool on the smallest one (upstream refuses). Needed for unequal cards, harmless on equal ones. |
| `FREETOKEN_HOST_ALLREDUCE` | unset | `1` (two ranks, ROCm): small all-reduces through one shared host-memory region and a one-block kernel instead of RCCL's proxy path (consumer boards have no GPU peer-to-peer). 6 us instead of ~12 us per decode all-reduce, **+2.5-4.8 % decode, bit-exact** (fp32 accumulation, rounded to bf16 like RCCL). |
| `FREETOKEN_HOST_ALLREDUCE_MAX` | `262144` | Largest message (bytes) that takes the host path; bigger ones stay on RCCL. |
| `FREETOKEN_HOST_ALLREDUCE_UNCACHED` | `1` | Register the shared pages as uncached fine-grained memory (`0`: default registration). |
| `FREETOKEN_HOST_ALLREDUCE_TIMEOUT_S` | `300` | How long a rank waits for its peer before trapping (a desynchronised pair must not hang forever); `0` = wait forever. |
| `FREETOKEN_TP_SYNC_TOKENS` | `1` | Every rank samples its own copy of the logits; `1` broadcasts rank 0's tokens after each sampled step, so two different GPUs can never continue on different tokens (~15 us per step). `0` = off; `check` also compares the ranks' draws and logs each disagreement (diagnostics, one sync per step). ROCm / RCCL path. |
| `FREETOKEN_FUSE_MOE_ALLREDUCE` | `1` | qwen4_exp at TP>1: one all-reduce per MoE block (routed + shared experts) instead of two. +2.7 % decode, -4.5 % prefill time. Reorders sums, *agent-validated*. |
| `FREETOKEN_RELAY_HANDSHAKE_MAX_HELLOS` | `2400` | TP>1: how many 50 ms hellos the rank-0 relay sends while the other ranks subscribe (2 minutes). |

## Dense layers

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_INT8_DENSE` | unset | `1`: every dense (non-expert) linear layer in int8, weights only, one scale per output channel; routers stay bf16. Converted at load. **+28 % decode**, frees ~2.3 GiB per card for the expert cache. Reorders sums, *agent-validated*. |
| `FREETOKEN_INT8_DENSE_SKIP` | empty | Comma-separated module names to keep in bf16. |
| `FREETOKEN_INT8_COMPACT` | `1` | Compact the device allocator after the conversion; without it the freed VRAM stays fragmented and is lost to the expert cache. |
| `FREETOKEN_INT8_ROWS_MAX` | `8` | Decode batches up to this many rows (concurrent requests) use the row-looped int8 GEMV, each row bit-identical to a batch of one; bigger batches dequantize the weight per call. |
| `FREETOKEN_TRITON_GEMV` | `1` on gfx11 | Split-K Triton GEMVs for the bf16 decode shapes of RDNA3 (+3.5 % decode). Only active on gfx11 GPUs. Reorders sums (split-K), *agent-validated*. |

## Experts (MoE)

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_NVFP4_PREFILL_TUNED` | unset | `1`: NVFP4 expert GEMM tiles for prefill swept on RX 7900 XTX / XT for this model. Only the M / N tiles, warps and stages change (a different K tile would reorder the sums): **bit-exact**, long prompts -11 %. |
| `FREETOKEN_NVFP4_DECODE_TUNED` | unset | `1`: same for decode, per (N, K) and batch size: **bit-exact**, +1.3 % decode. Shapes without a tuned entry fall back to the default tiles (logged once per shape). |
| `FREETOKEN_NVFP4_PREFILL_BLOCK` | `4096` | The prefill MoE runs the chunk 4096 tokens at a time on the same expert banks (`0` = the whole chunk). Bit-exact; lets 16k-token chunks fit in memory. |
| `FREETOKEN_BATCH_MEMCPY_ROCM` | unset | Copy engine for `--moe-prefill-hit-d2d` on ROCm: `batch` = `hipMemcpyBatchAsync` (HIP >= 7.1), `loop` = one copy per entry; unset tries `batch`, then `loop`. |
| `FREETOKEN_PREFILL_HIT_LOG` | unset | `1`: log, per prefill chunk, the share of expert rows reused from the GPU cache instead of crossing PCIe. |

## Qwen3.8-Flash-Next specifics

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_PLE_CONV_BLOCK` | `4096` | The per-layer-embedding (PLE) convolution runs 4096 tokens at a time, for one or several requests. Keeps big prefill chunks and batched prefills within memory (a batch of 4 prefills needed 2.6 GiB of temporaries, now 0.7 GiB). Reorders sums, *agent-validated*. |
| `FREETOKEN_QSA_KV_INT8` | unset | **Experimental, not recommended.** `1`: int8 KV cache for the attention (QSA) layers, 3.19 -> 1.71 GiB per card. Short benchmarks and 227k-token needle tests looked fine, but the agentic benchmark fell from 12-14 to 8-9 bugs fixed and the agent stopped exploring early. |

## Sampling

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_FAST_TOPK_MAX` | `256` on ROCm, `0` elsewhere | When every request of the batch has a `top_k` of at most this value (the model's default is 20), the batch samples from top-k candidates only: one `torch.topk` + one small kernel, ~0.07 ms per step, exact (at a tie on the k-th logit it keeps exactly k tokens, the sorted path keeps all tied ones). One request without `top_k`, or with a larger one, sends the batch to the full-vocabulary path: on ROCm an exact threshold from a sort, 0.2 ms (1 request) to 1 ms (4 requests). |

On ROCm, top-k / top-p sampling never uses upstream's cooperative kernels: their multi-block rows spin on a barrier
that RDNA3 does not guarantee (they hang), and with one block per row their top-p costs 10-60 ms per step. The
sorted-threshold path is exact (same kept set as the float64 definition, and for a given seed the same draws as
upstream's kernels, except when a draw lands within fp32 rounding of a boundary between two tokens).

## Conversations kept in RAM

For hybrid models (attention + GatedDeltaNet, e.g. Qwen3.8-Flash-Next). When the GPU context memory is full, the least
recently used conversation parts are copied to pinned RAM instead of being dropped, and copied back when their
conversation returns. Exact: a conversation brought back gives the same greedy answer as one that never left.

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_HOST_KV` | unset | `1`: enable. Four agents with 82-117k-token contexts, 3 turns each: 35 s instead of 599 s, time to first token ~2 s instead of ~47 s. |
| `FREETOKEN_HOST_KV_TOKENS` | `262144` | RAM tier size in tokens (Flash-Next, TP=2: ~4.6 + 4.3 GiB for the two ranks with the snapshots below). |
| `FREETOKEN_HOST_KV_SNAPSHOTS` | `24` | GatedDeltaNet state snapshots kept in RAM (50-65 MB each). |
| `FREETOKEN_HOST_KV_LOG` | unset | `1`: log every move to and from RAM. |

Beyond the GPU pool + the RAM tier, the oldest conversations are dropped as upstream does.

## Scheduling

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_DECODE_INTERLEAVE` | `0` | `N`: after each prefill chunk, the requests already decoding get up to N decode steps before the next chunk. Keeps other agents moving during a long read, but measured to hurt the short prefills of agent turns: keep `0`. |

## Diagnostics (off by default)

| Variable | Effect |
|---|---|
| `FT_STATS_EVERY=N` | Every N graph-replayed decode steps, log the host step period vs the GPU time of the forward. |
| `FT_MOE_STATS=1` | With `FT_STATS_EVERY`: also the expert-cache miss counters. |
| `FT_PROF_STEPS=N`, `FT_PROF_START=S` | torch.profiler over N decode steps from step S; tables and a chrome trace in `FT_PROF_DIR` (default `./ftprof`, i.e. `/opt/FreeToken/ftprof` inside the container, lost when it stops: set `FT_PROF_DIR=/root/.cache/freetoken-rdna3/ftprof` to keep it in the kernel-cache volume, or `docker cp` it out first). |
| `FT_PROF_EAGER=1` | Also profile eager decode (run with `--cuda-graph-max-bs 0`): ROCm's profiler does not see kernels replayed from a CUDA graph. |

## PyTorch TunableOp

The profiles mount `rdna3/tunableop/` read-only with `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=0`:
the GEMM algorithm for each of the 89 dense shapes is taken from a file tuned once on these cards (+27 % decode on the
bf16 build, mostly superseded by int8 but still used by the remaining bf16 GEMMs). Never enable tuning in service
(`PYTORCH_TUNABLEOP_TUNING=1`): each new prompt length would be tuned on the spot (16 s prefills). The files are tied to
the image's PyTorch / HIP / hipBLASLt versions, written in their header; to tune for another image, run a
representative workload once with `PYTORCH_TUNABLEOP_TUNING=1` and a writable `PYTORCH_TUNABLEOP_FILENAME`, then
freeze.

## Container settings the engine does not check

`PYTORCH_ALLOC_CONF=expandable_segments:False` and `OMP_WAIT_POLICY=PASSIVE` (set by `serve.sh`), an explicit
`HIP_VISIBLE_DEVICES` (rank 0 first), `--ulimit memlock=-1` (pinned RAM for the experts and the RAM tier) and
`--ipc=host` (shared memory between the two ranks).
