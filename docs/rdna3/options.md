# Options

Every tunable this build adds on top of upstream FreeToken is switched by an environment variable (set it in a
profile, or with `-e` on `docker run`). Unless stated otherwise the default is upstream's behavior. A few fixes have
no switch: the ROCm sampling routes, CUDA graphs for every batch size up to an explicit `--cuda-graph-max-bs` (8 or
less), the GDN boundary carried across prefill chunks, the TP>1 relay handshake, the refusal of a cache rebuild under
an uneven split, and RCCL instead of PyNCCL on ROCm ([limits.md](limits.md#does-it-work-on-nvidia-gpus)).
Measurements are from the reference machine (RX 7900 XTX + RX 7900 XT, Qwen3.8-Flash-Next NVFP4, TP=2): see
[benchmarks.md](benchmarks.md).

The labels:

- **Bit-exact**: the same results as without the option (identical greedy answers on a fixed set of 120 prompts, or a
  kernel check with identical outputs).
- **Reorders sums**: the same math, but floating-point additions happen in another order, so answers can differ at
  rounding level without being worse.
- **Lossy**: the numbers themselves change (8-bit weights).
- ***Agent-validated***: part of the two-card builds that scored 12-13 on the author's private agentic benchmark (29
  planted bugs, long OpenCode sessions; see [benchmarks.md](benchmarks.md#precision)). The one-card profiles were not
  run on it. `FREETOKEN_TP_SPLIT`, `FREETOKEN_INT8_DENSE`, `FREETOKEN_FUSE_MOE_ALLREDUCE` and `FREETOKEN_TRITON_GEMV`
  were also checked on a 671-item scored benchmark.

For every `ft serve` flag, see upstream's [CLI reference](../cli.md); the flags the profiles use are explained in
[profiles.md](profiles.md).

## Two or more GPUs

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_TP_SPLIT` | unset (even) | Uneven tensor parallelism: rank 0's share at TP=2 (`0.55`) or one weight per rank (`11:9`). XTX + XT at `0.55`: +4 % decode. Reorders sums (and, with `FREETOKEN_INT8_DENSE=1`, changes the int8 scales of two layers: see below), *agent-validated*. |
| `FREETOKEN_TP_ALLOW_IMBALANCE` | unset | `1`: accept ranks whose free memory differs by more than 2 GiB (upstream refuses them) and plan every pool on the smallest one. Needed for unequal cards; below a 2 GiB difference it changes nothing. |
| `FREETOKEN_HOST_ALLREDUCE` | unset | `1` (two ranks, ROCm): small all-reduces through one shared host-memory region and a one-block kernel instead of RCCL's proxy path. 6 µs instead of ~12 µs per decode all-reduce, **+2.5-4.8 % decode, bit-exact** (fp32 accumulation, rounded to bf16 like RCCL). |
| `FREETOKEN_HOST_ALLREDUCE_MAX` | `262144` | Largest message (bytes) that takes the host path; bigger ones stay on RCCL. |
| `FREETOKEN_HOST_ALLREDUCE_UNCACHED` | `1` | Register the shared pages as uncached fine-grained memory (`0`: default registration). |
| `FREETOKEN_HOST_ALLREDUCE_TIMEOUT_S` | `300` | How long a rank waits for its peer before trapping (a desynchronized pair must not hang forever); `0` = wait forever. |
| `FREETOKEN_TP_SYNC_TOKENS` | `1` | `1` broadcasts rank 0's sampled tokens to the other ranks after each sampled step (~15 µs per step). `0` = off; `check` also compares the ranks' draws and logs each disagreement (one sync per step). |
| `FREETOKEN_FUSE_MOE_ALLREDUCE` | `1` | qwen4_exp at TP>1: one all-reduce per MoE block (routed + shared experts) instead of two. +2.7 % decode, -4.5 % prefill time. Reorders sums, *agent-validated*. |
| `FREETOKEN_RELAY_HANDSHAKE_MAX_HELLOS` | `2400` | TP>1: how many 50 ms hellos the rank-0 relay sends while the other ranks subscribe (2 minutes). |

**`FREETOKEN_TP_SPLIT`** splits the experts' intermediate dimension (routed experts on the NVFP4 Triton expert kernels,
and the shared expert) and the GatedDeltaNet heads, the last two on qwen4_exp (Qwen3.8-Flash-Next's model code, the
only one tested); attention, the vocabulary and the PLE stay even or replicated
([how-it-works.md](how-it-works.md)). Each rank then holds a smaller or larger share of every expert, and because the
ranks all use the smallest card's expert-cache plan, the XT's smaller share raises the number of cached experts on
both cards (7,683 -> 8,708 slots when it was measured). With int8 dense layers the split also changes the int8 scales
of the two row-parallel layers it cuts unevenly (GDN output and shared-expert down projections), not only the order
of the sums.

**`FREETOKEN_TP_SYNC_TOKENS`**: every rank samples its own copy of the logits, so two different GPUs could in principle
continue on different tokens; the broadcast makes rank 1 follow rank 0. It changes no answer while the ranks agree (0
disagreements over ~8k sampled tokens in `check` mode). It is on the torch.distributed path (ROCm always; NVIDIA with
`--disable-pynccl`), new in the release build and not run on the agentic benchmark.

## Dense layers

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_INT8_DENSE` | unset | `1`: every dense (non-expert) linear layer in int8, weights only, one scale per output channel and rank; routers and the vision tower stay bf16. Converted at load. **+28 % decode**, frees ~2.4 GiB per card for the expert cache. Lossy, not bit-exact; no measurable loss on the scored benchmark, *agent-validated*. |
| `FREETOKEN_INT8_DENSE_SKIP` | empty | Comma-separated module names to keep in bf16. |
| `FREETOKEN_INT8_COMPACT` | `1` | Compact the device allocator after the conversion; without it the freed VRAM stays fragmented and is lost to the expert cache. |
| `FREETOKEN_INT8_ROWS_MAX` | `8` | Decode batches up to this many rows (concurrent requests) use the row-looped GEMVs (the int8 one and the bf16 Triton one), each row bit-identical to a batch of one; bigger batches dequantize the int8 weight per call (bf16 layers use `F.linear`). |
| `FREETOKEN_TRITON_GEMV` | `1` on gfx11 | Split-K Triton GEMVs for the bf16 decode shapes of RDNA3 (other GPUs: off). +3.5 % decode on the bf16 build; with `FREETOKEN_INT8_DENSE=1` only the routers are still bf16 and use it. Reorders sums (split-K), *agent-validated*. |

## Experts (MoE)

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_NVFP4_PREFILL_TUNED` | unset | `1`: NVFP4 expert GEMM tiles for prefill swept on RX 7900 XTX / XT for this model, chosen by token count (any split, one or two cards). Only the M / N tiles, warps and stages change (a different K tile would reorder the sums): **bit-exact**, long prompts -11 %. |
| `FREETOKEN_NVFP4_DECODE_TUNED` | unset | `1`: the same for decode, per (N, K) and batch size, for the two-card shapes at a 0.55 split; other shapes (one card, other splits) keep the default tiles, logged once per shape. **Bit-exact**; the tuned decode GEMMs are ~13 % faster at 1 request and 1.06-1.2x at 2-4 (kernel sweep; the full-model gain was not measured on its own). |
| `FREETOKEN_NVFP4_PREFILL_BLOCK` | `4096` | The prefill MoE runs the chunk 4096 tokens at a time on the same expert banks (`0` = the whole chunk). Bit-exact (kernel check); lets 16k-token chunks fit in memory. |
| `FREETOKEN_BATCH_MEMCPY_ROCM` | unset | Copy engine for `--moe-prefill-hit-d2d` on ROCm: `batch` = `hipMemcpyBatchAsync` (HIP >= 7.1), `loop` = one copy per entry; unset tries `batch`, then `loop`. |
| `FREETOKEN_PREFILL_HIT_LOG` | unset | `1`: log, per prefill chunk, the share of expert rows reused from the GPU cache instead of crossing PCIe. |

## Qwen3.8-Flash-Next specifics

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_PLE_CONV_BLOCK` | `4096` | The per-layer-embedding (PLE) convolution runs 4096 tokens at a time, for one or several requests. Keeps big prefill chunks and batched prefills within memory (3 requests prefilled together, 16k tokens, needed 2.6 GiB of temporaries, now 0.7 GiB). Reorders sums; *agent-validated* for one request, identical to the unblocked path in a GPU check with several. |
| `FREETOKEN_QSA_KV_INT8` | unset | **Experimental, not recommended.** `1`: int8 KV cache for the attention (QSA) layers, 3.19 -> 1.71 GiB per card. Short benchmarks and 227k-token needle tests looked fine, but the agentic benchmark fell from 12-14 to 8-9 bugs fixed and the agent stopped exploring early. |

## Sampling

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_FAST_TOPK_MAX` | `256` on ROCm, `0` elsewhere | When every request of the batch has a `top_k` of at most this value (the model's default is 20), the batch samples from its top-k candidates only: one `torch.topk` + one small kernel, ~0.07 ms per step, exact. Otherwise the batch takes the full-vocabulary path (below). |

On ROCm, top-k / top-p sampling never uses upstream's cooperative kernels: their multi-block rows spin on a barrier
that needs every block of a row running at once, which the cooperative launch does not give on this ROCm stack (they
hang on gfx1100), and with one block per row their top-p costs 10-60 ms per step. The full-vocabulary path takes an
exact threshold from a sort instead: 0.2 ms (1 request) to 1 ms (4 requests), the same kept set as the float64
definition, and for a given seed the same draws as upstream's kernels except when a draw lands within fp32 rounding of
a boundary between two tokens. One request without `top_k`, or with a larger one, sends its whole batch there. At a
tie on the k-th logit the top-k path keeps exactly k tokens, the sorted path keeps all the tied ones. This sampler
rework is new in the release build: checked by kernel tests, not run on the agentic benchmark (greedy decoding and the
model's default `top_k` of 20 do not use it).

## Conversations kept in RAM

For hybrid models (attention + GatedDeltaNet, e.g. Qwen3.8-Flash-Next). When the GPU context memory is full, the least
recently used conversation parts are copied to pinned RAM instead of being dropped, and copied back when their
conversation returns. Exact: a conversation brought back gives the same greedy answer as one that never left (four
conversations checked). Added after the agentic benchmark runs.

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_HOST_KV` | unset | `1`: enable. Four agents with 82-117k-token contexts, 3 turns each: after the first reads, 35 s instead of 599 s, time to first token ~2 s instead of ~47 s. |
| `FREETOKEN_HOST_KV_TOKENS` | `262144` | RAM tier size in tokens (Qwen3.8-Flash-Next, TP=2: 4.64 + 4.32 GiB pinned for the two ranks, with the snapshots below). |
| `FREETOKEN_HOST_KV_SNAPSHOTS` | `24` | GatedDeltaNet state snapshots kept in RAM (50-65 MB each at TP=2). |
| `FREETOKEN_HOST_KV_LOG` | unset | `1`: log every move to and from RAM. |

Beyond the GPU pool + the RAM tier, the least recently used conversations are dropped as upstream does.

## Scheduling

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_DECODE_INTERLEAVE` | `0` | `N`: after each prefill chunk, the requests already decoding get up to N decode steps before the next chunk. Keeps other agents moving during a long read, but measured to hurt the short prefills of agent turns: keep `0`. |

## Diagnostics (off by default)

| Variable | Effect |
|---|---|
| `FT_STATS_EVERY=N` | Every N graph-replayed decode steps, log the host step period vs the GPU time of the forward. |
| `FT_MOE_STATS=1` | With `FT_STATS_EVERY`: also the expert-cache miss counters. |
| `FT_PROF_STEPS=N`, `FT_PROF_START=S` | torch.profiler over N decode steps from step S; tables and a chrome trace in `FT_PROF_DIR`. |
| `FT_PROF_EAGER=1` | Also profile eager decode (run with `--cuda-graph-max-bs 0`): ROCm's profiler does not see kernels replayed from a CUDA graph. |

`FT_PROF_DIR` defaults to `./ftprof`, i.e. `/opt/FreeToken/ftprof` inside the container, which is lost when the
container stops: set `FT_PROF_DIR=/root/.cache/freetoken-rdna3/ftprof` to keep it in the kernel-cache volume, or
`docker cp` it out first.

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
