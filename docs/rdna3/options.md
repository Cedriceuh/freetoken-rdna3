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
| `FREETOKEN_TP_SPLIT` | unset (even) | Uneven tensor parallelism: rank 0's share at TP=2 (`0.55`) or one weight per rank (`11:9`). XTX + XT at `0.55`: +4 % decode. With an EXL3 checkpoint the expert slices round to whole 128-wide blocks (`0.55` gives 384 / 256, the even split is refused). Reorders sums (and, with `FREETOKEN_INT8_DENSE=1`, changes the int8 scales of two layers: see below), *agent-validated*. |
| `FREETOKEN_TP_ALLOW_IMBALANCE` | unset | `1`: accept ranks whose free memory differs by more than 2 GiB (upstream refuses them) and plan every pool on the smallest one. Needed for unequal cards; below a 2 GiB difference it changes nothing. |
| `FREETOKEN_HOST_ALLREDUCE` | unset | `1` (two ranks, ROCm): small all-reduces through one shared host-memory region and a one-block kernel instead of RCCL's proxy path. 6 µs instead of ~12 µs per decode all-reduce, **+2.5-4.8 % decode, bit-exact** (fp32 accumulation, rounded to bf16 like RCCL). |
| `FREETOKEN_HOST_ALLREDUCE_MAX` | `262144` | Largest message (bytes) that takes the host path; bigger ones stay on RCCL. |
| `FREETOKEN_HOST_ALLREDUCE_UNCACHED` | `1` | Register the shared pages as uncached fine-grained memory (`0`: default registration). |
| `FREETOKEN_HOST_ALLREDUCE_TIMEOUT_S` | `300` | How long a rank waits for its peer before trapping (a desynchronized pair must not hang forever); `0` = wait forever. |
| `FREETOKEN_HOST_ALLREDUCE_WAITLOG` | unset | Debug, `1`: each call stores how long its rank waited for the other one (100 MHz ticks, `[rank][seq % 65536]` after the four slots), and the shared region stays in `/dev/shm` (`freetoken-hostar-*-waitlog`) for a reader inside the container until a clean shutdown (a killed container leaves it: `--ipc=host` puts it in the host's `/dev/shm`, owned by root). Same speed. On `xtx-xt` it showed the XTX waiting ~0.9 ms a decode step for the XT, at every all-reduce. |
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
| `FREETOKEN_INT8_ROW_GROUPS` | `1` | With several rows (a speculative verify step, or requests decoding together), the int8 GEMVs whose input is longer than one tile (hyper-connection down projections, shared expert gate/up, index projection, LM head) read each weight tile once for up to 4 rows instead of once per row; same arithmetic per row, bit-exact. At 4 rows on the XT: HC down 17.7 -> 12.1 µs, LM head 704 -> 499 µs. `0`: each program runs the rows one after the other, reloading the tile for each. |
| `FREETOKEN_MOE_COPY_OVERLAP` | unset | `1`: in decode, the copy of the missing experts goes to a side stream ahead of the shared expert and its gate; the expert GEMVs wait for it (offload backend, GPU decode, Qwen3.8-Flash-Next). Bit-exact, but slower: measured end to end on 2026-10-03 (one request, agent turns at 77-99k of context, MTP head on), 47.5 tok/s decode against 71.1 without it (-33 %; 48.9 with `expandable_segments:False`) and cold reads 13 % longer. At short context it does save ~0.5 ms a step (GPU time, as measured before), but in the MTP verify steps at 77-88k of context the forward's GPU time grows from 30-35 to 40-48 ms with the same expert misses (cause not found). Leave it off. |
| `FREETOKEN_TRITON_GEMV` | `1` on gfx11 | Split-K Triton GEMVs for the bf16 decode shapes of RDNA3 (other GPUs: off). +3.5 % decode on the bf16 build; with `FREETOKEN_INT8_DENSE=1` only the routers are still bf16 and use it. Reorders sums (split-K), *agent-validated*. |

## Experts (MoE)

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_NVFP4_PREFILL_TUNED` | unset | `1`: NVFP4 expert GEMM tiles for prefill swept on RX 7900 XTX / XT for this model, chosen by token count (any split, one or two cards). Only the M / N tiles, warps and stages change (a different K tile would reorder the sums): **bit-exact**, long prompts -11 %. |
| `FREETOKEN_NVFP4_DECODE_TUNED` | unset | `1`: the same for decode, per (N, K) and batch size, for the two-card shapes at a 0.55 split; other shapes (one card, other splits) keep the default tiles, logged once per shape. **Bit-exact**; the tuned decode GEMMs are ~13 % faster at 1 request and 1.06-1.2x at 2-4 (kernel sweep; the full-model gain was not measured on its own). |
| `FREETOKEN_NVFP4_PREFILL_BLOCK` | `4096` | The prefill MoE runs the chunk 4096 tokens at a time on the same expert banks (`0` = the whole chunk). Bit-exact (kernel check); lets 16k-token chunks fit in memory. |
| `FREETOKEN_EXL3_PREFILL_BLOCK` | `4096` | EXL3 checkpoints ([how-it-works.md](how-it-works.md#exl3-checkpoints)): the prefill MoE runs 4096 tokens at a time (`0` = the whole chunk): 540 MiB of temporaries on the 384-wide expert slice, 620 MiB with a 16k chunk's output (measured; 500 / 580 MiB on the 256-wide slice, by the same formula), which the cache planner keeps free at start. Swept on the XT (one layer, 8192 tokens): 1024-token blocks cost 5.4 ms per 1k tokens before the down GEMM took the rotation out as its epilogue, 4096-token blocks 3.4, then 2.8-3.0 with that epilogue; 8192 saved 5-7 % more before that epilogue, for 0.5 GiB more. A block of 12 tokens or fewer (the last one included) takes the decode kernels for both rotations, one of 13-25 for the step between the products (at top-10), and rounds differently; otherwise the block size changes no result (`rdna3/tests/exl3_moe_check.py`: 1024-token blocks against the whole 3000-token chunk, bit-identical). |
| `FREETOKEN_BATCH_MEMCPY_ROCM` | unset | Copy engine for `--moe-prefill-hit-d2d` on ROCm: `batch` = `hipMemcpyBatchAsync` (HIP >= 7.1), `loop` = one copy per entry; unset tries `batch`, then `loop`. |
| `FREETOKEN_PREFILL_HIT_LOG` | unset | `1`: log, per prefill chunk, the share of expert rows reused from the GPU cache instead of crossing PCIe. |

## Qwen3.8-Flash-Next specifics

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_PLE_SYNC` | `auto` | How a captured decode step gets its PLE disk rows. `auto` probes the stream memops once at start: `wait-sync` (the step waits inside the graph for the host fill) where a captured wait holds three replays in a row (ROCm 10 on these cards); else `launch-gating` (the host fills before each launch). The log says which. `wait`: require memops, `gate`: never use them. ROCm 10 + wait-sync: +3.5-4 % decode, bit-exact. |
| `FREETOKEN_PLE_CONV_BLOCK` | `4096` | The per-layer-embedding (PLE) convolution runs 4096 tokens at a time, for one or several requests. Keeps big prefill chunks and batched prefills within memory (3 requests prefilled together, 16k tokens, needed 2.6 GiB of temporaries, now 0.7 GiB). Reorders sums; *agent-validated* for one request, identical to the unblocked path in a GPU check with several. |
| `FREETOKEN_FUSED_MOE_EPILOGUE` | `1` | The routed experts' sum, the shared expert's gate (a dot product and a sigmoid) and the mul-add of each MoE block in one kernel instead of three; the NVFP4 decode hands over its per-expert outputs. Same arithmetic in the same order on the same tiles: **bit-exact**. Only Triton NVFP4 / EXL3 experts on the GPU decode path, at TP=1 or with `FREETOKEN_FUSE_MOE_ALLREDUCE=1` (the bf16 kernel overwrites its input; hybrid and CPU layers add partial sums; without the block's single all-reduce the experts would reduce each expert's output): others keep the three kernels. NVFP4 `xtx-xt`, greedy decode without the MTP head: 57.8 -> 59.2 tok/s alone, with the MTP head 82.9 -> 83.8. `0`: three kernels. |
| `FREETOKEN_FUSED_HC_NORM` | `1` | Each hyper-connection combine and the grouped RMSNorm that reads the streams next (the MLP block's, the next layer's, the final mixer's; not into a PLE layer) in one kernel, the normed streams freed as soon as read. **Bit-exact**, two launches fewer per layer. With the epilogue above, `xtx-xt` greedy decode: NVFP4 57.8 -> 59.8-60.0 tok/s (82.9 -> 85.6 with the MTP head), EXL3 3.05 bpw 58.4 -> 59.7 (86.4 -> 89.2), 4.05 bpw 55.3 -> 56.5 (81.6 -> 83.3); prompt reading and VRAM peaks unchanged ([journey.md](journey.md#19-fewer-decode-kernels-2026-10-06)). `0`: separate kernels. |
| `FREETOKEN_QSA_KV_INT8` | unset | **Experimental, not recommended.** `1`: int8 KV cache for the attention (QSA) layers, 3.19 -> 1.71 GiB per card. Short benchmarks and 227k-token needle tests looked fine, but the agentic benchmark fell from 12-14 to 8-9 bugs fixed and the agent stopped exploring early. |

## Speculative decoding with the MTP head (Qwen3.8-Flash-Next, experimental)

The checkpoint ships a multi-token-prediction head (`mtp.*`: one decoder layer with its own 512 experts). With it, a
decode step of one request verifies the head's guesses in the same forward: m rows (the last token, then m - 1
drafts), of which it keeps the matching prefix. How it works and what it costs:
[decisions.md](decisions.md#speculative-decoding-with-the-mtp-head), measurements:
[benchmarks.md](benchmarks.md#speculative-decoding-mtp-head). The profiles turn it on (`FREETOKEN_MTP=1`,
`FREETOKEN_SPEC_VERIFY_M=4`); `rdna3/serve.sh --no-mtp` serves without it.

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_MTP` | unset (`1` in the profiles) | `1`: load the MTP head and draft with it; its experts share the expert cache (an NVFP4 checkpoint's stacked bf16 head experts are quantized to NVFP4 at load, an EXL3 checkpoint's are read as they are). Only with `FREETOKEN_SPEC_VERIFY_M` > 1 (else a warning, and no head) and a checkpoint whose head experts are in one of those two forms (others are refused). |
| `FREETOKEN_SPEC_VERIFY_M` | `1` (`4` in the profiles) | The most rows a verify step runs (1 + drafts). With the head, `4` is the measured setting (the adaptive depth then picks 2-4 rows per request); greedy answers stay identical to plain decode with `FREETOKEN_INT8_DENSE=1`, as every profile sets (bf16 dense layers outside the GEMV table go through `F.linear`, whose sums can change with the row count) and the default fp32 GDN state (`FREETOKEN_MAMBA_SSM_DTYPE`). Without the head, `m` > 1 runs placeholder rows (a cost measurement only). |
| `FREETOKEN_SPEC_VERIFY_MS` | `1,2,..,M` with the head | The row counts a step may run, each with its CUDA graphs (counts above the smallest only for the batch sizes a verify step can have). Near a request's length limit a step runs no more rows than tokens left. |
| `FREETOKEN_SPEC_DYNAMIC` | `1` | Each request picks its rows per step (2 to M; with `FREETOKEN_SPEC_BS_MAX` > 1, the most any of the step's requests wants) from its running acceptance per draft position and the step costs below: the count with the most expected tokens per millisecond (with two or more row counts above 1, as by default). `0`: always M. |
| `FREETOKEN_SPEC_COSTS` | measured, by card count | Step time per row count (ms; only the ratios matter): two cards (`xtx-xt`) `1:17.7,2:25.3,3:30.7,4:35.5`, one card (`xt`) `1:31,2:53,3:78,4:102`, where the extra rows' missing experts cross PCIe and the adaptive depth mostly keeps to 2 rows. Entries given here override. |
| `FREETOKEN_SPEC_BS_MAX` | `1`; `2` in `xtx-xt` | Verify only while at most this many requests decode (at most `--cuda-graph-max-bs`, checked at start). With more, steps run one row per request, the head only writes its KV (so drafts stay good when a request is alone again) and scheduling overlaps as without the head. `2` on `xtx-xt` (the other two-card profiles keep `1`: not measured there), measured 2026-10-03: two agents at ~100k of context decode 33.6-34.4 tok/s each instead of 30.8-31.1, 2-3 short requests finish 3-4 % sooner, 1 and 4 requests take the same time, a sweep to 248k runs without an out-of-memory retry (one of six 3-request rounds took 10.1 s instead of ~8, unexplained; greedy answers as with `1`: two at once identical to each alone, four at once 2 of 4 prompts differ in both, the batched-prefill rounding of [benchmarks.md](benchmarks.md#several-requests-and-agents-at-once-xtx-xt)); on 2026-10-04 with the reference machine's launcher settings (image input on), two agents 36.4 against 31.6 tok/s each (thinking on: 38.8 against 34.2 on the turns they decode together), and conversations of 115k and 121k tokens decoding together without an out-of-memory retry. The one crash seen during these loads (an illegal memory access while the weights loaded, once) happened before this setting is read. Before the adaptive depth, a fixed 4 rows for 2 requests cost 60.5 ms against 23.7 ms for one row each, for 4 requests 133 ms against 41.5 ms; past 8 rows per step answers can change. |
| `FREETOKEN_SPEC_SAMPLED_DRAFTS` | unset | `1`: a sampled request's drafts are draws from the head under the request's own temperature / top-k / top-p, verified against that distribution, instead of the head's argmax (both exact speculative sampling). Measured: 2-7 % more tokens kept per step, the same speed (120 items at the model's T 1.0: 439 s against 434 s). |

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
| `FREETOKEN_HOST_KV_TOKENS` | `262144` | RAM tier size in tokens (Qwen3.8-Flash-Next, TP=2: 4.91 + 4.59 GiB pinned for the two ranks with the MTP head, 4.64 + 4.32 without, with the snapshots below). |
| `FREETOKEN_HOST_KV_SNAPSHOTS` | `24` | GatedDeltaNet state snapshots kept in RAM (50-65 MB each at TP=2). |
| `FREETOKEN_HOST_KV_LOG` | unset | `1`: log every move to and from RAM. |

Beyond the GPU pool + the RAM tier, the least recently used conversations are dropped as upstream does.

## Scheduling

| Variable | Default | Effect |
|---|---|---|
| `FREETOKEN_DECODE_INTERLEAVE` | `0` | `N`: after each prefill chunk, the requests already decoding get up to N decode steps before the next chunk. Keeps other agents moving during a long read, but measured to hurt the short prefills of agent turns: keep `0`. |
| `FREETOKEN_MLOCK` | `1` | Lock the process's memory once the model is loaded (`mlockall`, current and future pages, on fault), so that with `vm.compact_unevictable_allowed=0` the kernel's memory compaction leaves the pages the GPU driver maps alone; each move stopped the GPU queues for seconds ([troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run)). Skipped, with a log line, under a finite `RLIMIT_MEMLOCK`. `0`: off. |

## Diagnostics (off by default)

| Variable | Effect |
|---|---|
| `FT_STATS_EVERY=N` | Every N graph-replayed decode steps, log the host step period vs the GPU time of the forward. |
| `FT_MOE_STATS=1` | With `FT_STATS_EVERY`: also the expert-cache miss counters. |
| `FT_PROF_STEPS=N`, `FT_PROF_START=S` | torch.profiler over N decode steps from step S; tables and a chrome trace in `FT_PROF_DIR`. |
| `FT_PROF_EAGER=1` | Also profile eager decode (run with `--cuda-graph-max-bs 0`): ROCm's profiler does not see kernels replayed from a CUDA graph. |
| `FREETOKEN_SPEC_TIMING=N` | Every N decode steps, the GPU and host time of each phase (forward, verify, MTP head, draft chain) per (requests, rows); one stream sync per report. |
| `FREETOKEN_SPEC_STALL_MS` | `500`; with `FREETOKEN_SPEC_TIMING`, every decode step whose GPU or host time, or distance to the previous step, exceeds it is logged on its own with its phases and the time of day (every rank): to catch the occasional stall. |
| `FREETOKEN_PLE_FILL_TIMING=N` | Every N decode steps, the host's wait for the step's tokens and the PLE disk fill time after it (the GPU idles through the fill when stream memops are unavailable). |

`FT_PROF_DIR` defaults to `./ftprof`, i.e. `/opt/FreeToken/ftprof` inside the container, which is lost when the
container stops: set `FT_PROF_DIR=/root/.cache/freetoken-rdna3/ftprof` to keep it in the kernel-cache volume, or
`docker cp` it out first.

## PyTorch TunableOp

Not used: tuning the dense GEMMs of the 16k-token prefill chunks gained 1.5-1.7 % on those GEMMs and < 0.2 % on
prompt reading ([journey.md](journey.md#17-dropped-along-the-way)). Never enable tuning in service
(`PYTORCH_TUNABLEOP_TUNING=1`): each new prompt length would be tuned on the spot (16 s prefills).

## Container settings the engine does not check

`PYTORCH_ALLOC_CONF=expandable_segments:True` and `OMP_WAIT_POLICY=PASSIVE` (set by `serve.sh`; without expandable
segments the allocator's cache filled the free VRAM and every 14-16k-token prefill chunk past ~26k of context failed
once and retried after emptying it, with or without the MTP head and with a larger margin too), an explicit
`HIP_VISIBLE_DEVICES` (rank 0 first), `--ulimit memlock=-1` (pinned RAM for the experts and the RAM tier) and
`--ipc=host` (shared memory between the two ranks).
