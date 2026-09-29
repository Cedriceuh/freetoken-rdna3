# Decisions and trade-offs

The main choices behind this build, what they cost, and the measurement that settled each one. The story in order is
in [journey.md](journey.md); the options themselves in [options.md](options.md).

## Rule 0: precision before speed

After a faster build lost a third of its score on an agentic benchmark (see "bf16 attention KV, not int8" below),
every change must be either **bit-exact** (same greedy answers on a fixed set of 120 prompts, or a kernel check with
identical outputs) or **validated by the agentic benchmark** (the author's, private: a coding agent fixing 29 planted
bugs over long sessions). Short fidelity tests (scored benchmarks, needle tests, teacher-forced probes) are used to
reject changes, never to accept them alone. The changes made after the last agentic run are bit-exact or exact by
kernel tests, except that prompts prefilled together in one batch (up to 4 requests) can differ at rounding level; the
docs say which.

- **Gives**: speed that did not cost quality on the agentic benchmark.
- **Costs**: slower progress; some real speedups were dropped (int8 KV, grouped prefills).

## Two cards in tensor parallel, rather than one card

| | |
|---|---|
| Choice | Split every layer over the XTX and the XT, both working on every token |
| Alternatives | One card with more experts offloaded; each card serving its own requests |
| For | Decode 55 tok/s instead of 36; 262k context instead of 131k; 44 GB of VRAM for one expert cache |
| Against | Every layer ends with an all-reduce through system RAM (no GPU peer-to-peer on the reference machine); both cards are busy for every request; more code (uneven split, host all-reduce) |
| Evidence | Depth sweeps of `xtx-xt` vs `xtx` ([benchmarks.md](benchmarks.md)) |

## Uneven split, 55 / 45

| | |
|---|---|
| Choice | The XTX (24 GB, ~14 % more compute) takes 55 % of the split work |
| Alternatives | Even split (this build's default; upstream has no tensor parallelism for this model); 57.5 % or 60 % |
| For | +4 % decode; both cards finish together; the XT's smaller share of each expert raises the expert-cache size of both cards |
| Against | ~+0.1 s first-token latency on short prompts; the tuned decode tiles are keyed to this split's shapes; attention and the vocabulary still split evenly |
| Evidence | 57.5 % and 60 % overload the XTX in prefill (compute-bound) |

## int8 dense layers

| | |
|---|---|
| Choice | Weight-only int8 for every dense linear layer (one scale per output channel and rank), routers in bf16 |
| Alternatives | bf16 (the checkpoint's format); int8 with one scale per 128 values |
| For | +28 % decode; ~2.4 GiB per card freed for the expert cache |
| Against | Lossy (weights rounded to 8 bits per output channel), not bit-exact; a conversion step at load and an allocator compaction |
| Evidence | Scored benchmark unchanged (635 vs 632-636 for bf16 builds); part of the builds that scored 13 and 12 on the agentic benchmark (the same build in bf16 also scored 12). Grouped int8 was closer on a probe but worse on the scored benchmark twice (625 vs 635): rejected |

## bf16 attention KV, not int8

| | |
|---|---|
| Choice | Keep the attention KV cache in bf16 |
| Alternative | int8 KV (3.19 -> 1.71 GiB per card) |
| For | The agentic benchmark: 12-14 bugs fixed with bf16 KV, 8-9 with int8 KV, and the int8 agent stopped exploring early |
| Against | ~1.5 GiB less expert cache per card; -4 % decode in isolation (~0 with the uneven split) |
| What missed it | Every other test (scored benchmark, 4/4 needles at 227k, a 46-183k-token bug-hunt proxy) had passed |

## Big prefill chunks: 16k tokens on two cards, 8k on one

| | |
|---|---|
| Choice | 16k-token chunks with the PLE layer and the MoE blocked by 4096 tokens inside, memory ratio 0.80 |
| Alternatives | 4k chunks with ratio 0.85 |
| For | 8.4k prompt 4.3 s instead of 5.5 s in the A/B test (4.5 s on the swept build, 4.1 s on the release build); 54-101k-token reads at depth 2-6 % faster |
| Against | -2 to -3 % decode beyond 100k context (fewer expert-cache slots); a single card cannot hold 16k chunks (8k there) |
| Evidence | At ratio 0.85, 16k chunks left 50-120 MiB free at 227k and slowed down |

## Up to four requests at once

| | |
|---|---|
| Choice | `--max-running-requests 4` with CUDA graphs for every batch size up to 4, per-row decode kernels |
| Alternatives | One request at a time (the earlier profiles' setting; upstream's default is 4) |
| For | 55 / 81 / 98 / 105 tok/s in total at 1 / 2 / 3 / 4 requests; four agents doing the same work finish ~2.7x sooner (97-100 s against 270 s one at a time); a request alone computes exactly as before |
| Against | Each request decodes slower when others run (~40 tok/s each at 2, ~26 at 4); prompts prefilled in one batch can differ at rounding level; not run on the agentic benchmark; GDN state for 4 requests (too much for one card: one card stays at 1) |

## Conversations kept in RAM

| | |
|---|---|
| Choice | Evicted KV pages and GDN snapshots go to pinned RAM and come back on demand |
| Alternatives | Drop them and re-read the conversation (upstream) |
| For | 4 agents with 82-117k-token conversations, 3 turns each: the rounds after the first reads take 35 s instead of 599 s; exact |
| Against | ~9 GiB of RAM locked for the whole run (5 GiB on one card); more code in the prefix cache; no gain for a single conversation that fits the GPUs |

## Host-memory all-reduce

| | |
|---|---|
| Choice | Small all-reduces through one shared host-memory region and a one-block kernel |
| Alternative | RCCL (which also goes through host memory without peer-to-peer, with proxy threads) |
| For | 6 µs instead of ~12 µs per decode all-reduce: +2.5-4.8 % decode, same bits |
| Against | Two ranks only; ROCm only; a custom synchronization, hence a timeout that traps if one rank stops answering |

## Sampling on ROCm

| | |
|---|---|
| Choice | Small `top_k`: sample from the top-k candidates. Otherwise: exact threshold from a sort + barrier-free draw kernels. Rank 0's tokens broadcast |
| Alternatives | Upstream's exact kernels (hang on gfx1100 with several blocks per row; 10-60 ms per step with one); the vendored module from before upstream PR #329 (wrong: up to 89 % of draws outside the top-k, and not faster) |
| For | Exact on any request; 0.07 ms with the model's default `top_k`, 0.2-1 ms otherwise; the two GPUs cannot drift apart |
| Against | The sort costs up to ~1 ms per step with 4 requests when one of them has no `top_k`; 15 µs per step for the broadcast; new in the release build: the sampler checked by kernel tests, the broadcast in `check`-mode server runs; not run on the agentic benchmark |

## Not done, on purpose

| Idea | Why |
|---|---|
| Grouping the prefills of several agents | -3 to -6 % total time, but not bit-exact, needs a ~150 ms wait, and agents already overlap naturally |
| Mixing prefill chunks and decode steps in one batch | estimated at ~3 % |
| Speculative decoding (MTP) | acceptance ~0.6 on real traffic, about break-even |
| Prefetching experts by prediction | 5 to 31 wasted copies per useful one |
| A smarter expert-cache policy | even the offline optimum only halves the misses; capacity is the lever |

## Packaging

| Choice | Why | Cost |
|---|---|---|
| Build the image locally (`Dockerfile.rdna3`) | the base is ~29 GB of ROCm + PyTorch, pinned by digest; the build itself takes minutes; the image records the commit it came from | a first build downloads the base |
| One profile per hardware setup | settings that belong together (chunk size, memory ratio, split, concurrency) travel together, and a profile states what was measured | untested setups start from a derived profile (`TESTED=0`) |
| Settings as environment variables, most of the tuning off by default | the upstream behavior stays one variable away, for comparisons and bug reports; what is on by default is bit-exact, agent-validated, or (the release's sampler rework, exact by kernel tests, and its token broadcast, checked in `check`-mode server runs) ([options.md](options.md)) | long command lines, hence the profiles |
