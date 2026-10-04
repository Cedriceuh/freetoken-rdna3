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
docs say which. The ROCm 10 base, whose greedy answers differ from ROCm 7.14 on 2 of 5 prompts, was run on the agentic
benchmark with the MTP head: 12 and 18 of 29 in two runs.

- **Gives**: speed that did not cost quality on the agentic benchmark.
- **Costs**: slower progress; some real speedups were dropped (int8 KV, grouped prefills).

## Two cards in tensor parallel, rather than one card

| | |
|---|---|
| Choice | Split every layer over the XTX and the XT, both working on every token |
| Alternatives | One card with more experts offloaded; each card serving its own requests |
| For | Decode 55 tok/s instead of 36 (ROCm 7.14, before the MTP head); 250k context instead of 131k; 44 GB of VRAM for one expert cache |
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
| For | 55 / 81 / 98 / 105 tok/s in total at 1 / 2 / 3 / 4 requests (ROCm 7.14 release build, without the MTP head); four agents doing the same work finish ~2.7x sooner (97-100 s against 270 s one at a time); a request alone computes exactly as before |
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

## Images: one tower on rank 0, embeddings broadcast

| | |
|---|---|
| Choice | Off by default (`serve.sh --vision`). The vision tower is never split: rank 0 alone holds it and encodes, then broadcasts the embeddings. Weights stream from RAM, in bf16. Images are scaled down to 1024 tokens |
| Alternatives | Split the tower like the text layers; keep a whole tower on both ranks; weights resident in VRAM; a larger image cap |
| For | No all-reduce while encoding. Both ranks get the same embeddings whatever the cards round. Two-card speed is unchanged up to 255k of context, and text answers are identical |
| Against | Rank 1 waits while rank 0 encodes. One-card profiles lose ~1.5-2 % of decode (the tower takes expert-cache room on the only card). The encode time grows faster than the image (0.1 s at 1024 tokens, 12 s at 16384) |
| Evidence | A tower on both ranks cost rank 1 351 experts of cache; resident weights take 1034 MiB instead of 294 MiB for the same encode speed; the two cards encode bit-identically, so the broadcast changes nothing on the reference pair ([benchmarks.md](benchmarks.md#images---vision-xtx-xt)) |

## Speculative decoding with the MTP head

| | |
|---|---|
| Choice | On in the profiles (`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4`; `rdna3/serve.sh --no-mtp` turns it off). The checkpoint's MTP layer drafts up to 3 tokens (chained on its own streams); the next decode step verifies them in the same captured forward, 2-4 rows chosen per request from its measured acceptance and the step costs measured for the card count (mostly 2 on one card; 1 near a request's length limit). Rejected rows are undone by restoring the GDN, conv and PLE states saved after each row; attention KV needs nothing (rejected positions are rewritten before any query sees them). Sampled requests use exact speculative sampling. Only while at most `FREETOKEN_SPEC_BS_MAX` requests decode (2 in `xtx-xt`, 1 in the other profiles): with more, steps run one row each and the head only writes its KV |
| Alternatives | Plain decode; a fixed draft depth; verifying several requests at once; the head's own forward captured in a graph |
| For | Agentic benchmark on ROCm 10 with the head: 12 and 18 of 29 (two runs). One request on `xtx-xt` (release, ROCm 10): +34 % decode at the model's sampling, +30-61 % greedy up to 248k of context (earlier runs: +11-39 % on chat, agent turns, a 108k-token context; +20-49 % on ROCm 7.14), x1.38 on 60 code and math items; one card +7-8 % (+6-12 % greedy); greedy answers identical to plain decode (bit-exact: 120/120 on ROCm 7.14, every check on ROCm 10) |
| Against | 3-4 % slower with 3-4 requests at once (the head still writes its KV every step); prompt reading 0-4 % slower (4-5 % on a 108k-token prompt; the head reads it too); the head's 512 experts compete for the expert cache; a CUDA graph per row count (captured up to `FREETOKEN_SPEC_BS_MAX` requests: 2 in `xtx-xt`) |
| Evidence | Step costs 17.7 / 25.3 / 30.7 / 35.5 ms at 1-4 rows on two cards, 31 / 53 / 78 / 102 ms on one (the extra rows' missing experts cross PCIe), 2.4-2.6 tokens kept per step on chat; a fixed 3 rows at 4 requests was -26 % before the head ran on the kept rows only ([benchmarks.md](benchmarks.md#speculative-decoding-mtp-head)) |

## EXL3 checkpoints: experts in EXL3, the rest in bf16

| | |
|---|---|
| Choice | Keep the routed experts in EXL3 (host banks and GPU cache) behind gfx1100 Triton kernels; decode every other linear to bf16 at load (then int8 like NVFP4's); expert slices cut on whole 128-wide Hadamard blocks (384 / 256 on two cards); the expert kernel declares its prefill temporaries and the cache planner keeps them free |
| Alternatives | Converting the experts to NVFP4 (loses the 3-bit size); serving every linear in EXL3; cutting the slices inside a Hadamard block (would need re-encoding); a bigger generic VRAM headroom |
| For | 42.6 GiB of experts instead of 63.3 at 3.05 bpw, 9.9k experts cached per card instead of 7.1k; at 3.05 bpw decode 83-107 tok/s up to 248k on `xtx-xt` (NVFP4 72-86), agent turns 0.9-1.6 s (1.3-2.0), 59 GiB of RAM instead of 82; the dense layers keep the tested int8 path |
| Against | One agentic-benchmark run per checkpoint so far (13 and 12 of 29; NVFP4 12 and 18 in two runs); the 60 / 40 split NVFP4 avoids; the even split of two equal cards refused; 0.6 GiB per card of prefill temporaries; more kernels to maintain (rotations per routed expert) |
| Evidence | [how-it-works.md](how-it-works.md#exl3-checkpoints), [journey.md](journey.md#16-exl3-checkpoints-2026-10-04) (the first agent test crashed on a full 7900 XTX before the temporaries were kept free) |

## Not done, on purpose

| Idea | Why |
|---|---|
| Grouping the prefills of several agents | -3 to -6 % total time, but not bit-exact, needs a ~150 ms wait, and agents already overlap naturally |
| Mixing prefill chunks and decode steps in one batch | estimated at ~3 % |
| Prefetching experts by prediction | 5 to 31 wasted copies per useful one |
| A smarter expert-cache policy | even the offline optimum only halves the misses; capacity is the lever |

## Packaging

| Choice | Why | Cost |
|---|---|---|
| Build the image locally (`Dockerfile.rdna3`) | the bases are ~31 GB of ROCm 10 + PyTorch and, for its Triton, the ~29 GB ROCm 7.14 one, pinned by digest; the build itself takes minutes; the image records the commit it came from | a first build downloads both bases |
| One profile per hardware setup | settings that belong together (chunk size, memory ratio, split, concurrency) travel together, and a profile states what was measured | untested setups start from a derived profile (`TESTED=0`) |
| Settings as environment variables, most of the tuning off by default | the upstream behavior stays one variable away, for comparisons and bug reports; what is on by default is bit-exact, agent-validated, or new and checked otherwise (the release's sampler rework, exact by kernel tests, and its token broadcast, checked in `check`-mode server runs) ([options.md](options.md)) | long command lines, hence the profiles |
