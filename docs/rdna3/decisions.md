# Decisions and trade-offs

The main choices behind this build, what they cost, and the measurement that settled each one. The story in order is
in [journey.md](journey.md); the options themselves in [options.md](options.md).

## Rule 0: precision before speed

After a faster build lost a third of its score on an agentic benchmark (see "bf16 attention KV, not int8" below), every
change must be either **bit-exact** (same greedy answers on a fixed set of 120 prompts, or a kernel check with identical
outputs) or **validated by the agentic benchmark** (the author's, private: a coding agent fixing 29 planted bugs over
long sessions). Short fidelity tests (scored benchmarks, needle tests, teacher-forced probes) are used to reject
changes, never to accept them alone. The changes made after the last agentic run are bit-exact or exact by kernel tests,
except these, each described where it is documented: prompts prefilled together in one batch (up to 4 requests) can
differ at rounding level; ROCm 10's own Triton 3.8 rounds differently from the 3.7.1 of that run
([journey.md](journey.md#18-rocm-10s-own-triton-2026-10-05)); upstream's NVFP4 decode kernel changed 5 of 15 greedy
answers ([journey.md](journey.md#23-upstream-up-to-ee9789f-2026-10-10)); the EXL3 dense weights are rebuilt in fp64
([options.md](options.md)); upstream's fp32 NVFP4 global scales and rewritten NVFP4 prefill GEMM changed 25 of 42 greedy
outputs, EXL3 none ([journey.md](journey.md#35-upstream-up-to-cc56b6b-2026-10-20)); and `--hybrid` moves its answers
with the expert cache's contents. The ROCm 10 base was run on the agentic benchmark with the MTP head: 12 and 18 of 29
in two runs.

- **Gives**: speed that did not cost quality on the agentic benchmark.
- **Costs**: slower progress; some real speedups were dropped (int8 KV, grouped prefills).

## Two cards in tensor parallel, rather than one card

| | |
|---|---|
| Choice | Split every layer over the XTX and the XT, both working on every token |
| Alternatives | One card with more experts offloaded; each card serving its own requests |
| For | Decode 88 tok/s instead of 46 at `rdna3/serve.sh`'s sampling (65.5 against 41 without the MTP head); 262k context instead of 131k; 44 GB of VRAM for one expert cache |
| Against | Every layer ends with an all-reduce through system RAM (no GPU peer-to-peer on the reference machine); both cards are busy for every request; more code (uneven split, host all-reduce) |
| Evidence | Depth sweeps of `xtx-xt` vs `xtx` ([benchmarks.md](benchmarks.md)) |

## Uneven split, 55 / 45

| | |
|---|---|
| Choice | The XTX (24 GB, ~14 % more compute) takes 55 % of the split work |
| Alternatives | Even split (this build's default; upstream has no tensor parallelism for this model); 57.5 % or 60 % |
| For | +4 % decode; both cards finish together; the XT's smaller share of each expert raises the expert-cache size of both cards |
| Against | ~+0.1 s first-token latency on short prompts; the tuned decode tiles are keyed to this split's shapes; attention and the input embedding still split evenly (the LM head with `FREETOKEN_LMHEAD_SPLIT` follows the shares) |
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
| Choice | 16k-token chunks with the PLE layer and the MoE blocked by 4096 tokens inside (the NVFP4 MoE by 8192 in `xtx-xt` and `xtx-xt-4x262k`), memory ratio 0.80 |
| Alternatives | 4k chunks with ratio 0.85 |
| For | 8.4k prompt 4.3 s instead of 5.5 s in the A/B test; 54-101k-token reads at depth 2-6 % faster |
| Against | -2 to -3 % decode beyond 100k context (fewer expert-cache slots); a single card cannot hold 16k chunks (8k there) |
| Evidence | At ratio 0.85, 16k chunks left 50-120 MiB free at 227k and slowed down |

## Up to four requests at once

| | |
|---|---|
| Choice | `--max-running-requests 4` with CUDA graphs for every batch size up to 4, per-row decode kernels |
| Alternatives | One request at a time (upstream's default is 4) |
| For | 1 / 2 / 3 / 4 greedy answers sent at once finish in 3.6 / 5.9 / 8.4-8.6 / 10.1 s; four agents doing the same work finish ~2.7x sooner than one at a time; a request alone computes exactly as with one request at a time |
| Against | Each request decodes slower when others run (two conversations at ~100k tokens: 33.6-34.4 tok/s each, against 71 alone); prompts prefilled in one batch can differ at rounding level; not run on the agentic benchmark; GDN state for 4 requests (too much for one card: one card stays at 1) |

## Conversations kept in RAM

| | |
|---|---|
| Choice | Evicted KV pages and GDN snapshots go to pinned RAM and come back on demand |
| Alternatives | Drop them and re-read the conversation (upstream) |
| For | When conversations together outgrow the GPU pool (parallel sub-agents), an evicted one comes back from RAM (~20 GB/s) instead of being read again: a returning 82-117k-token conversation's first token ~2 s instead of ~47 s (four agents, 3 turns each); exact |
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
| Alternatives | Upstream's exact kernels (hang on gfx1100 with several blocks per row; 10-60 ms per step with one); the vendored module from before upstream PR #329 (wrong: up to 89 % of draws outside the top-k, its top-p approximate, and not faster) |
| For | Exact on any request; 0.07 ms with the model's default `top_k`, 0.2-1 ms otherwise; the two GPUs cannot drift apart |
| Against | The sort costs up to ~1 ms per step with 4 requests when one of them has no `top_k`; 15 µs per step for the broadcast; the sorted path checked by kernel tests, the broadcast in `check`-mode server runs, neither run on the agentic benchmark |

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
| Choice | On in the profiles except `gre` (`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4`; `rdna3/serve.sh --no-mtp` turns it off). The checkpoint's MTP layer drafts up to 3 tokens; the next decode step verifies them in the same captured forward, 2-4 rows chosen per request from its measured acceptance and the step costs of the card count. Rejected rows are undone by restoring the GDN, conv and PLE states saved after each row (the attention KV needs nothing: rejected positions are rewritten before any query sees them). Sampled requests use exact speculative sampling, their drafts drawn under their own sampling. In `xtx-xt` and its K/V-in-RAM variants up to four requests at once verify, 2 rows each past two; in the other profiles one request at a time. With more, steps run one row each and the head only writes its KV |
| Alternatives | Plain decode; a fixed draft depth; more than 2 rows each for three or four requests at once; the head's own forward captured in a graph |
| For | Agentic benchmark on ROCm 10 with the head: 12 and 18 of 29 (two runs). One request on `xtx-xt`: +35 % decode at `rdna3/serve.sh`'s sampling, +35-60 % greedy up to 260k of context, x1.38 on 60 code and math items; one card +6-13 % (+9-16 % greedy); greedy answers identical to plain decode (bit-exact on every check) |
| Against | 3-4 % slower at 3-4 requests that do not verify (measured on `xtx-xt` with `FREETOKEN_SPEC_BS_MAX=1`, an earlier build); prompt reading 0-5 % slower (the head reads it too; an earlier build); the head's 512 experts compete for the expert cache; a CUDA graph per row count |
| Evidence | Step costs 31 / 53 / 78 / 102 ms at 1-4 rows on one card (the extra rows' missing experts cross PCIe); 1.9-3.9 tokens kept per request step ([benchmarks.md](benchmarks.md#speculative-decoding-mtp-head)) |

## EXL3 checkpoints: experts in EXL3, the rest in bf16

| | |
|---|---|
| Choice | Keep the routed experts in EXL3 (host banks and GPU cache) behind gfx1100 Triton kernels; decode every other linear to bf16 at load (then int8 like NVFP4's); expert slices cut on whole 128-wide Hadamard blocks (384 / 256 on two cards); the expert kernel declares its prefill temporaries and the cache planner keeps them free |
| Alternatives | Converting the experts to NVFP4 (loses the 3-bit size); serving every linear in EXL3; cutting the slices inside a Hadamard block (would need re-encoding); a bigger generic VRAM headroom |
| For | 42.6 GiB of experts instead of 63.3 at 3.05 bpw, 9.8k experts cached per card instead of 7.0k; at 3.05 bpw decode 95-117 tok/s up to 260k on `xtx-xt` (NVFP4 78-96), agent turns 0.70-0.94 s (0.80-0.93), 60 GiB of RAM instead of 82; the dense layers keep the tested int8 path |
| Against | One agentic-benchmark run per checkpoint so far (13 and 12 of 29; NVFP4 12 and 18 in two runs); the 60 / 40 split NVFP4 avoids; the even split of two equal cards refused; 0.6 GiB per card of prefill temporaries; more kernels to maintain (rotations per routed expert) |
| Evidence | [how-it-works.md](how-it-works.md#exl3-checkpoints), [journey.md](journey.md#16-exl3-checkpoints-2026-10-04) (the first agent test crashed on a full 7900 XTX before the temporaries were kept free) |

## Not done, on purpose

| Idea | Why |
|---|---|
| Grouping the prefills of several agents | -3 to -6 % total time, but not bit-exact, needs a ~150 ms wait, and agents already overlap naturally |
| Mixing prefill chunks and decode steps in one batch | estimated at ~3 % |
| Prefetching experts by prediction in decode | 5 to 31 wasted copies per useful one (a short prefill's many tokens let the next layer's router predict 79-84 % of its experts: done there) |
| A smarter expert-cache policy | even the offline optimum only halves the misses; capacity is the lever |

## Packaging

| Choice | Why | Cost |
|---|---|---|
| Build the image locally (`Dockerfile.rdna3`) | the base is ~25 GB of ROCm 10.1 + PyTorch, pinned by digest; the build itself takes minutes; the image records the commit it came from | a first build downloads the base |
| One profile per hardware setup | settings that belong together (chunk size, memory ratio, split, concurrency) travel together, and a profile states what was measured | untested setups start from a derived profile (`TESTED=0`) |
| Settings as environment variables, each switchable | the upstream behavior stays one variable away, for comparisons and bug reports; what is on by default is bit-exact, agent-validated, or checked otherwise (the ROCm sampler's sorted path, exact by kernel tests, and the token broadcast, checked in `check`-mode server runs) ([options.md](options.md)) | long command lines, hence the profiles |
