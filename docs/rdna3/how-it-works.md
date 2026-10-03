# How it works

What this build changes in FreeToken, and why, for Qwen3.8-Flash-Next on RDNA3. Upstream's own design (expert
offload, radix cache, FTW weights, APIs) is described in [FREETOKEN_UPSTREAM_README.md](../FREETOKEN_UPSTREAM_README.md).

## The problem

Qwen3.8-Flash-Next: 48 layers (12 attention "QSA" layers + 36 GatedDeltaNet (GDN) linear-attention layers), 512
experts per layer (24,576 in all), ~6B parameters active per token, per-layer n-gram embeddings (PLE),
hyper-connections, 262k context. In NVFP4 the experts alone are 68 GB and the n-gram tables 51 GB, against 44 GB of
VRAM on an XTX + XT. The two cards cannot read each other's memory (no GPU peer-to-peer on the reference machine, as
usual on consumer boards): every exchange between them goes through system RAM.

## Where everything lives

| Where | What |
|---|---|
| GPU VRAM | dense weights (int8), the KV cache of the attention layers, the GDN recurrent states, and an **expert cache** filling everything else (on `xtx-xt`: 7.1k of the 24,576 experts, on each card) |
| System RAM | **every expert** (~63 GiB, pinned; each rank holds its own share of every expert), conversations moved off the GPUs (the RAM tier), the ranks' exchange buffer |
| Disk | the n-gram embedding tables, read row by row on demand (the page cache keeps the hot rows) |

## One decode step

For each token, on both cards at once (tensor parallel): the dense projections (int8 GEMVs), then in each MoE layer
the router picks the experts, the ones in the GPU cache are used in place and the missing ones are copied from RAM over
PCIe (a least-recently-used cache), then an all-reduce joins the two halves of the layer, then sampling. Profiling
the untuned build: the GPUs were busy 99 % of the step; 53-56 % of it went to bf16 dense GEMMs running at ~20 % of VRAM
bandwidth, 14 % to PCIe expert copies, 7-14 % to all-reduces (the XTX waiting for the XT). The work below attacks those
three shares.

## Dense layers: int8 weights (+28 % decode)

The checkpoint keeps every non-expert tensor in bf16. At load, each dense linear layer is converted to int8 weights
with one scale per output channel and rank (routers stay bf16), and decode uses Triton GEMVs written for RDNA3 (2D
accumulator, one reduction, fused scale). Half the bytes per token, and ~2.4 GiB per card freed for the expert cache.
This is lossy (8-bit weights): no measurable loss on the scored benchmark, and it was in the builds validated on the
agentic one. The allocator is compacted after the conversion, or the freed memory stays fragmented and is lost.
Several concurrent requests run the same per-row kernel once per row, so a request's decode gets the same bits alone
or batched.

## Two unequal cards: uneven tensor parallelism (+4 %)

Upstream serves Qwen3.8-Flash-Next on one GPU only (its loader and NVFP4 expert kernels refuse TP>1) and refuses TP
ranks whose free memory differs by more than 2 GiB. This build's tensor-parallel port splits evenly by default. With
`FREETOKEN_TP_ALLOW_IMBALANCE` the memory plan uses the smaller card, and with `FREETOKEN_TP_SPLIT=0.55` the experts'
intermediate dimension (routed and shared) is split 55 / 45 and the GDN heads 9 / 7 (key) and 27 / 21 (value): the XTX has 20 % more VRAM and
~14 % more compute, so both finish together, and the XT's smaller share of each expert lets both cards cache more
experts (the ranks all use the smallest card's plan). Attention stays even (2 KV heads cannot be split unevenly). The
vocabulary stays even; the hyper-connection mixers, the PLE projections, the attention indexer, the norms and the
routers are replicated; with `--ple-backend disk` (every profile) each rank reads every PLE head from disk.

## Talking between the cards: host-memory all-reduce (+2.5-4.8 %)

Without peer-to-peer, RCCL stages each small decode all-reduce (5 KB) through host memory with its proxy threads:
~12 µs inside a CUDA graph, ~97 times per token. Both ranks now map one shared host-memory region and a one-block
kernel copies, signals, waits and adds (fp32, rounded to bf16 exactly as RCCL does): 6 µs, same bits. Separately, the
two MoE all-reduces of a layer (routed and shared experts) are merged into one (+2.7 % decode): that change reorders
sums and was validated on the agentic benchmark.

## The expert cache

Upstream's LRU cache over the GPU memory left after weights and KV (`--moe-cache-auto`). Policies were compared on
real traces: LFU = LRU, and even the offline optimum (Belady) only halves the misses; capacity, not policy, is the
lever, which is why every freed megabyte goes to the cache. Expert GEMM tiles were re-swept for this model on these
cards, changing only tiles that keep the summation order (bit-exact).

## Prefill

A prefill chunk runs every token through every layer, so each chunk streams all 512 experts of every layer over PCIe
(~700 MB per layer per card). Bigger chunks mean less streaming: in the A/B test, 16k-token chunks read an 8.4k prompt
in 4.3 s instead of 5.5 s (4.5 s on the swept build, 4.1 s on the release build: [benchmarks.md](benchmarks.md)), and 54-101k-token reads
at depth 2-6 % faster. To fit them, the prefill MoE and the PLE layer run 4096 tokens at a time inside the chunk (the
MoE blocks are bit-identical; the PLE blocks can round differently, checked to tolerance and validated on the agentic
benchmark for one request), and experts already in the GPU cache are gathered on the GPU instead of crossing PCIe
again (`--moe-prefill-hit-d2d`, with `hipMemcpyBatchAsync` on ROCm, or one copy per entry on older HIP): a third of
the experts no longer cross PCIe per chunk, an agent turn's first token 1.55 -> 1.18 s. A single card keeps 8k chunks
(its working memory is smaller).

## Sampling on ROCm

Upstream's exact top-k / top-p kernels coordinate several blocks per row with a spin barrier that needs every block
of the row running at once; the cooperative launch does not give that on this ROCm stack, and they hang on gfx1100.
Here:

- when every request of a batch has a small `top_k` (the model's default is 20), the batch samples from its top-k
  candidates: one `torch.topk`, then one small kernel does temperature, softmax, top-p and the draw (~0.07 ms);
- otherwise an exact threshold comes from a sort of the probabilities (top-k = the k-th largest value, top-p = the
  largest value whose cumulative mass reaches p), and the draw uses upstream's multi-block inverse-CDF kernels, which
  need no barrier: 0.2 ms for one request, 1 ms for four, on any distribution (upstream's kernels with one block per
  row took 10 ms on a peaked row and 33-60 ms on a flat one);
- every rank samples its own copy of the logits, and rank 0's tokens are broadcast after each sampled step, so two
  different GPUs can never drift onto different tokens.

The sorted path and the broadcast are new in the release build: the sorted path checked by kernel tests (same kept
sets as the float64 definition, seeded draws identical to upstream's kernels), the broadcast by
`FREETOKEN_TP_SYNC_TOKENS=check` server runs (0 disagreements over ~8k sampled tokens); neither was run on the agentic
benchmark. The vendored sampling
module from before upstream PR #329, used earlier, was removed: its top-k could keep the whole row (measured: 89 % of
the draws outside the top-20 on one distribution) and its top-p was approximate.

## Several requests at once

Up to 4 requests decode together (`--max-running-requests 4`), with CUDA graphs captured for every batch size up to 4.
The decode GEMVs run their unchanged per-row body once per request (the weight tile loaded once when the reduction fits
one tile), so batching changes the speed of a request's decode, not its tokens: 55 / 81 / 98 / 105 tok/s in total at
1 / 2 / 3 / 4 requests (ROCm 7.14 release build, without the MTP head). Prompts prefilled in the same forward can differ at rounding level (the dense GEMMs run at
another size), as prefix caching already makes them. Added after the agentic benchmark runs: a request alone on this
build gives the same answers as before (120 prompts), batched runs were not scored. Prefills are not mixed with
decode steps in one batch (estimated at ~3 %, not done); agents naturally overlap their prefills.

## Conversations kept in RAM

Upstream's hybrid radix cache keeps, per conversation prefix, the attention KV pages and a snapshot of the GDN
recurrent state at the end of the last prompt; a returning conversation resumes from there. When the GPU pool is full
it evicts the least recently used prefixes, and their conversations are re-read from scratch (~45-55 s for 100k
tokens). With `FREETOKEN_HOST_KV=1` an evicted node is **demoted** instead: its KV pages and GDN snapshot are copied to
pinned RAM and the node stays in the tree; a later match **promotes** it back (fresh GPU pages and state slot,
host-to-device copies, ~20 GB/s). The copies are exact. Eviction splits a node when only part of it has to leave (an
agent turn is one node), capacities are counted in pages and snapshots so both ranks take identical decisions, and
nodes being promoted are pinned against eviction. Four agents with 82-117k-token conversations, 3 turns each: after
the first reads, the rounds of turns take 35 s instead of 599 s.

## Images (`--vision`, off by default)

The Qwen3.8 vision tower (27 ViT blocks, ~0.45 B parameters) is upstream's, and it was never split across the GPUs.

- **Where the tower runs.** Rank 0 builds it whole and encodes every image. The embeddings are broadcast to the
  other rank (5 MiB for a 1024-token image), so both ranks put the same numbers in their residual stream. Rank 1
  holds no tower, and its expert cache keeps the room.
- **Weights.** The tower's blocks stream from pinned RAM two at a time, and the tower stays bf16 under
  `FREETOKEN_INT8_DENSE`.
- **Positions.** The text model switches to the 3-axis rope (mrope) that the image positions need. For text it gives
  the same rotation: greedy answers are identical, and speed is the same up to 255k of context on two cards.

Numbers: [benchmarks.md](benchmarks.md#images---vision-xtx-xt).

## What stays upstream's

The HTTP APIs and parsers and the FTW format are upstream's. The scheduler, the radix cache, the offload engine and the
qwen4_exp loader are upstream's with this build's hooks (RAM tier, GDN boundary carry, relay handshake, tensor-parallel
sharding, expert reuse in prefill on ROCm, tuned and blocked NVFP4 kernels); every changed file is listed in [changes-from-upstream.md](changes-from-upstream.md).
