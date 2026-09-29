# How it works

What this build changes in FreeToken, and why, for Qwen3.8-Flash-Next on RDNA3. Upstream's own design (expert
offload, radix cache, FTW weights, APIs) is described in [FREETOKEN_UPSTREAM_README.md](../FREETOKEN_UPSTREAM_README.md).

## The problem

Qwen3.8-Flash-Next: 48 layers (12 attention "QSA" layers + 36 GatedDeltaNet linear-attention layers), 512 experts per
layer (24,576 in all), ~6B parameters active per token, per-layer n-gram embeddings (PLE), hyper-connections, 262k
context. In NVFP4 the experts alone are 68 GB and the n-gram tables 51 GB, against 44 GB of VRAM on an XTX + XT. The
two cards cannot read each other's memory (no peer-to-peer on consumer boards): every exchange between them goes
through system RAM.

## Where everything lives

| Where | What |
|---|---|
| GPU VRAM | dense weights (int8), the KV cache of the attention layers, the GDN recurrent states, and an **expert cache** filling everything else (on `xtx-xt`: 8.3k of the 24,576 experts on the XTX, 7.2k on the XT) |
| System RAM | **every expert** (64.7 GiB, pinned, shared by both ranks), conversations moved off the GPUs (the RAM tier), the ranks' exchange buffer |
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
with one scale per output channel (routers stay bf16), and decode uses Triton GEMVs written for RDNA3 (2D accumulator,
one reduction, fused scale). Half the bytes per token, and ~2.3 GiB per card freed for the expert cache. The allocator
is compacted after the conversion, or the freed memory stays fragmented and is lost. Several concurrent requests run
the same per-row kernel once per row, so a request gets the same bits alone or batched.

## Two unequal cards: uneven tensor parallelism (+4 %)

Upstream splits every layer evenly and refuses cards with unequal free memory. Here the memory plan uses the smaller
card (`FREETOKEN_TP_ALLOW_IMBALANCE`), and the experts' intermediate dimension (routed and shared) and the GDN heads
are split 55 / 45 (`FREETOKEN_TP_SPLIT=0.55`): the XTX has 20 % more VRAM and ~14 % more compute, so both finish together and the XTX
gets more expert-cache slots. Attention, the vocabulary and the PLE heads stay even (their head counts do not split
unevenly).

## Talking between the cards: host-memory all-reduce (+2.5-4.8 %)

Without peer-to-peer, RCCL stages each small decode all-reduce (5 KB) through host memory with its proxy threads:
~12 us inside a CUDA graph, dozens of times per token. Both ranks now map one shared host-memory region and a one-block
kernel copies, signals, waits and adds (fp32, rounded to bf16 exactly as RCCL does): 6 us, same bits. The two MoE
all-reduces of a layer (routed and shared experts) are also merged into one.

## The expert cache

Upstream's LRU cache over the GPU memory left after weights and KV (`--moe-cache-auto`). Policies were compared on
real traces: LFU = LRU, and even the offline optimum (Belady) only halves the misses; capacity, not policy, is the
lever, which is why every freed megabyte goes to the cache. Expert GEMM tiles were re-swept for this model on these
cards, changing only tiles that keep the summation order (bit-exact).

## Prefill

A prefill chunk runs every token through every layer, so each chunk streams all 512 experts of every layer over PCIe
(~700 MB per layer per card). Bigger chunks mean less streaming: in the A/B test, 16k-token chunks read an
8.4k prompt in 4.3 s instead of 5.5 s (4.5 s on the release build) and a fresh 67k-token file ~1.5x faster. To fit them, the PLE layer and the prefill MoE run 4096
tokens at a time inside the chunk (same results), and experts already in the GPU cache are gathered on the GPU instead
of crossing PCIe again (`--moe-prefill-hit-d2d`, which needed `hipMemcpyBatchAsync` on ROCm): a third of the experts no
longer cross PCIe per chunk, an agent turn's first token 1.55 -> 1.18 s. A single card keeps 8k chunks (its working
memory is smaller).

## Sampling on ROCm

Upstream's exact top-k / top-p kernels coordinate several blocks per row with a spin barrier, which RDNA3 does not
guarantee to be co-resident: they hang. Here:

- requests with a small `top_k` (the model's default is 20) sample from their top-k candidates: one `torch.topk`, then
  one small kernel does temperature, softmax, top-p and the draw (~0.07 ms);
- the others take an exact threshold from a sort of the probabilities (top-k = the k-th largest value, top-p = the
  largest value whose cumulative mass reaches p) and draw with upstream's multi-block inverse-CDF kernels, which need
  no barrier: 0.2 ms for one request, 1 ms for four, on any distribution (upstream's kernels with one block per row
  took 10 ms on a peaked row and 33-60 ms on a flat one);
- every rank samples its own copy of the logits, and rank 0's tokens are broadcast after each sampled step, so two
  different GPUs can never drift onto different tokens.

The vendored sampling module from before upstream PR #329, used earlier, was removed: its top-k could keep the whole row (measured: 89 % of
the draws outside the top-20 on one distribution) and its top-p was approximate.

## Several requests at once

Up to 4 requests decode together (`--max-running-requests 4`), with CUDA graphs captured for every batch size up to 4.
The decode GEMVs run their unchanged per-row body once per request (the weight tile loaded once when the reduction fits
one tile), so batching changes speed, not answers: 57 / 82 / 113 tok/s in total at 1 / 2 / 4 requests. Prefills are
not mixed with decode steps in one batch (estimated at ~3 %, not done); agents naturally overlap their prefills.

## Conversations kept in RAM

Upstream's hybrid radix cache keeps, per conversation prefix, the attention KV pages and a snapshot of the GDN
recurrent state at the end of the last prompt; a returning conversation resumes from there. When the GPU pool is full
it evicts the least recently used prefixes, and their conversations are re-read from scratch (~45-55 s for 100k
tokens). With `FREETOKEN_HOST_KV=1` an evicted node is **demoted** instead: its KV pages and GDN snapshot are copied to
pinned RAM and the node stays in the tree; a later match **promotes** it back (fresh GPU pages and state slot,
host-to-device copies, ~20 GB/s). The copies are exact. Eviction splits a node when only part of it has to leave (an
agent turn is one node), capacities are counted in pages and snapshots so both ranks take identical decisions, and
nodes being promoted are pinned against eviction. Four agents with ~100k-token conversations: 35 s instead of 599 s.

## What stays upstream's

The HTTP APIs and parsers, the scheduler, the radix cache itself, the offload engine, the model loaders and the FTW
format are upstream's; the list of every changed file is in [changes-from-upstream.md](changes-from-upstream.md).
