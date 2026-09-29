# The journey

How this build came about: five days of measured experiments (2026-09-25 to 2026-09-29) on one machine, an RX 7900
XTX 24 GB + RX 7900 XT 20 GB (gfx1100, no GPU peer-to-peer), a Threadripper 3970X and 128 GB of DDR4, serving
Qwen3.8-Flash-Next NVFP4 to the author's coding agent (OpenCode) with a 262k-token context. Every number comes from an
A/B run on that machine. "tg" = decode tokens/s, "pp" = prefill tokens/s. The choices and their trade-offs are
summed up in [decisions.md](decisions.md).

## 0. Method, learned the hard way

- **One change at a time**, sessions interleaved (A B A B); a change counts only beyond the spread between sessions.
- **Never benchmark on predictable text**: speculative decoding and caches look great on repeated content (MTP
  acceptance 0.87-0.92 on code continuation, ~0.6 on real chat).
- **Watch every run**: a stall watchdog, a smoke test first (always with temperature > 0: greedy hides sampler bugs),
  abort a stall within minutes.
- **Never build images while measuring** (a concurrent build tripled a prefill), and give every image a persistent
  kernel cache (a cold JIT made a 52 tok/s build look like 23.9).
- **Microbenchmarks mislead twice**: weights stay hot in the 80-96 MB Infinity Cache, and the kernel mix differs from a
  real step. Only the full model decides.
- **Short fidelity tests are not enough** (section 8): an agentic, multi-turn benchmark is the referee.
- **Never run a kernel that may fault on the GPU driving your monitor**: a hard fault resets the card and takes the
  desktop session with it.

## 1. Starting point: FreeToken on two unequal cards

A community ROCm port of FreeToken, plus a patch so the two ranks accept unequal free memory: tensor parallelism over
both cards worked, 28 tok/s flat to 111k tokens, 1480-1680 tok/s of prefill, 4/4 needles found at 256k. Its MTP
branch gave nothing (acceptance 0.56-0.66 against a break-even of 0.618).

## 2. Profiling the decode step: 27 -> 36 tok/s

GPUs busy 99 % of every step. 53-56 % of the time went to bf16 dense GEMMs running at ~20 % of VRAM bandwidth (the
checkpoint keeps every non-expert tensor in bf16), 14 % to PCIe copies of missing experts, 7-14 % to all-reduces (the
XTX waiting for the XT). PyTorch TunableOp, read-only with tuned files, plus a larger memory ratio (0.85: more
expert-cache slots): **27.0 -> 35.9 tok/s**. Online TunableOp would retune every new prompt length (16 s prefills).
Power caps and undervolting did not limit decode; a VRAM overclock on the XT corrupted memory from ~88 C (29 bad reads
in 4 minutes) and was abandoned.

## 3. An own branch on upstream main

Merging upstream into the community port gave 46 conflicting files, so the needed pieces were ported onto upstream main
instead: upstream's pending ROCm PRs, an LDS budget clamp for RDNA, a relay handshake for TP>1, a GDN boundary fix, a
qwen4_exp tensor-parallel port, opt-in unequal-card planning, and a fallback for upstream's new sampling kernels, which
hang on RDNA3 (section 13 has the end of that story). Same speed as the port: 36.1 tok/s, 6.3 s for an 8.4k prompt.

## 4. Decode, first round: +46 %

| Change | Decode | Note |
|---|---:|---|
| baseline | 35.8 | |
| one all-reduce per MoE block instead of two | +2.7 % | also -4.5 % prefill time |
| split-K Triton GEMV for 5 bf16 shapes | +3.5 % | |
| weight-only int8 dense layers (routers stay bf16) | +20 % -> +28 % | quantize on the host and compact the allocator, or the freed memory is lost to fragmentation |
| int8 GEMV v2 (2D accumulator, one reduction, fused scale) | +8.5 % | |
| all together | **52.4** | 59/60 HumanEval-60 like bf16, 4/4 needles at 227k |

Expert-cache policy: LFU = LRU, and even the offline optimum (Belady) only halves the misses. Capacity, not policy, is
the lever.

## 5. Decode, second round: +6 %

- **Top-k-first sampler**: one `torch.topk` + one small kernel instead of ~20 launches: +1.7 %. It also turned out that
  the sampler used so far was inexact (it could draw outside the top-k and approximated top-p).
- **Uneven split**: the XTX takes 55 % of the GDN heads and expert widths: +4 %, expert cache 7683 -> 8708 slots.
  57.5 % or 60 % overload the XTX in prefill (compute-bound; the XTX has ~14 % more compute).
- **int8 KV cache** for the attention layers: 3.19 -> 1.71 GiB per card. The attention kernel took three versions to
  cost nothing (dequantizing every element: 48 us against 14 us; folding the scales into the tiles: 30 us; fp16 dots,
  native on RDNA3: 15 us). Later removed (section 8).
- Together: 55.5 tok/s, 8.4k prefill 5.5 s (-9 %); at 227k, prefill +17 % and decode +11 %.

## 6. Precision: is the speed paid for?

- A 12-prompt "identical text" count suggested drift (7/12 identical to bf16). Misleading: it counts differences, not
  errors.
- A scored benchmark (671 items: HumanEval 164, MBPP 257, GSM8K 250; greedy): baseline bf16 636, another bf16 build
  632, int8 builds 635. Two bf16 builds already differ by 4 items and on a quarter of the answers.
- A teacher-forced probe (next-token distributions against bf16 over 73k positions): merely reordering the bf16 additions
  of the MoE all-reduce moves the distributions half as much as int8 does. The model is extremely sensitive to rounding
  order, which is why answers diverge without getting worse.
- Grouped int8 (one scale per 128 values, 3x less weight error) got closer on the probe but lost the scored benchmark
  twice (625 against 635). Rejected: a prefill probe does not predict long greedy generations.

## 7. Faster long reads

Each prefill chunk streams all 512 experts of every layer over PCIe (~700 MB per layer per card), so bigger chunks
stream less: 16k-token chunks read an 8.4k prompt in 4.3 s instead of 5.5 s, and a fresh 67k-token file ~1.5x faster.
They only fitted at depth once the PLE layer and the prefill MoE ran 4096 tokens at a time inside the chunk (same
results) and the memory ratio went to 0.80. Cost: -2..3 % decode beyond 100k context (fewer cache slots).

## 8. The benchmark that mattered: an agent's own work

Everything so far had been judged on speed plus short fidelity tests (the scored benchmark, the probe, needles at
227k). The author's own (private) benchmark disagreed: a coding agent in OpenCode fixing 29 bugs planted in a ~50k-line codebase,
over long sessions. Every build with the int8 attention KV fixed 8-9 bugs, every bf16-KV build 12-14, and the int8-KV
agent stopped exploring early (it declared the task done at ~180k tokens instead of filling the context, compacting
and finding more). A single-turn bug-hunting proxy at 46k-183k tokens did not reproduce it. The int8 KV was removed
from every profile. From then on the rule was: **bit-exact changes, or changes validated by the agentic benchmark**.

## 9. Bit-exact speed after that

- **Expert reuse in prefill**: upstream's `--moe-prefill-hit-d2d` needed `cudaMemcpyBatchAsync`; HIP 7.1+ has
  `hipMemcpyBatchAsync`. A third of the experts no longer cross PCIe per chunk: agent turn 1.55 -> 1.18 s.
- **Host-memory all-reduce**: RCCL stages each 5 KB decode all-reduce through host memory with its proxies (~12 us in a
  graph). A shared host region and a one-block kernel do it in 6 us with the same bits: +2.5-4.8 % decode.
- **Expert GEMM tiles** re-swept for this model on these cards, changing only the tiles that keep the summation order:
  prefill MoE x1.5-1.8, long reads -11 %, decode +1.3 %.
- **Several requests**: the decode GEMVs run their unchanged per-row body once per request, so a request gets the same
  bits alone or batched; CUDA graphs for every batch size up to 4. 57 / 82 / 113 tok/s in total at 1 / 2 / 4
  requests on that first build; reserving the recurrent states of four requests then cost ~650 expert-cache slots, and
  the release build does 55 / 81 / 98 / 105 at 1 / 2 / 3 / 4. Four agents finish the same work ~2.6x sooner.
- One multi-request crash found on the way: several requests prefilled together made the PLE layer allocate ~2.6 GiB
  of temporaries and ran the XT out of memory; blocking it for several requests too brought it to 0.7 GiB, same output.

## 10. Conversations kept in RAM

With parallel sub-agents, the ~262k tokens of GPU context memory fill up and the least recently used conversation is
thrown away: its agent re-reads 90-110k tokens at its next turn (~45-55 s). The fix copies what must leave the GPUs to
pinned RAM and back (~20 GB/s), exactly. Making it work took three wrong theories (snapshot rows, promoting whole
paths) before the real cause of the lost contexts showed up in diagnostic logs: an agent turn is a single node of the
prefix tree, so eviction had to take a whole turn and overshot, dropping what it had just demoted. Splitting a node on
eviction fixed it. Result: 4 agents with 82-117k-token conversations, 3 turns each: **599 s -> 35 s**, first token
~47 s -> ~2 s; 4 agents x 12 short turns on a small pool: 143 s -> 94 s. Cost: ~9 GiB of RAM locked.

## 11. Tested and not kept: grouping prefills

Every prefill pass costs ~1.4 s whatever its size (it streams every expert) plus ~0.25 ms per new token, so holding a
prefill for a moment to batch it with another agent's looked attractive. Measured: -3..-6 % total time and median time
to first token -37 % with 4 agents, but agents already pair themselves (11 of 46 turns unaided), catching simultaneous
spawns needed a ~150 ms wait, and batched prefills are not bit-exact. Dropped. Mixing prefill chunks and decode steps
in one batch was estimated at ~3 % and not done.

## 12. One card

At TP=1 on the XTX alone: 36 tok/s, 131k context, one request. 16k prefill chunks do not fit a single card's working
memory (a 60k-token prompt took 166 s and decode fell to 1.9 tok/s); 8k chunks read it in 45 s. The XT alone works
too (28.5 tok/s, 0.8 GiB free at 124k).

## 13. Reviewing everything before publishing

Six parallel reviews of the whole diff against upstream, each finding checked by hand, found real bugs that the
benchmarks had not shown:

- The ROCm sampler fallback kept for requests without a small `top_k` was wrong: on one distribution 89 % of the
  `top_k=20` draws fell outside the top 20. It was also reached when a greedy request shared a batch with a sampled one.
  Upstream's exact kernels, run with one block per row so they cannot hang, were exact but took 10 ms per step for
  top-p alone and 33-60 ms on a flat distribution. The final ROCm path takes an exact threshold from a sort and draws
  with barrier-free kernels: 0.2-1 ms on any distribution, same draws as upstream's kernels for a given seed.
- Each rank sampled on its own; nothing stopped two different GPUs from continuing on different tokens. Rank 0's tokens
  are now broadcast (15 us per step).
- Memory planning with unequal cards used the larger card's free memory in one place; a runtime cache rebuild under an
  uneven split could make the ranks disagree and hang; three edge cases of the RAM tier; a copy engine that failed on
  HIP < 7.1; an unhelpful error for the vision tower at TP>1.

The GPU validation of the fixed build: 120/120 identical greedy answers, decode -0.4 % (noise), prefill ~1 % faster.

## 14. Dropped along the way

| Idea | Why not |
|---|---|
| Predicting the next layer's experts to prefetch them | co-occurrence: 31 wasted copies per useful one; a next-layer router: 65 % recall, still 5 wasted per useful one and an extra router per layer |
| More router warps | upstream's single warp is fastest on RDNA3 |
| Sampled-LRU expert cache | +0.3 % |
| Fusing the hyper-connection activation into the next GEMV | exact but twice slower |
| Interleaving decode steps between prefill chunks | hurts the short prefills of agent turns |
| Even split of the GDN heads under the uneven split | -2.5 % decode |
| Grouped int8, int8 KV | precision (sections 6 and 8) |
| Speculative decoding (MTP) | acceptance ~0.6 on real traffic, about break-even |
| Online TunableOp | retunes every new prompt length |

## 15. Still open

- An occasional 9-21 s stall, in about one run in three, on every build since the start: not understood yet.
- The model needs ~65 GiB of RAM for its experts alone (the engine keeps every expert in RAM, and NVFP4 is the smallest
  format it runs for this model): no 64 GB machine can serve it ([limits.md](limits.md)).
- More than two GPUs, other RDNA3 cards, RDNA4 and NVIDIA are untested.
