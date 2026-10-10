# The journey

How this build came about: measured experiments from 2026-09-25 to 2026-10-07 on one machine, an RX 7900
XTX 24 GB + RX 7900 XT 20 GB (gfx1100, no GPU peer-to-peer), a Threadripper 3970X and 128 GB of DDR4, serving
Qwen3.8-Flash-Next NVFP4 to the author's coding agent (OpenCode) with a ~250k-token context (262k until 2026-10-03). Every number was measured
on that machine (changes in interleaved A/B runs), except the few marked as estimates. The choices and their
trade-offs are summed up in [decisions.md](decisions.md).

## 0. Method, learned the hard way

- **One change at a time**, sessions interleaved (A B A B); a change counts only beyond the spread between sessions.
- **Never benchmark on predictable text**: speculative decoding and caches look great on repeated content (MTP
  acceptance 0.87 on code continuation, ~0.6 on real chat).
- **Watch every run**: a stall watchdog, a smoke test first (always with temperature > 0: greedy hides sampler bugs),
  abort a stall within minutes.
- **Never build images while measuring** (a concurrent build tripled a prefill), and give every image a persistent
  kernel cache (a cold JIT made a 52 tok/s build look like 23.9).
- **Microbenchmarks mislead twice**: weights stay hot in the 80-96 MB Infinity Cache, and the kernel mix differs from a
  real step. Only the full model decides.
- **Short fidelity tests are not enough** (section 8): an agentic, multi-turn benchmark is the referee.

## 1. Starting point: FreeToken on two unequal cards

A community ROCm port of FreeToken, plus a patch so the two ranks accept unequal free memory: tensor parallelism over
both cards worked, 28 tok/s flat to 111k tokens, 1480-1680 tok/s of prefill, 4/4 needles found at 256k. Its MTP
branch gave nothing (acceptance 0.56-0.66 against a break-even of 0.618).

## 2. Profiling the decode step: 27 -> 36 tok/s

GPUs busy 99 % of every step. 53-56 % of the time went to bf16 dense GEMMs running at ~20 % of VRAM bandwidth (the
checkpoint keeps every non-expert tensor in bf16), 14 % to PCIe copies of missing experts, 7-14 % to all-reduces (the
XTX waiting for the XT). PyTorch TunableOp, read-only with tuned files, plus a larger memory ratio (0.85: more
expert-cache slots): **27.0 -> 35.9 tok/s**. Online TunableOp would retune every new prompt length (16 s prefills).
Power caps and undervolting did not limit decode; a VRAM overclock on the XT corrupted memory from ~88 °C (29 bad reads
in 4 minutes) and was abandoned.

## 3. A branch of its own on upstream main

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

- **Top-k-first sampler**: one `torch.topk` + one small kernel instead of a chain of separate kernels: +1.7 %. It also turned out that
  the sampler used so far was inexact (it could draw outside the top-k and approximated top-p).
- **Uneven split**: the XTX takes 55 % of the expert widths and 9 of the 16 GDN key heads: +4 %, and the expert cache
  of both cards 7683 -> 8708 slots (the XT's smaller share of each expert; the ranks use the smaller card's plan).
  57.5 % or 60 % overload the XTX in prefill (compute-bound; the XTX has ~14 % more compute).
- **int8 KV cache** for the attention layers: 3.19 -> 1.71 GiB per card. The attention kernel took three versions to
  cost nothing (dequantizing every element: 48 µs against 14 µs; folding the scales into the tiles: 30 µs; fp16 dots,
  native on RDNA3: 15 µs). Later removed (section 8).
- Together: 55.5 tok/s, 8.4k prefill 5.5 s (-9 %); at 227k, prefill +17 % and decode +11 %.

## 6. Precision: is the speed paid for?

- A 12-prompt "identical text" count suggested drift (8-9/12 identical to bf16). Misleading: it counts differences, not
  errors.
- A scored benchmark (671 items: HumanEval 164, MBPP 257, GSM8K 250; greedy): the community port (bf16) 636, the
  upstream-based baseline (bf16) 632, int8 builds 635. Two bf16 builds already differ by 4 items and on a quarter of the answers.
- A teacher-forced probe (next-token distributions against bf16 over 73k positions): merely reordering the bf16 additions
  of the MoE all-reduce moves the distributions half as much as int8 does. The model is extremely sensitive to rounding
  order, which is why answers diverge without getting worse.
- Grouped int8 (one scale per 128 values, 3x less weight error) got closer on the probe but lost the scored benchmark
  twice (625 against 635). Rejected: a prefill probe does not predict long greedy generations.

## 7. Faster long reads

Each prefill chunk streams all 512 experts of every layer over PCIe (~700 MB per layer per card), so bigger chunks
stream less: 16k-token chunks read an 8.4k prompt in 4.3 s instead of 5.5 s, and 54-101k-token reads at depth 2-6 %
faster. They only fitted at depth once the PLE layer and the prefill MoE ran 4096 tokens at a time inside the chunk
(the MoE blocks bit-identical, the PLE blocks within rounding) and the memory ratio went to 0.80. Cost: -2 to -3 %
decode beyond 100k context (fewer cache slots).

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
- **Host-memory all-reduce**: RCCL stages each 5 KB decode all-reduce through host memory with its proxies (~12 µs in a
  graph). A shared host region and a one-block kernel do it in 6 µs with the same bits: +2.5-4.8 % decode.
- **Expert GEMM tiles** re-swept for this model on these cards, changing only the tiles that keep the summation order:
  prefill MoE 1.5-1.8x, long reads -11 %, decode GEMMs ~13 % faster at one request (kernel sweep).
- **Several requests**: the decode GEMVs run their unchanged per-row body once per request, so a request's decode gets
  the same bits alone or batched (prompts prefilled together can differ at rounding level); CUDA graphs for every batch
  size up to 4. 57 / 82 / 113 tok/s in total at 1 / 2 / 4 requests on that first build; reserving the recurrent states
  of four requests then cost ~650 expert-cache slots, and the release build does 55 / 81 / 98 / 105 at 1 / 2 / 3 / 4.
  Four agents doing the same work finished ~2.7x sooner (97-100 s against 270 s one at a time).
- One multi-request crash found on the way: several requests prefilled together made the PLE layer allocate ~2.6 GiB
  of temporaries and ran the XT out of memory; blocking it for several requests too brought it to 0.7 GiB, same output.

## 10. Conversations kept in RAM

With parallel sub-agents, the ~262k tokens of GPU context memory fill up and the least recently used conversation is
thrown away: its agent re-reads 90-110k tokens at its next turn (~45-55 s). The fix copies what must leave the GPUs to
pinned RAM and back (~20 GB/s), exactly. Making it work took two wrong theories (snapshot rows, promoting whole
paths) before the real cause of the lost contexts showed up in diagnostic logs: an agent turn is a single node of the
prefix tree, so eviction had to take a whole turn and overshot, dropping what it had just demoted. Splitting a node on
eviction fixed it. Result: 4 agents with 82-117k-token conversations, 3 turns each, rounds after the first reads: **599 s -> 35 s**, first token
~47 s -> ~2 s; 4 agents x 12 short turns on a small pool: 133 s -> 94 s. Cost: 9.0 GiB of RAM locked.

## 11. Tested and not kept: grouping prefills

A prefill pass has a large fixed cost whatever its size, since it streams every expert (in one test at a 20k-token
context: ~1.35 s, plus ~0.2 ms per new token), so holding a prefill for a moment to batch it with another agent's
looked attractive. Measured: -3 to -6 % total time and median time
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

- The ROCm sampler fallback, used for requests without a small `top_k` and for every row of a batch mixing greedy
  and sampled requests, was wrong: its top-k could keep the whole row (with `top_k=20` on one distribution, 89 % of
  the draws fell outside the top 20).
  Upstream's exact kernels, run with one block per row so they cannot hang, were exact but took 10 ms per step for
  top-p alone and 33-60 ms on a flat distribution. The final ROCm path takes an exact threshold from a sort and draws
  with barrier-free kernels: 0.2-1 ms on any distribution, the same draws as upstream's kernels for a given seed
  (except within fp32 rounding of a boundary between two tokens).
- Each rank sampled on its own; nothing stopped two different GPUs from continuing on different tokens. Rank 0's tokens
  are now broadcast (15 µs per step).
- Memory planning with unequal cards used the larger card's free memory in one place; a runtime cache rebuild under an
  uneven split could make the ranks disagree and hang; three edge cases of the RAM tier; a copy engine that failed on
  HIP < 7.1; an unhelpful error for the vision tower at TP>1.

The GPU validation of the review-fix branch (before its last sampler changes): 120/120 identical greedy answers,
decode -0.5 % (noise), prefill ~1 % faster. The published image itself was measured afterwards
([benchmarks.md](benchmarks.md)).

## 14. Images (2026-09-30)

- **The dead check.** The TP>1 error the review had added for the vision tower never fired: the loader renames the
  tower's keys to `visual.` before the check, which looked for `model.visual.`. So a TP=2 start with the tower got as
  far as a shape assertion (165 tower tensors the wrong size).
- **Never split.** The tower (0.45 B parameters, prefill only) is now never split. Rank 0 encodes and broadcasts the
  embeddings, which keeps the ranks' residual streams identical whatever the cards round. The two cards turned out to
  encode bit-identically, so the broadcast is insurance here.
- **Only rank 0 holds it.** A first version kept the tower on both ranks. Rank 1 then planned 351 fewer experts
  (-4.9 %), and decode measured 51.5-52.8 tok/s against 54.1 (single sessions). With the tower on rank 0 only, the
  plan loses 11 experts, and decode and prefill are within the spread between sessions.
- **Answers unchanged.** Greedy answers stayed identical with the switch to mrope.
- **Two cards, to 255k.** The depth sweep to 255k showed no difference beyond noise.
- **One card.** One-card profiles lose ~1.5-2 % of decode, because the tower shares the only card and the expert
  cache holds 165 fewer experts (436 MiB). Measured with the tower alone, the streamed weights take 130 MiB and the
  first encode takes ~170 MiB outside PyTorch's allocator, 94 MiB of that for MIOpen's Conv3d. Replacing that Conv3d
  with the equivalent matmul would win back ~36 experts but round differently, so it is left as it is.
- **Default cap.** 1024 image tokens: a full-HD screenshot of 16-pixel text stays readable, with half the prompt
  tokens of a 2048 cap (1.6 s instead of 1.9 s). Numbers: [benchmarks.md](benchmarks.md#images---vision-xtx-xt).

## 15. Speculative decoding with the MTP head (2026-10-01)

- **The head.** `mtp.*` is one more decoder layer (full attention, 512 bf16 experts, its own hyper-connection mixer,
  the main LM head). Replayed offline, its guess matched the next greedy token 0.875 of the time, the same with its
  experts quantized to NVFP4, so they ride the existing expert banks.
- **Verify inside the captured step.** A decode step runs m rows per request (the last token, then m - 1 drafts) and
  keeps the matching prefix. The recurrent states roll back to the last kept row from copies the kernels write after
  every row (GDN state, conv and PLE windows); attention KV needs nothing.
- **Where the gain is.** One request, greedy: 55.9 -> 76.9 tok/s at 3 rows once the head chained its own guesses and
  the kept tokens went to the host before drafting. Then the head's MoE, mixer and LM head ran on the kept row only, and
  each request picks 2-4 rows from its measured acceptance and the step costs. Up to 4 rows share each int8 weight
  tile in the GEMVs that loop over K (`FREETOKEN_INT8_ROW_GROUPS`, bit-exact).
- **Several requests.** Fixed 3 rows at 4 requests: -26 %. So it verifies only while one request decodes; with more,
  the head only writes its KV and the scheduler overlaps steps as without it. With the adaptive depth, two requests
  verifying together later paid on `xtx-xt` (+8-15 % each for two agents, 2026-10-04): `FREETOKEN_SPEC_BS_MAX=2` there.
- **Bugs found on the way.** The detokenizer repeated text when a request got several tokens in one step; the LM head
  gathered prompt rows twice (a GPU memory fault). 4 rows were not bit-exact: the GDN output norm picked its rows per
  program from the row count (also for 4 requests decoding together), and the QSA attention its tile profile; decode
  now pins both to the one-row choice. A review after the merge (2026-10-03): within a few tokens of the context's end
  the verify rows and the chained drafts indexed past the page table (now capped by the tokens left), image pad ids
  reached the head's embedding unchecked at TP=1, and a verify step in flight could let a finishing request donate an
  over-advanced GDN state to the prefix cache.
- **Validated.** Greedy answers identical to plain decode on the 120-item precision set (657 -> 426 s), a 124k-token
  conversation, a 35k-token read and a 1500-token answer. Sampled requests use exact speculative sampling. Numbers:
  [benchmarks.md](benchmarks.md#speculative-decoding-mtp-head).

## 16. EXL3 checkpoints (2026-10-04)

- **Why.** At 3 bits per weight the experts take 42.6 GiB instead of 63.3, the 64 GB question of
  [limits.md](limits.md#can-it-run-with-64-gb-of-ram); turboderp publishes EXL3 builds of this model. exllamav3's
  kernels are CUDA with PTX inline (`lop3`, `mma.sync`, `ldmatrix`, `cp.async`), so they were rewritten: first a torch
  reference of the format, checked on real tensors (5-bit dense layers against the bf16 originals: cosine 0.9993; a
  3-bit expert against its NVFP4 copy: 0.986; every wrong tile layout tried: ~0), then Triton kernels checked against
  it (the dequant bit-exact for K = 1-8 and the three codebooks).
- **Decode GEMV.** First version 1.3-2.9x slower than the NVFP4 one; then a 32-bit window extraction, the reduction
  moved out of the K loop, `v_dot4_u32_u8` for the codebook's byte sum (inline asm) and split-K: on par with the
  default NVFP4 tiles or faster, per shape. Skipping the codebook's per-weight fp16 rounding (an affine form of `mul1`)
  gained nothing once the byte sum was one instruction: dropped.
- **The rotations cost more than the products.** With the Hadamards as 16-row `tl.dot` tiles, the decode MoE took
  16.9 ms per step for 48 layers against 2.6 ms for NVFP4 (the step between the two products 164 us, three programs);
  one program per 128-vector with the matrix built from bit parity brought it to 3.06 ms, and decode to NVFP4's speed
  with the same settings. In prefill the same rotations went from 14.5 to ~5.4 ms per 2048-token layer (one fp16 dot
  instead of two, swept tile sizes), and a tile sweep of the grouped GEMM gained 35-42 %.
- **Prefill, second pass.** 1024-token blocks fed each expert ~20 routes for 64-row tiles: 4096-token blocks with
  32-row tiles cut the MoE time per token by 36 %. The down GEMM then took the rotation out as its epilogue (a program
  owns a 128-column block; per-route bf16 rows, then the existing per-token sum): another -15 %. The same epilogue
  for gate|up plus the step between the products was slower (7.8 against 5.7 + 1.5 ms per 4096-token layer: two
  accumulators per program, half the programs) and was dropped. Loading the Hadamard matrix once per program and
  looping over the 128-blocks helps prefill (-25 % on the input rotation) but kept the matrix in registers in decode,
  where the rotation out also ran on 2 warps: 12 -> 165 us per call, the decode step lost a fifth before the
  kernel profile caught it; decode now loads the matrix per dot, on 4 warps.
- **Second checkpoint, two loader bugs.** The 4.05 bpw branch stores its n-gram table as one tensor (not shards) and
  its vision tower in a file the index does not list, so its codebook markers were missed and the tower would have
  decoded as 3inst: cosine ~0 against the bf16 originals, no error. Both fixed; three vision linears per branch now
  at cosine 0.9994-1.0 against the originals. exllamav3 zero-pads the vision MLP to 128-multiples: cut back at load.
- **The first agent test crashed.** Every check so far had used prompts up to 8.3k tokens; the first opencode turn
  (a ~21k-token prompt, prefilled in 16k chunks while the title request decodes) killed the 7900 XTX's rank on an
  illegal memory access, and about one cold start in six hung at a collective in the same place. The driver's
  `evicted_ms` showed 1-3 s of stopped queues during the first such turn after a start, never later, ~0 for NVFP4.
  The EXL3 kernels alone in a fresh process stopped nothing. Sampling VRAM every 50 ms found it: the first 16k chunk
  took the 7900 XTX to 24,539 of its 24,560 MiB, and the queue stops came with it; NVFP4 peaked at 23,243 MiB
  there. With the expert intermediate cut 384 / 256, the 7900 XTX is the card that bounds the shared cache size for
  EXL3, so it kept only 2.65 GiB free after the CUDA graphs (3.98 for NVFP4), and an EXL3 layer's prefill needs
  620 MiB of temporaries (measured; 2048-token blocks would bring them to 350 MiB for +26 % MoE time per token). The expert
  kernel now declares them and the cache planner keeps them free (5 % fewer cached experts). Then, 10 cold starts
  (7 on 3.05 bpw, 3 on 4.05), each with that first turn and two agents taking five turns: no stopped queue, no
  error, the 7900 XTX at 24,360 MiB at most with `--memory-ratio 0.80`, 24,010-24,149 with 0.78.
- **Where it stands.** [how-it-works.md](how-it-works.md#exl3-checkpoints), measured on every tested profile the same
  day: at 3.05 bpw, decode 83-107 tok/s up to 248k tokens on `xtx-xt` (NVFP4: 72-86), agent turns 0.9-1.6 s (1.3-2.0),
  59 GiB of RAM locked instead of 82, 9.9k experts cached per card instead of 7.1k; at 4.05 bpw, NVFP4's speed with 74
  GiB. On the agentic benchmark (one run each) 13 of 29 at 3.05 bpw and 12 at 4.05, where NVFP4 scored 12 and 18 in
  two runs.

## 17. Dropped along the way

| Idea | Why not |
|---|---|
| Predicting the next layer's experts to prefetch them | co-occurrence: 31 wasted copies per useful one; a next-layer router: 65 % recall, still 5 wasted per useful one and an extra router per layer |
| More router warps | upstream's single warp is fastest on RDNA3 |
| Sampled-LRU expert cache | estimated +0.3 %, not worth an A/B |
| A gated activation computed inside the GEMV that reads it (hyper-connection up, expert down projections) | exact, but every output tile recomputes it: HC up 7.9 -> 14.2 µs a call on the XTX (8.8 -> 16.4 on the XT), routed NVFP4 down 10.8 -> 12.7 (13.1 -> 19.7), shared-expert int8 down 5.5 -> 4.6 at one row but 5.9 -> 6.3 at two (2026-10-06, graph replay); added to the kept fusions of section 19 on the server: 16.07 against 15.45 ms a step |
| Interleaving decode steps between prefill chunks | hurts the short prefills of agent turns |
| Even split of the GDN heads under the uneven split | -2.5 % decode |
| Grouped int8, int8 KV | precision (sections 6 and 8) |
| Speculative decoding (MTP), first estimate | acceptance ~0.6 on real traffic, about break-even; measured since (section 15) |
| Verifying drafts for several requests at once | 2 requests: 60.5 ms a step for 8 rows against 23.7 ms for one row each; 4 requests: 133 against 41.5 ms |
| Online TunableOp | retunes every new prompt length |
| TunableOp for the full 16k prefill chunks (ROCm 10) | the dense GEMMs gain 1.5-1.7 % (some shapes lose) and are 5-10 % of a chunk (~0.85 s of GEMMs at 16k rows on rank 0): < 0.2 % on prompt reading |
| Decoding the NVFP4 codes with integer ops instead of the 16-entry table | exact, but 0.59-0.89x: the expert GEMV is sensitive to every extra instruction (at 4 rows it already reads 764 GB/s, 80 % of the XTX's peak) |
| Missing experts copied by the expert GEMV itself (the route reads its host bank and fills its slot) | exact, but slower: the GEMV's tiles read PCIe at ~15-19 GB/s against 28 for the copy kernel (64-byte rows for `down`), and no other tile is bit-exact; per layer 1 miss 126 against 99 µs, 3 misses 339 against 207 |
| HIP graph segment scheduling (`DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1`) | a captured graph's independent branches run one after the other by default; with it they overlap a little in a microbenchmark, not measurably on the server |
| Thread trace (ATT) of the expert GEMV under ROCm 10 | the image's profiler could not load its aqlprofile library, and with it on the loader path the traced process spun for 20 min without a trace |
| EXL3: the codebook's per-weight fp16 rounding replaced by an affine form | gained nothing once the byte sum was one `v_dot4_u32_u8` |
| EXL3: the Hadamard rotations as 16-row `tl.dot` tiles in decode | 16.9 ms of decode MoE per step against 3.06 ms for one program per 128-vector |
| EXL3: gate\|up and the step between the products fused in one epilogue | 7.8 against 5.7 + 1.5 ms per 4096-token layer (two accumulators per program, half the programs) |
| EXL3: the Hadamard matrix loaded once per program in decode | it stayed in registers: the rotation out went from 12 to 165 us per call |
| Moving a verify step's PLE fill off its critical path | the ~2 ms wait seen under the profiler is probably its own doing (~30 ms steps under trace against ~16), and busy-waiting on the token readback changed nothing (70.9-85.5 against 71.0-85.5 tok/s) |
| A bigger tensor-parallel share for the XTX | in decode the XTX waits ~0.9 ms a step for the XT (the all-reduce wait log), but 0.575 and 0.6 overload it in prefill: 0.55 stays |
| Pinning the registered host memory from the engine (io_uring buffers) against the stall below | the pages stayed put, but compaction kept retrying around them and moved the runtime's own buffers: worse |
| A larger VRAM margin (`--memory-ratio 0.77`) against the prefill out-of-memory retries | the allocator's cache grew into it: the same retry at the same chunk (expandable segments removed them) |

The stall seen since 2026-09-28 (the GPU stopping for seconds mid-step, in bursts) was found on 2026-10-03: the
kernel's memory compaction moving host pages the GPU driver maps without pinning, which stops every GPU queue of the
process for 5-22 s. Turning off proactive compaction was not enough (background reclaim still woke the compaction
daemon: three stops in nine minutes, once 140 s and a dead server); the engine now locks its memory and
`vm.compact_unevictable_allowed=0` keeps compaction off it (a forced compaction during decoding: no stop). The prefill
out-of-memory retries at 14-16k-token chunks past ~26k of context (seen on the ROCm 10 image) went away with
expandable segments. After a pause, the VRAM clock stayed at 96 MHz for ~10 s under the default power profile. The
host settings: [troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run).

## 18. ROCm 10's own Triton (2026-10-05)

- **The detour.** On the ROCm 10 image its Triton 3.8 crashed the first prefill (an illegal memory access in an expert
  GEMM, then a hang), so the image carried an older base's Triton 3.7.1, a second 29 GB image pulled for one package.
  PyPI's Triton 3.7.1 was no way out: its library exports its LLVM symbols, which clash with the LLVM torch has already
  loaded (a segfault at `import triton` after `import torch`).
- **Not a miscompile.** The GEMM ran fine on random inputs, and its inputs captured from the crashing run crashed
  both Tritons: they came from `moe_align_block_size`, which sorts the routes by expert and had left its expert ids
  unwritten. Triton 3.8's AMD integer range analysis gives `tl.histogram`'s counts the range [0, -1] (the unsigned
  maximum read as signed; Triton's main branch has the fix), and the pass that folds always-true comparisons turned
  `j < nblk` and `h > 0` into `false`. Four lines show it: `tl.where(h > 0, h, -1)` over a histogram gives -1
  everywhere on 3.8 and the counts on 3.7.1.
- **The fix.** `moe_align` no longer compares histogram counts: the small path reads its block counts back from the
  scratch it already writes, the large one adds the zero bins too. The same integers on both Tritons;
  `tests/kernels/test_moe_align.py` fails on 3.8 without it. The sampler and the QSA top-k also call `tl.histogram`
  but compare only sums and prefix sums of it, which the analysis does not bound: the GPU checks, the sampler tests
  and the pytest suite run on the GPUs fail nothing on 3.8 that passes on 3.7.1.
- **Measured** (`xtx-xt`, the same code on both Tritons, one server per run, 3.8 / 3.7.1 / 3.8): decode within 1 %
  (greedy, 512 tokens: NVFP4 83.8-83.9 against 83.3-83.4 tok/s with the MTP head, 58.0-58.6 against 57.9-58.4
  without; EXL3 3.05 bpw 86.8-87.2 against 87.4-87.7 and 58.3-58.8 against 58.8-59.4), prompt reading ~5 % faster
  (a cold 8.3k-token read in 4.00-4.02 s against 4.20-4.22 s for NVFP4, 3.80-3.83 against 3.99-4.02 s for EXL3),
  agent turns the same. Greedy answers are identical from one 3.8 server to the next (11 of 11) but not to 3.7.1 (3
  to 5 of 11; the texts part at a word and stay coherent): another compiler rounds differently. Not yet run on the
  agentic benchmark. The two-card tables of [benchmarks.md](benchmarks.md) were measured again on it (2026-10-06):
  decode the same to +3 %, prompts read 3-6 % faster, from 9k to 248k tokens.

## 19. Fewer decode kernels (2026-10-06)

- **What a short kernel costs.** An eager profile of an NVFP4 decode step on `xtx-xt` (no MTP head) runs ~1770
  kernels, ~950 of them under 6 µs: ~3.2 ms of a 16 ms step. An earlier look counted only the gaps between the kernels
  of a captured step (~0.26 ms) and concluded fusing would not pay; but each short kernel's own time counts too. A test
  build that skipped 144 of them (wrong answers, timing only) saved 0.73 ms a step, ~5 µs per kernel once its extra
  expert misses were discounted.
- **Kept, bit-exact:** each fused kernel does the arithmetic of the kernels it replaces in the same order, on the same
  tile and warps (so the same sums); GPU tests compare the bits on both cards
  (`tests/kernels/test_moe_shared_gate.py`, `test_hc_combine_rmsnorm.py`, `test_gemv_bf16_rows.py`).
  - The MoE epilogue (`FREETOKEN_FUSED_MOE_EPILOGUE`): the routed sum, the shared expert's gate and the mul-add in one
    kernel, two launches fewer per layer.
  - The hyper-connection combine with the grouped RMSNorm that reads the streams next (`FREETOKEN_FUSED_HC_NORM`),
    within a layer, across layers and into the final mixer, two launches fewer per layer. A first version kept the
    normed streams alive through each block: 320 MiB more at a 16k-token prefill chunk, enough to bring the XT to its
    last 28 MiB with the MTP head, and the server hung at the first long prompt, three times out of three. They are now
    freed once read: XT peak 20213 MiB against 20193 without the fusion.
  - The bf16 split-K GEMV reduces every row of a verify step in one launch instead of one per row.
- **Measured** on `xtx-xt` (greedy, 512 tokens, median of four prompts, one server per build; 11 greedy prompts, up to
  34k tokens, identical between the builds):

  | Checkpoint | Without | With | With the MTP head: without | with |
  |---|---:|---:|---:|---:|
  | NVFP4 | 57.8 | 59.8-60.0 | 82.9 | 85.6 |
  | EXL3 3.05 bpw | 58.4 | 59.7 | 86.4 | 89.2 |
  | EXL3 4.05 bpw | 55.3 | 56.5 | 81.6 | 83.3 |

  GPU time of an NVFP4 step: 16.05 -> 15.45 ms, 29.68 -> 28.56 ms with the head; prompt reading the same (~1960 tok/s
  NVFP4, ~2150 EXL3). A depth sweep to 248k with the head: decode +2-3 % at every depth (71.9-90.4 against
  70.4-88.4 tok/s), prompt reading and agent turns the same, no new eviction. The image built with the fusions
  against the one before, sessions interleaved (A B A B, greedy, with the head): NVFP4 83.0 / 83.2 -> 85.6 / 85.7 tok/s,
  EXL3 4.05 bpw 81.5 -> 83.4, the same answers.

## 20. ROCm 10.1, and fewer copies (2026-10-07)

- **ROCm 10.1** (released 2026-10-05: HIP 7.16, RCCL 2.30.7, PyTorch 2.12-2.14 images; its notes list nothing for
  gfx1100). The image builds on `rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.13.0` with two changes: that base has
  no virtual environment (the system Python, `python3` only) and no git. Against the ROCm 10.0 image, the same code,
  sessions interleaved (`xtx-xt`, MTP head, greedy, 12 prompts of 512 tokens):
  - NVFP4: the same answers (12 of 12); decode 0.5 % slower on every prompt (83.0 against 83.5 tok/s); prompts read
    1.5-2 % slower (a 6.2k-token prompt 1895 against 1937 tok/s, 25k 1940 against 1969).
  - EXL3: other answers (0 of 12 the same; each build agrees with itself). At load the dense EXL3 weights are rebuilt
    with two fp32 Hadamard products in torch (`models/exl3_weights.py`), and 10.1's BLAS picks another kernel for the
    batched left one: 45 % of its fp32 outputs differ, 0.02-0.03 % of the bf16 weights by one unit in the last place.
    The Triton dequantization and the bf16 products give the same bits. Prompts 1.5-2 % slower, as with NVFP4.
  - Its Triton (3.8.0+git669b31ac) still has the `tl.histogram` range bug of section 18.
  - Adopted the next morning anyway (the newer stack is the one to be on): each component measured alone in both
    images on the XT gave the same time on 10.1 (the bf16 prefill GEMMs at 4.7k-16k rows with either BLAS, the NVFP4
    prefill MoE GEMMs, the QSA prefill attention, RCCL's all-reduce of a 16k chunk, kernel launches, pinned host-to-GPU
    copies), so the 1.5-2 % on prompts could not be pinned on one; tonight's +2.3-2.9 % of decode more than covers it.
    The EXL3 rebuild runs its two products in fp64 (`FREETOKEN_EXL3_RECON_FP64`, on): the bf16 weights are then the
    same on 10.0 and 10.1 (0 of 39M differ in a check), so are the greedy answers (12 of 12), at the cost of a one-time
    change from the fp32 rebuild. No BLAS setting reproduces 10.0's fp32 product on 10.1 (hipBLASLt changed its
    batched kernel; rocBLAS sums in a third order).
  - Profiling on 10.1 (rocprofv3 `--kernel-trace`, torch.profiler) hung RCCL's all-reduce of a prefill chunk on two
    cards, twice out of two (60 s watchdog); the same load without a profiler ran normally, and 10.0 profiled fine.
  - With PyTorch 2.14 (offered with 10.1; it needs C++20, the C++ extensions now build with `-std=c++20`): greedy
    answers identical to PyTorch 2.13 on 10.1 (NVFP4 and EXL3, 12 of 12), the same decode and prompt reading
    (NVFP4 84.9 tok/s, a 6.2k-token prompt at 1903 against 1896 tok/s). The image uses it.
- **Fewer copies.** A profile tied to the lines that launch each kernel (`FT_PROF_STACK=1`) showed ~110 small copies
  and elementwise kernels per verify step of the MTP head (q, the output gate, k, v, the GDN gate z, the indexer's q and
  k made contiguous; the gate's sigmoid and product) and the router's 48 split-K reduces. The norms and the indexer
  kernels now read the projections' outputs through their strides, the gate is applied in one kernel, and the router
  GEMV's last program sums the partials (`FREETOKEN_FEWER_COPIES`, `FREETOKEN_GEMV_LAST_REDUCES`): the same
  arithmetic, the same bits. In a graph, an attention block's part went from 22.3 to 11.8 µs at one row and from 26.0
  to 11.9 at four. On the server (interleaved, the same 12 answers): NVFP4 83.2 -> 85.1-85.2 tok/s, EXL3 3.05 bpw
  99.1 -> 101.9-102.0, the GPU time of the same decode steps 2.6-3.3 % lower; prompt reading unchanged, VRAM peaks
  ~100 MiB lower.
- **Where a verify step goes** (NVFP4, `xtx-xt`, MTP head, four rows; a kernel trace of captured steps with
  `rocprofv3 --kernel-trace`, which sees graph replays: started with `-P` after the load, the rank processes stopped
  with SIGTERM so they write their trace; the packaged `rocprofv3 --attach` lacks its helper on 10.0 and 10.1): ~2000
  kernels per step; the copies of the missing experts 30-35 % of the kernel time (rank 0 11.9 ms, rank 1 9.7 ms, under
  the profiler), the int8 GEMVs ~7.5 ms, the NVFP4 expert GEMVs 3.9 ms, the host all-reduce 2.2-2.9 ms, the kernels
  under 6 µs ~1000 per step for 2.9 ms. Without the profiler (`FT_MOE_STATS`, a long greedy answer): 20-25 experts per
  layer per step, 1.5-4 of them missing (7-17 %), and each extra miss per layer adds ~3.5 ms to a step (23 ms at 1.5,
  31.6 ms at 4). Four rows route to ~2.4x the experts of one, so misses weigh far more with the MTP head than the
  ~1 ms per token measured without it.
- **Expert-cache policies, replayed offline** on a recorded routing sequence (67,828 MoE calls of three answers,
  6979 slots as served): the engine's LRU 3.67 misses per call (15.3 %); a decaying-frequency policy -8 to -9 % with a
  12-16 step half-life but -1 % at 24 steps and +13 % at 32; segmented LRU (2Q) -1 %; the offline optimum (Belady)
  -56 %. Not implemented: ~2 % of decode at best, and tuned to one trace. What would move misses is more cache, smaller
  experts (EXL3 3.05 bpw moves ~2/3 of NVFP4's bytes per miss: 3.05 against 4.5 bits per weight) or copies that
  overlap compute.
- **KFD evictions at start.** The ~1 s per GPU queue that `evicted_ms` shows after a start (4.5 s summed over the
  four queues, EXL3 3.05 bpw) all happen 10-23 s into the weight load, while the expert banks are registered with the
  GPU, and not once while serving (unchanged through a whole agent run and depth sweep). Locking the process memory
  before the load instead of after it did not remove them (6.5 s).
- **Dropped:** the router reading the GEMV's fp32 partial sums itself. The experts chosen were the same, their
  weights not always: the partial sums load 4 values per lane where the bf16 logits load 8, Triton lays the tile out
  differently, and the softmax adds its terms in another order.
- **Method.** `--moe-cache-auto` sizes the expert cache from the VRAM free at start: a GPU test run on a card while
  another server loaded shrank that server's cache (6732 against 6979 slots) and its speed. And a source tree mounted
  into a server must not change during a series (two sessions picked up an unfinished change).

## 21. Prefill peaks, spills and the VRAM outside torch (2026-10-07)

- **Why.** Expert misses are a third of a verify step (section 20) and the cache is what VRAM leaves: `--memory-ratio`
  0.80 -> 0.83 on `xtx-xt` gives 7487 slots instead of 6977 and +3.8 % decode (85.1 -> 88.3 tok/s, the same answers).
  The first worst-case run at 0.83 (two agents growing to ~150k tokens, one's prefill chunks between the other's decode
  steps) ended in a `[gfxhub] page fault` on the 7900 XT, the card at 20420 of 20464 MiB, like the earlier faults and
  the hung all-gather of [troubleshooting.md](troubleshooting.md#a-gpu-fault-or-a-hung-collective-with-a-full-card).
- **The cap.** `FREETOKEN_VRAM_MARGIN_MIB` (512 on ROCm) caps the torch allocator once the engine is built, at the
  card's VRAM minus what lives outside torch then minus 512 MiB. The same run at 0.83 then stopped on a plain
  out-of-memory error instead of a fault, and the error named the peak: the MTP head's pass over a 16k-token prefill
  chunk.
- **Prefill peaks, the same bits.** At a 16k-token chunk each `[T, 5 x hidden]` tensor of the hyper-connection streams
  is 320 MiB:
  - the hc combines write over their input (`FREETOKEN_HC_INPLACE`): -312 MiB on the XT;
  - the GDN delta rule runs 4096 tokens of a request at a time (`FREETOKEN_GDN_PREFILL_BLOCK`): -289 to -327 MiB on
    the XT, -893 on the XTX;
  - the PLE output is added to the streams block by block for one request;
  - the MTP head frees the prompt's streams and their normed copy once read, and adds the embedding in place (two of
    five such tensors at its pass).

  At 0.83 the XT's allocated peak over the worst-case run is 18545 MiB, against a cap of 18989.
- **Outside torch**: ~1 GiB per card at start (962 MiB on the XT), +200-250 MiB while serving:
  - each Triton kernel module the HIP runtime loads takes 2 MiB of VRAM (20 trivial kernels: +40 MiB), whatever its
    size; no runtime setting changes it;
  - the scratch of kernels that spill registers, per GPU queue (the QSA prefill attention: 62 MiB, a 16k-token GDN
    prefill: 101 MiB);
  - RCCL: ~8 MiB (no P2P between these two cards, it goes through the host).
- **Spills.** The QSA sparse attention's prefill profile (tuned on GB300) runs 2 warps: on RDNA3 at head_dim 256 that
  is ~2.2 KB of scratch per lane. At 4 warps (`FREETOKEN_QSA_MIN_WARPS`) it spills nothing and gives the same bits (8
  shapes of 128-16k rows), 39-45 % faster. Two GDN prefill kernels spilled 484 and 796 B per lane at 4 warps; at 8
  (`FREETOKEN_GDN_PREFILL_WARPS`) the same bits, a 16k-token GDN prefill 8.56 -> 5.43 ms. Server, sessions at 0.83:
  prompt reading +2.6 to +3.5 % at each of 10 depths from 8.8k to 248k tokens, decode unchanged.
- **The copies of the missing experts already run at the bus's speed** (both cards on PCIe 4.0 x16 to the CPU): the
  shader copy from pinned RAM moves 26-28 GB/s from two experts on (20 GB/s for one), whatever its grid (8 to 64
  blocks of 256-1024 threads per bank); DMA copies, one `hipMemcpyAsync` per expert and bank, 15.6 GB/s. Only fewer
  misses (a bigger cache) or smaller experts make them cheaper.
- **The desktop decides, so the ratio stays 0.80.** The 7900 XTX drives the display, and the desktop's buffers on it do
  not show in the free memory HIP reports (sysfs `mem_info_vram_used` counts them; the cap reads it): 0.66 GiB at
  noon, 1.4-1.8 GiB later the same day with the desktop in use. The runs above had the light desktop. Through the
  launcher with the desktop in use:
  - 0.83 (7487 slots): the worst-case workload clean, but the XTX at most 107 MiB from full and the XT 62 MiB (a first
    start after a new image loads every autotune candidate's kernel module);
  - 0.81 (7147 slots, decode +1-2 %): the XTX 273 MiB from full;
  - 0.80 (6977 slots, as before): both cards ~970 MiB from full at worst (the desktop at 1.43 GiB), the same 12
    greedy answers, the workload clean.

  Each 0.01 of ratio is ~170 expert slots, ~260 MiB on the XTX and ~210 on the XT. Without today's reductions (~0.9
  GiB on the XTX), 0.80 would have left the XTX within ~0.1 GiB of full with a desktop that size, which fits the faults
  seen on it. The profiles keep 0.80, and the reductions are headroom. EXL3 3.05 bpw splits its experts 384 / 256, so
  the XTX bounds it: at 0.83 it came within 56 MiB of full with the light desktop; it keeps 0.77, where the XTX stayed
  531 MiB from full with the desktop at 1.5 GiB (the workload clean, the same answers).

## 22. Still open

- In NVFP4 the model needs ~63 GiB of RAM for its experts alone (the engine keeps every expert in RAM). The EXL3 3.05
  bpw checkpoint needs 42.6 GiB and served under a 56 GiB container limit, but a real 64 GB machine is untested
  ([limits.md](limits.md)).
- Triton 3.8 is not bit-exact with the 3.7.1 the agentic benchmark last ran on (section 18): not run on it yet.
- ROCm 10 / PyTorch 2.13 and ROCm 10.1 / PyTorch 2.14 (rechecked 2026-10-10), the image's bases since 2026-10-02:
  `F.linear` changes its sums with the row count for some shapes ((3072, 256), (6656, 2560)): serving does not use it
  for them (int8 or table GEMVs), but `test_qsa_spec_rows`' toy projections did, which made it fail on 3-4 rows;
  computed a row at a time they pass (the test now does so).
- A rare GPU page fault (`[gfxhub] page fault`, permission fault, `sq_intr` errors): those while serving came with a
  card within tens of MiB of full (section 21; the torch allocator is now capped 512 MiB short of that). One at start
  on 2026-10-07, ~3 s after the CUDA graphs were captured, with 3.9 GiB free on the XTX (EXL3 3.05 bpw; 1 start in
  ~32 that night, the 16 starts of a series after it were clean): cause not found.
- Expert misses are 30-35 % of a decode step with the MTP head (section 20): the next lever for decode, not a
  cache policy (an online one gains ~8 % of the misses at best on a recorded trace). More VRAM for the cache: ~170
  loaded Triton kernel modules at 2 MiB each per card (section 21), and a desktop-aware plan for the card that drives
  the display.
- EXL3's dense weights are rebuilt at load in fp64 since 2026-10-07 (section 20): the same bits on ROCm 10.0 and 10.1,
  a one-time change from the fp32 rebuild that the agentic benchmark has not run yet. The EXL3 profiles keep
  `--memory-ratio` 0.77 / 0.76 on two cards (3.05 bpw at 0.77 measured in section 21; 4.05 bpw not again).
- More than two GPUs, other RDNA3 cards, RDNA4 and NVIDIA are untested.
- Image input (`serve.sh --vision`) is measured on `xtx-xt`, `xtx` and `xt` only. The derived profiles (`xtx-xtx`,
  `xt-xt`, `gre`) are untested with it, as they are without it.
