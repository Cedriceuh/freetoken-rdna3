# The journey

How this build came about: measured experiments from 2026-09-25 to 2026-10-03 on one machine, an RX 7900
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
| Fusing the hyper-connection activation into the next GEMV | exact but twice as slow |
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
| Fusing kernels to save launches | the kernels of a captured step overlap (median gap -2.6 µs); the real bubbles are ~0.26 ms a step |
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
out-of-memory retries at 14-16k-token chunks past ~26k of context (seen on the ROCm 10 image; 7.14 was not checked) went away with
expandable segments. After a pause, the VRAM clock stayed at 96 MHz for ~10 s under the default power profile. The
host settings: [troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run).

## 18. Still open

- In NVFP4 the model needs ~63 GiB of RAM for its experts alone (the engine keeps every expert in RAM). The EXL3 3.05
  bpw checkpoint needs 42.6 GiB and served under a 56 GiB container limit, but a real 64 GB machine is untested
  ([limits.md](limits.md)).
- ROCm 10 / PyTorch 2.13, the image's base since 2026-10-02: its Triton 3.8 miscompiles these kernels, so the image
  carries the 7.14 image's Triton 3.7.1. On two cards decode and prompt reading match or beat 7.14, and the PLE
  wait-sync (`FREETOKEN_PLE_SYNC`) adds 3.5-4 % decode; on one card the same depth sweep gives the same decode on both
  images (29.5 against 29.1 tok/s at 124k on `xtx`). Greedy answers differ from 7.14 on 2 of 5 prompts; with the MTP head it scored 12 and 18 of 29 on the agentic benchmark. Its `F.linear`
  changes its sums with the row count for some shapes ((3072, 256), (6656, 2560)): serving does not use it for them
  (int8 or table GEMVs), but `test_qsa_spec_rows`' toy projections did, which made it fail on 3-4 rows; computed a row
  at a time they pass (the test now does so).
- More than two GPUs, other RDNA3 cards, RDNA4 and NVIDIA are untested.
- Image input (`serve.sh --vision`) is measured on `xtx-xt`, `xtx` and `xt` only. The derived profiles (`xtx-xtx`,
  `xt-xt`, `gre`) are untested with it, as they are without it.
