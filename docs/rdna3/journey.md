# The journey

How this build came about: measured experiments from 2026-09-25 to 2026-10-19 on one machine, an RX 7900
XTX 24 GB + RX 7900 XT 20 GB (gfx1100, no GPU peer-to-peer), a Threadripper 3970X and 128 GB of DDR4, serving
Qwen3.8-Flash-Next NVFP4 to the author's coding agent (OpenCode) with a 262k-token context (250k from 2026-10-03 to 2026-10-16). Every number was measured
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
eviction fixed it. Result: 4 agents with 82-117k-token conversations, 3 turns each, rounds after the first reads:
**599 s -> 35 s** (without the RAM tier, 1.15M tokens recomputed), first token ~47 s -> ~2 s (1.9-2.3 s for each
turn); 4 agents x 12 short turns on a small pool: 133 s -> 94 s. Cost: 9.0 GiB of RAM locked.

## 11. Tested and not kept: grouping prefills

A prefill pass has a large fixed cost whatever its size, since it streams every expert (short ones no longer do since
section 27; in one test at a 20k-token
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
  In the runs that chose it (2026-10-03) one of six 3-request rounds took 10.1 s instead of ~8, unexplained; the one crash seen
  during the loads of 2026-10-03 and 04 (an illegal memory access while the weights loaded, once) happened before this
  setting is read.
- **One card (2026-10-03).** The first one-card runs used the two-card step costs, so mostly 4 rows, and the head made
  one card slower (`xtx` / `xt`): chat prompts 38.0 -> 34.6 / 31.2 -> 27.6 tok/s, agent turns on a 3.5k-token prompt
  34.9 / 38.5 / 32.4 -> 34.4 / 29.9 / 29.8 and 28.8 / 30.7 / 27.4 -> 26.8 / 23.8 / 23.6, the 108k-token prompt and its
  follow-up 35.1 / 35.1 -> 32.3 / 34.6 and 29.1 / 28.8 -> 26.1 / 27.9, one 1500-token answer 41.4 -> 39.6 and 33.4 ->
  31.2, the 60 code / math items 516 -> 495 s and 613 -> 591 s, 2.1-3.9 tokens kept per request step (512-step
  averages). At 2 rows the head pays on one card too:
  [benchmarks.md](benchmarks.md#speculative-decoding-mtp-head).
- **Bugs found on the way.** The detokenizer repeated text when a request got several tokens in one step; the LM head
  gathered prompt rows twice (a GPU memory fault). 4 rows were not bit-exact: the GDN output norm picked its rows per
  program from the row count (also for 4 requests decoding together), and the QSA attention its tile profile; decode
  now pins both to the one-row choice. A review after the merge (2026-10-03): within a few tokens of the context's end
  the verify rows and the chained drafts indexed past the page table (now capped by the tokens left), image pad ids
  reached the head's embedding unchecked at TP=1, and a verify step in flight could let a finishing request donate an
  over-advanced GDN state to the prefix cache.
- **Validated.** Greedy answers identical to plain decode on the 120-item precision set (657 -> 426 s), a 124k-token
  conversation, a 35k-token read and a 1500-token answer. Sampled requests use exact speculative sampling. Numbers:
  [benchmarks.md](benchmarks.md#speculative-decoding-mtp-head). The head's experts take cache room: on `xtx-xt`
  (2026-10-03, ROCm 10, the 250k context of then) 7,127 experts on each card with the head, 7,395 without, of 24,576.

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

- **`FREETOKEN_EXL3_PREFILL_BLOCK`**, swept on the XT (one layer, 8192 tokens): 1024-token blocks cost 5.4 ms per 1k
  tokens before the down GEMM took the rotation out as its epilogue, 4096-token blocks 3.4, then 2.8-3.0 with that
  epilogue; 8192 saved 5-7 % more before that epilogue, for 0.5 GiB more.

## 17. Dropped along the way

| Idea | Why not |
|---|---|
| Predicting the next layer's experts to prefetch them in decode | co-occurrence: 31 wasted copies per useful one; a next-layer router: 65 % recall, still 5 wasted per useful one and an extra router per layer (for a prefill's many tokens the same router predicts 79-84 %: kept there, section 27) |
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
| The missing experts' copy on a side stream ahead of the shared expert in decode (`FREETOKEN_MOE_COPY_OVERLAP`) | bit-exact, but measured end to end on 2026-10-03 (one request, agent turns at 77-99k of context, MTP head on): 47.5 tok/s decode against 71.1 without it (-33 %; 48.9 with `expandable_segments:False`) and cold reads 13 % longer |
| HIP graph segment scheduling (`DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=1`) | a captured graph's independent branches run one after the other by default; with it they overlap a little in a microbenchmark, not measurably on the server |
| Thread trace (ATT) of the expert GEMV under ROCm 10 | the image's profiler could not load its aqlprofile library, and with it on the loader path the traced process spun for 20 min without a trace |
| EXL3: the codebook's per-weight fp16 rounding replaced by an affine form | gained nothing once the byte sum was one `v_dot4_u32_u8` |
| EXL3: the Hadamard rotations as 16-row `tl.dot` tiles in decode | 16.9 ms of decode MoE per step against 3.06 ms for one program per 128-vector |
| EXL3: gate\|up and the step between the products fused in one epilogue | 7.8 against 5.7 + 1.5 ms per 4096-token layer (two accumulators per program, half the programs) |
| EXL3: the Hadamard matrix loaded once per program in decode | it stayed in registers: the rotation out went from 12 to 165 us per call |
| Moving a verify step's PLE fill off its critical path | the ~2 ms wait seen under the profiler is probably its own doing (~30 ms steps under trace against ~16), and busy-waiting on the token readback changed nothing (70.9-85.5 against 71.0-85.5 tok/s) |
| A bigger tensor-parallel share for the XTX | in decode the XTX waits ~0.9 ms a step for the XT (the all-reduce wait log), but 0.575 and 0.6 overload it in prefill: 0.55 stays |
| Pinning the registered host memory from the engine (io_uring buffers) against the stall below | the pages stayed put, but compaction kept retrying around them and moved the runtime's own buffers: worse |
| The experts a short prefill's prediction missed copied by the copy engine while the GEMMs run the resident experts' blocks (two passes per GEMM, the same bits) | ~50 ms of copies a turn on the XTX in a kernel trace, but measured untraced: +0 to +2 % (2026-10-14) |
| A larger VRAM margin (`--memory-ratio 0.77`) against the prefill out-of-memory retries | the allocator's cache grew into it: the same retry at the same chunk (expandable segments removed them) |

The stall seen since 2026-09-28 (the GPU stopping for seconds mid-step, in bursts) was found on 2026-10-03: the
kernel's memory compaction moving host pages the GPU driver maps without pinning, which stops every GPU queue of the
process for 5-22 s. Turning off proactive compaction was not enough (background reclaim still woke the compaction
daemon: three stops in nine minutes, once 140 s and a dead server); the engine now locks its memory and
`vm.compact_unevictable_allowed=0` keeps compaction off it (a forced compaction during decoding: no stop). An endurance
run the next day (2026-10-04, `xtx-xt` as shipped then, `FREETOKEN_SPEC_BS_MAX=1`): 2 h 07 of rounds of agent turns
(one conversation, then two at once, 6 turns each, then 4 short requests and `quick_bench.py`; 90 conversations at
70-121k tokens, 540 turns), no GPU queue stop, no out-of-memory retry, no error, memory and VRAM flat after the first
round; one conversation decoded at 69.3 tok/s median over the 30 rounds (65.9-76.9), turns of two at once at 31.3
each (204 turns timed from round 12 on), its cold read (68-93k tokens) at 1860 tok/s, 4 short requests took 10.1-10.3 s. The prefill
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

## 22. Four conversations of 262k, or one of 1M: the K/V in host RAM (2026-10-10)

Qwen3.8-Flash-Next is trained to 262,144 positions and Qwen's model card extends it to 1M with YaRN. The attention K/V
costs 14,144 bytes per token on each card, so 1M tokens is 13.2 GiB per card: the plan refuses it on `xtx-xt`.
`FREETOKEN_QSA_KV_HOST=1` keeps the K/V in pinned, device-mapped host RAM that torch sees as a GPU tensor: decode and
verify read their ~2051 selected tokens of each row across PCIe in place, and a prefill first copies one layer's pages
at a time to a VRAM buffer (reading across PCIe instead ran at 789 tok/s at 118k). The answers are bit-identical to the
K/V in VRAM. Past 262,144 positions, YaRN now composes with the multi-axis rope (bit-identical to HF transformers'), or
`FREETOKEN_ROPE_MAX_POSITION` lengthens the plain rope's table. Two profiles, measured with EXL3 3.05 bpw:
`xtx-xt-1m` read 988k tokens at 1775 tok/s and decodes at 83.6 tok/s there, with ~75 GiB of RAM locked;
`xtx-xt-4x262k` takes four conversations of 262,144 tokens at once: four cold reads, 1,027,138 tokens in 463 s (2218
tok/s), then 29.4-30.1 tok/s each with the four decoding at 257-260k, 91.3 alone at 261k. Retrieval of planted values
100 % from 29k to 988k. Details: [long-context.md](long-context.md).

The first cold reads (stdlib code, then a 700-token greedy answer; PP / TG tok/s), before the decode changes below:
`xtx-xt` at 250k with the K/V in VRAM 2268 / 93.2 at 29k, 2185 / 90.6 at 118k, 2128 / 94.1 at 237k; `xtx-xt-1m` with
the K/V on the host 2376 / 84.3 at 118k, 2293 / 86.6 at 237k (YaRN), 2048 / 85.8 at 498k, 1891 / 75.1 at 699k, 1711 /
72.1 at 988k (9.6 min to read).

Decode with the K/V on the host, measured in the afternoon of 2026-10-10:

- **Coherent host memory.** One QSA decode attention layer, 4 verify rows of one request (2051 selected tokens each,
  90 % shared), on the 7900 XT: 122 us with the K/V in VRAM, 313 us from host memory allocated `Portable | Mapped` (8
  rows: 565 us; 1 row: the same as VRAM). A mapped allocation is coherent, so the GPU's L2 does not keep its lines and
  each row crosses PCIe again. Allocated `hipHostMallocNonCoherent` (now the default on ROCm): 129 us (8 rows: 220 us),
  the same bits. In the server it gave less than the kernel time suggests: decode 84.2 tok/s at 118k (84.3 before),
  88.5 at 237k, 87.9 at 498k (85.8 before); the MoE weights going through the same L2 probably evict the K/V lines
  between rows (not measured).
- **The indexer's top-k on a 1M-token table** fell back to one program per row: its split path held at most 16 chunks
  of 8192 columns, and the page table is 250,000 block columns wide. 504 us per decode layer at 988k visible tokens
  against 77-83 us split (4 rows, the same winners, ties included). The split first took more chunks past 16 (tables
  up to 131,072 columns, contexts up to 524,288 tokens, so the 262k profiles, kept their plan) and merged them in one
  spilled tile; since the merge of upstream `ee9789f` (section 23) it runs upstream's reduction
  levels (`5031e3d`): at 247,000 visible columns 53 / 64 / 96 us for 1 / 4 / 8 rows against 65 / 75 / 101 with the
  spilled tile, 4-7 us more at 30,000 visible columns, the same as before on the 65,536-column tables of the 262k
  profiles. Below ~100k visible tokens the wide table's split costs ~40 us per layer more than the one-program scan
  did.
- **With both changes** (the spilled tile): 498k read at 1973 tok/s (first request after a start), decode 91.9 tok/s
  (85.8 before); 988k read at 1775 tok/s in 9.3 min (1711 in 9.6 min), decode 83.6 tok/s (72.1); the same text as
  before both changes. With upstream's levels: the same text again at 498k (read 2014 tok/s, decode 93.0) and 988k (1780, 84.7), XTX peak 22.76 GiB.
- **Tried and dropped: the first 262,144 tokens' pages also in VRAM** (decode reads them there, the allocator hands
  them out first). Bit-identical, but those 3.5 GB per card leave 7,400 expert slots: decode 84.4 tok/s at 237k, below
  the host K/V alone (88.5). The prefix cache also keeps the low pages of finished conversations.
- **Measurement traps** of the quality runs: teacher-forcing a text sent as a chat *user* turn measures mostly the
  probability of `<|im_end|>` (~94 % at every position); the stdlib is memorized (raw NLL 0.06, top-1 98.5 %), so it
  only serves retrieval with planted values.
- **YaRN or the plain rope past 262,144.** YaRN is the model card's setting (`rope_type: yarn`, `factor: 4`,
  `original_max_position_embeddings: 262144`); upstream's `--hf-overrides` sets it without editing `config.json`, the
  section replaced whole: `{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
  "original_max_position_embeddings": 262144, "rope_theta": 10000000, "partial_rotary_factor": 0.25,
  "mrope_interleaved": true, "mrope_section": [11, 11, 10]}}}` (EXL3 3.05 at 497k: the same text as the `config.json`
  edit, 12/12, read at 2036 tok/s, decode 85; the config it builds is checked on the CPU by
  `tests/models/qwen4_exp/test_config.py`). Its table covers `original x factor` positions, bit-identical to HF
  transformers 5.16.1's rotary embedding, within two float32 ulps of 5.18's (`tests/layers/test_rotary_yarn.py`).
  Retrieval 100 % from 29k to 988k with both. Harder set (30 code questions, 6 notes, 3 two-hop notes): YaRN 19/19 at
  237k, 25/25 at 498k, 33/36 at 988k; the plain rope 32/36 at 988k; the two-hop questions fail at 988k with both (the
  answer is the pointer note). Short texts (12 texts of 2.6-5.2k tokens of this repository, teacher-forced, raw
  completions): next-token NLL 1.1298 without YaRN, 1.1339 with factor 4 (+0.4 %), 1.1307 with factor 2; KL to no-YaRN
  0.032 and 0.018, top-1 agreement 93.4 % and 95.1 %. A 988k-token text of this repository: the plain rope's NLL
  0.46-0.79 nats per token at every depth, YaRN factor 4 0.2-1.3 % worse up to ~650k and 1-3 % better beyond. Kept:
  the plain rope (`xtx-xt-1m`), the same bits as without it below 262,144.
- **What the host K/V and the wide table cost** (decode tok/s at 118k, EXL3 3.05): K/V in VRAM, 262k table (`xtx-xt`
  at 250k of context, 9,932 expert slots) 90.6; K/V on the host, 262k table (12,667 slots) 86.6; K/V on the host, 1M
  table (11,261 slots) 84.2.
- **Two cold requests at once** (498k + 237k): both 12/12 with the same answers as alone; the 7900 XTX peaked at 23.09
  of 23.98 GiB (sysfs, desktop included), the 7900 XT at 17.44 of 19.98. The 498k request decoded at 6.3 tok/s while
  the other one was read (prefill goes first, upstream's policy).

NVFP4 on `xtx-xt-4x262k`, 2026-10-15: the four-context pool locked 100.3 GiB and, with benchmark tools running beside
it, ran the 126 GiB host out of memory (a global OOM). With three contexts (`--kv-reserve-tokens 787500`) it locked
94.4 GiB, left 16 GB free and served 751 requests cleanly; that became the reference machine's NVFP4 launcher entry,
and the ~30 GiB `rdna3/serve.sh --check` keeps for the system sits between the two.

## 23. Upstream up to ee9789f (2026-10-10)

Upstream's top-k and YaRN replaced this tree's, and `--api-key` and `--hf-overrides` came with the merge
([changes-from-upstream.md](changes-from-upstream.md)). The NVFP4 decode tiles, swept on the previous kernel, ran with
layout conversions on upstream's rewritten marlin decode kernel: swept again, the down GEMV takes one trip per tile,
every entry bit-identical to the tiles shipped before. That kernel itself is not bit-exact with the previous one:
greedy answers changed on 5 of 15 prompts, all still right. Swept for it (`7644984`, merged with `ee9789f`), it takes
8-16 % less time per GEMV at 1 request than the previous kernel with its tuned tiles; decode with the MTP head on
`xtx-xt` 76.4 tok/s against 73.8 (12 sampled answers).

## 24. Expert misses on the CPU: `--hybrid` (2026-10-11)

`rdna3/serve.sh --hybrid` runs upstream's `--moe-strategy hybrid`: part of each decode step's expert misses is computed
by the CPU from RAM instead of crossing PCIe. It could not run as shipped on ROCm. The executor now signals with HIP's
stream memops, upstream's own since `2010271` (it fell back to two host callbacks per layer), sizes its buffers for the
MTP verify rows, runs under tensor parallelism with NVFP4 (each rank's pool on half the cores) and reads EXL3 experts
(AVX2). Opt-in: greedy answers differ from the offload path's at rounding level, the CPU summing in fp32 in another
order (measured as precise as the GPU, section 27). Decode, one request, re-measured on 2026-10-13: NVFP4 `xtx` 42.7 ->
62.3 tok/s (+46 %), `xt` 34.6 -> 52.1 (+51 %), `xtx-xt` 82.9 -> 84.5 and 25.2 -> 28.9 each with four at once; EXL3 3.05
bpw `xtx` 54.8 -> 57.8 (+5 %), `xtx-xt` even. Details and the settings tried without gain:
[cpu-experts.md](cpu-experts.md).

How it got there. A first try on 2026-10-01 fell back to two `hipLaunchHostFunc` callbacks per MoE layer (~1-1.5 ms
each) and decoded at 5.9-8.3 tok/s against 55-56 on `xtx-xt`; the idea was dropped on that number then. The first working version was slower than offload for one request on two cards (NVFP4 `xtx-xt` 71.0
tok/s against 78.4), so `FREETOKEN_HYBRID_MIN_REQS` kept such steps on the offload path; the hybrid layer's kernels
took it to 74.3, the rest of the changes to 78.3, and `serve.sh --hybrid` stopped setting it. Measured on 2026-10-11
(NVFP4, 12 sampled answers, MTP head on): `xtx` 39.4 -> 56.2 (+43 %), `xt` 31.4 -> 46.6 (+48 %), `xtx-xt` one request
78.2 -> 81.2-82.7 (+4-6 %), four at once 24.4 -> 28.2-28.6 each (87-96 -> 107-112 in total). The one-request gain on
`xtx-xt` is a mean over 12 answers (one run's answers spread over +-13 tok/s), about two standard errors; a second run
at 40 % gave the same answers and speeds, a third through `rdna3/serve.sh xtx-xt --hybrid` after a 118k-token prompt
(9/9, 61 s) other answers at 81.2 tok/s. The decode kernels of section 25 then sped up both columns, hence the
2026-10-13 numbers above. Every change after the first working version kept the greedy answers bit for bit (14/14 on `xtx` and on `xtx-xt`) until 2026-10-14. Two runs of one image gave the
same greedy answers on 2026-10-12, and on only 7 of 14 prompts on 2026-10-16 (cause not looked for).

## 25. A decode step, kernel by kernel (2026-10-11 and 12)

A `rocprofv3` trace of the decode step on `xtx-xt-4x262k`: 22.9 ms of GPU time a step on the 7900 XTX, and ~1750
kernels for a 4-row step, ~400 of them small torch ops. Kept, each with the bits of what it replaces (bitwise tests over
random inputs, greedy answers identical): the EXL3 output kernel's split count left unspecialized (Triton's
constant-1 variant ran ~2x slower on rank 1's single K split); the hyper-connection's and the shared expert's
activations in the stores of their GEMVs; and the glue in kernels (the QSA index-write plan, the PLE's 4-row decode
conv, the MTP head's KV-only pass, a verify step's rollback copies, token-pool writes, draft addressing). The PLE conv
matched torch only with Triton's FMA contraction off and torch's sum order on gfx1100, (p0 + p1) + (p2 + p3).
`xtx-xt-4x262k` 85.5 -> 93.9 tok/s at one request (sampled), 33.9 -> 36.0 each at four; NVFP4 `xtx-xt` 77.8 -> 82.5.
Every profile and checkpoint was re-measured on that image: greedy decode over the whole context on `xtx-xt`, NVFP4
77-96 tok/s, EXL3 3.05 bpw 98-117, 4.05 bpw 86-94 (2026-10-12; [benchmarks.md](benchmarks.md) has the 2026-10-16 figures).
The sampled benchmark answers used the checkpoint's temperature 1.0 until 2026-10-12, 0.6 (`rdna3/serve.sh`'s default)
since.

**The trace** (night of 2026-10-11; EXL3 3.05 bpw, `offload`, one request, sampled decode; `rocprofv3 --kernel-trace`
over a 20 s window, run with a writable `-w` directory and the GPU workers stopped with SIGTERM before the container:
killed, they write nothing; ROCm's `torch.profiler` in this image records no GPU kernel). GPU time per step on rank 0:
the int8 dense GEMVs 6.7 ms, the routed experts' EXL3 GEMVs 3.4, the host all-reduce 2.4 (99 calls, 24.6 us each on
average, mostly waiting for the other card), the expert misses copied from host RAM 2.0, the graph waiting at layer 1
for the PLE rows 0.6 (mostly the tracer's doing), RCCL 0.5 (4-6 calls a step, ~92 us each: the LM head's all-gathers
and the broadcasts of rank 0's sampled tokens). Under the tracer each kernel of the replayed graph starts 3.2-3.5 us
after the previous one ends (a GDN layer is ~28 kernels, an attention layer ~40; the median layer's attention part
164 us of wall time for 127 us of kernels, its MoE part 262 us for 197). A chain of tiny kernels replayed from a graph
takes 2.7-2.8 us per kernel on either card, the same with `HIP_FORCE_DEV_KERNARG=1` or `0` (in serving too: one request
90.3 / 90.4 tok/s against 90.5 unset): no runtime setting found, fewer kernels would help. Rank 1's trace covers another
stretch of the run (212 steps in its window, 404 on rank 0), so they compare per step only roughly. The two ranks'
kernels take about the same GPU time per step (20.5 and 21.1 ms without the all-reduces); at the median all-reduce
rank 0 waits 17-18 us against 8 on rank 1, so rank 1 arrives late, and its `_exl3_out_kernel` took 14.6 us against 6.9
on rank 0 at the same grid. On one card (`xtx`, EXL3 3.05, one request) the expert misses dominate: 12.0 of 32.4 ms a
step (37 %), 251 us a layer, the time `--hybrid` takes back. Four agents on distinct ~257k contexts (as in section
22): the last first token after 442 s, then 28.2-30.2 tok/s each decoding together (a step ~32 ms, 96 % of it GPU
time; ~33 experts active per layer, 3-6 % of them missing the cache), 92.0 alone at 259k. The profile's depth sweep
that night (`rdna3/bench/depth_sweep.py`, one request, one pass): an agent turn (~1k new tokens on the cached
conversation) 0.87 s at 25k, 1.76 s at 248k.

**Kept** (each against the code without it, same session: 12 sampled 512-token answers at one request, 4 requests x 3
rounds, 14 greedy prompts, `FREETOKEN_SPEC_TIMING` for the forward pass; greedy answers identical):

| Change | One request | Four requests, each | Forward pass, 2 / 3 / 4 rows; 4 requests |
|---|---:|---:|---|
| Rank 1's EXL3 kernels: `SPLITS` no longer specialized (Triton's constant-1 variant of `_exl3_out_kernel` ran ~2x slower), gate\|up with 4 N tiles a program | 85.8 -> 87.4 tok/s (+1.8 %) | +0.4 % | -2.3 / -2.2 / -1.6 %; -0.4 % |
| `hc_silu` in the down GEMV's store (`FREETOKEN_HC_SILU_EPILOGUE`) | 87.4 -> 88.5 (+1.2 %) | 34.3 -> 35.0 (+1.9 %) | -1.3 / -1.3 / -1.3 %; -1.9 % |
| `hc_gate_mix` inside the up GEMV, weights loaded once (`FREETOKEN_HC_MIX_IN_GEMV`) | 89.4 -> 89.9 (+0.6 %) | 34.6 -> 35.1 (+1.4 %) | -0.1 / -0.1 / -1.2 %; -1.5 % |
| SiLU-and-mul in the shared gate \| up GEMV's store, weight tiles shared across rows (`FREETOKEN_SHARED_ACT_IN_GEMV`) | 89.4 -> 90.0 (+0.8 %) | 34.6 -> 35.0 (+1.3 %) | -1.3 / -1.1 / -0.7 %; -1.4 % |
| All of these, against the code before the night (8.4 h apart) | 85.5 -> 90.6 (+5.9 %, 12 of 12 answers faster) | 33.9 -> 35.4 (+4.5 %) | |

Rank 1's EXL3 kernels, cold: `_exl3_out_kernel` without the constant-1 variant, the 7900 XT 13.6 -> 7.8 us at 1 token,
20.2 -> 6.8 at 4, the XTX 12.0 -> 8.5 and 17.7 -> 6.0; the gate|up GEMV with 4 N tiles a program 33.8 / 62.2 / 69.7 ->
29.4 / 53.0 / 53.5 us at 2 verify rows / 4 rows / 4 requests.

NVFP4 `xtx-xt` (the EXL3 kernels do not apply): `hc_silu` in the store 78.0 -> 78.8 tok/s (+1.1 %), four requests
24.2 -> 24.5; all of these 77.8 -> 79.8 (+2.6 %), four requests 24.5 -> 24.7 (+0.7 %); the mix +0.5 % / +1.1 %, the
activation +0.1 % / -0.1 % (its shapes are K-blocked). The mix and the activation together, against the night's final
code: `xtx-xt-4x262k` 89.0 -> 90.6 tok/s (+1.8 %) and +2.1 % at four requests, NVFP4 `xtx-xt` 79.2 -> 79.8 (+0.8 %)
and +1.0 %; a 256k-token prompt in `check` mode read in 114 s, 13/13 answers, 92.6 tok/s, no disagreement between the
ranks. The first version of the mix accumulated `sigmoid(g) * x`
through another expression than `_hc_gate_mix_kernel`: a single-seed test matched, but 53 of 120 random cases had
values one bf16 step apart and 7 of 14 greedy answers changed in serving; with the kernel's own expression, 0 of 240
cases (0 of 960 for the activation), and the tests draw several seeds. The first versions also reloaded the weights
for every row (on NVFP4 the mix -1.8 % at four requests, the activation -0.8 %); on `xtx-xt-4x262k`, one request, the
first mix (up to 3 rows) gave 88.4 -> 89.1 tok/s (+0.8 %), the first activation 89.0 -> 89.3 (+0.4 %).

**The glue** (afternoon of 2026-10-12). A 4-row step launched ~1750 kernels, ~400 of them small torch ops (rocprofv3,
rank 1), each costing its own run plus the ~2.8 us between two kernels: ~300 in the eager part of a verify step, ~25
for the first QSA layer's index plan and ~25 for the PLE layer's 4-row conv inside the graph. The kernels that replace
them give the torch ops' bits (bitwise tests over random batches, `tests/kernels/test_decode_glue.py`; greedy answers
identical, 14 / 14). For the PLE conv, with Triton's FMA contraction 90-121 of 150 random cases differed whatever the
sum order; without it, torch's order on gfx1100 matched 150 of 150 (the two other orders: 81 and 97 differed). Each
change against the same tree with it switched off ("greedy": the mean of 9 greedy 512-token answers whose texts do not
depend on the drafts; four requests over two passes):

| Change | Greedy, one request | Sampled, one request | Four requests, each |
|---|---:|---:|---:|
| plan + PLE conv + KV-only stop, `xtx-xt-4x262k` | 103.4 -> 105.3 tok/s (+1.8 %) | 90.7 -> 92.3 (+1.8 %) | 35.48 -> 36.15 (+1.9 %) |
| same, NVFP4 `xtx-xt` | 91.0 -> 92.5 (+1.6 %) | 80.2 -> 81.4 (+1.5 %) | 24.77 -> 25.02 (+1.0 %) |
| verify glue, `xtx-xt-4x262k` | 105.5 -> 107.1 (+1.5 %) | 92.4 -> 93.7 (+1.4 %) | 35.83 -> 36.02 (+0.5 %) |
| same, NVFP4 `xtx-xt` | 92.4 -> 93.9 (+1.6 %) | 81.3 -> 82.5 (+1.5 %) | 25.23 -> 25.15 (-0.3 %) |

The code as kept, one pass: `xtx-xt-4x262k` greedy 107.5 tok/s, sampled 93.9, four requests 35.9, a ~256k-token
conversation read in 111 s (13/13 answers, then 96.0 and 101.6 tok/s); NVFP4 `xtx-xt` greedy 93.8, sampled 82.5, four
requests 24.5. The four agents above on the night's final code: the same scores, the cold reads in 445 s, then
29.6-31.6 tok/s each decoding together and 95.1 alone at 259k.

**Dropped:**

- Rank 0's sampled tokens through the host all-reduce instead of RCCL broadcasts: one request -0.3 % (EXL3) and -0.2 %
  (NVFP4), four requests +2.2 % on EXL3 but -0.5 % on NVFP4 (+0.2 % for the plain decode step only): within the noise
  of that test.
- A RAM cache of the PLE rows (512 MiB per rank; the disk backend reads every row with O_DIRECT, since the 30.4 GiB
  table does not stay in the page cache: a buffered read took ~70 us a row). A fill of 4 tokens (64 rows, 16 hash
  heads) took 415 us alone in a microbenchmark, 500-560 with both ranks reading, 0.84-0.90 ms in serving at 4 rows (a
  pread pool of 32 threads instead of io_uring: 360 us). The cache took the fill at four requests 0.88 -> 0.57 ms, and
  decode nothing (one request +0.0 %, four 34.7 -> 34.3 tok/s; NVFP4 +0.1 % and 0.0 %). The fill is not on the step's
  critical path: the 1.6 ms PLE waits a trace shows (the median of runs of consecutive steps, 40 % of the steps, 0.68 ms
  a step on average, in the 2-row steps) are the tracer's doing (`hipGraphLaunch` held the host 1.3-1.9 ms under it; the launch takes 0.05 ms
  without). A fill on a helper thread from the start of the dispatch: one request +0.2 %, four +0.5 %;
  `ROC_ACTIVE_WAIT_TIMEOUT=5000` -0.1 %, a 0.5 ms GIL switch interval +0.1 %.
- Bit-exact int8 tiles for the uneven split at 2-4 rows (only BLOCK_N is free; a 4-wide tile for the GDN in_proj took
  3-11 % less per call cold): one request -0.3 % (EXL3) and -0.2 % (NVFP4), four +0.5 % and +1.6 % (within that test's noise). K blocks of 512
  are faster (forward pass -1.5 / -2.6 / -2.9 % at 2 / 3 / 4 rows, -2.3 % at four requests, 34.0 -> 34.8 tok/s each)
  but changed 6 of 14 greedy answers: opt-in (`FREETOKEN_INT8_TP_TILES=1`).
- `hc_silu` and SiLU-and-mul computed inside the consuming GEMV: 87.3 -> 78.4 tok/s (-10 %), forward +8 to +15 %
  (likely every program recomputing the activation of the whole row it reads, 10,240 programs for the up projection;
  not profiled).
- Grouping the verify rows that share an expert in one EXL3 GEMV (each tile decoded once for up to four routes,
  bit-identical): rank 0's shapes 104 -> 51 us (gate|up) and 54 -> 30 us (down) in a microbenchmark that called the
  same 22 experts over and over, which then sat in the Infinity Cache; as a step reads them, 7.95-8.09 ms for its
  expert GEMVs with or without the grouping; in serving the forward pass 26.4 -> 26.8 ms at 4 rows, EXL3 on `xtx`
  -1.5 %; NVFP4 grouped slower in 17 of 18 shapes, by up to 63 %.
- The MTP cost table measured again (`FREETOKEN_SPEC_TIMING`, GPU ms per step at 2 / 3 / 4 rows, one request): EXL3
  `xtx-xt-4x262k` 24.3 / 26.2 / 30.8, NVFP4 `xtx-xt` (`--hybrid`) 25.2 / 27.5 / 32.9, against 25.3 / 30.7 / 35.5 in
  `spec_decode.py`. With the measured table: EXL3 85.8 -> 86.4 tok/s, NVFP4 80.8 -> 81.6, within the noise (the depth
  already ran 3-4 rows most of the time, 1202 + 1465 of 3300 steps). On one card (`xtx`, `--hybrid`) the depth stays at 2 rows (4593 of 4593
  steps), and forcing 3 or 4 rows was slower (NVFP4 50.6 / 44.5 tok/s, EXL3 47.4 / 40.4, against 56.1 and 52.8 at 2
  rows).
- `num_stages` 1-4 on the decode GEMVs, the router's top-k in one packed int64 reduction (~0.5 us a call saved), several
  hidden indices per program of the fused mix: nothing beyond the ~2 % spread of the cold timings. Run-to-run noise,
  the same code in two sessions: one request +-0.1 %, four requests +-1 %.
- Verifying drafts with three or four requests (`FREETOKEN_SPEC_BS_MAX=4`): four requests 26.3 tok/s each against
  35.35. The depth, costed for one request, picks 4 rows for each (a forward pass of 94 ms against 27 for one row each,
  ~2.55 tokens kept per request and step; the only 2-row step seen cost 64 ms); greedy answers of four at once changed
  in 3 of 12.
- MTP drafts over part of the vocabulary (the draft's argmax over 38-75k frequent token ids): the head and the chained
  guesses 1.36 -> 1.0-1.07 and 2.25 -> 1.5-1.65 ms of GPU time at 4 rows, but fewer drafts kept (greedy 103.7 against
  103.5 and 104.1 tok/s with 75k and 52k ids) and the copied LM-head rows (~92 MiB a card with 75k ids) came off the
  expert cache: four requests 35.9 -> 34.8 tok/s (one pass each, -3 %; forward pass +2.8 %). A first set built from
  code only missed 4.6 % of English prose tokens; with man pages, READMEs and word lists, 0.5-1.5 % of held-out ones.
- Narrower V tiles in the GDN recurrent kernel (the same bits): 1-3 us a call, ~0.25 % of a step (estimated), not run
  in serving.

## 26. The PLE fill on one card (2026-10-13)

Without the MTP head the scheduler overlaps its decode steps: the host launched step N+1, then waited for step N's
readback before filling N+1's PLE rows, while N+1's captured wait held the GPU, ~11 ms more a step on one GPU.
`FREETOKEN_PLE_SYNC=auto` takes launch-gating there (one GPU, no MTP head): NVFP4 `xtx --no-mtp` 26.6 -> 38.3 tok/s, the
same tokens. Two cards and the MTP head keep the wait-sync, 1.6-4.3 % faster in their A/B.

## 27. Agent turns: short prefills through the expert cache (2026-10-14)

An agent turn is a short prefill (a tool result of 100-3000 tokens on a cached conversation) and a short answer. In
NVFP4 `xtx-xt`'s depth sweep of 2026-10-12 the ~1k-token agent turn took 1.3 -> 2.1 s along the sweep (0.8-0.9 s at
every depth after the changes below: [benchmarks.md](benchmarks.md#depth-sweep-xtx-xt)). On
`xtx-xt-4x262k` (EXL3 3.05 bpw) a turn of 128-334 new tokens took 0.77-0.82 s to its first token at 25k tokens of
context and 1.13-1.17 s at 150k, about as long as a turn of 1.2k tokens. In a rocprofv3 trace of those turns the 7900
XTX computed for 0.17-0.52 s of a ~0.9 s prefill and waited for the rest. Each prefill chunk streams every expert the
cache lacks into the two full-layer buffers (the 40 % the cache held are gathered device-side; the other ~15k of the 25k
layer-experts crossed PCIe, ~17 GB on the XTX by the slice sizes), whatever its length: at 128 tokens 62 % of the
layer-experts are routed to (71 % at 334, 82 % at 1.2k, 90 % at 3.1k). With the copies skipped (a measurement with wrong
answers) the same turns took 0.39-0.48 s. The bus itself was not the limit: 28.1-28.4 GB/s per card, by DMA or the
shader copy, alone or both cards at once, and the batched DMA of a layer's misses 26 GB/s; the shader copy instead of
the DMA gave nothing on short turns and cost 6-15 % from 1k tokens on (it takes compute units from the GEMMs).

What was tried, in order (each against the code without it, same session; answers compared token for token):

- **Only the routed misses, into the slot cache**, by decode's copy, then the prefill GEMMs reading each expert from its
  slot (the tiles are still built on expert ids, so the same bits): slower (1.2k tokens 0.88 -> 1.44 s). The copy no
  longer overlapped anything, since a layer's routing is known only when it runs, and the prefill's ~15k experts pushed
  the decode's working set out of the 11k-slot cache: the answer after the turn decoded slower too.
- **The experts a prefill brought in demoted** to the next victims once their GEMMs are queued: the decode after a turn
  became faster than before (the full-layer path also drops the 1024 slots its buffers borrow), short turns -10 to
  -22 %, 1-3k-token turns still slower.
- **The next layer's experts predicted and copied while this layer computes.** Layer L+1's router applied to layer L's
  MoE input, the union of each token's top-10 (its own top-k), recalls 79-84 % of the experts layer L+1 then routes to
  (88-91 % with the top-14, which predicted 14-27 % more experts, with 53-97 % more of them not routed to; those copies
  cost more than the misses they saved: short turns up to 6 % slower with the top-14 than with the top-10, 1-3 % slower
  with the top-8). They
  cross PCIe on a copy stream during layer L's GEMMs and layer L+1's attention; layer L+1 waits for that copy and
  fetches only what the prediction missed. A wrong prediction
  costs a copy, never a result: the GEMMs read only the experts the routing picked, all resident by then.
- **The slot bookkeeping in three kernels** (marks and hit stamps, LRU keys, the slots of the routed misses and of the
  predicted experts claimed together, one top-k over the cache per layer) instead of small torch ops (~74 more kernels
  a layer than a full-layer prefill in a trace) and up to five host syncs a layer: 7-10 % lost at 3k tokens before,
  2-4 % after. One host sync per layer remains (the DMA takes host pointer lists).

Kept as `FREETOKEN_PREFILL_CACHED_MAX`: on by default up to 1280 tokens for Qwen3.8-Flash-Next (the model that
predicts). Measured (with the top-14 prediction of the time) against the code without it on the
same image (greedy answers on 14 prompts and the answers of the agent conversations identical on every profile; time to
the first token of a turn, seconds; one conversation per profile):

| Profile | Context | 132-148 new tokens | 329-339 | 910-1046 | 2.7-3.2k |
|---|---|---|---|---|---|
| EXL3 3.05 `xtx-xt-4x262k` | 25k | 0.76-0.79 -> 0.50 | 0.76-0.78 -> 0.55-0.56 | 0.84-0.88 -> 0.76-0.82 | 1.46-1.54 -> 1.47-1.56 |
| EXL3 3.05 `xtx-xt-4x262k` | 173k | 1.13-1.23 -> 0.88-0.99 | 1.21-1.25 -> 1.01-1.03 | 1.29-1.33 -> 1.18-1.21 | 1.95-2.02 -> 1.96-2.02 |
| NVFP4 `xtx-xt` | 25k | 1.19-1.21 -> 0.77-0.80 | 1.20-1.22 -> 0.87-0.88 | 1.29-1.32 -> 1.10-1.15 | 1.75-1.84 -> 1.74-1.85 |
| EXL3 3.05 `xtx` | 25k | 1.50-1.51 -> 1.06-1.08 | 1.50 -> 1.10-1.19 | 1.54-1.55 -> 1.34-1.39 | 1.95-2.04 -> 1.95-2.04 |

On a 224k-token conversation (`xtx-xt-4x262k`, the same answers): turns of 142-360 tokens 1.31-1.43 -> 1.15-1.27 s,
1-3k-token turns unchanged, the cold read 95.9 / 96.2 s. Four agents at once on distinct 25k-token conversations (one
run each; batched answers differ from run to run, the code without it included): the median turn under the threshold
0.89 -> 0.77 s, the median decode 21.2 -> 23.3 tok/s per agent, the four conversations' summed turn time 447 -> 380 s.
The answer after the turn decodes faster too (EXL3 `xtx-xt-4x262k` at 25k: +0.5 to +1.9 tok/s over the 88-131 tokens
answered after the turns under the threshold); the cold read of 173k tokens is unchanged (71.4 / 71.6 s). Past 1.2-1.4k
tokens the turn is compute-bound and the full-layer path, which overlaps its copy with the previous layer, stays: there
the cache path was 2-4 % slower.

Under `--moe-strategy hybrid` it is on as well. There the cache contents decide which decode routes the CPU computes, so
the answers move with them, as they did before. With the cache path off (the answers of main: the same 9 of 9 on a
25k-token conversation on EXL3 `xtx`), that conversation gave other answers from its second turn on when the 14
greedy prompts had run before it in the same server. The CPU is not the less precise side. At the
model's sizes (hidden 2560, expert width 640, top-10, NVFP4 banks, 300 tokens), the relative error of one MoE output
against an fp64 reference: GPU decode kernel (bf16 between the steps) 3.9e-3, CPU (fp32 inside) 2.3e-3, 8 to 2 of the
10 routes on the CPU (two bf16 partial sums added) 3.3-4.0e-3; rounding the exact result to bf16 alone is 1.7e-3. The
CPU and GPU results differ from each other by 4.5e-3, which is what moves a greedy answer. Time to the first token of a
turn of 132-337 new tokens at 25k, path off -> on: NVFP4 `xtx-xt` 1.19-1.24 -> 0.78-0.89 s, EXL3 `xtx` 1.50-1.52 ->
1.05-1.18 s, NVFP4 `xtx` 2.32-2.34 -> 1.63-1.80 s.

**Then the tokenizer.** A short turn at 173k of context still took 0.93-1.03 s against 0.49-0.56 s at 25k. The GPU part
of a turn grows little with the context (the K/V staged from RAM: 0.1-0.2 s of it at 196k, measured with the K/V in VRAM
instead); the rest was on the host. The tokenizer worker encoded the whole conversation at every request: with the
model's tokenizer on one core of the 3970X, 41-47 ms at 24k tokens, 0.27-0.40 s at 174k, 0.36-0.51 s at 224k. An
agent's prompt extends its previous one, and a fast tokenizer encodes the text between two added tokens on its own,
so the tokens up to an added token depend only on the text up to it. The worker now keeps its last prompts, takes the
tokens of the longest shared prefix up to the last `<|im_start|>` inside it and encodes only the rest
(`FREETOKEN_TOKENIZE_PREFIX_CACHE`): 1.0-2.6 ms a turn. The same tokens: compared with the whole encoding on the 8
turns that reuse a prompt in an agent run (`FREETOKEN_TOKENIZE_PREFIX_CHECK=1`, no difference; prompts under 16k
characters, such as the greedy prompts, are encoded whole), and in tests on agent
conversations, edited and branched ones, and prompts cut at random places (inside a word, a multi-byte character, a
special token's text). That worker also detokenizes the running streams between two encodes, so with several agents
their streams waited for each encode too. Time to the first token on EXL3 3.05 `xtx-xt-4x262k` (one conversation each,
the same answers):

| Context | 132-360 new tokens | 910-1118 | 2.7-3.2k |
|---|---|---|---|
| 25k | 0.49-0.56 -> 0.42-0.51 | 0.74-0.80 -> 0.69-0.73 | 1.43-1.51 -> 1.36-1.45 |
| 173k | 0.93-1.03 -> 0.53-0.58 | 1.23-1.29 -> 0.81-0.84 | 1.91-1.98 -> 1.48-1.54 |
| 224k | 1.17-1.21 -> 0.58-0.70 | 1.45-1.57 -> 0.90-0.98 | 2.23-2.35 -> 1.60-1.73 |

(At 224k the earlier run had the mid step at 16 columns a program instead of 32, the same bits.)

What still grows with the context in a short turn is mostly on the GPU. With the K/V in RAM (`xtx-xt-4x262k`), each
prefill copied every page of the conversation to VRAM before each of the 12 attention layers: in kernel traces, 1.1 ms
a layer at 25k, 6.3-6.9 ms in a cold read's chunk at 164-180k (the trace of the 224k read stopped there), for a turn whose ~220 rows read 2048 tokens each. Those rows select 11-21 %
of the history's tokens but touch 40-66 % of its 64-token pages (21-58 % of the tokens and 73-95 % of the pages for
600-3200 rows). A prefill now copies only the K/V rows of the selected tokens, marked from the attention's own indices
with the attention kernel's mapping (`FREETOKEN_QSA_STAGE_SELECTED`): the same rows read, so the same bits (tested with
NaN in every row left out). At 224k, every page / the selected pages / the selected rows: 142-360-token turns 0.57-0.70
/ 0.52-0.66 / 0.49-0.62 s, 1.0-1.1k 0.90-0.98 / 0.88-0.94 / 0.84-0.89, 2.8-3.2k 1.61-1.74 / 1.59-1.73 / 1.56-1.70; the
same answers (greedy and agent conversation), and unchanged at 25k.

Where a short turn's time goes after that, counted per layer during an agent run at 25k with 64-token answers between
the turns: a turn of 131-232 rows routes ~220-290 distinct experts a layer, 61-73 % of them already cached; the rest
cross PCIe. For a 131-row turn that is 5275 expert copies, about 5.9 GB on the XTX and 3.9 GB on the XT (estimated
from the bank sizes), 0.15-0.23 s at 26 GB/s of a 0.34 s turn. Half of the predicted copies were for experts the next
layer did not route to (2441 of 4693 with each token's top-14): the top-10 is the default now (above). The two EXL3
prefill GEMMs also run with two pipeline stages (and 16-block groups for gate|up), the same bits: -5 to -8 % on gate|up
and -2 to -3 % on down in isolation on both cards. Both together, against the image of the previous commit: 132-337-token
turns 0.42-0.50 -> 0.39-0.47 s at 25k, 0.44-0.52 -> 0.43-0.50 at 173k, and with the launcher's arguments on the two
images 0.49-0.62 -> 0.46-0.59 at 224k (cold read 94.1 -> 92.2 s); 1-3k-token turns and the other cold reads unchanged;
the same answers.

Found on the way:

- **A hang, and the kernel behind it.** A rocprofv3 trace of a short turn after a 196k-token cold read hung both
  ranks: an all-reduce timed out after 60 s (the RCCL watchdog then stops the server). With a profiler or a tracer
  slowing the host, the cache path hung there in 7 of 8 runs; the code without it in none of 3 runs (24 turns); without
  instrumentation, 5 runs of that conversation ran clean. `rocgdb` on the two ranks at the hang: rank 1 inside the
  RCCL kernel; rank 0 with no kernel running, its compute queue stopped at a valid dispatch of the EXL3 mid kernel
  (between the two expert products) that was never launched, every packet before it complete. That kernel's 64-row
  tile version spilled 861-866 VGPRs, 2.7 KB of scratch per lane; the dispatch waiting for a scratch grant fits what
  the queue showed (not proven: the runtime has no symbols). The same kernel ran in every EXL3 prefill before, the
  full-layer path included. It now takes the Hadamard 32 columns at a time, 32 rows and 4 warps a program
  (`FREETOKEN_EXL3_MID_COLS`): the same bits (each output is one dot of a row with one column), 572 B of scratch,
  1.7-3.3x faster alone (7900 XTX: 1394 -> 826 us for a 4096-token block, 416 -> 125 us for 222 tokens); the cold read
  of 173k tokens 71.4 -> 69.8 s. 6 runs of the same instrumented test with it (with 32 columns, and an earlier 16-column
  version at 264 B), 24 of 24 turns, no hang. 8 or 16 HIP hardware queues (`GPU_MAX_HW_QUEUES`) also avoided it,
  which only moves the kernel to another queue.
- **The JIT modules were rebuilt at every server start.** tvm-ffi builds them under `~/.cache/tvm-ffi`, outside the
  per-image cache volume: the first request after a start took ~10 s, and the first long prefill 1.5-5.6 s more. The
  image and `serve.sh` now set `TVM_FFI_CACHE_DIR` in the volume: first request 1.3-1.9 s.
- **That exposed a race in the batched-DMA probe**: it filled its target on one stream and copied on another without
  ordering them, and the seconds of compilation used to hide it. With the module loaded from the cache the probe failed
  at random, and that rank fell back to one copy per expert: full-layer short prefills ~10 % slower. Fixed (the copy
  waits for the fill), and a failed mode is now logged.

All of it against the `main` before (time to the first token at 25k of context, arms alternated, 9/9 answers
identical): 130-340-token turns NVFP4 `xtx-xt-4x262k` (3 x 262k pool) 1.22-1.24 -> 0.58-0.70 s and EXL3 3.05 bpw
0.86-0.89 -> 0.40-0.48 s, ~1k-token turns 1.33-1.34 -> 0.93-0.98 s and 0.94-0.99 -> 0.69-0.74 s, 3k-token turns -2 to
-8 %; decode unchanged (-2 to +4 %). At 173k / 224k on EXL3 3.05 bpw: 1.13-1.25 -> 0.43-0.50 s and 1.31-1.43 ->
0.46-0.59 s. The tokenizer's prompt prefix: ~0.5 s -> 1-3 ms at 224k.

## 28. Upstream up to 9b68e00 (2026-10-16)

Five commits cherry-picked, then the rest merged. Upstream had rewritten the files this tree's `--hybrid` lives in: its
CPU MoE executor split into weight formats and GPU sync backends (with HIP memops of its own), its hybrid ensure kernel
now splitting the routes (what this tree's split kernel did), and the expert cache cut into slot pools. The port kept
upstream's structure and moved this tree's pieces onto it: the EXL3 CPU experts behind a whole-task hook of the format
interface, the publish / merge kernels on upstream's task-word handshake (until then they counted step numbers in per-(layer,
batch) slots; now two pinned words, rung and reset by one lane:
a review of the port found that a reset repeated by every thread could make the coordinator run a step twice), the
prefill through the cache gated to one slot pool of at least two layers of experts. transformers 5.18 rounds a few
values of its YaRN reference up to two ulps away from 5.16: the test now allows that, the engine's rows are unchanged.
Upstream's fix of the Triton 3.8 histogram bug (a staged align, `0bca88f`) was left out: `moe_align.py` here has no such
compare. The image took transformers 5.18 (the config, the chat-template tokens, tool calls, a ~105k-token text and the image
preprocessing read the same as with 5.16, bit for bit), flashlib 0.4.0 and numpy 2.5.3. Against the build before the
merge (`xtx-xt-4x262k`, NVFP4 with three 262k contexts and EXL3 3.05 bpw with four): greedy with the MTP head 14/14 and
a 25k-token agent conversation 9/9, the same answers on both; decode per turn -0.8 to +0.9 %; agent-turn time to the
first token 0.59-1.69 s against 0.59-1.66 (NVFP4), 0.40-1.46 against 0.40-1.45 (EXL3); cold reads of 36k / 46k tokens
2375 / 2512 tok/s against 2387 / 2525 (NVFP4), 2565 / 2680 against 2552 / 2661 (EXL3); four 20k-token requests decoding
together (NVFP4) 29.0-29.9 tok/s each against 28.9-29.9; a 259,728-token prompt read at 2338 tok/s, 400 tokens at
74.7 tok/s. One card without `--hybrid`: greedy 14/14 the same. `--hybrid` was not compared: two runs of the build
before the merge already gave different greedy answers (7 of 14 the same).

## 29. Three and four requests, and cold reads (2026-10-16)

Decode at three and four requests (NVFP4 `xtx-xt-4x262k` 3 x 262k, 20k-token conversations decoding together,
`FT_STATS_EVERY` + `FT_MOE_STATS`): a step takes 26-29 ms at three and 31-35 ms at four, the GPU busy 98-99 % of it;
33-35 experts are active a layer, 2-4.6 of them missing the cache (8-13 %), each copied across PCIe in ~55 us: ~9 ms of
a four-request step by that count (misses x 55 us, not timed apart), its largest part. The PLE fill costs 0.9 ms a step
(rows read with O_DIRECT, no RAM cache); launch-gating instead of wait-sync -3 %; the side-stream miss copy
(`FREETOKEN_MOE_COPY_OVERLAP`) +0.7 %. Prefetching layer L+1's predicted experts on a side stream while layer L computes
cut the misses by 42 % but made the step 20 % longer at four requests and 2.2x at one (probably the fork and join edges
inside the replayed graphs, not measured apart; a narrower copy grid was slower still): dropped. The sparse attention
reads its 2048 selected rows a request from the host K/V each step, 66-81 % of them the same as the step before: a cache
that kept them in VRAM per request and layer gave the same bits and steps 0.6-3.5 % shorter at three or four requests,
0.8-1.7 % longer for one (the VRAM it took from the expert cache): not kept, too little for ~440 lines of code. A kernel
trace of these steps lost 37 % of its records in whole chunks of dispatches
([testing.md](testing.md#micro-benchmarks-rdna3bench)).

Cold reads on NVFP4 `xtx-xt-4x262k` (a kernel trace of 16k-token chunks on the 7900 XTX, busy 98 % of the time inside
the reads): the NVFP4 expert GEMMs 37 % of the GPU time, the RCCL all-reduces 17 % (7.9 ms for 80 MiB when both ranks
arrive together, ~10 GB/s: the cards have no peer access and stage through host memory), the sparse attention 11 %,
hipBLASLt's dense GEMMs 13 %, the indexer 4.5 %, strided copies of the GatedDeltaNet q / k / v 3.5 % (gone with
`FREETOKEN_GDN_CONV_ROWS`). A many-block all-reduce through shared host memory, pipelined in pieces so both PCIe
directions work, gave RCCL's bits but took 9.0 ms: dropped. The sparse attention reads each query row's 2048 selected
rows apart; sharing them between neighbouring rows would change the sums' order.

The NVFP4 prefill MoE ran 4096 tokens at a time since the 16k-token chunks (section 7), when the memory was short at
depth. On two cards an 8192-token block reads 8 % faster (NVFP4 `xtx-xt-4x262k`: 98k tokens 43.4 -> 40.2 s, 245k 112.6
-> 103.7 s) and 16384 13 % (38.7 and 99.9 s); an expert gets ~160 / ~320 rows a block instead of ~80 on average. Every
tile keeps the 32-byte K step, so no bit changes. 16384 took the XT's VRAM peak of a 255k-token read from 19,146 to
19,806 MiB (0.65 GiB left), 8192 to 19,346: `xtx-xt` and `xtx-xt-4x262k` take 8192. In a one-layer kernel benchmark
(random routing, the XTX's slice), the gate_up GEMM took 1.53 / 1.51 / 1.15 ms per 1k tokens at 4096 / 8192 /
16384-token blocks and the down GEMM 1.09 / 0.85 / 0.79; gate_up's 128-row tiles are on average 62 % full at 4096 and
8192 tokens (80 or 160 rows an expert) and 83 % at 16384, which fits its times (not measured apart). A 12288-token
block, the fastest per token there (gate_up 1.11 ms per 1k), read 98k / 116k tokens in 40.5 / 47.1 s, as 8192 did (a
16k-token chunk also runs a 4096-token block there; its share was not measured apart). Two variants of the GEMM with the
same values were slower and dropped: the E2M1 codes decoded by bit arithmetic instead of the 16-entry table (+17 to +23
%), and the activations loaded as one contiguous tile and split into even / odd columns in registers (gate_up up to
2.3x).

With the token-major GatedDeltaNet conv, the NVFP4 MoE in 8192-token blocks (NVFP4) and two pipeline stages in the
EXL3 prefill GEMMs (EXL3), against the `main` before (the same texts, arms alternated): NVFP4 `xtx-xt-4x262k` (3 x 262k pool) 98k tokens 44.8-44.9 -> 40.2-40.3 s, 116k
52.8 -> 46.8-46.9 s (+11 to +13 %); EXL3 3.05 bpw 98k 40.2 -> 38.2 s, 116k 47.0-47.1 -> 44.3-45.5 s (+3 to +6 %).

## 30. A rank killed by the host's OOM killer (2026-10-17)

Three times in a morning (07:26, 08:16, 09:55) the host's 125.6 GiB filled up (no swap) and the kernel's OOM killer took
a rank, the largest process (52 GiB of RSS, the other rank 43 GiB; rank 0 the second time), while 86-90 `MainThread`
processes (Node.js; 63-64 of them one burst) held 13-14 GiB beside the desktop. Rank 1 kept its 18.7 GiB on the 7900 XT
and looped on `TCPStore ... Broken pipe`; the port no longer answered, but the container stayed up, its service stayed
active, and stopping it ran into the 30 s timeout. The logs went with the container; the cause found in the code and
reproduced on the CPU: on a worker's death the API server sent itself a SIGTERM, and uvicorn's graceful stop (0.54,
`timeout_graceful_shutdown=None`) waits for open connections first. A stream to a dead backend only sends keepalives, so
the lifespan shutdown that terminates the workers never ran. The server now stops the workers itself and exits with code
1 (a test with an open stream and a worker ignoring SIGTERM: no exit in 60 s before, exit 1 after; live, rank 0 killed
under a stream: the container exited 37 s later, both cards freed, `llm.service` restarted it 5 s after), and `serve.sh`
gives the container `--oom-score-adj -999`. What filled the RAM, from the kernel's task dumps of these events: a burst
of 63-64 Node.js processes, consecutive pids started together, 11.6-13.8 GiB, under an `npm exec jest` the first time
(Jest's default pool is one worker per core but one: 63 on the 64 threads); the later dumps show the same burst, its
command not named. Five more events the same day (bursts of 63-66 workers, 12.2-17.3 GiB as the dumps went on, two of
them under `timeout ... | grep | head` and `timeout ... | tail` pipelines) filled the RAM again: with `--oom-score-adj
-999` the kernel killed browsers, an editor's helper process, a VPN window and a command-line tool instead, and the
server kept serving.

## 31. Short agent turns (2026-10-17 and 18)

Measured on `xtx-xt-4x262k` through the launcher's arguments (NVFP4 with a 3 x 262k pool, EXL3 3.05 bpw), arms
alternated, each on a fresh server. The workloads: an agent conversation (a 34k-token document, then 9 questions of
~200-250 new tokens, each after a 200-token greedy answer), 9 one-word turns of ~20-30 new tokens on a 26k-token
conversation, 15 greedy prompts; their answers were identical to the build before on every arm.

**Where a short turn goes.** A kernel trace of the agent turns (NVFP4, rank 0): ~4,020 kernels a turn, the GPU busy
80-83 % of its 400-500 ms, ~1,700 gaps over 10 us adding to 80-100 ms, 14-23 ms of which a copy kernel on another queue
covers; the routed experts' GEMMs 100-160 ms, the missing experts' copies on the compute stream 34-65 ms, the int8
weights' dequantization 26-28 ms, hipBLASLt 16-23 ms. PyTorch's sync debug mode (`torch.cuda.set_sync_debug_mode`)
found the host waiting on the GPU ~83 times a forward: 36 in the GDN prefill conv, which read the longest request back
(`.item()`) in every GDN layer, and ~47 where the predicted experts' copy builds the copy engine's host pointer lists.

**Kept, the same bits:**
- The GDN conv takes the longest request from the host bounds the metadata already has: NVFP4 agent turns (turns 2-9)
  0.447-0.448 -> 0.415-0.416 s (two starts), one-word turns 0.394 -> 0.350 s.
- NVFP4 prefill tiles for a few rows an expert (`FREETOKEN_NVFP4_PREFILL_TUNED`): the tiles were swept for chunks of
  thousands of tokens (64-row tiles at 8 warps above 64 tokens, the shipped 16-row tile at 8 warps and 4 stages below),
  where a 250-token turn gives an expert ~5 rows. 4-warp tiles with the same K step, compared bit for bit with the tiles
  they replace on both cards at 22 sizes from 1 to 1537 tokens (and by `rdna3/tests/moe_block_check.py`): one layer's
  gate_up / down 5-13x faster at 32-64 tokens, 1.7-2.5x at 128-512, 1.4-1.8x at 768-1536, within 9 % from 2048 (left as
  they were). With the GDN fix: agent turns 0.448 -> 0.398-0.399 s, one-word turns 0.394 -> 0.227 s; the first use of a
  new tile compiles it (1.3-2.5 s, once per image).
- EXL3: 16-row tiles instead of 32 up to 512 tokens (the whole MoE block 11-19 % faster at 32-512 tokens on both cards,
  slower from 1024; bit-identical): with the GDN fix, agent turns 0.328 -> 0.287 s, the first turn 1.18-1.21 -> 1.13 s,
  one-word turns 0.230 -> 0.218 s.
- Decode (the agent passes' decode speeds within 0.6 tok/s), four requests (NVFP4 35.0-35.7 tok/s each, EXL3 37.3-37.5)
  and a 133k-token cold read (NVFP4 56.3-56.5 s, EXL3 53.3-53.5 s): unchanged.
- Through the launcher's two entries with an image built from the tree (fresh servers, the image before against this
  one): the 15 greedy prompts, the agent conversation, the one-word turns and a 256,393-token prompt answered
  identically; that prompt read in 108.5 / 108.1 s on EXL3 (2,363 / 2,373 tok/s) and decoded at 93.6 tok/s on both;
  NVFP4 114.3 / 113.8 s (2,242 / 2,254 tok/s; 118.6 s on the new image's first start, still compiling), decoded at 85.5
  / 82.8-85.8 tok/s (after another workload on each start). The first start of a new image compiles each new tile once
  (1.3-2.5 s on the turns that first use it). Three agent conversations at once (25k-token documents, 9 turns each,
  NVFP4, two passes an image): the turns' median time to the first token 0.481-0.502 -> 0.426-0.427 s, the first reads
  and the decode as before (once the new image's tiles had compiled: in its first pass the last of the three documents
  was read after 43 s against 34-35).
  One card (`xtx`, EXL3 3.05 bpw, 131k, two starts an image): the same answers, the agent turns unchanged (the turns
  there wait on the experts crossing PCIe), the one-word turns 0.281-0.283 -> 0.264-0.267 s.

**Dropped:**
- The predicted experts through the shader copy (device indices, no host sync) instead of the copy engine: agent turns
  0.427-0.436 s against 0.401 without it (three passes of 9 turns each): slower.
- The missing experts of a short prefill copied on a side stream while the shared expert runs (eager, no captured
  graph): agent turns 0.398 s against 0.404-0.407 (-2 %), one-word turns 0.223-0.227 against 0.235-0.239 (-5 %), the
  same answers, but every decode after it slower: 4-6 % for one request, 23.2-23.5 tok/s each for four instead of
  35.0-35.2.
  The cause was not found (`GPU_MAX_HW_QUEUES=8` changes nothing, with or without it; the decode graphs are captured at
  start either way). On the prefill copy stream instead, kept (section 32).
- The int8 dense weights dequantized for a prefill by a Triton kernel instead of torch's mixed-dtype multiply (the same
  bits; 4.5-7.7x faster on the weights of 4M elements or more): agent turns as before, one-word turns 0.246 s with it
  against 0.240 without, a 133k-token cold read at 2378 tok/s (2368-2369 before): the GPU ran them while it waited
  anyway.
- Decode, a kernel trace of one-request steps (MTP verify, 4 rows): 1,560 kernels, ~24 ms traced, 22 % of it between
  kernels; the int8 dense GEMVs 7.2 ms, the routed experts' GEMVs 2.7 ms and their copies 2.2 ms, the host all-reduces
  1.6 ms. The HIP runtime's settings against the gap between dependent kernels (a graph of 1,000 tiny kernels: 2.9-3.3
  us a kernel): `DEBUG_CLR_AQL_BARRIER_OPT`, `DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING`, `DEBUG_HIP_GRAPH_BATCH_SIZE`,
  `ROC_SYSTEM_SCOPE_SIGNAL`, `GPU_FLUSH_ON_EXECUTION`, `HIP_FORCE_DEV_KERNARG` change nothing,
  `DEBUG_HIP_GRAPH_CLASSIC_PATH=1` and `ROC_USE_FGS_KERNARG=1` lengthen it.
- The missing experts' copy in the same launch as the shared expert's gate_up GEMV, so that the PCIe copy and the GEMV
  overlap without the fork and join that slowed the side-stream copy: the copy rewritten in Triton moves 24-27 GB/s
  alone (the C++ kernel 25-28), but the launch must keep the GEMV's single warp (its bits change at 2 or 4 warps), where
  the copy is slower: 1-5 us saved a layer with 0-1 misses, up to 12 us lost with 2-5.
- The GDN gated norm inside the recurrent kernel: the kernel splits a head's 128 value columns over 4 programs and the
  norm needs all of them; one program per head spills at one warp and more warps reorder the sums over K. The conv
  inside it: its state is shared by the value heads of a key head, written by one program while others read it.
- Cold reads: overlapping one half of a chunk's all-reduce with the other half's compute first needs the dense GEMMs to
  keep their bits on fewer rows; hipBLASLt gives other bits for the shared expert's gate_up (704 x 2560) at 8192 rows
  and the hyper-connection's down projection (336 x 10240) at 4096 (29 of 32 shape and split pairs identical).
- The host side of a short prefill, measured with a forward timed on the host and then synchronized: the host finishes
  queuing a 30-280-token prefill only 0.1-8 ms before the GPU finishes it (a 16k-token chunk 170 ms before), so the host
  paces these prefills. Under cProfile (inflated times, 9 one-word turns): ~1,500 Triton launches a prefill took the
  largest share, about a third of it the AMD backend's specialization of every tensor argument at each launch
  (`get_tensor_specialization`, `is_within_2gb`, active while `AMDGCN_USE_BUFFER_OPS` is on); the ~47 `.cpu()` syncs of
  the predicted experts' copy plan, `torch.topk` (the prediction and the victim candidates, neither of which syncs on
  its own: 11 and 39 us of host time measured apart) and `F.linear` (hipBLASLt, ~44 us of host time a call) the rest.
  Measured apart, a Triton launch costs 25 us of host time, 3 us of which a leaner version of that specialization (the
  same keys) saves: ~4.5 ms of a short prefill, not kept. The CPU ran the `powersave` governor (`amd-pstate-epp`,
  `balance_performance`); its energy preference set to `performance` shortens the one-word turns by 10 % (section 32).
  Without the buffer ops (`AMDGCN_USE_BUFFER_OPS=0`, which drops that specialization and compiles every kernel without
  them; the same answers): short turns as before, decode 2.5-3 % slower for one request, a 133k-token cold read 85.9 s
  against 56.4.
- A short prefill's all-reduces on the host path instead of RCCL (`FREETOKEN_HOST_ALLREDUCE_MAX` at 1 and 4 MiB instead
  of 256 KiB, so that 200-800-token messages take it; the same answers): agent turns, one-word turns, a 133k-token cold
  read and decode as before (RCCL's ~230 us a call there were not on the turn's critical path).
- More VRAM for the expert cache from fewer Triton modules: 145 (7900 XTX) and 179 (7900 XT) distinct Triton kernels ran
  in a session (load, prefill, decode), all in use, at 2 MiB each (section 21).

## 32. Streams, queues and the GDN chain (2026-10-18)

The dropped leads of section 31 measured again, on NVFP4 `xtx-xt-4x262k` with the launcher's arguments, each arm a
fresh server on the image of section 31 with the changed files mounted: 15 greedy prompts, the agent conversation, the
one-word turns, three agent conversations at once, then one-request decode, a 133k-token cold read and four requests
decoding together (some arms with `FT_STATS_EVERY=50` and `FT_MOE_STATS=1`: the GPU time of each decode step, the
expert misses). Every arm gave the 15 greedy answers of the build without its change, except the first version of
the conv prototype below.

**A short prefill's fix-up copy overlapping the shared expert, on the prefill copy stream (kept,
`FREETOKEN_PREFILL_COPY_OVERLAP`).** Section 31 dropped it on a side stream of its own for a slower decode afterwards.
That slowdown is real but came only in the runs without the step statistics: four requests 22.7-23.5 tok/s each (three
runs, one with `GPU_MAX_HW_QUEUES=8`) against 34.4-35.7 in every run without the change, three agents decoding at 27.5
tok/s against 32.6, one agent's turns slowing from 30 to 20 tok/s over the conversation; with `FT_STATS_EVERY=50` (a
device synchronize every 50 decode steps), 34.2 and 35.6 (two runs). The cause was not found: a single-card
micro-benchmark (a graph of 1,800 tiny kernels replayed after 6,000 fork / join pairs around host copies on a side
stream, with and without device-wide synchronizes) barely slowed (at most 3 %, on the 7900 XT after 6,000). The same
copy on the prefill copy stream, which already carries the predicted experts (the next layer's prefetch then queues
behind it): four requests 35.4-35.5, three agents 31.5-33.4 tok/s, turns that do not slow, in two runs; one-word turns
0.239 -> 0.225-0.228 s (-5 %), agent turns (2-9) 0.407 -> 0.398-0.399 s (-2 %). EXL3 3.05 bpw, four runs against two:
agent turns 0.286 -> 0.274 s, one-word turns 0.225 -> 0.218 s, three agents' median turn 0.307 -> 0.286 s, but the
decode after them slower in every run (one request 93.4 -> 91.7 tok/s on average, four 37.2-37.3 -> 36.6-37.1 each; two
of the four runs with a host synchronize of the copy stream at each prefill, 36.8 and 37.1, not clearly better): for an
agent turn of a few hundred tokens a net loss, so the switch defaults to NVFP4 experts decoded on the GPU (`--hybrid`
was not measured); that default, through the tree's files, gave one-word turns of 0.219 s. The ~256k-token prompt of
section 31 through either checkpoint gave the same answer as there, read in 113.6 s (NVFP4) and 108.4 s (EXL3).

**The CPU's energy preference.** The host paces short prefills (section 31), and the reference machine's CPU (Ryzen
Threadripper 3970X, `amd-pstate` in active mode) ran the `powersave` governor with the energy preference
`balance_performance`. The same image and workload under each setting, two runs each: with the preference at
`performance` (under either governor, `performance` or `powersave`), one-word turns 0.240-0.242 -> 0.216-0.217 s
(-10 %), agent turns (2-9) 0.408-0.410 -> 0.403-0.404 s, three agents' median turn 0.403-0.405 -> 0.395-0.397 s; the
decode at four requests 34.4 -> 35.5-35.7 tok/s each, within what other runs with the original setting gave that day
(34.4-35.7); a 133k-token cold read 56.4-56.5 s either way. The governor itself made no difference. The power the
preference costs at idle was not measured.

**Decode prefetch of the next layer's predicted experts, again (dropped).** The patch of section 29 on today's cache
(the router of layer L+1 on layer L's input, the absent experts copied on a side stream inside the replayed graphs):
with the top-10 a row, the misses at four requests 2.07 -> 1.34 a layer and at one request (4 verify rows) 2.16 ->
1.33, but a step's GPU time 29.1 -> 37.0 ms and 22.7 -> 32.8 ms; with the top-4, 1.30 and 1.63 misses, 29.9 and 24.4
ms. Each miss costs ~2.7 ms a step at four requests and ~4.3 at one (a fit of the step's GPU time over its misses,
48 layers x ~55 us), so the prefetch cost more than the misses it saved. Two causes, measured apart:
- The side stream shares a hardware queue: `GPU_MAX_HW_QUEUES=4` (HIP's default) and each rank already has five
  compute queues (KFD's `/sys/class/kfd/kfd/proc/<pid>/queues`). With it, even the prefill slowed, where the prefetch
  does not run (the speed workload's ~1k-token turns 0.68-0.78 -> 0.87-1.07 s; the cold read 56.7 -> 58.1 s with the
  top-10); with `GPU_MAX_HW_QUEUES=8` the prefill was back (0.67-0.79 s, 56.8-56.9 s) and the baseline unchanged (34.8
  tok/s at four).
- On a queue of its own the copy runs beside the compute and slows it: a copy kernel reading host memory holds CUs for
  the whole PCIe transfer. With 8 queues, top-10: 32.6 ms a step at four requests, 35.7 at one; top-4: 31.4 and 36.1.
The copy engine (SDMA) takes no CUs, but its addresses come from the host, which a replayed graph does not have for
experts the GPU picks. Fork / join edges themselves are cheap: a graph of 1,800 tiny kernels replays 0.10-0.24 ms
slower with 48 pairs and 0.26-0.39 ms with 96 (2-5 us a pair, the XTX the lower).

**The GDN decode chain.** Per GDN layer, in a graph at rank 0's shapes (9 key heads): the conv 3.1-3.4 us, the
recurrent kernel 13.3 us for a 4-row verify step (5.4 for one row), the gated norm 2.7-3.0 us: ~0.73 ms of a ~21 ms
verify forward over 36 layers. Two prototypes, each bit-exact against the kernels they replace over random inputs on
both cards and in serving (15/15: the prefetch alone and both together; the conv alone ran only in its first
version, below):
- The conv in the int8 in_proj GEMV's store (a channel's conv state belongs to one program there, unlike in the
  recurrent kernel where several programs read it).
- The recurrent kernel loading step t+1's inputs while step t computes: 16.8 -> 14.7 us a layer (7900 XTX) and 19.1 ->
  16.1 (7900 XT) at 4 rows, 6.7 -> 5.3 and 7.9 -> 5.9 at one, ~0.1 ms of a step.
Both together against neither (one run each, the step's GPU time fitted over its misses): -0.01 +- 0.33 ms at one
request, +0.59 +- 0.08 ms at four. Not kept.
A trap found on the way: the recurrent kernel gives other bits when its q / k / v rows are 9,270 elements apart (the
in_proj output) instead of 5,760 (the conv's own output), 41k-161k elements of its output and state in each of 6
random cases (Triton specializes a kernel on whether its integer arguments are multiples of 16; the variant built for
the other stride gives other bits, why was not looked into). The first version of the conv prototype passed its
bitwise test (which checked the conv's outputs) and changed 7 of 15 greedy answers. A fused kernel's test has to run
the kernel after it on the layout it hands over.

## 33. Short turns: the host waiting at every layer (2026-10-18 and 19)

Measured on `xtx-xt-4x262k` with the launcher's arguments, each arm a fresh server, the changed files mounted over the
image of section 32, arms alternated. The workloads: 15 greedy prompts; one-word turns (9 turns on a 34k-token
conversation, 40-90 tokens to read each); an agent conversation (a 45k-token document, then 9 turns of 500-750 tokens to
read, each after a 200-token greedy answer); one request decoding 512 tokens; four requests of ~8k tokens each decoding
together (the window where all four decode); cold reads of 164k and 214k tokens. Every arm gave the answers of the build
without its change (the greedy prompts, both conversations and the cold reads).

**Where a short turn's host time goes.** A wrapper around Triton's launch counted ~1,550 Triton launches a one-word turn
and ~78 ms of host time inside them (~50 us a launch; ~18 us for a small kernel in isolation). Launching faster (below)
gained little on NVFP4, so the host was not the bound: under cProfile, then py-spy (native stacks), the main thread of
rank 0 spent ~74 % of its samples (EXL3) waiting in the `.cpu()` of the predicted experts' copy plan, ~47 a turn, ~3 ms
each, and `torch.topk`, which cProfile showed at ~1 ms a call, took 50 us in isolation and was barely sampled (a
cProfile artefact). Each of those syncs emptied the GPU's queue, which the host then refilled at its launch pace.

**Kept, the same bits:**
- The copy plan read once the layer's expert GEMMs are queued (`FREETOKEN_PREFILL_PREFETCH_DEFER`): the plan goes to
  pinned memory with an event; the host waits on that event, by then usually passed, while the GPU still has the layer's
  GEMMs to run. EXL3 one-word turns 0.246 -> 0.220 s, agent turns 0.508 -> 0.494 s; NVFP4 0.263 -> 0.258 s, agent turns
  unchanged.
- A shorter launch path for every Triton kernel (`FREETOKEN_TRITON_FAST_LAUNCH`): the compiled kernel is kept under a
  key computed from what Triton's specialization reads, and later launches call its C launcher with the same arguments
  (empty launch hooks passed as none). A small kernel 17.4 -> 9.7 us of host time; a check mode runs Triton's binder at
  every launch and compares the kernels (clean through every NVFP4 workload, the two long reads included). EXL3 one-word
  turns 0.235 -> 0.214 s alone, 0.188-0.195 s with the deferred plan read; NVFP4 0.260 -> 0.255 s.
- The prefill's int8 dequantization in one Triton pass (`FREETOKEN_INT8_PREFILL_DEQUANT`), bit for bit `torch.mul`'s:
  EXL3 agent turns 0.495 -> 0.487 s, NVFP4 0.718 -> 0.713 s.
- With the deferred plan read, the EXL3 fix-up copy on the prefill copy stream (`FREETOKEN_PREFILL_COPY_OVERLAP`,
  section 32) no longer slows the decode after it (two runs each: one request 92.0 against 91.9-92.1 tok/s, four
  36.1-36.2 each): agent turns 0.487-0.488 -> 0.475-0.476 s, one-word turns 0.197-0.201 -> 0.191-0.192 s. `auto` now
  takes it with EXL3 too.

All four, the image of section 32 against its tree (one run each, the answers identical):

| | NVFP4 3 x 262k | EXL3 3.05 bpw |
|---|---|---|
| one-word turns | 0.261 -> 0.251 s | 0.243 -> 0.192 s |
| agent turns | 0.718 -> 0.717 s | 0.507 -> 0.477 s |
| one request, decode | 92.9 -> 93.2 tok/s | 91.9 -> 91.9 tok/s |
| four requests, each | 33.8 -> 34.0 tok/s | 36.1 -> 36.2 tok/s |

NVFP4's agent turns wait on the routed experts crossing PCIe (section 31), which none of this changes. Fully cold reads
(on a fresh server after the greedy prompts, two starts each): EXL3 164k tokens 66.8 s on the image, 66.7 s with these
four, 96k 37.5 and 37.4 s. The same 164k-token read after the four-request workload, one of whose prompts is its
beginning (9,856 tokens found in the cache): NVFP4 65.7-66.1 s in three runs (the image, these four, these four in the
check mode), EXL3 62.6 s with these four but 67.2 s on the image, a slow run without a cause found (with the NVFP4 read
of 23.4 s below, two slow reads among the night's ~40).

**Cold reads: the GatedDeltaNet output's all-reduce behind the next block (kept, `FREETOKEN_GDN_AR_OVERLAP`).** The
delta rule of a 16k-token chunk already runs in 4096-token blocks. A one-request chunk of whole blocks now runs each
block's gated norm (with the whole chunk's rows per program) and output projection as the block finishes, and starts its
all-reduce while the next block's delta rule runs. hipBLASLt keeps a row's bits across row counts for some shapes only:
the output projections (2560 x 3456 / 2688 on the two ranks with GDN, 2560 x 3328 / 2816 for attention) gave the whole
call's rows in 4096-row blocks of 8k-16k-row calls on both cards, but 2560 x 2688 and 2560 x 2816 not in 2048-row
blocks, and the shared expert's 704 x 2560 not even in 8192-row ones. So the first chunk of each size checks a block
against the whole call on random data, and the ranks keep the overlap only if all pass (one all-reduce of the verdict).
The same bits (greedy prompts and both cold reads identical). Two runs each, with the MoE overlap above: NVFP4 164k
tokens 66.9-67.0 -> 65.8-65.9 s, 96k 37.5 -> 37.0 s; EXL3 96k 35.0-35.1 -> 34.7 s, 164k 62.7 -> 62.1 s in one run (the
other took 67.5 s, two of its 16k-token chunks 10 and 8 s instead of 6, the slowdown seen twice before).

**The decode after the overlapped all-reduces.** A validation of the whole tree with the decode measured after the cold
reads (the overlap series had measured only reads) found four NVFP4 requests decoding at 22.7 tok/s each against 33.9:
the compute stream waits for RCCL's on the GPU, the mechanism of the next paragraph. A device synchronize at the end of
every prefill forward that ran such an all-reduce restored them (34.2) but lengthened the 164k-token read by 1.4 s (67.3
against 65.9 s: the host prepares each chunk while the GPU sits idle); one before the next decode step instead restored
them (34.0) and left the reads unchanged (65.9 s). EXL3's four requests had not slowed (36.5).

**Cold reads: the PLE rows read ahead (kept, `FREETOKEN_PLE_PREFILL_PREFETCH`).** The GPUs' busy percentage sampled at
20 Hz during a 164k-token read showed both cards idle for 0.9-1.4 s at the start of every 16k-token chunk, ~16 % of the
read. A prefill reads its tokens' per-layer-embedding rows from the checkpoint on the host, before the forward starts:
1.09-1.21 s per 16,384 tokens on each rank (each reads every head). While a chunk of a longer prompt computes, a thread
now reads the next chunk's rows (assumed of the same length, or the rest of the prompt) into the other pinned buffer
through a store of its own (the store's calls release the GIL); the next chunk takes them when its tokens are exactly
those, and every chunk of a long read did. The scheduler hands the whole prompt to each chunk for it. The same rows, the
same bits. Fully cold reads, two runs each: NVFP4 164k tokens 65.9-66.0 -> 56.7 s, 96k 37.0 -> 32.0 s, the 45k-token
document of the agent conversation 17.0 -> 15.3 s; EXL3 3.05 bpw 62.1 -> 53.4 s, 34.7 -> 30.1 s, 15.9 -> 14.3 s; short
turns and decode unchanged. Against the image of section 32, with the two all-reduce overlaps: NVFP4 reads 70.2 -> 56.7
s and 39.5 -> 32.0 s, EXL3 66.9 -> 53.4 s and 37.6 -> 30.1 s (-19 to -20 %). The slow reads above (two 16k-token chunks
taking 10 and 8 s instead of 6) fit a slower disk read for a while, which the read-ahead now hides; not verified.

**Short prefills: the PLE rows read on the thread too.** The same read takes 46-66 ms for an agent turn of 500-740
tokens (EXL3, both ranks reading at once) and 4-10 ms for a one-word turn of 40-94 tokens, before the forward. A
micro-benchmark of the row store on the checkpoint's table (9,600 random rows of 160 bytes, one process) took 36-44 ms
with io_uring at a queue depth of 64, no less at 128-1024, and the same with 64 threads of `pread`: ~250k random reads a
second, taken then for the disk's limit for this pattern (section 34: it was the store, one thread driving one io_uring
ring, and its 8 KiB reads of the rows across a page, which halve this disk's rate; a container without
`seccomp=unconfined` falls back to 16 `pread` threads, 53 ms). So the thread now also reads the current prefill's rows,
while the host launches the layers before the PLE layer (layer 2), and the lookup waits for them; an eager fill on the
host thread (a decode step without a graph) waits for the thread and drops what it read ahead. Two runs each, the same
answers: EXL3 one-word turns 0.199-0.202 -> 0.194 s, agent turns 0.477 -> 0.465-0.466 s, the 45k-token document 15.8 ->
14.1 s; NVFP4 (one run) 0.253 -> 0.243 s, 0.714 -> 0.694 s, 17.0 -> 15.1 s; decode unchanged.

**One rank reads the PLE rows (kept, `FREETOKEN_PLE_PREFILL_RANK0`).** Each rank read every head, the same rows,
from the same disk at the same time. Now only rank 0 reads a prefill's rows and broadcasts their bytes (~1.5 MB for a
600-token turn) at the lookup; the same bytes, the same answers, and the four requests after it did not slow (EXL3
36.2, NVFP4 34.1 tok/s each). One run each: EXL3 agent turns 0.466 -> 0.435 s, one-word turns 0.192 -> 0.188 s, the
45k-token document 14.1 -> 13.8 s; NVFP4 0.693 -> 0.662 s, 0.243 s either way, 15.1 -> 14.7 s.

**NVFP4: copy ahead only the experts several tokens predict (kept, `FREETOKEN_PREFILL_PREDICT_MIN_VOTES`).** Counted
in the plans, an NVFP4 agent turn of 500-750 tokens copies 8,400-10,000 predicted and ~1,600 missed experts on rank 0,
14.5-17 GiB at 1.46 MiB each over the turn's 0.65-0.70 s: PCIe is busy throughout, so an expert copied ahead that the
next layer does not route to costs the turn its transfer time. Keeping only the experts that at least 2 tokens predict
(the others go through the fix-up copy if routed; only copies, the same answers), two runs each: NVFP4 agent turns
0.663 -> 0.624-0.625 s, one-word turns 0.244-0.245 -> 0.228-0.229 s; at 3, 0.616 and 0.235-0.236 s. EXL3, whose turns
lean on the copies overlapping the compute, lost at 2 (0.435-0.436 -> 0.440-0.441 s, 0.191-0.193 -> 0.199-0.205 s): the
default takes 2 for NVFP4 only.

**The decode slowed by copies on another stream (cause narrowed, section 32).** The fix-up copy put back on a third
stream of its own (NVFP4): one request decoded at 88.4 tok/s after the turns against 92.9-93.0, the four requests after
long prefills unchanged (33.9). With a device synchronize after every prefill: 92.7. With a host synchronize of that
stream alone after every prefill: 92.9 and 33.9. With the compute stream waiting on it on the GPU (`wait_stream`) after
every prefill instead: 92.9 for one request but 22.3 tok/s each for four, the slowdown of section 32. So a cross-stream
wait on the GPU leaves the decode graph replayed after it slow until the host synchronizes the other stream; what in the
HIP runtime keeps that state was not found (its stream wait inserts a marker that depends on the other queue's command).

The NVFP4 read of 23.4 s: an arm read the agent conversation's 45k-token document in 23.4 s instead of ~18, two
16k-token chunks taking 10 and 7 s instead of 6 and 5, and decoded the answer at 87 tok/s instead of 93, around
midnight; the system journal shows nothing.

**Cold reads: the MoE all-reduce block by block (kept, `FREETOKEN_PREFILL_AR_OVERLAP`).** A 16k-token chunk's routed
experts already run in token blocks (8192 tokens with NVFP4, 4096 with EXL3) whose rows are independent. Each block now
finishes its rows (the fused epilogue, with the warps the whole chunk would take, since the gate's sum follows them) and
starts their all-reduce on RCCL's stream while the next block computes; the layer waits for all of them at its end.
Every element is the same epilogue, then the sum of the two ranks' values: the same bits (greedy prompts and both cold
reads identical). Two runs each, on a fresh server after the greedy prompts, nothing of either text cached: NVFP4 164k
tokens 70.1-70.2 -> 67.0-67.1 s, 96k 39.4-39.5 -> 37.6 s (-4.5 %); EXL3 66.6-66.7 -> 62.7-62.9 s and 37.4 -> 35.0-35.1 s
(-6 %).

**Cold reads: the indexer's scoring 16 rows at a time (kept, `FREETOKEN_QSA_SCORE_ROWS`).** The QSA indexer scores
every query row against every visible compressed key (a quarter of the tokens) in a program of its own: a [64 x 128]
x [128 x 16] dot a tile, its 4 heads padded to 16, the keys reloaded for every row; 47.3 ms a layer for a 16k-token
chunk 150k deep on the 7900 XTX, 12 ms at 30k (12 attention layers a chunk). A kernel for a prefill of one request
scores 16 rows a program, one [64 x 128] x [128 x 64] dot a tile for their 64 (row, head) pairs: 11.1 ms. A score
keeps its bits: the dot's K order is the same, and of three explicit orders for the sum of the 4 heads, (h0 + h2) + (h1
+ h3) is the one the padded reduction gives (the two others differed in ~14k and ~12k of ~1M scores). Bit for bit
against the one-row kernel from 70 to 16,384 rows and 0 to 150k deep on both cards; the same answers. Cold reads: NVFP4
164k tokens 56.0-56.2 -> 52.7-53.7 s, 96k 31.5-31.6 -> 30.5 s (two runs each); EXL3 52.6 -> 49.7 s and 29.5 -> 28.5 s
(one); short turns unchanged.

**Long agent turns.** Turns of ~2k tokens (beyond `FREETOKEN_PREFILL_CACHED_MAX`, they stream whole layers) took 1.19
s (NVFP4) and 0.89 s (EXL3); with the slot cache up to 4096 tokens, 1.17 and 0.90 s, the same answers: the threshold
stays.

**Dropped:** the row store's reads through the page cache instead of O_DIRECT (a micro-benchmark on the table: on text
not read before, 600-token turns 31-55 ms against 23-40 and 16k-token chunks 650-780 ms against 510-710; faster only on
rows already read, for RAM the experts and the K/V need); a deeper io_uring queue (above).

**The night's build against the image of section 32**, `xtx-xt-4x262k` with the launcher's arguments, two fresh
servers each, alternated, every answer identical (the greedy prompts, both conversations, the 164k and 96k-token
prompts, and with NVFP4 the four requests at once; with EXL3 two of those four differ between two starts of the image
itself as well, batching, so they are not compared):

| | NVFP4 3 x 262k | EXL3 3.05 bpw |
|---|---|---|
| one-word turns at 34k (40-90 tokens to read) | 0.256-0.261 -> 0.227-0.228 s | 0.237-0.243 -> 0.185-0.197 s |
| agent turns at 45k (500-750 tokens to read) | 0.718 -> 0.621-0.622 s | 0.509-0.512 -> 0.432-0.434 s |
| a 45k-token document, cold | 18.2 -> 14.5 s | 17.2 -> 13.6 s |
| a 96k-token prompt, cold | 39.5-39.6 -> 30.3-30.4 s | 37.5-37.6 -> 28.4-28.7 s |
| a 164k-token prompt, cold | 70.2-70.3 -> 52.7-52.8 s | 66.8-67.0 -> 49.5-49.9 s |
| one request, decode | 92.8-92.9 -> 93.3 tok/s | 91.2-91.7 -> 91.7-92.1 tok/s |
| four requests, each | 33.8-33.9 -> 33.8-34.0 tok/s | 35.9-36.2 -> 36.0 tok/s |

One card (`xtx`, EXL3 3.05 bpw, 131k, one request at a time; the launcher's third entry), two starts each, the same
answers: one-word turns 0.408-0.411 -> 0.388 s, agent turns 1.070-1.074 -> 1.009-1.010 s, the 45k-token document 24.6
-> 23.4 s, the 96k-token prompt 53.4 -> 50.6-50.7 s, decode 51.3-51.6 -> 52.2-52.4 tok/s (there the all-reduce and
rank-0 changes do not apply; measured before the indexer's 16-row scoring, not run on one card).

## 34. Disk reads on four rings, tail tiles, the input embedding in RAM (2026-10-19)

Measured on `xtx-xt-4x262k` with the launcher's arguments (NVFP4 and EXL3 3.05 bpw, both with the NVFP4 entry's 3 x 262k
pool, `--kv-reserve-tokens 787500`), each arm a fresh server, the changed files mounted over the image of section 33,
arms alternated, two runs each. The workloads: 13 greedy prompts; one-word turns on a 55k-token conversation; an agent
conversation (a 45k-token document, then 9 turns of ~520 tokens to read); one request decoding 512 tokens; four requests
decoding together; cold reads of 40k, 113k, 159k and 210k tokens. Every arm gave the answers of the image (the four
requests at once excepted, now and then in a run of either build: the load's timing). The final build below is the image
with the PLE rows on four rings and the tail tiles, then the PLE row cache; the input embedding in RAM came after it
(its own paragraph, measured against the final build). The figures are the final build's, the image against it, unless
stated.

**The PLE rows: the store, not the disk.** Section 33 took ~250k random reads a second for the disk's limit. A C
benchmark of `O_DIRECT` reads on the checkpoint's table (512 B or 4 KiB, the same) gave ~190k at 16 threads, 416k at 64,
490k at 128; with io_uring, one ring on one thread stayed at 240-290k whatever its depth (deeper was slower: 64 -> 256
entries, 33 -> 41 ms for 9,600 reads), while four rings on four threads read 9,600 rows in 19.5 ms. The engine's store
with four rings, though, still took ~40 ms: on this disk (the reference machine's NVMe, PCIe 3 x4), reads spanning two 4
KiB units halve the rate of the whole batch, whatever their share (all 4 KiB: 19.3 ms; 2.5 % of 8 KiB reads among them:
41.7 ms; 1 KiB reads at 512-byte offsets: 48 ms, at 1 KiB offsets: 19.0 ms), and the store read a row that crosses a
page as one 8 KiB read. Read as its two pages instead, those rows were still slow: the block layer merged the two
adjacent reads back (~750 merges in 7 batches); with another row's read between them, no merge. Kept
(`FREETOKEN_ROW_STORE_RINGS`, 4 rings, the second page deferred): 9,600 random rows NVFP4 42.1 -> 25.9 ms, EXL3 40.4 ->
20.7 ms, the bytes of the previous store. In serving (the rings alone, EXL3): agent turns 0.523-0.526 -> 0.494 s (one
ring on the new code 0.522), a 40k-token cold read 12.46-12.49 -> 12.09-12.11 s, longer reads within 1 %; NVFP4 agent
turns 0.741 -> 0.730 s with the tail tiles below (PCIe-bound: section 33's 14.5-17 GiB of expert copies a turn);
one-word turns within their spread (0.191-0.202 s between arms of one build), decode unchanged.

**Tail tiles.** An 8192-token NVFP4 block gives each expert ~160 rows, which the gate-up GEMM's 128-row tiles cover with
two tiles, the second mostly padding (section 29: 62 % full). Each expert's full 128-row tiles now run in one launch and
its remaining rows in 32-row tiles in a second (`moe_align_split` sorts the routes into both; the down GEMM, 64-row
tiles before, takes the same split): the same K order, so the same bits (`tests/moe/test_prefill_tail_tiles.py`, against
the one-launch tiles). One MoE block, this model's shapes: 20.3 -> 16.5 ms (XTX slice), 19.8 -> 16.6 ms (XT), 16.4 ->
15.1 ms with skewed routing. In serving (`FREETOKEN_NVFP4_PREFILL_TAIL_TILES`, with the rings): cold reads of 113k
tokens 36.2-36.3 -> 34.1-34.2 s, 159k 51.5-51.6 -> 48.6-48.8 s, 210k 68.7 -> 64.9-65.0 s, 40k 13.4 -> 12.4-12.5 s.
EXL3's prefill tiles are 32 rows (80 routes an expert in a 4096-token block): full 64-row tiles plus 32-row remainders
gave the gate-up GEMM 4.27 -> 3.74 ms (XTX) and 4.03 -> 3.67 ms (XT), the down GEMM nothing; the whole MoE block on 96
dumped 4096-token blocks 919 -> 884 ms (XTX) and 790 -> 762 ms (XT); kept for the gate-up from 3072 tokens
(`FREETOKEN_EXL3_PREFILL_TAIL_TILES`, the final build with it off -> on): cold reads of 113k / 159k / 210k tokens 33.8 /
48.1 / 64.2 -> 33.2-33.3 / 47.4-47.5 / 63.1-63.2 s. The image -> the final build, EXL3: 40k 12.5 -> 11.9 s, 113k
34.0-34.1 -> 33.2-33.3 s, 210k 64.4 -> 63.1-63.2 s.

**Through the depth sweep** (`rdna3/bench/depth_sweep.py`, a conversation grown to 260k with blocks of new text, median
of two passes, the image -> the final build without the PLE row cache below; PP of the new block in tok/s, a following
~1k-token agent turn in s; decode within 0.4 tok/s at every depth but EXL3's first, 93.8 -> 92.6):

| Depth | NVFP4 PP | NVFP4 agent turn | EXL3 PP | EXL3 agent turn |
|---:|---:|---:|---:|---:|
| 8.8k | 2653 -> 2820 | 0.63 -> 0.60 | 2809 -> 3059 | 0.50 -> 0.46 |
| 16.0k | 2635 -> 2953 | 0.65 -> 0.62 | 2941 -> 3219 | 0.52 -> 0.48 |
| 25.0k | 2689 -> 2979 | 0.67 -> 0.66 | 3000 -> 3313 | 0.52 -> 0.48 |
| 40.0k | 2948 -> 3297 | 0.59 -> 0.56 | 3124 -> 3446 | 0.52 -> 0.48 |
| 60.0k | 2851 -> 3093 | 0.63 -> 0.62 | 3074 -> 3321 | 0.53 -> 0.49 |
| 86.9k | 3081 -> 3279 | 0.62 -> 0.59 | 3196 -> 3416 | 0.53 -> 0.49 |
| 122.9k | 2949 -> 3148 | 0.63 -> 0.61 | 3145 -> 3325 | 0.54 -> 0.50 |
| 164.9k | 2964 -> 3165 | 0.64 -> 0.61 | 3137 -> 3295 | 0.55 -> 0.51 |
| 206.9k | 2902 -> 3091 | 0.65 -> 0.63 | 3062 -> 3217 | 0.55 -> 0.52 |
| 259.8k | 2871 -> 3028 | 0.66 -> 0.63 | 3010 -> 3138 | 0.54 -> 0.52 |

**One card** (`xtx`, EXL3 3.05 bpw, a 131k context, the launcher's one-card entry), the image -> the final build, two
runs each: the same answers (40 of 40), cold reads of 40k tokens 21.0-21.1 -> 20.5-20.6 s and 113k 59.1-59.2 ->
58.1-58.2 s; one-word turns 0.429-0.431 s, agent turns 1.119-1.121 s and decode 52.6-52.8 tok/s unchanged.

**Three agents at once**, the final build, NVFP4 with the 3 x 262k pool: three conversations sampled at 0.6 (a 30k-token
document, then tool outputs of 0.5-8k tokens and answers of up to 300 tokens, each restarted near 230k) for 25
minutes: 476 requests, no error, contexts up to 234k tokens; the container's memory 95.66 -> 95.81 GiB, VRAM steady
(XT 19,599 of 20,464 MiB, XTX 23,424 of 24,560); 20.5 GiB of the host's RAM left available. The same on EXL3 3.05
bpw (the same 3 x 262k pool): 595 requests, no error, contexts up to 237k, 73.25 -> 73.37 GiB, 43 GiB left available.

**Starts.** One start of the final build's EXL3 server crashed at its first request (an illegal memory access on the XT,
`sq_intr` in the kernel log, surfaced in the copy-plan code of the first prefill), like section 35's rare start fault;
the cause was not found. 12 more starts of the image and 12 of the final build (start, one answer, stop) all came up:
that night's EXL3 starts, 16 of 16 for the image and 24 of 25 for the final build.

**A RAM cache of the PLE rows** (`FREETOKEN_PLE_ROW_CACHE_ROWS`, 8-way, 1M rows, 176 MiB with NVFP4, 118 MiB with EXL3):
in a simulated agent session reading this repository's source files one a turn, ~40 % of a turn's rows had been read
by an earlier turn, and a 1M-row cache kept them all. Put first in the store of the eager and graph fills, it changed
nothing (EXL3 agent turns 0.495 against 0.496 s): a prefill's rows are read by the thread's own store. In that store,
two runs each, the same answers: turns of ~2,600 tokens on a 45k conversation NVFP4 1.208-1.214 -> 1.160-1.163 s,
EXL3 0.987-0.990 -> 0.940-0.941 s; ~8k-token turns NVFP4 2.89-2.91 -> 2.81 s, EXL3 2.70 -> 2.62 s (median of four,
the last two of which gained nothing in either); EXL3 agent turns 0.492-0.494 -> 0.485-0.488 s, NVFP4's unchanged;
cold reads unchanged.

**The input embedding in host memory (kept, `FREETOKEN_EMBED_HOST`).** The vision tower already stays in host memory;
the input embedding (248,320 x 2560 bf16, not tied to the LM head: 606 MiB a rank at TP=2) was the largest tensor left
in VRAM that a token reads one row of. In pinned, device-mapped host memory (`kernel/host_mapped.py`, as the K/V of
section 22) the same lookup kernel reads the same bytes across PCIe. The table moves after the int8 compaction, so the
memory reading that sizes the pools sees it gone: 8,768 -> 9,255 expert slots with NVFP4, 11,554 -> 12,095 with EXL3.
Two runs each, every answer the image's (the four requests excepted in one EXL3 run, as above): NVFP4 agent turns
0.728-0.729 -> 0.707-0.711 s, ~2,600-token turns 1.162-1.165 -> 1.142-1.148 s, decode 89.7-89.9 -> 91.2-91.7 tok/s, four
requests 29.0-29.3 -> 30.5 tok/s each; EXL3 agent turns 0.485-0.487 -> 0.472 s, ~2,600-token turns 0.940-0.942 against
0.943 s, decode 93.5-93.6 -> 94.3-94.5 tok/s, four requests 36.2 -> 36.8-37.1. Cold reads unchanged (NVFP4 40k / 113k /
159k tokens 12.45-12.53 / 34.14-34.16 / 48.63-48.66 s without, 12.47 / 34.14 / 48.61 s with in one run; the other run's
first two reads took 13.58 and 34.70 s, not seen again). One card (`xtx`, EXL3 3.05 bpw, 131k), two runs each, the same
40 answers as the image: 4,783-4,793 -> 5,459-5,468 slots, one-word turns 0.432-0.433 -> 0.410-0.413 s, agent turns
1.120-1.126 -> 1.083-1.086 s, decode 52.3-52.7 -> 55.4 tok/s, cold reads of 40k / 113k tokens unchanged (20.50-20.70 /
58.17-58.87 against 20.59-20.76 / 58.18-58.66 s); the container's locked memory 1.18-1.19 GiB more (the table), which
`serve.sh --check` now counts.

**Dropped:**
- Lossless compression of the NVFP4 experts for the PCIe copies: the FP4 codes carry 3.968 bits of entropy of 4, the
  FP8 block scales 4.64 of 8: 5.4 % fewer bytes at most, before the cost of decoding them on the GPU.
- A smaller VRAM margin for more expert-cache slots (8,768 slots with the 3 x 262k pool before the embedding moved):
  at start the XT has 2.9 GiB free, of which a 255k-token read takes all but ~1.1 GiB (section 29); ~400 more slots
  (+5 %, estimated) would leave a deep read ~0.5 GiB. (The embedding's move above gave as many from VRAM the weights
  held instead.)
- `--hybrid` made bit-exact: on this profile (K/V in RAM) it did not gain ([cpu-experts.md](cpu-experts.md)).
- NVFP4 prefill tiles chosen on the model's own routing. The tile tables had been swept with random routing; this
  model's routing (dumped from serving, then the first M tokens of 48 real 8192-token blocks replayed, both cards) puts
  the 16- -> 32-row switch at ~352 tokens instead of 640 (32-row tiles 6 % faster at 512, 10 % at 640) and the 64-row
  tiles nowhere (1.2x the 32-row time at 1536-2048; 128-row tiles with the tail tiles win from ~2048). In serving, the
  same answers but no change: agent turns 0.729-0.730 against 0.729 s, 2,600-token turns 1.197-1.204 against
  1.189-1.201 s (PCIe-bound, section 33). EXL3's switch at 512 tokens is where its replay put it (~480).
- The 12 attention layers' all-reduce block by block (estimated ~1.7 % of a cold read from section 29's trace): the
  sparse attention run by 4096-row query blocks (the kernel computes each row alone, with one tile profile above its
  small-batch limits: the same bits, checked on both cards with bf16 and int8 K/V), each block's gate, `o_proj` and
  all-reduce started behind the next block's attention. The same answers; two runs each, cold reads of 113k / 159k /
  210k tokens NVFP4 33.89-33.94 / 48.33-48.38 / 64.55-64.66 -> 33.84-33.86 / 48.20-48.23 / 64.36-64.45 s, EXL3
  33.05-33.09 / 47.12-47.13 / 62.85-62.90 -> 33.01-33.05 / 47.00-47.09 / 62.66-62.82 s: 0.1-0.3 %, too little for ~70
  lines (why it fell short of the estimate was not measured).
- The GatedDeltaNet delta rule: a 4096-token block on the 7900 XTX's 9 / 27 heads takes 1.77 ms (`recompute_w_u` 0.46,
  the state recurrence 0.39, the output 0.31, the fused `kkt` + solve 0.23 ms), ~5 % of a 16k-token chunk over 36
  layers (estimated). A sweep of the first three kernels' tile widths, warps and stages (every variant bit-identical)
  found the shipped launches fastest.
- EXL3 MoE tiles on the model's own routing: 48 dumped agent-turn calls (540 tokens, the XTX's 384 columns), one
  parameter of either GEMM at a time (tile width, warps, stages, grouping; all bit-identical): none faster than the
  shipped tiles (111.09 ms; within 0.3 % or slower).
- 256-row NVFP4 tiles: no gain (a 256 x 128 tile would also exceed the 64 KiB of LDS).
- A RAM cache in the decode fills' store as well: on text that repeats the conversation, the fill on the step's path
  0.86 -> 0.38 ms, the decode unchanged (one request 93.1 / 94.2 against 92.3 / 94.5 tok/s, four requests 38.0 each in
  both): the fill hides behind the GPU's work.
- Graph-capturing the MTP head's pass (not attempted): at one request (NVFP4) the head and its chain run eagerly, host 3
  + 2.6-5.7 ms a step against GPU 1.2 + 1-2 ms, but the host stays ahead; the GPU idles 0-6 % of a 24-25 ms step, which
  bounds the gain.
- Waiting block by block for the first chunk's PLE rows (not attempted): the layers after the first need the whole
  chunk's rows.
- Thermal throttling: through cold reads the NVMe peaked at 63 °C, the XT's junction at 90 °C, the XTX's at 76 °C.
- The expert cache after agent turns: the decode right after a turn 88.4 against 88.3 tok/s, no pollution.
- The hyper-connection mix (11 % of a 16k-token chunk, NVFP4): hipBLASLt's GEMMs at 673-813 GB/s, and fusing them would
  change their sums.
- `rocprofv3 --attach` is not usable here (it needs a rocprofiler-register built with attach support), and a server
  launched under `rocprofv3` lost its workers (signal 6) ~2 minutes into the trace.
- The QSA top-k (the indexer's selection, ~2 % of a 16k-token chunk): a 512-row call at ~60k visible blocks takes
  0.6-1.0 ms, reading its scores at ~125-145 GB/s. Its launch geometry (warps of both kernels, 4- to 8-bit digits, 4k
  or 8k slices; every variant the same indices) found the shipped one the fastest (the best within 1.6 %); a faster
  select needs another algorithm, for ~1 % of a cold read: not attempted.

## 35. Upstream up to cc56b6b (2026-10-20)

29 commits merged, the authors' kept. Two of them change NVFP4's numbers on purpose: the global scales stay fp32
(`0db5023`; as fp16 most were subnormal, up to 1.7e-3 relative error per row) and the Triton prefill GEMM is rewritten
(`03c2c9c`: the weight tile is the MMA's A operand, one dot per K step instead of two). EXL3 kept its bits: 42 of 42
answers the same in two runs on `xtx-xt-4x262k`. Nine files conflicted; upstream had moved the NVFP4 launches to
`kernel/nvfp4_moe.py`, given the CPU executor a format per layer and rewritten the qwen4_exp dense reader, and this
tree's pieces moved onto those. The CPU and GPU suites passed where they passed before, yet the first two NVFP4 starts
failed while loading the experts: upstream's expert reader now hands the family hook a `layers` argument, and its
pieces now carry the experts' static activation scales, which the tensor-parallel slicing had no rule for (both fixed
with a test each; only a real load reaches them).

The RDNA3 prefill tiles had been swept on the old GEMM. On the new one, with the same tiles, an 8192-token MoE block was
1.8x faster (19.4 -> 10.7 ms) but a 2048-token one 1.4x slower (8.2 -> 11.6 ms; the XT with the XTX's 352-wide slice):
the 8-warp 1-stage tile of 1537-3071 tokens suits it badly. A grid of 72 tiles at one K step (all of them the same bits)
per size and card found 4 warps and 2 stages best from 1537 tokens, and the 64-row tile ahead of the 128-row one with
tail tiles up to 6144 tokens (from 7168 the 128-row one wins). The grid's first-timed config read 1.5x slow (the same
tile timed again later was fast), so the table was settled by an interleaved A/B of whole MoE blocks. Those ran each
card on the other's slice width (`HIP_VISIBLE_DEVICES=0` is the XT here, 1 the XTX); run again on each card's own slice:
1600-2560 tokens 2.8-3.0x faster, 1280 1.4-1.5x, 32 1.3-1.4x, 4096-6144 1.1-1.2x, 7168-8192 even, the same bits
everywhere. Whole model, `xtx-xt-4x262k` (NVFP4 with a 3 x 262k pool, EXL3 3.05 bpw with 1,050,000 tokens), the build before
(three NVFP4 runs, two EXL3) against this one (two each): NVFP4 cold reads of 40k / 113k / 159k / 210k tokens
12.39-12.42 / 34.11-34.25 / 48.66-48.78 / 64.77-64.93 s -> 10.89-10.91 / 29.87-29.90 / 42.57-42.58 / 56.85-56.95 s
(-12 %), decode 90.1-91.1 -> 92.4-92.5 tok/s, four requests 30.2-30.4 -> 31.2-31.3 tok/s each, agent turns
0.710-0.720 -> 0.709-0.728 s, 2k-token turns 1.142-1.153 -> 1.132-1.133 s; NVFP4 greedy outputs 17 of 42 the same (a
sample of the others, read by hand, says the same in other words). EXL3 42 of 42 answers the same in both runs, agent
turns 0.490-0.497 -> 0.485-0.487 s, decode 92.4-93.4 -> 92.8-93.7 tok/s, cold reads within 0.5 %. NVFP4 expert slots
-0.5 % (9,255 -> 9,209: the globals take 4 bytes). The CPU suite fails only where it failed before, minus one; the GPU
tests (the 93 files with GPU-gated tests, each card) only where they failed before, minus three.

## 36. Verify rows from the head's confidence, sampled drafts, three and four requests verifying (2026-10-20 and 21)

A published inference engine schedules its verification from a trained confidence head that predicts each draft's
acceptance. Qwen3.8-Flash-Next's MTP head has none (no such tensor in the NVFP4 or EXL3 checkpoints); its own top-1
probability stood in. Runs on `xtx-xt-4x262k` with a 300k-token pool, 14 prompts x 1024 tokens greedy and at T 0.6.

- **The head's probability predicts acceptance.** Over ~10k verify steps per quant, the head's top-1 probability (T 1)
  predicts whether the first draft is kept far better than the running per-position acceptance that picks the rows
  (Brier 0.126 against 0.177 greedy, 0.133 against 0.197 at T 0.6; EXL3 the same), and it is close to calibrated
  (drafts at 0.5-0.7 kept 0.67 of the time, above 0.999 0.997). The first draft's probability also predicts the deeper
  ones (draft 1 kept 0.58 of the time after a first draft below 0.5, 0.92 after one above 0.999; the running acceptance
  stays at 0.72-0.82). Computed in a fixed-order kernel so both cards get the same bits and pick the same rows.
- **But the choice comes too late to pay.** The rows of the next step are chosen while the GPU still drafts. Waiting
  for every draft's probability (the head always drafting 3): 1.8-3.3 % slower (the host waits 2.7 ms a step, a 2-row
  step pays the whole chain: 25 ms against 20-21). Waiting for the first draft only (0.6 ms, while the GPU runs the
  chain), the running acceptance past it: greedy +0.4-0.5 % on NVFP4 (101.8-102.0 against 101.0-101.4 tok/s), +0.1-0.4 %
  on EXL3, the same answers. At T 0.6 the arms read +2 % (NVFP4) and -1.1 % (EXL3): the row count changes how the
  draws fall, so each arm samples other texts; a simulation with the real step costs (the chain is drafted before the
  choice, so fewer rows save only the rows) gives +0.4-0.7 % for both. A logistic fit of the deeper drafts on the
  first one: +0.4 % greedy and -2.6 % at T 0.6 against the first draft alone. A 1-row step when the first draft is
  doubtful: with the head such a step takes 17.6 (NVFP4) and 18.2 ms (EXL3) against 18.6 and 19.9 at 2 rows, so it
  pays only below ~0.06-0.09. All dropped.
- **Real step costs.** The period of a step (one request, every step at the same row count; NVFP4 from the decode rate
  and the tokens kept per step, EXL3 3.05 bpw on the host clock): NVFP4 18.6 / 22.6 / 25.6 ms at 2 / 3 / 4 rows, EXL3
  19.9 / 23.5 / 27.6: about the ratios of the shipped table. A table built from the GPU events (forward + verify + head
  + chain) has the wrong shape, as the captured step's GPU time includes its wait for the host's PLE fill: 1.5-3.3 %
  slower with it. Both cards stay 99 % busy (`gpu_busy_percent`) at one request.
- **Sampled drafts at T 0.6.** `FREETOKEN_SPEC_SAMPLED_DRAFTS=1`, measured only at T 1.0 before: a drawn draft is kept
  with probability Σ min(p, q), the argmax with p(argmax), and the arms kept more tokens per step in almost every
  512-step window. 14 prompts x 1024 tokens NVFP4 91.0-91.5 -> 94.0-94.2 tok/s, EXL3 92.9-93.0 -> 97.0; 5 prompts at
  28k-130k of context NVFP4 82.6-82.7 -> 82.8-83.0, EXL3 85.8-85.9 -> 91.9; 2 prompts at 75k-89k EXL3 88.5-88.6 ->
  87.9-88.0; two requests NVFP4 62.3 -> 61.5-61.6, EXL3 65.0 -> 65.9; single prompts -9 to +18 % (other texts). Greedy
  answers unchanged. Now the default.
- **Three and four requests verifying, 2 rows each.** Verifying was off past two requests (4 rows each at four requests
  had given 26.3 tok/s against 35.35). On this build, four requests with every step at one row count (NVFP4, one run
  each): 31.3 tok/s each at 1 row, 35.2 at 2, 31.7 at 3, 30.4 at 4 (GPU time per step 30.5 / 49.0 / 73.3 / 86.8 ms). So
  `FREETOKEN_SPEC_BS_DEEP`: past it, up to `FREETOKEN_SPEC_BS_MAX`, a step verifies 2 rows each. `xtx-xt-4x262k` with 4
  / 2: three requests +6.4 % (NVFP4) and +8.9 % (EXL3), four +3 to +6 % greedy (at T 0.6 -0.3 % and +2.5 %). `xtx-xt`
  (K/V in VRAM, smaller expert cache, one run each, two series): three requests NVFP4 +2 % and +17 %, EXL3 +5 % and +12
  %, four at T 0.6 -1 to +8 %; its sweep to 260k the same and no out-of-memory with the two more graphs. `xtx-xt-1m`
  (EXL3): +10 % at three, +4-12 % at four. All three take 4 / 2. The same greedy answers on prompts read apart
  ([benchmarks.md](benchmarks.md#several-requests-and-agents-at-once-xtx-xt)).
- **A capture-order bug on the way.** The first runs of that setting stopped on an illegal memory access as soon as two
  requests verified 4 rows. The verify step's rollback buffers (GDN states, PLE conv and n-gram windows) are sized by
  the eager forward before each capture and grow with the requests, freeing the old ones; the captures ran the most
  rows first, so 2 requests x 4 rows was captured before 4 requests x 2 rows, whose eager forward freed the buffers the
  first graph writes. The shapes with the most rows, then the most requests, now run eagerly before any capture. No
  profile before captured a shape with more requests after one with fewer (`FREETOKEN_SPEC_VERIFY_MS=2,4` with
  `FREETOKEN_SPEC_BS_MAX=2` would have).
- **The rare start fault again.** Once in ~70 starts over the night (NVFP4, the first prefill, the XTX this time: a
  write permission fault at a host-range address, surfaced in the expert cache's copy plan), with the changes above
  switched off. The test servers' memory maps were saved at each start since, for the next one to name its buffer.

## 37. A decode step's glue: the LM head's candidates, one-kernel sampling, the router, the all-reduce (2026-10-21 and 22)

A kernel trace of a decode step (`xtx-xt-4x262k`, EXL3 3.05 bpw, one request at T 0.6, `rocprofv3 --kernel-trace`):
1,833 kernels a step, 346 of them small torch ops (1.25 ms of kernels and 1.3 ms of gaps in a ~27 ms traced step), most
of them the sampling of the verify step and of each draft; after each of the ~3 LM-head passes a step makes (the verify
step and the drafts, up to 4), the two cards all-gathered the whole vocabulary's logits (113-122 µs pure on the XT for 3
rows, 51-55 for one; the XTX waited 180 and 114, its half of the head being done 75 µs earlier: 366 against 440 µs). All
measurements below on the same setting, 12 prompts x 1024 tokens at T 0.6 (`calib6`), 3 greedy 512-token answers
(`speed`), four requests started 1.5 s apart (`fourst` greedy, `four6` T 0.6), arms interleaved with the image.

- **The candidates instead of the logits** (`FREETOKEN_SPARSE_LOGITS`). Each rank takes the k largest logits of its own
  shard (the request's top-k, 20 by default; 1 for greedy rows), the ranks all-gather those (2k floats a row), and a
  kernel merges them. What decides a sampled token is their order, which has to be `torch.topk`'s over the whole row:
  the int8 LM head returns bf16 logits, so a row's top 20 often hold equal values, and topk orders them differently by
  the number of rows in its call (one row: ascending index; 2 to 16 rows: the layout of its gather sorted by its bitonic
  sort; checked over rows of many equal bf16 values). The merge reproduces both. The first version, ascending ids for
  every call, gave the same greedy answers but other sampled texts (0 of 12 identical), as did the bitonic order alone
  (drafts are drawn over one row a request). Not followed: `torch.topk` puts -0.0 below +0.0 and NaN above everything,
  where `torch.argmax` takes them as equal or ignores them; the merge keys follow argmax (a review found it; a logit of
  exactly 0 or NaN among a row's 20 largest is not expected from the head, not checked).
- **The draw in one kernel** (`FREETOKEN_SAMPLE_KERNELS`). On the candidates, a verify step's speculative sampling
  (~60 torch ops) and a draft's draw (~25) run as one Triton kernel each, with the torch ops' bits: torch's `exp` and
  `log` on gfx1100 are LLVM's AMDGPU expansions of `llvm.exp.f32` / `llvm.log.f32` (`v_exp_f32` / `v_log_f32` with
  `fma` corrections; Triton's own `exp` and the device library's differ in 88 % of values, compiled for flushed
  denormals), written out with inline assembly and FMA contraction off; softmax sums per lane then a butterfly over
  min(next power of two, 32) lanes; a row sum of K <= 64 over the largest power of two <= K lanes (at most 32), then
  adjacent pairs (wider rows change their order with the row count); `cumsum` depends on the call's shape: one row as
  rocprim (4 elements a thread, the threads' totals summed in order, a thread's first element plus the total before
  it), several rows as ATen's Sklansky scan over blocks of 2^(clamp((9 + log2 K - log2 rows) / 2, 4, 9) + 1). The
  uniforms come from the same `torch.rand` calls. Checked bit for bit against the torch ops (88 shapes and parameter
  sets), and in serving the same sampled texts as the torch ops on the candidates (12 / 12, two runs).
- **An uneven LM head** (`FREETOKEN_LMHEAD_SPLIT`). The vocabulary rows follow `FREETOKEN_TP_SPLIT`: 136,576 on the
  XTX, 111,744 on the XT at 0.55. Each logit is one rank's dot product with the even head's K tiles: the same bits
  (the int8 GEMV keyed its tiles on the head's shape: any shard height now takes them).
- **Measured** on `xtx-xt-4x262k`, EXL3 3.05 bpw, against the previous build (two interleaved runs each; greedy answers
  identical in every arm, the sampled `calib6` texts identical to the previous build's in every run): one request greedy
  94.1 -> 95.1 tok/s (+1.0 %), at T 0.6 95.2 -> 99.8 (+4.8 %); four requests greedy 40.7 -> 41.2 each (+1.1 %), at T 0.6
  39.7 -> 40.7 (+2.5 %). NVFP4 (a 300k-token pool): one request 94.8 -> 96.5 (+1.8 %), at T 0.6 94.0 -> 99.3 (+5.6 %);
  four requests 36.1 -> 36.9 (+2.1 %), at T 0.6 34.5 -> 36.7 (+6.4 %). That build also held the flat positions buffer and
  the PLE change below (the PLE one alone changed nothing on two cards; on one card the two together: +1.1 % greedy,
  +0.6 % at T 0.6).
  By part (EXL3), on the same texts: the fused sampling +2.0 % at T 0.6 against the torch ops on the candidates; the
  uneven LM head +0.5 % greedy, +0.3 % at T 0.6.
- **Smaller things.** Each rank's top k comes from two Triton kernels (each block's k largest, then the row's)
  instead of `torch.topk`'s radix select: 85 -> 40 µs for one row at k = 20, 176 -> 48 for three (CUDA graph, XT).
  The graphs' mrope positions sit in a flat buffer viewed `[3, rows]` per shape: each attention layer copied a strided
  slice before its rope (12 kernels a step). The disk PLE store's lookups no longer hash the n-grams on the GPU (~40
  int64 kernels a step it never read): no change measured. A step now launches 1,625 kernels, 133 of them torch ops.
- **The router's top-k in one reduction a pick** (`FREETOKEN_ROUTER_KEYS`). The router kernel (a softmax over 512
  experts, the top 10, one wave a row) took 10-15 µs a call in serving (traced), ~50 calls a step, against 3-6 µs alone:
  each pick ran three reductions over the 512 values (the largest, the lowest id holding it, its weight by a masked
  sum). One reduction of 64-bit keys (the value's ordered bits, -0.0 as 0.0, then the inverted id) picks the same
  expert, and the weight is the softmax's own operations on that one value (`exp(x - max) / sum`; the masked sum only
  added zeros to it). A picked lane keeps a -inf key with its id: in a row with fewer values above -inf than picks, the
  old loop picks the lowest id among the -inf lanes, picked ones included, with that lane's weight (the first version
  moved on to the next lane; found by a review, out of reach of real router logits). The same experts and weight bits
  over bf16 and fp32 rows full of equal values, signed zeros, infinities, NaN, rows of fewer finite values than picks,
  widths of 4 warps and not powers of two; 5-7 µs a call in serving. Not the instruction cache: the old kernel took the
  same time between twelve code-heavy kernels as back to back. Measured (before that fix, which adds one check a pick):
  one request 94.5 -> 95.6 tok/s (+1.2 %), four requests +1.0 % (greedy, two runs each); at T 0.6 101.4 -> 102.7 (+1.3
  %, one run each); greedy and sampled texts identical to the image's. A trap on the way: a sampled workload run after
  four concurrent sampled requests (`four6`) starts from a generator state that depends on timing (how many steps a
  request decodes alone before the next joins), so its texts differ between runs of the same build; the identity checks
  run the sampled prompts first.
- **The host all-reduce's reads ahead** (`FREETOKEN_HOST_ALLREDUCE_AHEAD`). A decode step runs ~100 of them, 14.5 µs
  on average on the XT and 19 on the XTX in the one-request trace (the wait for the other rank included). After the
  handshake each thread read the peer's 16-byte chunks one at a time (two 8-byte system-scope atomic loads, the sum and
  the store before the next chunk's loads): a 20 KB message (four rows) took three rounds of PCIe reads. Four chunks a
  thread loaded before any sum, in single 16-byte loads (the pages are registered uncached and read after the
  system-scope acquire), the same sums: per call in a CUDA graph on both cards 15 KB 8.4 -> 7.6-7.9 µs, 20 KB 10.0 ->
  8.5, 40 KB 14.1 -> 12.0, 80 KB 23.8 -> 19.5, one row unchanged (~8 µs: the write, the flag's round trip, the data's
  round trip). Bit-identical to RCCL at every size checked. In serving: +0.2 % greedy, +0.5 % at T 0.6, +0.45 % at four
  requests (two runs each, the same answers).
- **The first request's all-gather.** With the decode logits kept local, the graph capture no longer runs the LM head's
  whole-row gather, and the first all-gather of a process sets up RCCL's path for it: the first request after a start
  read its 41k-token prompt in 14.1 s against 13.1 for the image (the 112k- and 159k-token reads after it equal). The
  engine now gathers once at start (0.9 s): 13.1 s.
- **No rollback copy for the requests that kept every row** (`FREETOKEN_SPEC_ROLLBACK_SKIP`). After a verify step
  each request's GatedDeltaNet states are put back where its last kept row left them, one copy of every layer's state
  a call (~35 µs at one request, ~130 at four, several calls a step); a request that kept every row wrote its copy to
  the padding slot, the fixed-shape sink of the torch path. Its programs now exit: +0.5 % at one request (greedy and
  T 0.6), +0.6 % at four greedy, +0.1 % at T 0.6 (two runs each, the same answers).
- **All of this section against the image** (two rounds each, greedy and sampled texts identical in every arm): one
  request 94.7 -> 97.8 tok/s greedy (+3.3 %), 96.3 -> 103.1 at T 0.6 (+7.1 %); four requests 43.2 -> 44.6 greedy (+3.4
  %), 42.5 -> 43.3 at T 0.6 (+1.9 %). NVFP4 (one pair): 94.1 -> 97.3 greedy, 93.3 -> 100.2 at T 0.6, four requests +3.8
  / +3.0 %. `xtx-xt` and `xtx-xt-1m` (EXL3, one pair each): 90.8 -> 93.5 and 92.6 -> 95.4 greedy, 92.7 -> 99.1 and 93.8
  -> 100.0 at T 0.6. One card (`xtx`: the router before its fix, the flat positions buffer and the skipped PLE hashing):
  +1.9 % greedy, +1.8 % at T 0.6. Long contexts (before the router's fix, the start-up gather and the rollback skip):
  cold reads of 41-159k tokens and agent turns on a warm conversation unchanged, five 28-130k prompts at T 0.6 +6.2 %;
  with all of it, the cold reads again equal to the image's.
- **Dropped: a RAM cache for the decode steps' PLE rows.** At layer 1 the GPU waits for the host to stage the step's PLE
  rows (0.85-1.65 ms a step under the tracer): the host's fill takes 0.5-0.9 ms at one request and 1.0-1.2 at four
  (timed without the tracer), most of it likely one NVMe read round (the store alone reads 16 rows in ~210 µs, 64 in
  ~470), after the GPU has run 0.4-0.9 ms of the step. The prefill store's row cache in the decode store too: 15-30 % of
  the rows hit (an empty cache, new prompts), the fill ~0.1 ms shorter, speeds within the noise (one run each).
- **Dropped: a VRAM ring of evicted experts on the XT** (`FREETOKEN_MOE_VICTIM`, patch kept). Under the uneven split
  every rank plans its expert cache on its own memory and all take the smallest plan; the XT's own plan held 12,956
  slots against the common 11,638 (EXL3 3.05 bpw), 2.8 GiB of its VRAM unused. Replayed on a recorded decode routing
  (294,000 MoE calls), an LRU of 12,956 slots misses 24 % less than one of 11,638 (5.09 -> 3.85 %). Kept as a ring of
  the experts its decode evicts (the cache's decisions unchanged on both ranks, the XT copying a miss from the ring
  instead of host RAM): its copies took 19.4 µs a layer instead of 24.8, but its plan kernel 9.6 µs against
  flashlib's 4.8 (traced): decode 0.7 % slower at one request. Its plan simplified (a save landing on an entry the call
  reads is skipped): even with the rest, within 0.6 % at one and four requests. Its test found two bugs on
  the way (two saves into one entry within a call; Triton's `%` keeps a negative operand's sign), and a trap: the JIT
  extensions' cache keys on the `#include` line, not the header's text, so an edited `.cuh` is not rebuilt.
- **Dropped: 128-row tiles for the EXL3 prefill expert GEMM** (each weight tile decoded once per 128 routes instead
  of 64): 3.86 -> 3.72 ms for a 4096-token block's gate|up on the XT, nothing at 1024 tokens; the same bits.

## 38. A decode step's waits and launches: the PLE rows ahead, fewer copies, wider programs (2026-10-22 and 23)

All on `xtx-xt-4x262k`, EXL3 3.05 bpw (a 300k-token pool), arms interleaved with the image of section 37's build:
three greedy 512-token answers (`speed`), 12 prompts x 1024 tokens at T 0.6 (`calib6`), four requests started 1.5 s
apart (`fourst`); greedy and sampled answers identical to the image's in every arm (13 short answers, the 12 sampled
ones, the 3 and the 4 long ones).

- **The PLE rows read while the GPU drafts** (`FREETOKEN_PLE_DRAFT_READS`). A decode step's fill (the host reading its
  rows from the NVMe after the drafts, while the GPU runs layer 0, then signalling the graph's wait at layer 1) took
  0.72-0.79 ms at one request, 0.65-0.69 of it one disk round for the 48-64 rows (16 a token, both ranks reading their
  own), the rest under 0.1 ms; 1.07-1.20 ms at four requests. A thread now reads the next step's first token's rows
  once the verify step's kept tokens reach the host and each draft's once it is drawn, into a 16,384-row cache of the
  decode store: 99-100 % of the fill's rows found there, the fill 0.17 ms at one request (0.1 ms of it waiting for the
  last draft's read), 0.17-0.28 at four. Decode +0.7 % greedy, +0.5 % at T 0.6 at one request (two pairs); four
  requests unchanged, the host reaching each step there after the GPU has finished the one before (its readback wait
  ~0; py-spy: the scheduler thread busy 15 % of the time, mostly waiting for the step's tokens). The thread first
  spun a core in HIP's event wait for most of every step (checked: the wait releases the GIL but burns its core); it
  now polls with short sleeps except for the last draft. A first version crashed: the table is built under inference
  mode, which a thread does not inherit, and the thread wrote an inference tensor.
- **v stored from the qkv projection's rows.** The K/V store kernel took one row stride for k and v, so each
  attention layer copied v contiguous first (a 7-10 µs torch copy, 17 a step with the MTP head's): k and v now each
  take their own (the JIT module renamed, since its cache keys on the wrapper, not the `.cu`). With the PLE reads:
  +1.7 % greedy, +1.4 % at T 0.6, +0.9 % at four requests (two pairs).
- **Two launches fewer per QSA layer** (`FREETOKEN_QSA_FUSED_WRITES`, `FREETOKEN_QSA_FUSED_GATE`): after the group
  compression, the index keys' and the queries' norm and rope and the pending ring's store in one launch; the output
  gate in the sparse attention's split merge. +0.4-0.9 % at one request against both off (one pair and two image
  runs).
- **Wider programs for small decode kernels.** The EXL3 input rotation ran one program per (vector, 128-block), ~1,600
  programs at four rows doing a 128 x 128 product and one reduction each: 14.7 µs on the 7900 XTX, 28.3 at eight rows;
  five blocks a program (the same tile, the same bits) 9.4 and 18.7 (the XT 37.9 -> 20.3 at eight)
  (`FREETOKEN_EXL3_ROTATE_BLOCKS`); one row keeps one block a program (6.7 against 8.5 µs). The mid step lost from the
  same grouping (it has 5x fewer programs: 8.1 -> 13.9 µs at four rows), but both kernels run faster at 2 warps a
  program than at 4, with the same reduction order (1, 2 and 4 warps give the same bits, 8 does not)
  (`FREETOKEN_EXL3_VEC_WARPS`): the mid step at four rows 8.7 -> 5.1 µs on the XTX, 10.7 -> 5.6 on the XT; the rotation
  9.8 -> 7.2 and 11.9 -> 8.3. The output step's `tl.dot` tiles at other warps were slower or changed bits. The router's
  bf16 GEMV loaded its weight tile once per verify row: once for the 2-4 rows, each row summed in the same order
  (`FREETOKEN_GEMV_SHARED_TILES`), 11.2 -> 7.4 µs at four rows on the XTX, 14.0 -> 9.3 on the XT. The sampling
  parameters' device tensors are reused while they do not change (`FREETOKEN_SAMPLE_ARGS_REUSE`, four copies fewer a
  step).
- **The SiLU in the GEMVs that feed it, NVFP4.** A trace of the NVFP4 build (one request) still ran 86 SiLU-and-mul
  kernels a step: one after each MoE layer's gate | up GEMV (the marlin kernel), and the shared expert's at 3-4 rows,
  whose K-blocked shapes had the pair kernel only up to 2 rows (its four accumulators of [2, 512] a warp). The marlin
  GEMV now computes both halves of a program's columns on its own tiles and stores the activation
  (`FREETOKEN_NVFP4_ACT_IN_GEMV`), and the shared expert's pair kernel takes 3-4 rows one column a program
  (`FREETOKEN_INT8_ACT_PAIR4`): +0.8 % greedy, +0.9 % at T 0.6 against both off (two pairs).
- **One decode of an expert's trellis for the routes that share it, EXL3** (`FREETOKEN_EXL3_SHARED_ROUTES`). The
  decode GEMV ran one program a route, each decoding its expert's trellis tiles for that route alone: a route whose
  expert another route of the step also reads cost about as much as a distinct one (the decode is the cost, not the
  bytes). On recorded routing a third of the routes share an expert at four rows of one request (81 / 74 / 63 % distinct
  at two / three / four rows), and 32 % at four requests of two rows. The routes of a slot now share a program, up to
  four: each tile decoded once, each route summed in its own accumulator in the single-route order (the same bits).
  Which program takes which routes is written by the input rotation's kernel: a launch of its own cost 3.3 µs, and
  finding the group inside every GEMV program (a scan of the step's slots before the K loop, the routes' branches in
  it) made the GEMV slower than one program a route. The whole decode MoE layer in a CUDA graph on that routing, the
  XTX's 384-wide slice: four rows 131 -> 111 µs, four requests 227 -> 195, two rows 79 -> 78; the XT's 256: 117 -> 101,
  199 -> 168, 79 -> 75. In serving against the tree without it (two pairs): one request 101.8-102.2 -> 103.7-103.8
  tok/s greedy (+1.8 %), 107.6-107.8 -> 110.0-110.2 at T 0.6 (+2.3 %); four requests started apart 44.4-46.2 ->
  47.6 (greedy), four sampled ones together 46.3-46.4 -> 47.0-47.4.
- **The missing experts read by the GEMVs that need them** (`FREETOKEN_MOE_MISS_READS`, `FREETOKEN_BANK_COARSE`).
  At four requests the copies of the missing experts took 5.0 ms of an EXL3 step on the XTX (51 copies at 99 µs,
  traced), each on the main queue between its layer's cache lookup and its GEMVs: the PCIe transfer and the GEMV work
  ran one after the other. A GEMV program whose slot the cache fills this step now finds the slot in the step's copy
  plan, reads its tiles from the layer's pinned host bank and stores them into the slot as it loads them (an EXL3
  tile's windows cover all its words for K <= 8; NVFP4's GEMV loads each word once); the cache copies only the small
  banks first (the rotation scales, the global scales). The same words, the same sums, and the slot ends up as the
  copy would have left it (tests on both cards). From the banks as registered (fine-grained, so not cached in the
  GPU's L2) the EXL3 gate | up GEMV moved a 0.74 MB slice in 58.8 µs, twice the copy kernel's 33.6: it loads most
  words twice. Registered coarse-grained (`hipExtHostRegisterCoarseGrained`) it takes 32.5, and the copy kernel 33.6
  either way. In serving, EXL3 3.05 bpw against the tree without both (two pairs): one request 103.6-103.8 ->
  104.4-104.9 tok/s greedy, 110.1-110.2 -> 111.8-112.1 at T 0.6; four requests started apart 47.6 -> 49.7-52.2, four
  sampled ones together 47.1-48.1 -> 50.4-50.9; NVFP4: 101.1-101.2 -> 103.2 greedy, 104.2-104.3 -> 106.8-107.1 at T
  0.6, 38.0 -> 40.5 and 38.3 -> 38.6-40.1; the same answers. The coarse-grained banks cost NVFP4's cold reads 0.8 %
  (the launcher's pool, 164k tokens 42.56-42.61 -> 42.85-43.03 s, 214k 56.78-56.84 -> 57.20-57.33, two pairs); EXL3's
  are unchanged.
- **Dropped: each rank its own expert slots.** Under the uneven split both ranks take the smaller plan: 12,943 slots
  where the XT could hold 14,842 (EXL3 3.05 bpw, a 300k-token pool). A slot decides only whether an expert is copied
  or found, so each rank could keep its own count (the pages, which the scheduler decides on, stay common): the same
  answers, but 104.6-105.0 against 104.7-105.1 tok/s at one request and the same at four (two pairs): the XT is not
  the card the step waits for.
- **Dropped: shared routes in NVFP4's GEMVs.** There too a shared route costs nearly a distinct one at eight rows (gate
  | up 60 µs against 41 for the distinct routes alone on the XTX), but the FP4 decode is cheap and no kernel before them
  can write the table (the router runs one program a row): with a launch of its own for it, the whole layer at four
  requests 106.3 -> 106.1 µs on the XTX, 107.7 -> 104.3 on the XT, and slower at one request (two rows 44.2 -> 48.1).
  Bit-exact (its test passed), not kept.
- **All of this section against the image** (every arm's greedy and sequential sampled answers identical to the
  image's; two runs of the final tree against one of the image): EXL3 3.05 bpw one request 98.2 -> 104.4-104.9 tok/s
  greedy (+6.5 %), 103.6 -> 111.8-112.1 at T 0.6 (+8.1 %); four requests started 1.5 s apart (`fourst`, greedy) 44.9
  -> 49.7-52.2 each (+14 %), four sampled ones together (`four6`) 43.6 -> 50.4-50.9 (+16 %). NVFP4: 98.2 -> 103.2
  (+5.1 %), 101.0 -> 106.8-107.1 (+5.9 %), `fourst` 37.6 -> 40.5 (+7.7 %), `four6` 36.5 -> 38.6-40.1 (+8 %). One card
  (`xtx`, EXL3 3.05 bpw, before the shared routes and the miss reads, where the expert misses took 37 % of a step in
  section 25's trace): 57.1 -> 57.5 greedy, 58.0 -> 58.4 at T 0.6 (+0.6 %, two runs against one). Cold reads of 45k /
  96k / 164k tokens, agent turns, 2k-token turns and one-word turns on a 45k conversation unchanged (164k 46.9 s both,
  agent turns 0.68 s both, one pair after the shared routes; NVFP4's 0.8 % longer with the coarse-grained banks, above),
  the same answers. The texts of `four6` differ between any
  two runs, the image's too (2 of 4 identical from one image run to the next): four sampled requests at once draw in an
  order set by timing (which request joins a step when), so they check speed, not bits.
- **Dropped: the host all-reduce's data and call number in one 8-byte word** (an LL protocol: the peer polls the data
  itself, no fence, flag store or flag round trip). The same sums as RCCL, but per call in a CUDA graph 5 KB 6.13 ->
  6.06 µs (one lane a wave polling), 6.63 (every lane polling), 10 KB 7.00 -> 7.62, 20 KB 8.25 -> 12.2: the floor is
  the PCIe round trips, not the protocol, and the words double the bytes.
- **Dropped: one host-to-device copy for a decode step's inputs.** The ~40 small operations between the drafts and the
  graph (39 kernels, 353 µs of wall time for 137 µs of kernels in the trace) come from the scheduler's positions and
  mappings, the QSA metadata, the graph's input copies and the sampling parameters (attributed with a profiler over
  whole scheduler iterations); packing them would touch the scheduler every model shares for ~0.1-0.2 ms a step.
- **Not reproduced: the rare start fault.** One more at the first request after a start (the XTX, an illegal memory
  access surfaced in `offload_cache._subset_copy_plan`, the kernel log only `sq_intr: error, type 2`, rate-limited,
  no page-fault address). Ruled out: the batched copy reading its pointer arrays after the call (it reads them at the
  call: rewriting them right after changed nothing); ROCr's asynchronous scratch reclaim on a failed allocation taking
  scratch from a running kernel (77 failed allocations while the QSA prefill attention, 468-592 bytes of scratch a
  lane, looped on another stream: no fault, with or without `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0`).
- **A cold read's kernels by rank: not traced.** `rocprofv3 -P` stops the workers (signal 6) two minutes into a
  prefill window, as in section 34.
- **The server's stop left 5 semaphores** in the host's `/dev/shm` each time (`docker run --ipc=host`): uvicorn
  re-raises the SIGTERM it stopped on and the process died of it before its exit handlers. It now reaps its workers,
  runs multiprocessing's exit handler and the resource tracker's cleanup, then dies of the signal (0 left, against 5
  for the image).

## 39. Fewer launches a decode step: the inputs, the cache admission, the norms, the router (2026-10-25 and 26)

A kernel trace of the build of section 38 (`xtx-xt-4x262k`, EXL3 3.05 bpw, one request at T 0.6): a step 24.3 ms,
1,558 kernels, 18.7 ms of them in kernels and 5.6 ms between them (~3.5 µs each); 37 kernels and 321 µs between the
token-pool write that ends a step and the graph of the next one, 35 of them copies, casts and gathers. At one request
the step is ~1,300 graph kernels of the 48 layers (a MoE layer launches 12), ~130 for the three drafts (each a decoder
layer, the 390 µs LM head and its sampling), ~30 for the verify step's sampling and rollback and ~37 for the step's
inputs. All measurements below on `xtx-xt-4x262k` with arms interleaved with the image of section 38: three greedy
512-token answers (`speed`), 12 prompts x 1024 tokens at T 0.6 (`calib6`), two greedy 1024-token answers at once
(`two`), four requests started 1.5 s apart (`fourst`); greedy and sampled answers compared with the image's.

- **A step's inputs in one copy and one kernel** (`FREETOKEN_STEP_INPUTS`): section 38 dropped it as too invasive for
  ~0.1-0.2 ms; measured, it pays. The scheduler writes each row's position and mrope positions, each request's
  page-table row, GatedDeltaNet slot and K/V length, and the write tuple into one pinned buffer; one copy and one kernel
  write the graph's input buffers (the token and the K/V slot each row reads, the QSA block table), the graph's own
  copies and the QSA restage skipped. EXL3, two pairs: +0.8 % greedy, +0.7 % at T 0.6, +0.7 % at two requests, +1.2 %
  at four.
- **The cache admission and the copy of the small banks in one launch** (`FREETOKEN_MOE_ENSURE_COPY`): flashlib's LRU
  admission (its code, so its slots and plan) in the first program; the others wait for its plan on a release /
  acquire flag, then copy the missing experts' small banks (EXL3's rotation signs; the big banks are read by the GEMVs,
  section 38). EXL3, two pairs: +0.7 % greedy, +0.5 % at T 0.6, four requests within the noise.
- **The q / k norms and the rope in one launch** (`FREETOKEN_QK_NORM_ROPE`): one program per row and head, the norm as
  the Triton norm kernel (same code, tile and warps), stored, then the head's first 64 dims rotated as the rope kernel
  does. NVFP4, two pairs: +0.5 % greedy and at T 0.6, +0.4 % at two requests.
- **Three constant ops fewer in the PLE layer's decode** (its indptr is the GatedDeltaNet's, the placeholder restore
  only in prefill, an unused arange) and no copy of the free-page list when nothing was freed.
- These four against two runs of the image, EXL3: one request 105.05 / 104.67 -> 106.90 / 107.20 tok/s (+2.1 %), T
  0.6 112.00 / 111.75 -> 113.68 / 113.77 (+1.7 %), two requests 76.63 / 76.54 -> 78.58 / 78.50 (+2.5 %), four 51.32 /
  51.37 -> 52.55 / 52.10 (+1.9 %); 22 greedy and 12 sampled answers identical. A trace of that build: 1,452 kernels a
  step (1,558 before), 23.7 ms.
- **Fixed: the chained drafts of a second request** wrote their MTP K/V at the first request's page-table row
  (`chain_meta` read the strided per-request rows as contiguous; CHANGELOG). Only two requests chain drafts on this
  profile (three and four verify two rows, no chain): `two` is part of the +2.6 % above.
- **The router GEMV with the shared expert's gate | up, the shared expert's down projection with the router's top-k**
  (`FREETOKEN_ROUTER_SHARED`): each pair reads the same input or nothing of each other, so one launch runs both
  programs' unchanged bodies at one warp count. A first version ran them at 1-2 warps (the shapes it was tested on kept
  their bits there) and changed the answers: EXL3's shared expert is 768 rows a rank, whose default tiles run at 4
  warps, and its sums changed with them (found by computing both paths in the server, eager decode, and comparing every
  layer's outputs). The launch now takes the shared expert's own warps (the router GEMV keeps its bits at 1, 2, 4 and
  8). In a graph of 48 layers: 13.3 -> 9.5 µs a layer at one row, 14.1 -> 9.9 at two, 17.8 -> 13.6 at four.
  `xtx-xt-4x262k`, two pairs: EXL3 one request +2.5 % greedy, +2.3 % at T 0.6, two and four requests +0.2 / +0.4 %;
  NVFP4 +2.4 %, +2.3 %, +0.1 / +0.3 %; the same answers. The cost a microbenchmark gave (~4 µs a layer, ~0.8 % of a
  step) understates it: the step loses two launches a layer and their waits.
- **Long reads and the other requests** (`FREETOKEN_DECODE_INTERLEAVE_LONG`): section 17 dropped the interleaving for
  the agent turns it delayed; it now follows only a chunk that leaves prompt to read. One request decoding while a
  130k-token prompt is read cold (EXL3, one run each): without it the read 38.1 s and the decoding request stopped for
  38.0 s; 8 steps a chunk 39.1 s (+2.7 %), stops of one chunk (4.9 s) at most; 32 steps 42.2 s (+10.8 %). Off by
  default.
- **Dropped: the GatedDeltaNet output norm inside the delta-rule kernel.** The head's last V block to finish (a counter)
  normed the head's rows with the norm kernel's own program body (one warp, the same tile): the same bits (outputs,
  normed rows and states, both cards), one launch fewer per GDN layer (36 a step), but EXL3 `xtx-xt-4x262k` 110.1 /
  109.7 tok/s against 109.9 without it, 116.6 / 116.4 against 116.7 at T 0.6 (likely the last block normalizing every
  row of its head alone, on one warp, after the others: not timed apart). Patch not kept.
- **Tests that failed on ROCm, fixed in the tests**: the chunked-prefill comparisons of `test_qsa_backend.py` (the toy
  layer's `F.linear` changes its sums with the row count: computed a row at a time, all three pass), the block top-k
  case with ties (`torch.topk` keeps an unspecified order of equal values, the kernel the lowest column: the reference
  now sorts stably) and three sizing tests that named flashinfer's backend.
- **All of the above** (two runs against one of the image): EXL3 104.92 -> 109.68 / 109.92 tok/s greedy at one
  request (+4.7 %), 112.04 -> 116.35 / 116.50 at T 0.6 (+3.9 %), two requests 76.56 -> 78.77 / 78.68 (+2.8 %), four
  51.35 -> 52.74 / 52.57 (+2.5 %); NVFP4 103.00 -> 107.45 / 107.54 (+4.4 %), 106.82 -> 110.73 / 110.83 (+3.7 %), two
  requests 74.10 -> 73.95 / 73.90 (unchanged, below), four 40.47 -> 40.74 / 40.75 (+0.7 %); the same greedy and
  sampled answers.
  A kernel trace of this build (EXL3, one request, T 0.6): 1,348 kernels a step, 22.8 ms (24.3 before). One card
  (`xtx`, the launcher's 131,072-token context, one run each): EXL3 61.57 -> 63.27 tok/s greedy (+2.8 %), 62.43 ->
  64.10 at T 0.6 (+2.7 %), agent turns 1.054 -> 1.053 s; the same answers. Twenty minutes of four agent-like
  sessions on each checkpoint with the launchers' settings (tool outputs of 200-3000 tokens, 64-600-token answers
  at T 0.6, one request in ten dropped mid-answer, contexts to 137k / 143k): 367 and 333 requests, no error; forty
  minutes more on EXL3: 727 requests, contexts to 154k, no error.
- **NVFP4 at two requests gains nothing** where EXL3 gains 2.8 %: each change turned off in turn (three runs of `two` an
  arm) 72.2-72.4 tok/s, the image 72.3 twice, the build with all of them 72.4. Yet in a kernel trace of two sampled
  requests the build's step is shorter (36.5 against 38.9 ms, 1,478 against 1,646 kernels). Sampled (three runs of
  `two6` an arm) the build is slower: 65.4 / 65.7 tok/s against 67.3 / 67.8, with or without any one of the four launch
  changes (65.3-65.7), and its drafts are kept less often (2.6-2.9 tokens a request step against 2.8-3.0 in the `[mtp]`
  log). Put back for an experiment, the old chained-draft addressing (above) gave 66.7 / 67.8 against 65.5 / 66.5 for
  the fixed one, its drafts kept more often (2.8-3.0 against 2.6-2.9). A possible reading: with the old addressing the
  second request's chained guesses read the K/V the head's pass over the verify rows had written at those positions,
  where now they read their own. Tried for every request (next item), it does not carry over. The fix stays (the old
  addressing wrote into another request's slots); why NVFP4's two sampled requests lose ~2 % is open.
- **Dropped: the chained guesses keeping the K/V of the head's pass over the verify rows.** Where a chained guess's
  position is one of the step's verify rows, its K/V went to a junk slot and the guess read the pass's instead (EXL3,
  two pairs): greedy drafts kept more often over the first 512 steps (2.53 -> 2.87 tokens a request step) but greedy
  decode +0.5 % only, T 0.6 -2.2 % (116.7 / 116.9 -> 113.3 / 115.0), two sampled requests unchanged; greedy answers
  identical. Not kept.

## 40. Upstream up to ba31db2 (2026-10-26)

27 commits merged (upstream's 0.2.0 and 0.2.1), the authors' kept; fourteen files conflicted, and two changes that
merged without a conflict needed this tree's attention:

- **The shards the index names, and the files beside it.** Upstream's loader now lists only the shards
  `model.safetensors.index.json` names (`0bf7dbb`, for checkpoints that keep them in a subdirectory). The EXL3
  checkpoints keep tensors in files the index does not name: 3.05 bpw the n-gram hash constants
  (`ngram_embedding.safetensors`) and the MTP head's mixer, 4.05 bpw the hash constants and a vision patch. Loaded as
  merged, the hash constants would not have reached the model (the loader test: `KeyError` on `layer_multipliers`).
  qwen4_exp lists the index's shards, then the folder's other safetensors files.
- **The PLE wait-sync on HIP.** Upstream added HIP's memops with a signal-memory flag and a capture probe (`32056cf`),
  which this tree had by loading the memops at run time: upstream's are kept, this tree's check that a captured wait
  holds three replays runs after its probe, and both profiles still pick wait-sync. Upstream fills a decode step's
  rows on a worker thread on HIP, one row a request; this tree's fill (the verify rows, the draft reads) stayed, run
  after the launch. Upstream's test of a 16,384-kernel graph behind an unsatisfied wait then hangs: the launch call
  blocks once the hardware queue is full, before the host reaches the fill that would release it (8,192 kernels
  still pass; a decode step's graph launches ~1,350 at one request). On a thread started before the launch
  (`FREETOKEN_PLE_FILL_THREAD`, on by default on ROCm) it passes; the same answers, and on `xtx-xt-4x262k` (EXL3 3.05
  bpw, two pairs) 110.13 / 110.15 tok/s greedy against 110.27 / 110.15 with the fill after the launch, 116.90 /
  117.07 at T 0.6 against 116.87 / 116.99, two requests 79.67 / 79.76 against 79.69 / 79.71, four 53.2-53.3 in both
  (the first four-request run after a start reads ~47 with either).
- The hybrid ensure kernel took upstream's fetch cap, rounding modes and chunked scan of a large pool on top of its
  vector route reads, with upstream's barrier between the route loads and stores; the router body that
  `FREETOKEN_ROUTER_SHARED` reuses took the sigmoid and grouped routing, the softmax path unchanged; this tree's ROCm
  batch copy (with its per-copy fallback) stays beside upstream's; the package keeps its ROCm classifier.

Measured on `xtx-xt-4x262k` with a 300k-token pool, the merge's image against the build before (two runs against
one): EXL3 3.05 bpw 109.25 / 110.08 tok/s greedy against 109.67, 116.17 / 116.82 at T 0.6 against 116.53; NVFP4
107.48 / 108.13 against 107.81, 111.08 / 111.30 against 111.25; cold reads of 40k / 112k / 159k tokens the same
(12.09 / 33.05 / 47.61 s against 12.12 / 33.03 / 47.60, EXL3; the first start of a new image compiles, its first 112k
read 36.6 s); agent turns, two and four requests within their run-to-run spread. Answers: the same as the build before
(EXL3 40 of 40 greedy and 12 of 12 sampled, NVFP4 36 of 36 and 12 of 12), and as the runs of section 39; in the second
run of each, the two greedy requests started together came out otherwise, as they do between two runs of the same image
(when each request starts sets its batches). GPU tests (43 files, each card): 1,478 passed, the one failure the
large-graph test above, which passes with the fill thread; CPU suite 1,849 passed.
Fifteen minutes of four agent-like sessions with the launcher's EXL3 settings: 243 requests to 124k tokens, no error.
`/v1/cache/flush` (upstream's, `e0a7e23`) answers "unsupported" with two cards.

## 41. Still open

- In NVFP4 the model needs ~63 GiB of RAM for its experts alone (the engine keeps every expert in RAM). The EXL3 3.05
  bpw checkpoint needs 42.6 GiB and served under a 56 GiB container limit, but a real 64 GB machine is untested
  ([limits.md](limits.md)).
- Not run on the agentic benchmark since its last run: Triton 3.8, not bit-exact with that run's 3.7.1 (section 18);
  upstream's NVFP4 decode kernel (section 23, 5 of 15 greedy answers changed); the fp64 rebuild of EXL3's dense
  weights (section 20); `--hybrid` (section 24), whose answers also move with the expert cache's contents (section 27);
  the two profiles with the K/V in RAM, `xtx-xt-1m` and `xtx-xt-4x262k` (section 22); and the drafts drawn under each
  request's sampling, with three and four requests verifying on `xtx-xt-4x262k` (section 36).
- ROCm 10 / PyTorch 2.13 and ROCm 10.1 / PyTorch 2.14 (rechecked 2026-10-10), the image's bases since 2026-10-02:
  `F.linear` changes its sums with the row count for some shapes ((3072, 256), (6656, 2560)): serving does not use it
  for them (int8 or table GEMVs), but `test_qsa_spec_rows`' toy projections did, which made it fail on 3-4 rows;
  computed a row at a time they pass (the test now does so).
- A rare GPU page fault (`[gfxhub] page fault`, permission fault, `sq_intr` errors): those while serving came with a
  card within tens of MiB of full (section 21; the torch allocator is now capped 512 MiB short of that). One at start
  on 2026-10-07, ~3 s after the CUDA graphs were captured, with 3.9 GiB free on the XTX (EXL3 3.05 bpw; 1 start in
  ~32 that night, the 16 starts of a series after it were clean); one more at the first request of a start on
  2026-10-19 (on the XT, 1 of the night's 41 EXL3 starts, section 34), one on 2026-10-20 (NVFP4, the XTX, the first
  prefill, 1 in ~70 starts, section 36) and one on 2026-10-22 (EXL3, the XTX, the first request, 1 in ~20 starts,
  section 38): cause not found; two hypotheses ruled out in section 38. Not seen in ~80 starts on 2026-10-25 and 26
  (section 39's build, both checkpoints), 31 of them in a row with the launcher's pools (19 EXL3, 12 NVFP4), a
  7.5k-token first request and ROCm's debug agent loaded to name the faulting kernel
  (`HSA_TOOLS_LIB=librocm-debug-agent.so.2`, in the image; it costs ~0.2 µs a kernel, so only for such a hunt).
- Expert misses (section 20): the decode GEMVs now read them from host memory while computing the cached experts
  (section 38), so their PCIe time shows where it exceeds that work (a layer's GEMVs at one request, EXL3: 40.5 µs
  against 32.5 when the copy came first, the copy 34.6 -> 4.2 µs). Fewer of them is the lever left: a larger cache,
  not a cache policy (an online one gains ~8 % of the misses at best on a recorded trace); on one card `--hybrid`
  computes part of them on the CPU (section 24). More VRAM for the cache: a desktop-aware plan for the card that drives
  the display (the ~145-179 Triton kernels a card loads are all in use, section 31).
- On two cards `serve.sh` lowers `--memory-ratio` to 0.77 / 0.76 for EXL3 (3.05 bpw at 0.77 measured in section 21; 4.05
  bpw at 0.76 not measured again).
- Short agent turns (section 27): about half of a 130-row turn at 25k is the routed experts the cache lacks crossing
  PCIe (4-6 GB a card; estimated from the bank sizes at 26 GB/s, not timed apart), the rest compute and the host. An
  NVFP4 agent turn of 500-750 tokens at 45k copies 8,400-10,000 predicted and ~1,600 missed experts on rank 0 (counted
  in the plans: 14.5-17 GiB at 1.46 MiB each, ~24-28 GB/s over the turn's 0.65-0.70 s): PCIe is saturated throughout.
  The levers left are a larger expert cache or fewer bytes per expert; a cache policy gains ~8 % of the misses at best
  (above). The host's part (section 33): the per-layer sync of the predicted experts' copy plan is gone, Triton launches
  cost ~10 us instead of ~50 and the PLE rows are read by one rank on a thread over four io_uring rings (section 34:
  ~20-26 ms for a 600-token turn's rows, which likely ends within the first two layers; not measured in serving);
  `F.linear` (hipBLASLt) takes ~44 us of host time a call (~500 a turn).
- A decode step's PLE rows (section 38): read while the GPU drafts, the step's fill takes 0.17 ms, most of it waiting
  for the last draft's rows (16 a request, one disk round of ~0.1 ms). Alone on the host the store reads 16 rows in
  ~210 µs and 64 in ~470 (4 KiB direct reads, io_uring). Not tried: polled NVMe queues, the IOMMU in passthrough (it
  translates, lazy invalidation) -- machine settings.
- The host all-reduce's floor: ~6 µs a call for one row in a CUDA graph (the write, the flag's round trip, the data's
  round trip). The data and the call number in one word did not lower it (section 38).
- Decode at four requests (EXL3 3.05 bpw, `xtx-xt-4x262k`, a kernel trace of section 38's build): per step on the
  XTX the EXL3 GEMVs take 9.9 ms with the missing experts read inside them (8.6 + 5.0 ms of copies before), the int8
  dense GEMVs ~11 ms (eight rows; by their operation count likely ALU-bound, not measured), the sparse attention 2.5;
  the two cards now wait about as long in the all-reduces (2.5 and 2.3 ms). Prefetching the misses on a side stream
  inside the replayed graphs made the step longer (section 29): a copy kernel beside the compute holds CUs for the
  whole PCIe transfer, and the copy engine would need host-known addresses (section 32).
  At four requests with `FREETOKEN_SPEC_BS_MAX=2` (no verify step then) the host spends 23.7-25.0 ms of a 26.3-27.7 ms
  forward inside it (`FREETOKEN_SPEC_TIMING`, one run), then 1.4-1.5 ms launching the MTP head's KV pass (0.33 ms of GPU
  time). What holds it there (py-spy on rank 0, NVFP4, one run): 72 % of the scheduler thread's samples wait on the
  step's sampled tokens before the PLE fill, so the host is ahead of the GPU, not its limit.
- NVFP4 decoding two requests: greedy as fast with section 39's changes as without them (72.3-72.4 tok/s), sampled ~2 %
  slower after the chained-draft fix, the drafts kept less often (section 39); the chained guesses reading the K/V of
  the head's pass over the verify rows instead did not help EXL3 (section 39). Not understood.
- A decode step at one request (EXL3 3.05 bpw, T 0.6, section 39) launches 1,348 kernels (1,558 before it), ~3.5 µs
  apart under the tracer (4.9 ms of a 22.8 ms traced step). Left of the small torch ops and copies: 7 of the 10 kernels
  between a step's token writes and its first layer (among them the PLE rows' copy, the tokens read back for the PLE
  store, the embedding lookup), 23 in the layers (most in the PLE layer: its gate in bf16 torch ops, whose sum order a
  fused kernel would have to reproduce), 17 around the verify step and 17 in the drafts (most keeping `torch.topk`'s
  order, the draws' random numbers, the drafts' copies to the host). Inside a layer every launch left reads what the one
  before wrote: a MoE layer's ten (two of them two kernels' programs each), a GatedDeltaNet layer's eight; one launch
  for such a chain would need a wait across the grid, and its parts run at different warps, which set their sums.
  Sampling with a `top_k` above 64 keeps the torch ops (ATen's sum order then changes with the row count).
- A cross-stream wait on the GPU leaves the decode graph replayed after it slow until the host synchronizes the other
  stream (section 33: the fix-up copy on a stream of its own, one request -5 %; joined on the GPU after each prefill,
  four requests a third slower); what in the HIP runtime keeps that state was not found. The shipped copies on the
  prefill copy stream do not show it.
- Cold reads on two cards: the RCCL all-reduces take 17 % of the GPU time, staged through host memory (the cards have no
  peer access); a host all-reduce of the same bits was slower (section 29). The MoE's and the GatedDeltaNet output's
  now run block by block behind the next block's work (section 33); the attention layers' output (12 of 48) and the
  last, partial chunk of a read still all-reduce in one piece (the attention's by blocks gained 0.1-0.3 %, section
  34). Where a 16k-token chunk's GPU time goes (NVFP4, rank 0, timed by parts): the MoE 45 %, the GatedDeltaNet layers
  17 %, the sparse attention kernel 14 %, the hyper-connection mix 11 %, `o_proj` with its gate and all-reduce 6 %, the
  indexer's selection 2 %.
- Cold reads: in a kernel trace of 16k-token chunks on the 7900 XTX (5.2-5.9 s of kernels each), the all-reduce took
  ~1 s (most of it likely waiting for the slower XT, which was not traced), the EXL3 expert GEMMs ~1.35 s, the sparse
  attention ~0.66 s (close to the bandwidth its K/V gathers need) and the indexer 0.15-0.55 s, growing with the
  context.
- More than two GPUs, other RDNA3 cards, RDNA4 and NVIDIA are untested.
- Image input (`serve.sh --vision`) is measured on `xtx-xt`, `xtx` and `xt` only. The derived profiles (`xtx-xtx`,
  `xt-xt`, `gre`) are untested with it, as they are without it.
