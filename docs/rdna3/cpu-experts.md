# Expert misses computed on the CPU (`--hybrid`)

Experimental, measured on 2026-10-11 and again on 2026-10-13 after the decode kernels of 35afbb3 (Threadripper
3970X: 32 Zen 2 cores, AVX2, 4 x 32 GB DDR4-3200, STREAM read 72-74 GB/s; RX 7900 XTX + 7900 XT on PCIe 4.0 x16). Not
agent-validated.

With `--moe-strategy offload` (every profile), the experts live in RAM and the free VRAM is a cache of them; an expert
a token routes to that is not in the cache (a miss) is copied over PCIe first (26-28 GB/s). Upstream's `hybrid`
strategy computes part of each step's misses on the CPU instead, straight from the RAM banks: only the token's
activation crosses PCIe, the weights do not. Each MoE layer of a decode step then splits its misses: some are fetched
into the cache as before, the rest go to a pool of pinned CPU threads that runs while the GPU computes the hits.

`rdna3/serve.sh <profile> --hybrid` turns it on with the split measured best here (`FREETOKEN_HYBRID_FETCH_FRACTION`,
the share of each step's misses fetched): with NVFP4 a quarter on one card (what `ft bench bw` picks on this machine)
and 40 % on two, 60 % with EXL3.

## Re-measured on 2026-10-13 (image of 0205cb3)

Decode tok/s, MTP head on, the profiles' splits: one request = median of 12 sampled 512-token answers (6 prompts x 2,
as in [benchmarks.md](benchmarks.md#how-the-numbers-are-taken)), its `offload` column the benchmark cell of 2026-10-12
on the same code; 4 at once = 3 rounds of 4 sampled answers, both columns the same evening.

| Profile | `offload` | `--hybrid` |
|---|---:|---:|
| NVFP4 `xtx`, one request | 42.7 | **62.3 (+46 %)** |
| NVFP4 `xt`, one request | 34.6 | **52.1 (+51 %)** |
| NVFP4 `xtx-xt`, one request | 82.9 | 84.5 (+2 %) |
| NVFP4 `xtx-xt`, 4 at once | 25.2 each, 92-100 in total | **28.9 each (+15 %), 89-114 in total** |
| EXL3 3.05 bpw `xtx`, one request | 54.8 | 57.8 (+5 %) |
| EXL3 3.05 bpw `xtx-xt`, one request | 88.9 | 89.1 (0 %) |

The decode kernels sped up both columns; on one card the CPU's share now gains a little more than on 2026-10-11 (+46 /
+51 % against +43 / +48 %). The two-card one-request differences are within the spread of sampled answers (below).
The first round of 4 is the slowest in both columns (92.0 and 89.0 tok/s in total).

## Measured with NVFP4 on 2026-10-11 (decode tok/s, 12 sampled answers, MTP head on; 4 at once: per request, 3 rounds)

| Profile | `offload` | `--hybrid` |
|---|---:|---:|
| `xtx`, one request | 39.4 | **56.2 (+43 %)** |
| `xt`, one request | 31.4 | **46.6 (+48 %)** |
| `xtx-xt`, one request | 78.2 | **81.2-82.7 (+4-6 %)** |
| `xtx-xt`, 4 at once | 24.4 each, 87-96 in total | **28.2-28.6 each (+16-17 %), 107-112 in total** |

On one card a quarter is the best split measured: on `xtx` 15 / 25 / 35 % fetched gave 51.9 / 53.2-54.2 / 50.1 tok/s
(before the last changes below). On `xtx-xt`, everything on, 25 / 40 / 55 % fetched gave 78.3 / 82.6-82.7 / 78.6 tok/s
for one request and 28.6 / 28.6 / 26.9 each with 4. A likely reason, not measured: each rank's pool has half the cores
and of the RAM's bandwidth but its own PCIe link. A sampled answer's speed depends on its text, which changes with the
split
(answers of one run spread over +-13 tok/s): the one-request gain is a mean over 12 answers, about two standard
errors; a second run at 40 % gave the same answers and speeds, a third through `rdna3/serve.sh xtx-xt --hybrid` after
a 118k-token prompt (9/9, 61 s) other answers at 81.2 tok/s.

A 118k-token prompt read at the usual speed (83 s on `xtx`, 61 s on `xtx-xt`: prefill streams whole layers to the GPU,
unchanged) and answered 9/9 planted questions on both. Greedy answers are not bit-identical to offload (the CPU sums in
fp32 in another order): on `xtx`, 8 of 14 identical, the others diverge after 150-720 characters, all sensible. They
are not less precise: against an fp64 reference at the model's sizes, one MoE output computed on the CPU is off by
2.3e-3 (relative), on the GPU by 3.9e-3 (bf16 between its steps), split between them by 3.3-4.0e-3
([journey.md](journey.md#27-agent-turns-short-prefills-through-the-expert-cache-2026-10-14)). Every
change after the first working version kept the greedy answers bit for bit (14/14 on `xtx` and on `xtx-xt`) until
2026-10-14: since then prefill chunks of up to 1280 tokens (agent turns) read their routed experts through the cache
here too, which moves the answers as the cache's contents do. Time to the first token of a 132-337-token turn at 25k:
NVFP4 `xtx-xt` 1.19-1.24 -> 0.78-0.89 s, EXL3 3.05 `xtx` 1.50-1.52 -> 1.05-1.18, NVFP4 `xtx` 2.32-2.34 -> 1.63-1.80.

The CPU pool spins on its cores while a decode runs and parks 50 ms after the last layer: with `--hybrid` on one card,
31 of the 32 cores are busy during generation (15 + 1 per card on two).

## What it took on this hardware

- **The GPU <-> CPU handshake.** Upstream signals the CPU pool with stream memory operations (`cuStreamWriteValue64` /
  `cuStreamWaitValue64`) looked up in `libcuda`; on ROCm there is none, so every MoE layer fell back to two
  `hipLaunchHostFunc` callbacks, ~1-1.5 ms each: a test on 2026-10-01 decoded at 5.9-8.3 tok/s against 55-56
  (`xtx-xt`) and the idea was dropped on that number. HIP's own `hipStreamWriteValue64` / `hipStreamWaitValue64` work,
  in captured graphs too, on ROCm 10.1 (the PLE disk reads already rely on them).
- **MTP verify rows.** The executor was sized for `max(--max-running-requests, --cuda-graph-max-bs)` rows, but a
  verify step runs up to `FREETOKEN_SPEC_VERIFY_M` rows per request: the C++ scratch then grew on the first verify
  step, in the middle of serving (and the EXL3 scratch below has no such path). The bound now counts the verify rows,
  and a larger task is refused.
- **Two cards.** Upstream refused CPU decode with NVFP4 experts under tensor parallelism; each rank's executor reads
  its own slices of the banks and its partial joins the GPU partial before the all-reduce, like the GPU path, so the
  check is gone. Both ranks' pools took every physical core; each rank now pins a contiguous half.
- **The hybrid path's fixed cost.** Each MoE layer of a hybrid step added ~12 small graph nodes (a clone of the
  routing, masks, two `where`, a clamp, three device-to-host copies, three stream memops, the copy back and an add); a
  node of a captured graph costs ~3.2 us here, a memop as much as a tiny kernel. On two cards, where one layer of a
  one-request step lasts ~270 us, that cost ~9 % at one request whatever share of the misses the CPU took. What is
  left of it:
  - one kernel writes the CPU's inputs straight into its pinned buffers (a device view of them) and splits the
    routing, another adds the CPU's partial back from them (`FREETOKEN_HYBRID_SPLIT_KERNEL`);
  - the two kernels carry the handshake (`FREETOKEN_HYBRID_FLAG_IN_KERNEL`): the first bumps the layer slot's step
    number and raises the host `ready` flag to it with a system-scope release, the coordinator hands the number back
    in `done`, the second spins on `done` with a system-scope atomic; no flag is reset, so no memop is left. A first
    try of this spun with a `volatile` load and hung the server: in a bounded test that load returned in 0.03 us a
    poll, too fast to reach host memory, and never saw the CPU's store; the atomic took 1.46 us a poll and saw it;
  - the kernel that picks which misses to fetch read the routes one dependent scalar load at a time and compared
    the whole cache per route: with vector loads and the slots used this step told apart by their usage stamp, it
    takes the same decisions (state compared after every call at the profiles' cache sizes, synthetic Zipf routing)
    in 11.5-19.4 us a layer instead of 17.3-30.4 at 4 rows (159 -> 31 us at 16 rows and an 11k-slot cache), and
    writes the slots apart, so the routing needs no copy;
  - the coordinator answers a step that leaves the CPU no route with zeros without waking its pool.

  EXL3 `xtx-xt-4x262k` with every miss fetched (the CPU gets nothing), against offload 85.8 / 33.9 tok/s: 80.3 / 31.6
  before the last three changes, 82.8 / 33.1 after; NVFP4 `xtx-xt`, one request: 71.0 (first version), 74.3 (fused
  kernels), 78.3 (all).
- **Rows routed to the same expert.** The CPU computed each (row, expert) route on its own, so an expert two rows of a
  step use (MTP verify rows, several requests) was read from RAM, or for EXL3 decoded, once per row: 8-13 % of the
  CPU's routes in these runs (`FREETOKEN_CPU_MOE_ROUTE_STATS=n` prints the share). NVFP4: one task per (expert, row
  block) runs its routes back to back, the second from the core's cache (`FREETOKEN_CPU_MOE_GROUP`); EXL3: each tile
  row is decoded once for all of them. Both bit-identical to one route at a time.
- **EXL3 experts** had no CPU path. The executor reads the EXL3 banks in place (mul1 codebook, 2-4 bits): rotate in
  (`H(x * suh)`), the gate|up products, silu and the rotation into down's domain, the down products, the rotation out
  and the routed sum, AVX2 + F16C. The trellis states are decoded 8 at a time from one 64-bit window, the bytesum of
  `s * 0x83DCD12D` through `vpmaddubsw` / `vpmaddwd`, and the GEMV accumulates `x * bytesum` and adds the affine part
  of mul1 once per column (no per-weight fp16 rounding: relative error ~1.7e-3 on a layer's output with bf16 in and
  out, `tests/moe/test_cpu_moe_exl3.py`). The tiles of a column are 7.7 KB apart and come cold from RAM in serving:
  software prefetch 6 rows ahead took a cold expert from 307 to 95 us, and the group plan as compile-time constants
  the executor's time per route from 104 to 44-48 us warm (31 threads), against ~66 us to copy one over PCIe (the
  cold time was not measured again after that last change).

## Five more levers, measured (evening of 2026-10-11)

Each behind a switch, every arm bit-identical (greedy answers 14/14 between the first and the last arm of each
profile, as long as the expert cache got the same size: it follows the free VRAM at start, and on the 7900 XTX the
desktop's share moves it by a few slots, which changes the evictions and so the texts); decode tok/s, 12 sampled
answers (their texts differ from run to run: the sampler is exact only up to fp32
atomic order, so they compare as means, ~2 % run to run) and two greedy 1024-token answers (same texts across arms).

| Lever | NVFP4 `xtx-xt` (fetch 0.4) | NVFP4 `xtx` (0.25) | EXL3 `xtx` (0.6) | EXL3 `xtx-xt-4x262k` (0.6) |
|---|---|---|---|---|
| none | 80.1, 28.3 each with 4 | 54.0 | 52.5 | 83.3, 34.3 |
| the GPU skips the CPU's routes | **82.6 (+3 %), 28.9 (+2 %)**; greedy +2 / +7 % | **54.5 (+1 %)**; greedy +2 / +3 % | **53.0 (+1 %)**; greedy +2 / +4 % | 82.6, 34.2 |
| + cores by slice size | 82.3, 28.9 | - | - | 82.3, 33.6 |
| + SMT siblings | 53.4, 16.3 | 46.5 (with the skip only) | - | 61.5, 23.5 |
| the skip + EXL3 column blocks of 4 | - | - | 54.0; greedy +2 / 0 % against the skip alone | - |
| `offload` | 78.2 (earlier run), 24.4 | 39.4 (earlier run) | 50.0 | 85.9, 34.2 |

- **The GPU computed the CPU's routes too** (`FREETOKEN_HYBRID_SKIP_CPU_ROUTES`, on): the NVFP4 and EXL3 decode
  kernels run one program per route and read no "inactive" slot, so a route the CPU took went to the GPU as slot 0
  at weight 0 and was computed in full (for EXL3 a whole trellis decode). They now write zeros for a slot of -1
  without a read; the EXL3 rotation and scale kernels read slot 0's scales for it (weight 0). +1-3 %.
- **Cores by slice size** (`FREETOKEN_CPU_MOE_CORES_BY_SLICE`, off): the ranks' slices are uneven (NVFP4 352 / 288
  rows at 0.55, EXL3 384 / 256: its 128-row Hadamard blocks round 0.55 to 0.6), so rank 0 had ~22 % (NVFP4) and 50 %
  (EXL3) more CPU work per core; the pools now could split 18 / 14 and 19 / 13 cores (checked in the logs). No gain:
  the CPU is not what the step waits for at these miss rates (below).
- **SMT** (`FREETOKEN_CPU_MOE_SMT`, off): the pool also on the worker cores' SMT siblings. Alone on the CPU it
  takes a cold EXL3 expert 5-11 % faster (16 cores, 15 -> 30 threads: 62.6-64.0 -> 55.7-56.0 us; 3 runs); in serving
  it costs 14-42 % (the spinning workers then share every core with the server's own threads; not measured further).
- **EXL3 column blocks** (`FREETOKEN_CPU_MOE_EXL3_COLS`): a task takes n adjacent tile columns, the first prefetching
  their contiguous row segments. Alone on the CPU (8 cold experts per task, 2 runs): 31 threads, slice 640, 71 ->
  64-66 us a route with 4 or 8; with 15-19 threads no gain (8: a loss). In serving on `xtx` within the noise (54.0
  against 53.0), so it stays at 1.
- **EXL3 on two cards** has little to win: with `offload`, 1.0-1.4 experts per layer miss at one request (4.5-6 % of
  the active ones) and 2.2-2.5 at 4 (`FT_STATS_EVERY` + `FT_MOE_STATS`); the hybrid split gives the CPU 0.3-0.9 of
  them, which bounds the gain to a few percent, about what the hybrid path still costs. Alone on the CPU, a cold
  expert of rank 0's 384-row slice takes 61 us on 15 threads (54 on 19), against ~42 us to copy it (1.1 MB at the
  26-28 GB/s measured for PCIe, estimated): slower, but it runs alongside the copies; the miss rate is the limit.

## EXL3: +6 % on one card, a loss on two (2026-10-11)

EXL3 3.05 bpw, `FREETOKEN_HYBRID_FETCH_FRACTION` = share of the misses fetched (`ft bench bw` does not bench EXL3):

| Profile | `offload` | `hybrid` |
|---|---:|---:|
| `xtx`, one request, fetch 0.6, everything on | 49.6; 50.0 later | **52.5 (+6 %); 53.0 with the GPU skip** |
| `xtx`, fetch 0.6 / 0.75 (fused kernels) | 49.6 | 51.7 / 50.1 |
| `xtx`, fetch 0.4 / 0.25 (before the fused kernels) | 49.6 | 50.1 / 47.0 |
| `xtx-xt-4x262k`, one request / 4 at once, fetch 0.6, everything on | 85.8 / 33.9 each | 82.5 / 33.3 |
| `xtx-xt-4x262k`, 4 at once, fetch 0.6 fused / 0.6 / 0.4 (before) | 33.9 each | 31.5 / 30.3 / 28.2 |

Decoding the trellis is compute-bound on the CPU (~3 G weights/s per thread, about half that per thread with every
core busy) where NVFP4 is bound by the RAM (~120 G weights/s on the whole CPU), and an EXL3 expert is a third smaller
than an NVFP4 one, so its misses cost less to copy. On two cards, with every miss fetched (the CPU idle) the hybrid
path is still 2-4 % slower than offload (82.8 / 33.1 tok/s), and giving the CPU 40 % of the misses does not win it
back (82.5 / 33.3): few experts miss there (the section above).

## NVFP4 with the K/V in RAM, three and four requests (2026-10-16)

`xtx-xt-4x262k` with `--kv-reserve-tokens 787500` (3 x 262k), 20k-token conversations decoding together, the decode step's period over five 300-step windows (`FT_STATS_EVERY` + `FT_MOE_STATS`; MTP head on, no drafts at three or more):

| Requests at once | `offload` | `hybrid` (the CPU takes 1.8-2.8 of 2.7-4.4 misses a layer) |
|---|---:|---:|
| 3 | 26.1-28.8 ms (32.1-36.4 tok/s each) | 28.1-29.4 ms (31.9-35.3) |
| 4 | 31.5-35.4 ms (29.1-29.8) | 32.9-34.4 ms (29.5-30.0) |

Neither better nor worse within the windows' spread (-4 to +8 %): the +15 % of four requests on `xtx-xt` (K/V in VRAM) does not carry over to this profile (why was not measured; the K/V's reads cross the same host memory).

## Also measured

| NVFP4, first working version | `offload` | `hybrid` |
|---|---:|---:|
| `xtx`, auto split (24.7 % fetched) | 39.4 | 51.7 |
| `xtx`, every miss on the CPU (`--moe-hybrid-max-fetch 0`): the cache stops learning | 39.4 | 29.2 |
| `xtx-xt`, auto split, one request / 4 at once | 78.4 / 24.4 | 71.0 / 27.0 |
| `xtx-xt`, fetch caps 2 / 4, 4 at once | 24.4 | 26.3 / 25.1 |
