# Expert misses computed on the CPU (`--hybrid`)

Off by default, not agent-validated. Measured on a Threadripper 3970X (32 Zen 2 cores, AVX2, 4 x 32 GB DDR4-3200,
STREAM read 72-74 GB/s) with an RX 7900 XTX + 7900 XT on PCIe 4.0 x16.

With `--moe-strategy offload` (every profile), the experts live in RAM and the free VRAM is a cache of them; an expert
a token routes to that is not in the cache (a miss) is copied over PCIe first (26-28 GB/s). Upstream's `hybrid`
strategy computes part of each step's misses on the CPU instead, straight from the RAM banks: only the token's
activation crosses PCIe, the weights do not. Each MoE layer of a decode step then splits its misses: some are fetched
into the cache as before, the rest go to a pool of pinned CPU threads that runs while the GPU computes the hits.

## When it helps

Decode tok/s, MTP head on, the fetch fractions `rdna3/serve.sh --hybrid` sets ([below](#using-it)). One request:
median of 12 sampled 512-token answers (6 prompts x 2, as in
[benchmarks.md](benchmarks.md#how-the-numbers-are-taken)), the `offload` column the benchmark's cell on the same code.
4 at once: 3 rounds of 4 sampled answers, both columns in the same session. Measured on an earlier build than the
[benchmarks.md](benchmarks.md#summary) summary and not re-measured: compare the two columns with each other, not
with the summary.

| Checkpoint, profile | `offload` | `--hybrid` |
|---|---:|---:|
| NVFP4 `xtx`, one request | 42.7 | **62.3 (+46 %)** |
| NVFP4 `xt`, one request | 34.6 | **52.1 (+51 %)** |
| NVFP4 `xtx-xt`, one request | 82.9 | 84.5 (+2 %) |
| NVFP4 `xtx-xt`, 4 at once | 25.2 each, 92-100 in total | **28.9 each (+15 %), 89-114 in total** |
| EXL3 3.05 bpw `xtx`, one request | 54.8 | 57.8 (+5 %) |
| EXL3 3.05 bpw `xtx-xt`, one request | 88.9 | 89.1 (0 %) |

The two-card one-request differences are within the spread of sampled answers: a sampled answer's speed depends on its
text, which changes with the split (the answers of one run spread over +-13 tok/s). The first round of 4 is the
slowest in both columns (92.0 and 89.0 tok/s in total).

**EXL3** gains less. Decoding the trellis is compute-bound on the CPU (~3 G weights/s per thread, about half that per
thread with every core busy) where NVFP4 is bound by the RAM (~120 G weights/s on the whole CPU), and an EXL3 expert is
a third smaller than an NVFP4 one, so its misses cost less to copy. On two cards few of its experts miss: with
`offload`, 1.0-1.4 experts per layer at one request (4.5-6 % of the active ones) and 2.2-2.5 at 4 (`FT_STATS_EVERY` +
`FT_MOE_STATS`); the hybrid split gives the CPU 0.3-0.9 of them, which bounds the gain to a few percent, about what the
hybrid path still costs. Alone on the CPU, a cold expert of rank 0's 384-row slice takes 61 us on 15 threads (54 on
19), against ~42 us to copy it (1.1 MB at the 26-28 GB/s measured for PCIe, estimated): slower, but it runs alongside
the copies; the miss rate is the limit. On `xtx-xt-4x262k` (EXL3 3.05 bpw, measured on an older build than the table
above) `--hybrid` does not pay: one request / 4 at once 82.6 / 34.2 tok/s against 85.9 / 34.2 with `offload`. With every
miss fetched (the CPU idle) the hybrid path is still 2-4 % slower than `offload` there (82.8 / 33.1 against 85.8 /
33.9; measured with the step-number handshake, not with the current task word), and giving the CPU 40 % of the misses
does not win it back (82.5 / 33.3).

**NVFP4 with the K/V in RAM**: `xtx-xt-4x262k` with `--kv-reserve-tokens 787500` (3 x 262k), 20k-token conversations
decoding together, the decode step's period over five 300-step windows (`FT_STATS_EVERY` + `FT_MOE_STATS`; MTP head
on, no drafts at three or more with `FREETOKEN_SPEC_BS_MAX=2` then):

| Requests at once | `offload` | `--hybrid` (the CPU takes 1.8-2.8 of 2.7-4.4 misses a layer) |
|---|---:|---:|
| 3 | 26.1-28.8 ms (32.1-36.4 tok/s each) | 28.1-29.4 ms (31.9-35.3) |
| 4 | 31.5-35.4 ms (29.1-29.8) | 32.9-34.4 ms (29.5-30.0) |

Neither better nor worse within the windows' spread (-4 to +8 %): the +15 % of four requests on `xtx-xt` (K/V in VRAM)
does not carry over to this profile (why was not measured; the K/V's reads cross the same host memory).

## Using it

`rdna3/serve.sh <profile> --hybrid` passes `--moe-strategy hybrid` and the share of each step's misses fetched
(`FREETOKEN_HYBRID_FETCH_FRACTION`, [options.md](options.md)) measured best here: with NVFP4 a quarter on one card (what
`ft bench bw` picks on this machine) and 40 % on two, 60 % with EXL3 (`ft bench bw` does not bench EXL3). With EXL3
on two cards it prints a note: measured even to 2-4 % slower than without.

How the fractions were chosen (decode tok/s; measured on older builds than the first table):

- NVFP4 `xtx`, one request: 15 / 25 / 35 % fetched gave 51.9 / 53.2-54.2 / 50.1 tok/s.
- NVFP4 `xtx-xt`: 25 / 40 / 55 % fetched gave 78.3 / 82.6-82.7 / 78.6 tok/s for one request and 28.6 / 28.6 / 26.9
  each with 4. Why two cards want a larger share fetched, a likely reason, not measured: each rank's pool has half the
  cores and of the RAM's bandwidth but its own PCIe link.
- EXL3 `xtx`, one request (`offload` 49.6): fetch 0.6 / 0.75 gave 51.7 / 50.1; without the hybrid layer's kernels
  ([below](#how-it-works-on-this-hardware)), 0.4 / 0.25 gave 50.1 / 47.0.
- EXL3 `xtx-xt-4x262k`, 4 at once (`offload` 33.9 each): fetch 0.6 gave 31.5; without the hybrid layer's kernels, 0.6
  / 0.4 gave 30.3 / 28.2.
- Every miss on the CPU (`--moe-hybrid-max-fetch 0`; NVFP4 `xtx`, the first working version): the cache stops
  learning, 29.2 tok/s against 39.4 with `offload` and 51.7 with the split `ft bench bw` picks (24.7 % fetched).
- Fetch caps instead of a fraction (`--moe-hybrid-max-fetch` 2 / 4; NVFP4 `xtx-xt`, 4 at once, the first working
  version): 26.3 / 25.1 each, against 27.0 with the `ft bench bw` split and 24.4 with `offload`.

## Costs and caveats

- **The CPU.** The pool spins on its cores while a decode runs and parks 50 ms after the last layer: with `--hybrid`
  on one card, 31 of the 32 cores are busy during generation (15 + 1 per card on two).
- **Other answers than `offload`.** Greedy answers are not bit-identical to offload (the CPU sums in fp32 in another
  order): on `xtx`, 8 of 14 identical, the others diverge after 150-720 characters, all sensible. They are not less
  precise: against an fp64 reference at the model's sizes, one MoE output computed on the CPU is off by 2.3e-3
  (relative), on the GPU by 3.9e-3 (bf16 between its steps), split between them by 3.3-4.0e-3
  ([journey.md](journey.md#27-agent-turns-short-prefills-through-the-expert-cache-2026-10-14)).
- **Answers that move with the expert cache.** Prefill chunks of up to 1280 tokens (agent turns) read their routed
  experts through the cache here too (`FREETOKEN_PREFILL_CACHED_MAX`), which moves the answers as the cache's contents
  do. Time to the first token of a 132-337-token turn at 25k, that path off -> on: NVFP4 `xtx-xt` 1.19-1.24 -> 0.78-0.89
  s, EXL3 3.05 `xtx` 1.50-1.52 -> 1.05-1.18, NVFP4 `xtx` 2.32-2.34 -> 1.63-1.80.
- **Not reproducible.** Two runs of one image (`xtx`, NVFP4, the same settings, greedy) gave the same answers on only 7
  of 14 prompts: `--hybrid` is not reproducible from one run to the next (the cause was not looked for).
- **Long prompts** read at the usual speed (a long prefill streams whole layers to the GPU, as without `--hybrid`): a
  118k-token prompt in 83 s on `xtx`, 61 s on `xtx-xt`, and 9/9 planted questions answered on both.

## How it works on this hardware

- **The GPU <-> CPU handshake.** Upstream's executor signals the CPU pool with stream memory operations. HIP's own
  `hipStreamWriteValue64` / `hipStreamWaitValue64` work, in captured graphs too, on ROCm 10.1 (the PLE disk reads rely
  on them too), and upstream's executor has a HIP backend for them. The hybrid layer's kernels below carry the
  handshake themselves.
- **MTP verify rows.** A verify step runs up to `FREETOKEN_SPEC_VERIFY_M` rows per request. The executor's row bound
  counts them, sized up front, and a larger task is refused; sized for `max(--max-running-requests,
  --cuda-graph-max-bs)` rows only, the C++ scratch grew on the first verify step, in the middle of serving (the EXL3
  scratch below has no such path).
- **Two cards.** Upstream refuses CPU decode with NVFP4 experts under tensor parallelism. Here each rank's executor
  reads its own slices of the banks and its partial joins the GPU partial before the all-reduce, like the GPU path.
  Each rank pins a contiguous half of the cores (without it, both ranks' pools took every physical core).
- **The hybrid layer's fixed cost.** As ported, each MoE layer of a hybrid step added ~12 small graph nodes (a clone of
  the routing, masks, two `where`, a clamp, three device-to-host copies, three stream memops, the copy back and an
  add); a node of a captured graph costs ~3.2 us here, a memop as much as a tiny kernel. On two cards, where one layer
  of a one-request step lasts ~270 us, that cost ~9 % at one request whatever share of the misses the CPU took. What
  this tree does instead:
  - one kernel writes the CPU's inputs straight into its pinned IO rows (a device view of them), another adds the
    CPU's partial back from them; the routing split runs in upstream's hybrid ensure kernel;
  - the two kernels carry the handshake (`FREETOKEN_HYBRID_FLAG_IN_KERNEL`, on) over two pinned words the coordinator
    polls beside upstream's doorbell, with its protocol: the first resets `done` and raises `ready` to the step's task
    word (layer, rows) with system-scope releases, or sets `done` at once when no route goes to the CPU; the
    coordinator takes the task and sets `done`; the second spins on `done` with a system-scope atomic; no stream memop
    is left. The spin needs the atomic: in a bounded test a `volatile` load returned in 0.03 us a poll, too fast to
    reach host memory, and never saw the CPU's store (it hung the server); the atomic took 1.46 us a poll and saw it;
  - the kernel that picks which misses to fetch reads the routes with vector loads and tells the slots used this step
    apart by their usage stamp, where upstream's reads them one dependent scalar load at a time and compares the whole
    cache per route: the same decisions (state compared after every call at the profiles' cache sizes, synthetic Zipf
    routing) in 11.5-19.4 us a layer instead of 17.3-30.4 at 4 rows (159 -> 31 us at 16 rows and an 11k-slot cache);
    it writes the slots apart, so the routing needs no copy;
  - a step that leaves the CPU no route rings nothing (the publish kernel sets `done` itself; upstream's coordinator
    also answers such a doorbell with zeros without waking its pool).

  Measured with the step-number handshake these kernels used before the task word (the task word was not measured):
  EXL3 `xtx-xt-4x262k` with every miss fetched (the CPU gets nothing), against offload 85.8 / 33.9 tok/s: 80.3 / 31.6
  without the last three of these changes, 82.8 / 33.1 with them; NVFP4 `xtx-xt`, one request: 71.0 (first version),
  74.3 (the hybrid layer's kernels), 78.3 (all).
- **The GPU skips the CPU's routes** (`FREETOKEN_HYBRID_SKIP_CPU_ROUTES`, on). The NVFP4 and EXL3 decode kernels run
  one program per route; with the skip off they read no "inactive" slot, so a route the CPU takes goes to the GPU as
  slot 0 at weight 0 and is computed in full (for EXL3 a whole trellis decode). With it they write zeros for a slot of
  -1 without a read; the EXL3 rotation and scale kernels read slot 0's scales for it (weight 0). +1-3 %
  ([the table below](#levers-measured)).
- **Rows routed to the same expert.** Computed one (row, expert) route at a time, an expert two rows of a step use
  (MTP verify rows, several requests) is read from RAM, or for EXL3 decoded, once per row: 8-13 % of the CPU's routes
  in these runs (`FREETOKEN_CPU_MOE_ROUTE_STATS=n` prints the share). NVFP4: one task per (expert, row block) runs its
  routes back to back, the second from the core's cache (`FREETOKEN_CPU_MOE_GROUP`); EXL3: each tile row is decoded
  once for all of them. Both bit-identical to one route at a time.
- **EXL3 experts** have no CPU path upstream. The executor reads the EXL3 banks in place (mul1 codebook, 2-4 bits):
  rotate in (`H(x * suh)`), the gate|up products, silu and the rotation into down's domain, the down products, the
  rotation out and the routed sum, AVX2 + F16C. The trellis states are decoded 8 at a time from one 64-bit window, the
  bytesum of `s * 0x83DCD12D` through `vpmaddubsw` / `vpmaddwd`, and the GEMV accumulates `x * bytesum` and adds the
  affine part of mul1 once per column (no per-weight fp16 rounding: relative error ~1.7e-3 on a layer's output with
  bf16 in and out, `tests/moe/test_cpu_moe_exl3.py`). The tiles of a column are 7.7 KB apart and come cold from RAM in
  serving: software prefetch 6 rows ahead takes a cold expert from 307 to 95 us, and the group plan as compile-time
  constants the executor's time per route from 104 to 44-48 us warm (31 threads), against ~66 us to copy one over PCIe
  (the cold time was measured before that last change).

## Levers measured

Each lever behind a switch, every arm bit-identical (greedy answers 14/14 between the first and the last arm of each
profile, as long as the expert cache got the same size: it follows the free VRAM at start, and on the 7900 XTX the
desktop's share moves it by a few slots, which changes the evictions and so the texts). Decode tok/s, 12 sampled
answers (their texts differ from run to run: the sampler is exact only up to fp32 atomic order, so they compare as
means, ~2 % run to run) and two greedy 1024-token answers (same texts across arms); measured on an older build than the
first table.

| Lever | NVFP4 `xtx-xt` (fetch 0.4) | NVFP4 `xtx` (0.25) | EXL3 `xtx` (0.6) | EXL3 `xtx-xt-4x262k` (0.6) |
|---|---|---|---|---|
| none | 80.1, 28.3 each with 4 | 54.0 | 52.5 | 83.3, 34.3 |
| the GPU skips the CPU's routes (kept, on) | **82.6 (+3 %), 28.9 (+2 %)**; greedy +2 / +7 % | **54.5 (+1 %)**; greedy +2 / +3 % | **53.0 (+1 %)**; greedy +2 / +4 % | 82.6, 34.2 |
| + cores by slice size | 82.3, 28.9 | - | - | 82.3, 33.6 |
| + SMT siblings | 53.4, 16.3 | 46.5 (with the skip only) | - | 61.5, 23.5 |
| the skip + EXL3 column blocks of 4 | - | - | 54.0; greedy +2 / 0 % against the skip alone | - |
| `offload` | 78.2 (another session), 24.4 | 39.4 (another session) | 50.0 | 85.9, 34.2 |

- **Cores by slice size** (`FREETOKEN_CPU_MOE_CORES_BY_SLICE`, off): the ranks' slices are uneven (NVFP4 352 / 288
  rows at 0.55, EXL3 384 / 256: its 128-row Hadamard blocks round 0.55 to 0.6), so rank 0 has ~22 % (NVFP4) and 50 %
  (EXL3) more CPU work per core; the pools can split 18 / 14 and 19 / 13 cores instead (checked in the logs). No gain:
  the CPU is not what the step waits for at these miss rates ([When it helps](#when-it-helps)).
- **SMT** (`FREETOKEN_CPU_MOE_SMT`, off): the pool also on the worker cores' SMT siblings. Alone on the CPU it takes a
  cold EXL3 expert 5-11 % faster (16 cores, 15 -> 30 threads: 62.6-64.0 -> 55.7-56.0 us; 3 runs); in serving it costs
  14-42 % (the spinning workers then share every core with the server's own threads; not measured further).
- **EXL3 column blocks** (`FREETOKEN_CPU_MOE_EXL3_COLS`, 1): a task takes n adjacent tile columns, the first
  prefetching their contiguous row segments. Alone on the CPU (8 cold experts per task, 2 runs): 31 threads, slice 640,
  71 -> 64-66 us a route with 4 or 8; with 15-19 threads no gain (8: a loss). In serving on `xtx` within the noise
  (54.0 against 53.0), so it stays at 1.
