# Benchmarks

Everything here was measured on one machine: RX 7900 XTX 24 GB + RX 7900 XT 20 GB (gfx1100, PCIe 4.0 x16 each, no GPU
peer-to-peer), Threadripper 3970X, 128 GB DDR4-3200, Qwen3.8-Flash-Next NVFP4 (RadixArk checkpoint; the experimental
EXL3 checkpoints are measured in
[how-it-works.md](how-it-works.md#exl3-checkpoints)). The `xtx-xt` rows of the
summary, the two-card depth sweeps, the several-request table and the MTP step costs were measured on 2026-10-06 on
the ROCm 10.0 / PyTorch 2.13 image with its own Triton 3.8: each configuration as shipped (MTP head on) and the same
with `--no-mtp`, one server per configuration, with the host settings of
[troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run). The one-card rows and sweeps were taken
on 2026-10-03/04 with the image's earlier Triton 3.7.1 (on two cards Triton 3.8 decodes within 1 % of it and reads
prompts ~5 % faster: [journey.md](journey.md#18-rocm-10s-own-triton-2026-10-05)); the one-card sweeps with the head
ran with a variant of the adaptive depth dropped before the release, and the released image gave the same within
4 % (`xtx` 33.1-43.2, `xt` 27.1-34.9 tok/s). Other sections give their own date.

## How the numbers are taken

- **Decode, one request**: tokens/s of 512-token answers after the first token, thinking off: the median of 12 answers
  (6 prompts x 2: English and French prose, code, a recipe) at the model's default sampling, and 3 greedy answers.
- **Cold prefill (PP)**: an 8.3k-token prompt with nothing cached, time to first token (`rdna3/bench/quick_bench.py`).
- **Depth sweep** (`rdna3/bench/depth_sweep.py`): one conversation grown in steps up to the profile's maximum context.
  The first step is a cold ~8.8k-token read; each later step adds a 6-41k-token block the cache has not seen, like a
  large tool result (ten steps to 248k on two cards, seven to 124k on one). At each step: **TG** = greedy decode over
  256 tokens, **PP** = speed of reading that block, **agent turn** = time to the first token of a following ~1k-token
  turn. Median of two passes, each with its own text. With the MTP head, TG follows how many of its guesses the text
  lets it keep (1.6-3.5 tokens per step): the spread between steps is the text, not noise.

## Summary

| Profile | Decode, one request: sampled / greedy | TG, 9k to max depth | Cold 8.3k prompt (PP) | PP of a new block | Agent turn, 9k -> max | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| `xtx-xt` | 77.9 / 73-85 tok/s | 70-88 tok/s | 4.0 s, ~2070 tok/s | 1740-2050 tok/s | 1.3 -> 2.0 s | 248k |
| `xtx-xt --no-mtp` | 58.5 / 59-62 tok/s | 53.5-58.5 tok/s | 4.0 s, ~2080 tok/s | 1800-2060 tok/s | 1.3 -> 2.0 s | 248k |
| `xtx` | 38.8 / 38-44 tok/s | 33-42 tok/s | 7.7 s, ~1080 tok/s | 1130-1510 tok/s | 2.4 -> 2.7 s | 124k |
| `xtx --no-mtp` | 35.8 / 35-38 tok/s | 29.5-39 tok/s | 7.6 s, ~1095 tok/s | 1140-1500 tok/s | 2.3 -> 2.7 s | 124k |
| `xt` | 31.7 / 30-36 tok/s | 27-34 tok/s | 8.3 s, ~1000 tok/s | 1040-1340 tok/s | 2.5 -> 2.8 s | 124k |
| `xt --no-mtp` | 29.5 / 28-32 tok/s | 25-32 tok/s | 8.2 s, ~1020 tok/s | 1070-1340 tok/s | 2.5 -> 2.8 s | 124k |
| `xtx-xt`, EXL3 3.05 bpw | 80.5 / 76-89 tok/s | 86-107 tok/s | 3.8 s, ~2190 tok/s | 1830-2170 tok/s | 0.9 -> 1.6 s | 248k |
| `xtx-xt`, EXL3 4.05 bpw | 74.9 / 70-80 tok/s | 77-90 tok/s | 3.8 s, ~2170 tok/s | 1810-2170 tok/s | 1.2 -> 2.0 s | 248k |
| `xtx`, EXL3 3.05 bpw | 50.0 / 48-54 tok/s | 42-52 tok/s | 6.7 s, ~1245 tok/s | 1280-1590 tok/s | 1.5 -> 1.9 s | 124k |
| `xtx`, EXL3 4.05 bpw | 40.5 / 38-45 tok/s | 36-44 tok/s | 7.3 s, ~1140 tok/s | 1200-1590 tok/s | 2.1 -> 2.4 s | 124k |
| `xt`, EXL3 3.05 bpw | 38.7 / 37-43 tok/s | 33-42 tok/s | 7.3 s, ~1145 tok/s | 1200-1420 tok/s | 1.6 -> 2.0 s | 124k |
| `xt`, EXL3 4.05 bpw | 32.0 / 30-35 tok/s | 28-35 tok/s | 7.9 s, ~1055 tok/s | 1120-1420 tok/s | 2.2 -> 2.6 s | 124k |

The EXL3 rows ([how-it-works.md](how-it-works.md#exl3-checkpoints)) were measured the same way, one server per row
started by `rdna3/serve.sh` (on two cards it lowers `--memory-ratio` to 0.77 / 0.76 for EXL3), MTP head on, the same
scripts: the 12 sampled and 3 greedy answers, `quick_bench.py` for the
cold read, the depth sweep. The checkpoints answer differently, so the MTP head keeps a different share of its
guesses: read the decode columns across checkpoints as an order of magnitude.

With the head, one request decodes 33 % faster on two cards at the model's sampling (+29 to +64 % greedy along the
sweep) and 7-8 % faster on one card (+6 to +12 % greedy); reading a prompt costs 0-4 % more, since the head reads it
too. On one card one of the six sampled prompts (a long story) is 3-4 % slower with it. With more requests
decoding than `FREETOKEN_SPEC_BS_MAX` (2 in `xtx-xt`, 1 in the other profiles), the head only writes its KV: see
the table further down.

Decode loses ~6 % from 9k to 248k on two cards and 20-24 % from 9k to 124k on one card. Where the
one-card drop comes from (`xtx`, one pass with `FT_STATS_EVERY=64 FT_MOE_STATS=1`): the decode step takes ~27 ms of
GPU time at 9-16k and ~33 ms from 25k to 124k, flat beyond 25k, while the expert cache misses ~2.2 then 3.0-3.6 of
the 10 experts a layer reads: the longer context spreads the routing over more experts (the misses do not fall during
an answer, so it is not the prefill evicting the cache), and attention itself adds little.

## Depth sweep, `xtx-xt`

| Depth | TG with the head / without | PP new, with / without | Agent turn, with / without |
|---:|---:|---:|---:|
| 8.8k | 74.7 / 58.1 | 1323 / 2059 | 1.29 / 1.27 s |
| 16.0k | 83.8 / 58.5 | 1959 / 1938 | 1.31 / 1.29 s |
| 25.0k | 87.5 / 54.7 | 2012 / 1996 | 1.33 / 1.30 s |
| 40.0k | 82.8 / 54.1 | 2046 / 2027 | 1.35 / 1.33 s |
| 60.0k | 88.4 / 54.0 | 1952 / 2005 | 1.41 / 1.38 s |
| 86.9k | 78.3 / 53.5 | 1970 / 2008 | 1.48 / 1.46 s |
| 122.9k | 84.4 / 54.2 | 1899 / 1953 | 1.57 / 1.56 s |
| 164.9k | 73.6 / 53.5 | 1856 / 1911 | 1.75 / 1.71 s |
| 206.9k | 72.8 / 53.7 | 1795 / 1852 | 1.92 / 1.94 s |
| 247.9k | 70.4 / 54.7 | 1744 / 1797 | 2.01 / 2.02 s |

The first step with the head ran right after the server's start (one-time kernel preparation: its PP is low); the
run without the head came after other measurements on the same server.

## Depth sweep, one card

| Depth | `xtx` TG with / without the head | `xtx` PP new | `xtx` turn | `xt` TG with / without | `xt` PP new | `xt` turn |
|---:|---:|---:|---:|---:|---:|---:|
| 8.8k | 41.7 / 38.8 | 1129 / 1143 | 2.37 / 2.34 s | 33.7 / 31.7 | 1040 / 1069 | 2.48 / 2.45 s |
| 16.0k | 37.5 / 34.7 | 1434 / 1431 | 2.38 / 2.35 s | 30.3 / 28.5 | 1276 / 1284 | 2.48 / 2.46 s |
| 25.0k | 35.5 / 32.0 | 1507 / 1498 | 2.39 / 2.38 s | 28.9 / 26.6 | 1338 / 1340 | 2.50 / 2.47 s |
| 40.0k | 35.6 / 31.7 | 1454 / 1463 | 2.43 / 2.41 s | 29.4 / 26.6 | 1297 / 1343 | 2.55 / 2.53 s |
| 60.0k | 33.8 / 30.2 | 1372 / 1400 | 2.46 / 2.46 s | 27.9 / 25.3 | 1250 / 1303 | 2.57 / 2.56 s |
| 86.9k | 33.5 / 29.9 | 1315 / 1349 | 2.59 / 2.54 s | 27.4 / 25.0 | 1200 / 1249 | 2.68 / 2.67 s |
| 123.9k | 33.1 / 29.5 | 1358 / 1398 | 2.73 / 2.65 s | 27.1 / 24.8 | 1234 / 1280 | 2.82 / 2.81 s |

PP is the same with and without the head within 4 %; one card reads 8k-token chunks (16k on two).

## Depth sweep, EXL3 checkpoints

Median of two passes, MTP head on; two cards on 2026-10-06 (Triton 3.8), one card on 2026-10-04 (Triton 3.7.1):

| Depth | `xtx-xt` 3.05 bpw TG / PP new / turn | `xtx-xt` 4.05 bpw TG / PP new / turn |
|---:|---:|---:|
| 8.8k | 85.7 / 2173 / 0.87 s | 77.2 / 2168 / 1.21 s |
| 16.0k | 97.6 / 2076 / 0.91 s | 81.5 / 2064 / 1.23 s |
| 25.0k | 95.1 / 2099 / 0.90 s | 89.5 / 2091 / 1.25 s |
| 40.0k | 90.9 / 2132 / 0.94 s | 82.9 / 2115 / 1.29 s |
| 60.0k | 102.5 / 2074 / 0.98 s | 88.4 / 2013 / 1.33 s |
| 86.9k | 91.6 / 2064 / 1.05 s | 80.0 / 2040 / 1.43 s |
| 122.9k | 107.0 / 2013 / 1.17 s | 86.1 / 1965 / 1.52 s |
| 164.9k | 90.3 / 1950 / 1.30 s | 79.9 / 1926 / 1.70 s |
| 206.9k | 87.3 / 1890 / 1.45 s | 79.0 / 1874 / 1.79 s |
| 247.9k | 87.1 / 1832 / 1.57 s | 79.5 / 1809 / 2.00 s |

| Depth | `xtx` 3.05 bpw | `xtx` 4.05 bpw | `xt` 3.05 bpw | `xt` 4.05 bpw |
|---:|---:|---:|---:|---:|
| 8.8k | 52.4 / 1284 / 1.51 s | 44.2 / 1201 / 2.07 s | 41.7 / 1201 / 1.64 s | 34.6 / 1117 / 2.20 s |
| 16.0k | 49.4 / 1594 / 1.53 s | 41.4 / 1569 / 2.10 s | 39.0 / 1413 / 1.65 s | 32.5 / 1395 / 2.21 s |
| 25.0k | 44.0 / 1574 / 1.55 s | 37.5 / 1594 / 2.12 s | 34.5 / 1419 / 1.68 s | 29.2 / 1423 / 2.23 s |
| 40.0k | 43.9 / 1522 / 1.60 s | 36.7 / 1524 / 2.16 s | 34.2 / 1379 / 1.71 s | 28.8 / 1396 / 2.28 s |
| 60.0k | 44.9 / 1478 / 1.64 s | 36.3 / 1441 / 2.21 s | 35.0 / 1342 / 1.77 s | 28.4 / 1322 / 2.34 s |
| 86.9k | 44.4 / 1414 / 1.75 s | 36.5 / 1378 / 2.31 s | 35.4 / 1292 / 1.87 s | 28.3 / 1259 / 2.44 s |
| 123.9k | 41.8 / 1428 / 1.88 s | 35.8 / 1427 / 2.43 s | 33.2 / 1298 / 2.00 s | 27.8 / 1289 / 2.57 s |

Each cell: TG (tok/s) / PP of the new block (tok/s) / agent turn. The agent turns get their first token sooner than
with NVFP4 (0.9-1.6 s against 1.3-2.0 s on two cards at 3.05 bpw), consistent with a turn's prefill streaming every
expert of a layer over PCIe ([how-it-works.md](how-it-works.md#prefill)) and a 3.05 bpw expert weighing 1.86 MB
against 2.76 MB.

## Agent turns on real code (`xtx-xt`)

Conversations built from this repository's own source and docs (77-113k tokens), then 8 turns each: a ~2k-token tool
result and a ~250-token answer, temperature 0.6 (what the reference machine's agent client sends), 2026-10-03:

| Per request | With the head | Without |
|---|---:|---:|
| One conversation: decode | 71.1 tok/s | 55.8 tok/s |
| One conversation: time to the first token of a turn | 1.75 s | 1.73 s |
| Two conversations at once (sub-agents), `FREETOKEN_SPEC_BS_MAX=1`: decode | 31.1 tok/s | 33.2 tok/s |
| Same, `FREETOKEN_SPEC_BS_MAX=2` (the profile's since 2026-10-04): decode | 33.6-34.4 tok/s | |
| Two conversations at once: time to the first token | 1.96 s | 1.95 s |

Two requests share each decode step. With `FREETOKEN_SPEC_BS_MAX=1` the head then only writes its KV, so each gets
~31-33 tok/s at ~100k of context, with or without it; with `2` it keeps verifying drafts for both. On the reference
machine's own launcher settings (image input on, 2026-10-04): 36.4 against 31.6 tok/s each, and with thinking on 38.8
against 34.2 on the turns the two decode together; two conversations of 115k and 121k tokens decoded together with no
out-of-memory retry.

Endurance, 2026-10-04: 2 h 07 of rounds of this load as shipped then (`FREETOKEN_SPEC_BS_MAX=1`; one conversation, then two at once, 6 turns each, then
4 short requests and `quick_bench.py`; 90 conversations at 70-121k tokens, 540 turns). One conversation decoded at 69.3
tok/s median over the 30 rounds (65.9-76.9), turns of two at once at 31.3 each (204 turns timed from round 12 on), its
cold read (68-93k tokens) at 1860 tok/s, 4 short requests took 10.1-10.3 s; no GPU queue stop, no out-of-memory retry,
no error, memory and VRAM flat after the first round.

## Several requests and agents at once (`xtx-xt`)

| Workload | Result |
|---|---|
| Decode, 1 / 2 / 3 / 4 requests at once, greedy 512-token answers, counted while every request decodes (2026-10-06, Triton 3.8, median of two passes) | with the head 72.3 / 81.9 / 82.2 / 89.2 tok/s in total (72.3 / 40.9 / 27.4 / 22.3 per request); without it 59.5 / 79.2 / 85.6 / 92.3 (59.5 / 39.6 / 28.6 / 23.1) |
| 1 / 2 / 3 / 4 requests at once, greedy answers of 93-200 tokens, time from the send to the last token (ROCm 10, two rounds) | with the head 3.6 / 5.9 / 8.4-8.6 / 10.1 s; without it 4.1-4.2 / 5.9-6.0 / 8.2-8.3 / 9.8 s |
| 4 agents, 82-117k-token conversations, 3 turns each (together more than the GPUs hold) | the 3 rounds of turns after the first reads: 35 s, first token of each turn 1.9-2.3 s; without the RAM tier 599 s and ~47 s per turn (1.15M tokens recomputed) |
| 4 agents x 12 short turns, small GPU pool | 94 s instead of 133 s without the RAM tier |
| 5 requests with different sampling settings (greedy, T=1, top-p only, top-k 1000 + top-p decoding together; top-k 20 + top-p queued behind them), 256-token answers | ~27-28 tok/s for each of the 4 decoding together |

With the head and `FREETOKEN_SPEC_BS_MAX=1`, only a request decoding alone verifies drafts: with two or more, steps run
one row per request and the head only writes its KV, an eager pass that costs 3-4 % at 3-4 requests; a lone request finishes 13 % sooner.

A request's decode is computed the same way alone or batched (per-row kernels); prompts prefilled in the same batch can
differ at rounding level, as they already do with prefix caching.

## Speculative decoding (MTP head)

`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4` (adaptive depth), 2026-10-03, each profile without the head against the same plus the head (one server
per configuration; the odd half of the 120-item set). Greedy answers identical with and without the head on every
profile (5 chat prompts, the 60 items, the 108k-token turns, the functional checks). The one-card columns ran with
the two-card step costs, so mostly 4 rows; their 2-row figures follow the table:

| One request, decode tok/s (or time) | `xtx-xt` | `xtx` (one card, 4 rows) | `xt` (one card, 4 rows) |
|---|---:|---:|---:|
| 3 chat prompts x 2, median | 57.7 -> 74.8 | 38.0 -> 34.6 | 31.2 -> 27.6 |
| 3 agent turns on a 3.5k-token prompt | 51.5 / 58.9 / 54.3 -> 71.8 / 65.4 / 66.5 | 34.9 / 38.5 / 32.4 -> 34.4 / 29.9 / 29.8 | 28.8 / 30.7 / 27.4 -> 26.8 / 23.8 / 23.6 |
| 108k-token prompt, then a follow-up turn | 56.4 / 55.4 -> 71.0 / 76.2 | 35.1 / 35.1 -> 32.3 / 34.6 | 29.1 / 28.8 -> 26.1 / 27.9 |
| Time to the first token of that prompt | 56.8 -> 59.2 s | 78.7 -> 82.4 s | 84.5 -> 88.4 s |
| One 1500-token answer | 59.8 -> 73.9 | 41.4 -> 39.6 | 33.4 -> 31.2 |
| 60 code / math items (time) | 296 -> 215 s | 516 -> 495 s | 613 -> 591 s |
| Tokens kept per request step (512-step averages) | 1.9-3.9 | 2.1-3.9 | 2.1-3.9 |

On one card the extra rows cost more: its expert cache holds ~2.3k (XT) or ~3.5k (XTX) of the 24.6k experts against
~7k per card on `xtx-xt`, and a 4-row step on the XT misses 8-10 experts per layer against 2-3 in plain decode (all
copied over PCIe): 93-98 ms against ~31. At 2 rows (53 ms) the head pays; the one-card step costs make the adaptive
depth keep to 2 rows (`xt`: chat 32.6 / 29.8 / 36.3 tok/s against 30.9 / 28.9 / 32.2 without the head). Measured at 2
rows, answers identical to plain decode:

| One request, decode tok/s (or time) | `xtx` | `xt` |
|---|---:|---:|
| 3 chat prompts x 2, median | 38.0 -> 40.3 | 31.2 -> 32.7 |
| 3 agent turns on a 3.5k-token prompt | 34.9 / 38.5 / 32.4 -> 37.6 / 40.3 / 34.7 | 28.8 / 30.7 / 27.4 -> 29.8 / 31.7 / 28.5 |
| 108k-token prompt, then a follow-up turn | 35.1 / 35.1 -> 38.0 / 39.0 | 29.1 / 28.8 -> 31.5 / 31.7 |
| 60 code / math items (time) | 516 -> 473 s | 613 -> 563 s |

Step costs on the ROCm 10 image with Triton 3.8 (`xtx-xt`, NVFP4, `FREETOKEN_SPEC_TIMING=64` over the decode, concurrency and
prompt-reading runs of 2026-10-06, contexts up to ~41k tokens; GPU ms per step, phases summed):

| Requests x rows | 1 x 2 | 1 x 3 | 1 x 4 | 2 x 2 | 2 x 3 | 2 x 4 | 3 x 1 | 4 x 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Step (forward + verify + head + draft chain) | 25.3 | 28.8 | 36.4 | 39.7 | 53.1 | 65.0 | 36.3 | 44.5 |
| Of which the forward | 21.4 | 25.2 | 31.7 | 35.5 | 48.6 | 59.1 | 35.6 | 43.9 |
| Steps measured | 1640 | 1745 | 1434 | 25 | 68 | 272 | 1022 | 1022 |

With three or four requests a step runs one row each and the head only writes its KV.

## Precision

- **Scored benchmark** (671 items: HumanEval 164, MBPP 257, GSM8K 250; greedy, thinking off): the community ROCm port
  (bf16) 636, the ROCm port this build started from (bf16) 632, the int8-dense builds 635. Two bf16 builds already
  differ by 4 items: no measurable loss.
- **Agentic benchmark** (the author's, private: a coding agent in OpenCode fixing 29 bugs planted in a ~50k-line
  codebase, long sessions, single runs): builds with the int8 attention KV cache fixed 8-9 bugs, builds with the bf16 KV
  12-14. The int8 KV is out of every profile ([options.md](options.md)).
- **What the agentic benchmark covered**: the two-card builds that scored 12-13 had the uneven split, int8 dense
  layers, one MoE all-reduce per block, the split-K GEMVs, the top-k-first sampler, 16k chunks with blocked PLE,
  expert reuse in prefill, the host-memory all-reduce and the tuned prefill tiles. Added after the last scored build,
  and checked for exactness instead: the tuned decode tiles, up to 4 concurrent requests (a request's decode is
  identical alone or batched; prompts prefilled together can differ at rounding level), the RAM tier, the sampler
  rework for requests without a small `top_k`, and rank 0's token broadcast. The one-card profiles were not run on it.
  The ROCm 10 image with the MTP head (`xtx-xt`) scored 12 and
  18 in two runs; the same build with the EXL3 checkpoints ([how-it-works.md](how-it-works.md#exl3-checkpoints))
  scored 13 at 3.05 bpw and 12 at 4.05 bpw, one run each.
- **Exactness checks**: a fixed set of 120 greedy prompts (not published) gave identical answers for the host-memory
  all-reduce, the tuned NVFP4 tiles (prefill and decode), expert reuse in prefill, a single request on the 4-request
  build, and the review-fix branch against the swept one. Kernel checks prove the blocked prefill MoE and batched decode
  rows identical (`moe_block_check.py`, `int8_rows_check.py`); four conversations pushed out to the RAM tier and brought
  back give the same greedy answers (and `host_kv_check.py` checks the copies).

## Images (`--vision`, `xtx-xt`)

Measured on 2026-09-30 on a build of the release tree plus the image changes (`rdna3/serve.sh --vision`), with images
capped at 1024 tokens unless stated otherwise.

**Text with the vision build.** Four sessions ran interleaved (text-only, vision, text-only, vision), each with three
`quick_bench.py` runs; the ranges below span both sessions of each mode:

| | text-only | `--vision` |
|---|---:|---:|
| decode, 512 tokens | 53.1-55.1 tok/s | 53.3-54.3 tok/s |
| cold read, 8.3k tokens | 4.28-4.40 s | 4.37-4.40 s |
| agent turn, ~1.5k new tokens | 1.27-1.29 s | 1.27-1.28 s |
| expert cache (both ranks take the smaller plan) | 7,228 | 7,217 |
| greedy answers, 34 prompts of 20 to 20k tokens (thinking off) | reference | identical in every session |

With the tower on both ranks (the first version), rank 1 planned 351 fewer experts (6,877): rank 1 never encodes, so
it now holds no tower.

**Depth, two cards.** The depth sweep of this page ran four times, interleaved (text-only, vision, text-only,
vision). Each ran on a fresh server, with the same unseen text for the two modes of a pass and greedy 256-token
answers (`ignore_eos`). The depths land a little below the sweeps above:

| Depth | TG text (2 passes) | TG `--vision` | PP new, text / vision | Agent turn, text / vision |
|---:|---:|---:|---:|---:|
| 9.2k | 55.3 / 55.4 | 54.5 / 54.0 | 1496 / 1505 | 1.27 / 1.27 s |
| 16.6k | 55.3 / 56.5 | 55.8 / 56.2 | 1840 / 1838 | 1.28 / 1.28 s |
| 35.9k | 54.9 / 55.7 | 55.4 / 55.4 | 1902 / 1890 | 1.33 / 1.33 s |
| 73.9k | 55.3 / 56.1 | 52.5 / 55.3 | 1942 / 1944 | 1.44 / 1.44 s |
| 103.2k | 56.5 / 56.2 | 56.0 / 55.0 | 1878 / 1876 | 1.51 / 1.53 s |
| 139.1k | 54.5 / 55.5 | 56.7 / 53.2 | 1811 / 1806 | 1.66 / 1.67 s |
| 166.3k | 56.1 / 56.1 | 56.4 / 54.6 | 1751 / 1756 | 1.74 / 1.72 s |
| 202.9k | 55.4 / 55.3 | 55.1 / 55.1 | 1727 / 1731 | 1.88 / 1.85 s |
| 232.3k | 56.5 / 57.0 | 54.5 / 56.1 | 1676 / 1676 | 1.94 / 1.93 s |
| 253.5k | 54.7 / 55.2 | 54.5 / 54.3 | 1617 / 1606 | 1.98 / 2.00 s |

At the full depth (255.6k tokens), vision was asked about a screenshot at the end of the conversation. It read the
id in both passes (first token in 2.5 s).

**Depth, one card** (`xtx` and `xt`, two passes of each mode, same method, to 124k):

| Depth | `xtx` TG text | `xtx` TG vision | `xtx` PP text / vision | `xt` TG text | `xt` TG vision | `xt` PP text / vision |
|---:|---:|---:|---:|---:|---:|---:|
| 8.7k | 35.3 / 35.4 | 34.2 / 34.7 | 731 / 939 | 28.1 / 29.0 | 27.9 / 27.9 | 860 / 879 |
| 16.4k | 43.9 / 35.6 | 34.3 / 34.3 | 1417 / 1412 | 34.2 / 28.1 | 27.5 / 28.0 | 1258 / 1264 |
| 32.8k | 35.2 / 34.3 | 34.8 / 34.6 | 1468 / 1468 | 27.7 / 28.2 | 33.5 / 27.2 | 1340 / 1344 |
| 66.1k | 34.8 / 34.7 | 34.0 / 34.2 | 1452 / 1456 | 28.5 / 37.4 | 27.4 / 28.0 | 1328 / 1328 |
| 97.7k | 35.1 / 37.7 | 40.3 / 43.9 | 1397 / 1408 | 28.7 / 36.8 | 33.9 / 27.3 | 1274 / 1277 |
| 123.2k | 35.3 / 38.7 | 34.9 / 41.8 | 1374 / 1378 | 28.9 / 37.3 | 36.0 / 29.0 | 1248 / 1254 |


On one card the tower sits on the only rank, and the expert cache shrinks by ~165 experts: `xtx` 3,653 -> 3,487
(-4.5 %), `xt` 2,420 -> 2,255 (-6.8 %). That is 436 MiB. Measured with the tower alone, the streamed weights and
their staging take 130 MiB, and the first encode takes ~170 MiB more outside PyTorch's allocator (94 MiB of it for
the patch embedding's Conv3d, in MIOpen). Decode loses ~1.5-2 % outside the steps where both modes jump (35-44 tok/s on
the XTX, 33-37 on the XT, depending on the generated text): `xtx` ~35.2 -> ~34.5 tok/s, `xt` ~28.4 -> ~27.8 tok/s.
Prompt reading is unchanged, and the agent turn is 0-1.5 % longer. The screenshot at 125.3k was read in all four vision
passes (first token in 3.2-3.3 s).

**The tower alone** (`rdna3/bench/vision_tower_bench.py`; one card, weights streamed from RAM, seeded random pixels):

| image tokens | pixels | XTX | XT | peak VRAM beyond the weights |
|---:|---:|---:|---:|---:|
| 256 | 0.3 MP | 33 ms | 33 ms | 25 MiB |
| 1024 | 1.0 MP | 91 ms | 107 ms | 97 MiB |
| 2048 | 2.1 MP | 268 ms | 307 ms | 142 MiB |
| 4096 | 4.2 MP | 871 ms | 1012 ms | 253 MiB |
| 8192 | 8.4 MP | 3.2 s | 3.8 s | 371 MiB |
| 16384 | 16.8 MP | 12.1 s | 15.0 s | 742 MiB |

- **Attention memory is linear.** PyTorch's attention runs its AOTriton kernel (`attn_fwd`) on gfx1100, so memory
  grows linearly with the image; nothing ran out of memory up to 16384 tokens.
- **Weight placement.** Streaming the weights from RAM leaves 294 MiB of them in VRAM; keeping them all resident
  (`--mm-encoder-weights gpu`) takes 1034 MiB and gains nothing (106 against 107 ms at 1024 tokens on the XT).
- **Both cards agree.** The XTX and the XT gave bit-identical embeddings at nine sizes from 64 to 8192 tokens.

**Requests** (thinking off, 16 to 120 output tokens):

- **Small print.** A 1920x1080 screenshot of 30 lines of 16-pixel text becomes 1055 prompt tokens at the 1024 cap. 3
  ids of 3 were read, in 1.6 s; at a 2048 cap it was 2087 tokens, 1.9 s, same answers.
- **Concurrent requests.** Four sampled image requests at once were all correct, with 0 rank disagreements under
  `FREETOKEN_TP_SYNC_TOKENS=check`.
- **Repeated image and follow-up turn.** The same image twice in one prompt was answered correctly. A follow-up turn
  reused the cached image prefix (2048 cached tokens, 52 new).
- **Latency.** The first image request after a start took 3.7 s; later ones took 1.8-2.0 s for a 1034-token prompt.
  The first request of any kind after a start takes ~10 s in both modes.

## Memory while serving (`xtx-xt`)

| | Used (ROCm 10, MTP head on, 2026-10-03) |
|---|---|
| XTX VRAM | ~21.3 GiB of 24.0 while decoding (the desktop's share included), up to ~23.7 GiB while a 16k-token chunk is read deep in the context |
| XT VRAM | ~18.0 GiB of 20.0 while decoding, up to ~19.9 GiB while a 16k-token chunk is read deep in the context |
| Host RAM | 82 GiB resident over the two processes, all of it locked (`FREETOKEN_MLOCK`): ~63 GiB of experts (each rank pins its own share of every expert), the RAM tier (4.6 + 4.9 GiB with the snapshots), the rest the processes |
| Expert cache | 7,127 experts on each card with the head (7,395 without), of 24,576: each rank plans on its own free memory, then both take the smaller plan |

VRAM is the driver's count (`mem_info_vram_used`) sampled twice a second over a depth sweep to 248k: the ~2 GiB left
free per card while decoding is the prefill's working memory, which a deep 16k-token chunk uses almost entirely (56 MiB
left on the XT). Host RAM is the kernel's `Mlocked` count after start.

**Every profile and checkpoint** (2026-10-04, one server each, `rdna3/serve.sh`): RAM is the growth of the kernel's
`Mlocked` count from before the start to after it; the expert cache is the planned size (the same on both cards); VRAM
is the highest `mem_info_vram_used` sampled every 50 ms over an agent-like first turn (a title request decoding while
a ~21k-token prompt is read), the decode answers, `quick_bench.py` and the depth sweep, the desktop's share included.

| | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| RAM locked, `xtx-xt` / `xtx` / `xt` | 81 / 72 / 72 GiB | 59 / 51 / 51 GiB | 74 / 65 / 65 GiB |
| Expert cache, `xtx-xt` / `xtx` / `xt` | 7,127 / 3,567 / 2,328 | 9,900 / 4,871 / 3,026 | 7,312 / 3,663 / 2,276 |
| VRAM peak, `xtx-xt` (XTX / XT of 24,560 / 20,464 MiB) | 23,524 / 20,291 MiB | 24,284 / 19,228 MiB | 24,035 / 19,197 MiB |
| VRAM peak, `xtx` / `xt` | 23,506 / 19,367 MiB | 22,943 / 19,072 MiB | 22,868 / 19,081 MiB |

No GPU queue was stopped by a full VRAM in these runs; the two 4.05 bpw one-card servers stopped their queues once,
for 47-59 ms, on their first request (the VRAM was 1.7 GiB from full; the cause was not traced). The NVFP4 servers ran
on the same build as the EXL3 ones, for their memory only.

How to reproduce or check a change: [testing.md](testing.md).
