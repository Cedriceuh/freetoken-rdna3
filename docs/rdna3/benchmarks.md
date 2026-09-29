# Benchmarks

Everything here was measured on one machine: RX 7900 XTX 24 GB + RX 7900 XT 20 GB (gfx1100, PCIe 4.0 x16 each, no
GPU peer-to-peer), Threadripper 3970X, 128 GB DDR4-3200, Qwen3.8-Flash-Next NVFP4 (RadixArk checkpoint), ROCm 7.14 /
PyTorch 2.11 in the container. The depth sweeps ran on the build before the final review fixes; the release build was
re-measured against it: decode -0.4 % (within noise), 8.4k prefill ~1 % faster, identical greedy answers.

## How the numbers are taken

- **Decode**: tokens/s over a 512-token answer, after the first token.
- **Cold prefill**: an 8.4k-token prompt with nothing cached, time to first token.
- **Depth sweep**: one conversation that grows turn by turn up to the profile's maximum context (about ten steps
  from 10k to 255k tokens; the tables give the exact depths); each turn adds a new block to a conversation the cache already holds, like
  an agent's tool result. At every step: **TG** = decode speed at that depth, **PP** = speed of reading the new part,
  **agent turn** = time to first token of that turn. Median of two passes.
- **Baseline**: the same checkpoint on FreeToken ported to ROCm with TP=2 and no tuning (bf16 dense layers, even split,
  one request at a time), same cards.

## Summary

| Profile | Decode | Decode, 10k to max depth | Cold 8.4k prompt | PP of new tokens | Agent turn, 10k -> max | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| `xtx-xt` | 55.3 tok/s | 48-53 tok/s | 4.5 s | 1400-1850 tok/s | 1.2 -> 1.9 s | 255k |
| baseline, 2 cards | 35.9 tok/s | 33-35 tok/s | 6.3 s | 1290-1620 tok/s | 1.4 -> 2.0 s | 255k |
| `xtx` | 36.5 tok/s | 32.5-34.3 tok/s | 10.5 s | 1180-1300 tok/s | 2.3 -> 2.7 s | 124k |
| `xt` | 28.5 tok/s | 27-28 tok/s | 8.9 s | 1100-1190 tok/s | 2.4 -> 2.7 s | 124k |

Decode holds to the maximum context on every profile (at most ~10 % lower at 255k than at 10k). Reading the new part
of an agent turn costs about the same with or without the tuning: every prefill pass streams the experts it uses from
RAM, a fixed cost per prompt that dominates short reads.

## Depth sweep, `xtx-xt`

| Depth | TG tok/s (two passes) | PP new tok/s | Agent turn |
|---:|---:|---:|---:|
| 10.6k | 48.1 (44.2-52.0) | 1436 | 1.23 s |
| 18.0k | 53.3 (53.2-53.3) | 1843 | 1.25 s |
| 37.3k | 52.0 (51.8-52.1) | 1826 | 1.29 s |
| 75.3k | 41.3 (33.5-49.1) | 1748 | 1.39 s |
| 104.6k | 51.1 (48.6-53.5) | 1763 | 1.44 s |
| 140.5k | 50.9 (47.7-54.0) | 1572 | 1.56 s |
| 167.7k | 49.5 (46.8-52.2) | 1676 | 1.76 s |
| 204.3k | 48.7 (47.6-49.9) | 1648 | 1.71 s |
| 233.7k | 49.6 (47.2-52.1) | 1619 | 1.78 s |
| 254.9k | 52.3 | 1574 | 1.86 s |

The 75k row is one slow pass (33.5 tok/s): an occasional 9-21 s stall seen in about one run in three on every build,
not an effect of depth (see [troubleshooting.md](troubleshooting.md)).

## Depth sweep, baseline (2 cards, untuned)

| Depth | TG tok/s | PP new tok/s | Agent turn |
|---:|---:|---:|---:|
| 10.6k | 34.5 | 1606 | 1.42 s |
| 37.3k | 34.8 | 1608 | 1.49 s |
| 104.6k | 34.3 | 1290 | 1.61 s |
| 167.7k | 33.8 | 1499 | 1.78 s |
| 233.6k | 33.6 | 1467 | 1.96 s |
| 254.9k | 34.4 | 1370 | 2.04 s |

It also leaves 4.6 GiB of the XTX unused (the even split sizes everything for the smaller card).

## Depth sweep, one card

| Depth | `xtx` TG | `xtx` PP new | `xtx` turn | `xt` TG | `xt` PP new | `xt` turn |
|---:|---:|---:|---:|---:|---:|---:|
| 10.0k | 32.5 | 1182 | 2.33 s | 27.1 | 1101 | 2.44 s |
| 17.8k | 33.3 | 1294 | 2.52 s | 27.3 | 1173 | 2.45 s |
| 34.2k | 32.6 | 1223 | 2.39 s | 27.1 | 1135 | 2.50 s |
| 67.5k | 33.2 | 1299 | 2.47 s | 27.4 | 1194 | 2.58 s |
| 99.1k | 34.3 | 1296 | 2.54 s | 28.1 | 1191 | 2.66 s |
| 124.6k | 33.5 | 1254 | 2.66 s | 27.7 | 1160 | 2.73 s |

The XT ends at 124k with 0.8 GiB of VRAM free: 131k is its limit. A 60k-token prompt read from scratch: 45 s on the XTX
(then 30.5 tok/s), 48 s on the XT (then 26.7 tok/s).

## Several agents at once (`xtx-xt`)

| Workload | Result |
|---|---|
| Decode throughput, 1 / 2 / 4 requests | 57 / 82 / 113 tok/s in total |
| 4 agents, 82-117k-token conversations, 3 turns each (together more than the GPUs hold) | 35 s, first token of each turn 1.9-2.3 s; without the RAM tier 599 s and ~47 s per turn (1.15M tokens recomputed) |
| 4 agents x 12 short turns, small GPU pool | 94 s instead of 143 s without the RAM tier |
| 4 requests with different sampling settings (greedy, top-p only, top-k 1000, top-k 20) | ~28 tok/s per request |

A request is computed the same way alone or batched (per-row kernels); prompts prefilled in the same batch can differ
at rounding level, as they already do with prefix caching.

## How the speed was reached (2 cards, decode)

| Step | Decode |
|---|---:|
| FreeToken ported to ROCm, TP=2, first working build | 27.0 tok/s |
| + PyTorch TunableOp (read-only) + memory ratio 0.85 | 35.9 |
| + int8 dense layers, split-K GEMVs, one MoE all-reduce per block | 52.4 |
| + top-k-first sampler, uneven 0.55 split | 55.5 |
| + host-memory all-reduce, tuned NVFP4 tiles, bf16 attention KV (int8 KV dropped) | 56-57 |
| + up to 4 requests, 16k prefill chunks, conversations kept in RAM (`xtx-xt`) | 55 alone, 113 at 4 |

Details of every step, including what did not work: [journey.md](journey.md).

## Precision

- **Scored benchmark** (671 items: HumanEval 164, MBPP 257, GSM8K 250; greedy, thinking off): baseline bf16 636, a
  second bf16 build 632, the int8-dense builds 635. Two bf16 builds already differ by 4 items: no measurable loss.
- **Agentic benchmark** (a coding agent in OpenCode fixing 29 bugs planted in a ~50k-line codebase, long sessions,
  single runs): builds with the int8 attention KV cache fixed 8-9 bugs, builds with the bf16 KV 12-14. The int8 KV is
  therefore out of every profile ([options.md](options.md)); every other option in the profiles was part of the builds
  that scored 12-14.
- **Bit-exact changes** are checked on a fixed set of 120 greedy prompts (identical answers; the set is not published): host all-reduce, tuned NVFP4 tiles,
  blocked prefill MoE, expert reuse in prefill, the RAM tier (a conversation brought back from RAM answers the same),
  and the release build against the swept one.

## Memory while serving (`xtx-xt`)

| | Used |
|---|---|
| XTX VRAM | 23.7 of 24.0 GiB (dense weights, ~55 % of the split layers, KV, expert cache) |
| XT VRAM | 19.8 of 20.0 GiB |
| Host RAM | 80.5 GiB (64.7 GiB of experts shared by both ranks, ~9 GiB RAM tier, the rest the two processes) |
| Expert cache | 8,287 experts on the XTX, 7,228 on the XT, of 24,576 (start-up log) |

How to reproduce or check a change: [testing.md](testing.md).
