# Benchmarks

Everything here was measured on one machine: RX 7900 XTX 24 GB + RX 7900 XT 20 GB (gfx1100, PCIe 4.0 x16 each, no
GPU peer-to-peer), Threadripper 3970X, 128 GB DDR4-3200, Qwen3.8-Flash-Next NVFP4 (RadixArk checkpoint), ROCm 7.14 /
PyTorch 2.11 in the container. The depth sweeps ran on the build before the final review fixes; the review-fix branch
(before its last sampler changes, which greedy decoding does not use) was re-measured against it: decode -0.5 % (within
noise), 8.4k prefill ~1 % faster, identical greedy answers. The rows marked "release build" were measured on the
published image itself (`rdna3-v0.1.0`).

## How the numbers are taken

- **Decode**: tokens/s over a 512-token answer, after the first token, at short context.
- **Cold prefill (PP)**: an 8.4k-token prompt with nothing cached, time to first token.
- **Depth sweep**: one conversation grown in steps up to the profile's maximum context. The first step is a cold
  ~10.5k-token read; each later step adds a 6-42k-token block the cache has not seen, like a large tool result (ten
  steps to 255k on two cards, six to 124k on one; depths are the mean of the two passes). At each step: **TG** = decode
  over 256 tokens, **PP** = speed of reading that block, **agent turn** = time to the first token of a following
  ~1k-token turn. Median of two passes (the 255k step: one pass, the second one overshot the context).
- **Baseline**: the same checkpoint on FreeToken ported to ROCm, TP=2, with PyTorch TunableOp (read-only) and memory
  ratio 0.85 but none of this build's other changes (bf16 dense layers, even split, one request at a time), same cards.

## Summary

| Profile | Decode | Decode, 10k to max depth | Cold 8.4k prompt (PP) | PP of a new block at depth | Agent turn (~1k new tokens), 10k -> max | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| `xtx-xt` | 55.3 tok/s | 48-53 tok/s | 4.5 s, ~1870 tok/s | 1400-1850 tok/s | 1.2 -> 1.9 s | 255k |
| baseline, 2 cards | 35.9 tok/s | 33-35 tok/s | 6.3 s, ~1330 tok/s | 1290-1620 tok/s | 1.4 -> 2.0 s | 255k |
| `xtx` | 36.5 tok/s | 32.5-34.3 tok/s | 10.5 s, ~800 tok/s | 1180-1300 tok/s | 2.3 -> 2.7 s | 124k |
| `xt` | 28.5 tok/s | 27-28 tok/s | 8.9 s, ~940 tok/s | 1100-1190 tok/s | 2.4 -> 2.7 s | 124k |

At the maximum depth, decode is 3-8 % below short-context decode and about the same as at 10k: it does not degrade
with context (the `xtx-xt` range leaves out one slow pass at 75k, below). On the release build a cold 8.3k-token
prompt took 4.1 s (~2030 tok/s). Reading new tokens costs about the same with or without the tuning: every prefill
pass streams the experts it uses from RAM, a large fixed cost that dominates short reads.

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
| 254.9k | 52.3 (one pass) | 1574 | 1.86 s |

The 75k row includes one slow pass (33.5 tok/s, ~2.5 s lost over 256 tokens): the occasional stall described in
[troubleshooting.md](troubleshooting.md), not an effect of depth.

## Depth sweep, baseline (2 cards, TunableOp only)

| Depth | TG tok/s | PP new tok/s | Agent turn |
|---:|---:|---:|---:|
| 10.6k | 34.5 | 1606 | 1.42 s |
| 37.3k | 34.8 | 1608 | 1.49 s |
| 104.6k | 34.3 | 1290 | 1.61 s |
| 167.7k | 33.8 | 1499 | 1.78 s |
| 233.6k | 33.6 | 1467 | 1.96 s |
| 254.9k | 34.4 (one pass) | 1370 | 2.04 s |

The baseline also leaves 4.6 GiB of the XTX unused (the even split sizes everything for the smaller card).

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

## Several requests and agents at once (`xtx-xt`)

| Workload | Result |
|---|---|
| Decode, 1 / 2 / 3 / 4 requests at once (release build) | 55 / 81 / 98 / 105 tok/s in total (55 / 40 / 33 / 26 per request) |
| Cold 8.3k-token prompts (PP), 1 alone; 4 at once (release build) | 1 alone: 4.1 s, ~2030 tok/s; 4 at once: all four in 15.7 s, ~2120 tok/s in total (the first of them answers after 11.5 s) |
| 4 agents, 82-117k-token conversations, 3 turns each (together more than the GPUs hold) | the 3 rounds of turns after the first reads: 35 s, first token of each turn 1.9-2.3 s; without the RAM tier 599 s and ~47 s per turn (1.15M tokens recomputed) |
| 4 agents x 12 short turns, small GPU pool | 94 s instead of 133 s without the RAM tier |
| 5 requests with different sampling settings (greedy, T=1, top-p only, top-k 1000 + top-p decoding together; top-k 20 + top-p queued behind them), 256-token answers | ~27-28 tok/s for each of the 4 decoding together |

Release-build decode is the median of two passes over 512-token answers, counted only while every request is decoding.
The PP row is the second pass: the first one, right after start, took 6.0 s alone and 21.5 s for four (one-time kernel
preparation). Four requests share each decode step, so each gets ~26 tok/s; real agents also wait for each other's
prefills.

A request's decode is computed the same way alone or batched (per-row kernels); prompts prefilled in the same batch can
differ at rounding level, as they already do with prefix caching.

## How the speed was reached (2 cards, decode)

| Step | Decode |
|---|---:|
| FreeToken ported to ROCm, TP=2, first working build | 27.0 tok/s |
| + PyTorch TunableOp (read-only) + memory ratio 0.85 (the baseline) | 35.9 |
| + int8 dense layers, split-K GEMVs, one MoE all-reduce per block | 52.4 |
| + top-k-first sampler, uneven 0.55 split (measured with the int8 KV, dropped later) | 55.5 |
| + 16k prefill chunks at ratio 0.80, expert reuse in prefill, host-memory all-reduce, tuned NVFP4 prefill tiles, bf16 attention KV | 56-57 |
| + up to 4 requests, tuned decode tiles, conversations kept in RAM (`xtx-xt`, release build) | 55 alone, 105 at 4 |

Details of every step, including what did not work: [journey.md](journey.md).

## Precision

- **Scored benchmark** (671 items: HumanEval 164, MBPP 257, GSM8K 250; greedy, thinking off): the community ROCm port
  (bf16) 636, the baseline of these tables (bf16) 632, the int8-dense builds 635. Two bf16 builds already differ by 4
  items: no measurable loss.
- **Agentic benchmark** (the author's, private: a coding agent in OpenCode fixing 29 bugs planted in a ~50k-line
  codebase, long sessions, single runs): builds with the int8 attention KV cache fixed 8-9 bugs, builds with the bf16 KV
  12-14. The int8 KV is out of every profile ([options.md](options.md)).
- **What the agentic benchmark covered**: the two-card builds that scored 12-13 had the uneven split, int8 dense layers,
  one MoE all-reduce per block, the split-K GEMVs, the top-k-first sampler, 16k chunks with blocked PLE, expert reuse in
  prefill, the host-memory all-reduce and the tuned prefill tiles. Added after the last scored build, and checked for
  exactness instead: the tuned decode tiles, up to 4 concurrent requests (a request's decode is identical alone or
  batched; prompts prefilled together can differ at rounding level), the RAM tier, the sampler rework for requests
  without a small `top_k`, and rank 0's token broadcast. The one-card profiles were not run on it.
- **Exactness checks**: a fixed set of 120 greedy prompts (not published) gave identical answers for the host-memory
  all-reduce, the tuned NVFP4 tiles (prefill and decode), expert reuse in prefill, a single request on the 4-request
  build, and the review-fix branch against the swept one. Kernel checks prove the blocked prefill MoE and batched decode
  rows identical (`moe_block_check.py`, `int8_rows_check.py`); four conversations pushed out to the RAM tier and brought
  back give the same greedy answers (and `host_kv_check.py` checks the copies).

## Memory while serving (`xtx-xt`, release build)

| | Used |
|---|---|
| XTX VRAM | 22.4 of 24.0 GiB after an 8.3k-token prompt (0.8 GiB of it the desktop); ~23.9 GiB deep in a 255k conversation |
| XT VRAM | 18.8 of 20.0 GiB after an 8.3k-token prompt; ~19.9 GiB deep in a 255k conversation |
| Host RAM | 81 GiB: ~63 GiB of experts (each rank pins its own share of every expert), 9.0 GiB RAM tier (4.64 + 4.32), the rest the two processes |
| Expert cache | 7,228 experts on each card, of 24,576: each rank plans on its own free memory, then both take the smaller plan (the XTX's own was 8,287) |

Host RAM is the drop of the host's available memory when the server starts (80.7 GiB) and after a short workload
(81.0 GiB). The deep figures come from the depth sweep's free-memory readings.

How to reproduce or check a change: [testing.md](testing.md).
