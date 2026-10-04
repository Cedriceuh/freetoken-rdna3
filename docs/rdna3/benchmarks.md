# Benchmarks

Everything here was measured on one machine: RX 7900 XTX 24 GB + RX 7900 XT 20 GB (gfx1100, PCIe 4.0 x16 each, no
GPU peer-to-peer), Threadripper 3970X, 128 GB DDR4-3200, Qwen3.8-Flash-Next NVFP4 (RadixArk checkpoint). The numbers
from the summary to the one-card sweeps were taken on 2026-10-03 on the ROCm 10.0 / PyTorch 2.13 image as released:
each profile as shipped (MTP head on) and the same with `--no-mtp`, one server per configuration, with the host
settings of [troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run). The one-card
sweeps with the head ran with a variant of the adaptive depth dropped before the release; the released image itself gave
the same within 4 % (`xtx` 33.1-43.2, `xt` 27.1-34.9 tok/s). Sections marked ROCm 7.14 were measured on the earlier image (PyTorch 2.11): the baseline, the 0.1.0 release build
("release build" rows, image `rdna3-v0.1.0`) and the MTP head's development.

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
- **Baseline** (ROCm 7.14): the same checkpoint on FreeToken ported to ROCm, TP=2, with PyTorch TunableOp (read-only)
  and memory ratio 0.85 but none of this build's other changes (bf16 dense layers, even split, one request at a time),
  same cards.

## Summary

| Profile | Decode, one request: sampled / greedy | TG, 9k to max depth | Cold 8.3k prompt (PP) | PP of a new block | Agent turn, 9k -> max | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| `xtx-xt` | 77.2 / 70-90 tok/s | 72-86 tok/s | 4.1 s, ~2020 tok/s | 1670-1990 tok/s | 1.3 -> 2.0 s | 248k |
| `xtx-xt --no-mtp` | 57.5 / 57-60 tok/s | 52-57 tok/s | 4.2 s, ~2010 tok/s | 1730-1990 tok/s | 1.3 -> 2.0 s | 248k |
| `xtx` | 38.8 / 38-44 tok/s | 33-42 tok/s | 7.7 s, ~1080 tok/s | 1130-1510 tok/s | 2.4 -> 2.7 s | 124k |
| `xtx --no-mtp` | 35.8 / 35-38 tok/s | 29.5-39 tok/s | 7.6 s, ~1095 tok/s | 1140-1500 tok/s | 2.3 -> 2.7 s | 124k |
| `xt` | 31.7 / 30-36 tok/s | 27-34 tok/s | 8.3 s, ~1000 tok/s | 1040-1340 tok/s | 2.5 -> 2.8 s | 124k |
| `xt --no-mtp` | 29.5 / 28-32 tok/s | 25-32 tok/s | 8.2 s, ~1020 tok/s | 1070-1340 tok/s | 2.5 -> 2.8 s | 124k |
| baseline, 2 cards (ROCm 7.14) | 35.9 tok/s | 33-35 tok/s | 6.3 s, ~1330 tok/s | 1290-1620 tok/s | 1.4 -> 2.0 s | 255k |

With the head, one request decodes 34 % faster on two cards at the model's sampling (+30 to +61 % greedy along the
sweep) and 7-8 % faster on one card (+6 to +12 % greedy); reading a prompt costs 0-4 % more, since the head reads it
too. On one card one of the six sampled prompts (a long story) is 3-4 % slower with it. With more requests
decoding than `FREETOKEN_SPEC_BS_MAX` (2 in `xtx-xt`, 1 in the other profiles), the head only writes its KV: see
the table further down.

Decode loses 4-9 % from 9k to 248k on two cards and 20-24 % from 9k to 124k on one card. The ROCm 7.14 image
(code 3a380bd, TunableOp on) measured the same way on `xtx` without the head: 36.9 / 32.3 / 30.2 / 29.1 tok/s at
8.8k / 25k / 60k / 124k against 38.8 / 32.0 / 30.2 / 29.5 here, prompt reading and agent turns alike: the drop with
depth is the same on both images (the flatter 7.14 one-card numbers this page gave before came from another sweep script). Where the
one-card drop comes from (`xtx`, one pass with `FT_STATS_EVERY=64 FT_MOE_STATS=1`): the decode step takes ~27 ms of
GPU time at 9-16k and ~33 ms from 25k to 124k, flat beyond 25k, while the expert cache misses ~2.2 then 3.0-3.6 of
the 10 experts a layer reads: the longer context spreads the routing over more experts (the misses do not fall during
an answer, so it is not the prefill evicting the cache), and attention itself adds little.

## Depth sweep, `xtx-xt`

| Depth | TG with the head / without | PP new, with / without | Agent turn, with / without |
|---:|---:|---:|---:|
| 8.8k | 74.4 / 57.1 | 1994 / 1992 | 1.28 / 1.27 s |
| 16.0k | 85.1 / 55.8 | 1864 / 1857 | 1.32 / 1.30 s |
| 25.0k | 86.3 / 53.8 | 1922 / 1919 | 1.33 / 1.31 s |
| 40.0k | 73.2 / 52.4 | 1944 / 1943 | 1.35 / 1.33 s |
| 60.0k | 85.4 / 52.9 | 1870 / 1928 | 1.40 / 1.37 s |
| 86.9k | 74.8 / 53.0 | 1875 / 1926 | 1.48 / 1.45 s |
| 122.9k | 83.8 / 53.6 | 1809 / 1873 | 1.57 / 1.56 s |
| 164.9k | 73.1 / 53.0 | 1779 / 1838 | 1.74 / 1.70 s |
| 206.9k | 72.1 / 52.7 | 1717 / 1774 | 1.88 / 1.83 s |
| 248.4k | 71.5 / 52.0 | 1669 / 1728 | 2.01 / 1.95 s |

## Depth sweep, baseline (2 cards, TunableOp only, ROCm 7.14)

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

| Depth | `xtx` TG with / without the head | `xtx` PP new | `xtx` turn | `xt` TG with / without | `xt` PP new | `xt` turn |
|---:|---:|---:|---:|---:|---:|---:|
| 8.8k | 41.7 / 38.8 | 1129 / 1143 | 2.37 / 2.34 s | 33.7 / 31.7 | 1040 / 1069 | 2.48 / 2.45 s |
| 16.0k | 37.5 / 34.7 | 1434 / 1431 | 2.38 / 2.35 s | 30.3 / 28.5 | 1276 / 1284 | 2.48 / 2.46 s |
| 25.0k | 35.5 / 32.0 | 1507 / 1498 | 2.39 / 2.38 s | 28.9 / 26.6 | 1338 / 1340 | 2.50 / 2.47 s |
| 40.0k | 35.6 / 31.7 | 1454 / 1463 | 2.43 / 2.41 s | 29.4 / 26.6 | 1297 / 1343 | 2.55 / 2.53 s |
| 60.0k | 33.8 / 30.2 | 1372 / 1400 | 2.46 / 2.46 s | 27.9 / 25.3 | 1250 / 1303 | 2.57 / 2.56 s |
| 86.9k | 33.5 / 29.9 | 1315 / 1349 | 2.59 / 2.54 s | 27.4 / 25.0 | 1200 / 1249 | 2.68 / 2.67 s |
| 123.9k | 33.1 / 29.5 | 1358 / 1398 | 2.73 / 2.65 s | 27.1 / 24.8 | 1234 / 1280 | 2.82 / 2.81 s |

PP is the same with and without the head within 4 %; one card reads 8k-token chunks (16k on two). On ROCm 7.14 the XT
ended at 124k with 0.8 GiB of VRAM free (131k is its limit), and a 60k-token prompt read from scratch took 45 s on the
XTX (then 30.5 tok/s), 48 s on the XT (then 26.7 tok/s).

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
| 1 / 2 / 3 / 4 requests at once, greedy answers of 93-200 tokens, time from the send to the last token (ROCm 10, two rounds) | with the head 3.6 / 5.9 / 8.4-8.6 / 10.1 s; without it 4.1-4.2 / 5.9-6.0 / 8.2-8.3 / 9.8 s |
| Decode, 1 / 2 / 3 / 4 requests at once (release build, ROCm 7.14) | 55 / 81 / 98 / 105 tok/s in total (55 / 40 / 33 / 26 per request) |
| Cold 8.3k-token prompts (PP), 1 alone; 4 at once (release build, ROCm 7.14) | 1 alone: 4.1 s, ~2030 tok/s; 4 at once: all four in 15.7 s, ~2120 tok/s in total (the first of them answers after 11.5 s) |
| 4 agents, 82-117k-token conversations, 3 turns each (together more than the GPUs hold) | the 3 rounds of turns after the first reads: 35 s, first token of each turn 1.9-2.3 s; without the RAM tier 599 s and ~47 s per turn (1.15M tokens recomputed) |
| 4 agents x 12 short turns, small GPU pool | 94 s instead of 133 s without the RAM tier |
| 5 requests with different sampling settings (greedy, T=1, top-p only, top-k 1000 + top-p decoding together; top-k 20 + top-p queued behind them), 256-token answers | ~27-28 tok/s for each of the 4 decoding together |

Release-build decode is the median of two passes over 512-token answers, counted only while every request is decoding.
The PP row is the second pass: the first one, right after start, took 6.0 s alone and 21.5 s for four (one-time kernel
preparation). Four requests share each decode step, so each gets ~26 tok/s; real agents also wait for each other's
prefills.

With the head and `FREETOKEN_SPEC_BS_MAX=1`, only a request decoding alone verifies drafts: with two or more, steps run
one row per request and the head only writes its KV, an eager pass that costs 3-4 % at 3-4 requests; a lone request finishes 13 % sooner.

A request's decode is computed the same way alone or batched (per-row kernels); prompts prefilled in the same batch can
differ at rounding level, as they already do with prefix caching.

## Speculative decoding (MTP head)

`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4` (adaptive depth) on top of the `xtx-xt` profile, release image (ROCm
7.14), measured on 2026-10-01/02 against the same build without the head, one server per configuration, before the
int8 row groups (`FREETOKEN_INT8_ROW_GROUPS`) were added. Decode is per request, after the first token.
The final code, same image, 2026-10-02: the 3 chat prompts 54.9-56.8 -> 70.4-84.6 tok/s, the 120-item set 585 s
(release build) -> 412 s, answers identical to the release build with and without the head.

ROCm 10 image with the review fixes, 2026-10-03, each profile without the head against the same plus the head (one server
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

| Workload, one request | Without the head | With the head | Answers |
|---|---:|---:|---|
| 3 chat prompts (FR prose, EN explanation, FR recipes), 96-256 tokens, two rounds, greedy | 54.5-56.7 tok/s | 66.9-81.9 tok/s | identical |
| 120-item precision set (40 HumanEval, 40 MBPP, 40 GSM8K; greedy, up to 768 tokens) | 657 s | 426 s (x1.54) | 120/120 identical |
| 113k-character prompt (35k tokens, three prefill chunks) + 300 tokens | 54.9 tok/s | 76.2 tok/s | identical |
| One 1500-token answer | 59.1 tok/s | 74.5 tok/s | identical |
| 124k-token conversation: first answer, then a follow-up turn (300 tokens each) | 54.2 / 55.8 tok/s | 70.8 / 83.2 tok/s | identical |
| Same, time to first token of the 124k read / of the follow-up | 70.8 / 1.3 s | 73.3 / 1.4 s | |
| 3 agent turns on a 3.4-3.6k-token prompt (questions on `docs/cli.md`) | 47.5 / 55.8 / 54.5 tok/s | 64.1 / 67.0 / 65.5 tok/s | |
| Model's default sampling (T 1.0, top-k 20, top-p 0.95), 9 answers of up to 256 tokens | 56.2 (54.1-57.4) tok/s | 72.1 (59.4-81.5) tok/s | |
| T 0.6, same | 56.3 (55.0-57.1) tok/s | 74.0 (69.6-79.7) tok/s | |
| The 120-item set at the model's default sampling (the drawn answers differ: 24.6k / 24.0k tokens) | 600 s | 434 s (x1.34 per token) | |
| Cold 6.8k-token prompt (PP) | ~1990 tok/s | ~1960 tok/s | |

| Several requests (`FREETOKEN_SPEC_BS_MAX=1`: the head only writes its KV; greedy texts identical to each request alone in both) | Without the head | With the head |
|---|---:|---:|
| 2 requests at once, greedy, 200 + 96 tokens: time from the send to the last token | 6.10-6.22 s | 5.98-5.99 s (-3 %) |
| 4 requests at once, greedy, 200 tokens: time from the send to the last token | 9.23-9.36 s | 9.63-9.74 s (+4 %) |
| Same, GPU time of a 4-request decode step | 40.0-40.8 ms | 41.2-45.3 ms, + 0.8 ms for the head |
| 4 requests started 1.5 s apart (each prompt read alone), 200 tokens: time to the last token | 12.09-12.10 s | 12.33-12.34 s (+2 %) |

Tokens kept per verify step: 2.4-2.6 on chat (greedy and sampled), 3.6-3.7 on code and math. Drafts drawn from the
head (`FREETOKEN_SPEC_SAMPLED_DRAFTS=1`) instead of its argmax keep 2-7 % more tokens per step for sampled requests,
at the same speed: 70.9 / 74.5 tok/s on the chat rows above, 439 s on the 120 items. Where a step's time goes
(`FREETOKEN_SPEC_TIMING`, one request, a fixed row count, GPU ms):

| Rows | Forward | Verify | Head | Draft chain | Step | Tokens per step | Decode |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1, no head | 17.7 | | | | 17.7 | 1 | 54.5-56.7 tok/s |
| 2 | 23.2 | 0.51 | 1.6 | | 25.3 | 1.82 | 68-76 tok/s |
| 3 | 27.2 | 0.54 | 1.56 | 1.43 | 30.7 | 2.41 | 71.5-82 tok/s |
| 4 | 30.5 | 0.54 | 1.58 | 2.78 | 35.5 | 2.88 | 72-84 tok/s |

Each extra row costs 3.3-5.5 ms in the forward. An eager profile from 1 to 3 rows put it in the dense int8 GEMVs
(+1.8 ms in a graph), the expert GEMMs (+2.7), the GDN recurrence (+0.7) and expert copies (+0.6): the rows of one
step choose ~20 distinct experts per layer instead of 10, and the ones missing from the GPU cache (1.7-2.9 per layer
at 3 rows, against 0.4-1.3 at one) are copied from RAM over PCIe. The head itself, run eagerly while the GPU still
executes the forward's graph, adds ~3 ms.

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
  without a small `top_k`, and rank 0's token broadcast. The one-card profiles were not run on it. The ROCm 10 image
  with the MTP head (`xtx-xt`; greedy answers differ from ROCm 7.14 on 2 of 5 prompts) scored 12 and 18 in two runs.
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

How to reproduce or check a change: [testing.md](testing.md).
