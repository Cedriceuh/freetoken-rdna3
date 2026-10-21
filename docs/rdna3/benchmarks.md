# Benchmarks

Everything here was measured on one machine: RX 7900 XTX 24 GB + RX 7900 XT 20 GB (gfx1100, PCIe 4.0 x16 each, no GPU
peer-to-peer), Threadripper 3970X, 128 GB DDR4-3200, Qwen3.8-Flash-Next NVFP4 (RadixArk checkpoint; the
EXL3 checkpoints in [their depth sweep](#depth-sweep-exl3-checkpoints) and
[how-it-works.md](how-it-works.md#exl3-checkpoints)).

The summary and the three depth sweeps are the numbers re-measured at each release
([maintaining.md](maintaining.md#releasing)). The sections after them come from earlier builds than the summary and
give orders of magnitude; a section that mixes the two says so.

## How the numbers are taken

- **Servers**: one server per row, started by `rdna3/serve.sh`, each profile as shipped (MTP head on) and NVFP4 again
  with `--no-mtp`, with the host settings of
  [troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run). The first server of each checkpoint and
  card count runs a second time once the image's kernel cache is warm, and that run is the one shown: in the first, the
  kernels compile during the depth sweep's first pass (NVFP4 `xtx-xt` read its 16k step at 1416 tok/s, 2079 in the
  second pass).
- **Decode, one request**: tokens/s of 512-token answers after the first token, thinking off: the median of 12 answers
  (6 prompts x 2: English and French prose, code, a recipe) at the sampling of a request that sets none (the
  checkpoint's top-k / top-p, temperature 0.6 from `rdna3/serve.sh`), and 3 greedy answers.
- **Cold prefill (PP)**: an 8.3k-token prompt with nothing cached, time to first token (`rdna3/bench/quick_bench.py`).
- **Depth sweep** (`rdna3/bench/depth_sweep.py`): one conversation grown in steps up to the profile's maximum context.
  The first step is a cold ~8.8k-token read; each later step adds a 6-52k-token block the cache has not seen, like a
  large tool result (ten steps to 260k on two cards, seven to 124k on one). At each step: **TG** = greedy decode over
  256 tokens, **PP** = speed of reading that block, **agent turn** = time to the first token of a following ~1k-token
  turn (an earlier build for this column: since the short-prefill changes, 200-250-token turns take 11-13 % less time to
  their first token on another workload, [journey.md](journey.md#31-short-agent-turns-2026-10-17-and-18)). Median of two
  passes, each with its own text. With the MTP head, TG follows how many of its guesses the text lets it keep (1.6-3.5
  tokens per step): the spread between steps is the text, not noise.

## Summary

| Profile | Decode, one request: sampled / greedy | TG, 9k to max depth | Cold 8.3k prompt (PP) | PP of a new block | Agent turn, 9k -> max | Max depth |
|---|---:|---:|---:|---:|---:|---:|
| `xtx-xt` | 88.2 / 86-90 tok/s | 78-96 tok/s | 3.8 s, ~2220 tok/s | 1910-2180 tok/s | 0.9 -> 0.9 s | 260k |
| `xtx-xt --no-mtp` | 65.5 / 66-69 tok/s | 58-65 tok/s | 3.8 s, ~2200 tok/s | 1980-2170 tok/s | 0.9 -> 0.9 s | 260k |
| `xtx` | 45.9 / 42-48 tok/s | 36-44 tok/s | 5.6 s, ~1500 tok/s | 1520-1670 tok/s | 1.5 -> 1.5 s | 124k |
| `xtx --no-mtp` | 40.6 / 37-43 tok/s | 32-38 tok/s | 5.5 s, ~1520 tok/s | 1560-1680 tok/s | 1.5 -> 1.6 s | 124k |
| `xt` | 35.8 / 33-38 tok/s | 29-34 tok/s | 6.2 s, ~1340 tok/s | 1350-1480 tok/s | 1.7 -> 1.7 s | 124k |
| `xt --no-mtp` | 33.6 / 30-35 tok/s | 26-31 tok/s | 6.0 s, ~1380 tok/s | 1410-1510 tok/s | 1.6 -> 1.7 s | 124k |
| `xtx-xt`, EXL3 3.05 bpw | 94.4 / 86-95 tok/s | 95-117 tok/s | 3.6 s, ~2340 tok/s | 1940-2280 tok/s | 0.7 -> 0.9 s | 260k |
| `xtx-xt`, EXL3 4.05 bpw | 80.5 / 79-84 tok/s | 86-94 tok/s | 3.6 s, ~2330 tok/s | 1970-2270 tok/s | 0.8 -> 1.0 s | 260k |
| `xtx`, EXL3 3.05 bpw | 55.8 / 56-61 tok/s | 52-58 tok/s | 5.0 s, ~1680 tok/s | 1700-1820 tok/s | 1.0 -> 1.2 s | 124k |
| `xtx`, EXL3 4.05 bpw | 45.3 / 44-48 tok/s | 37-47 tok/s | 5.1 s, ~1630 tok/s | 1640-1820 tok/s | 1.3 -> 1.4 s | 124k |
| `xt`, EXL3 3.05 bpw | 43.2 / 42-45 tok/s | 41-46 tok/s | 5.5 s, ~1500 tok/s | 1500-1600 tok/s | 1.2 -> 1.3 s | 124k |
| `xt`, EXL3 4.05 bpw | 34.9 / 34-36 tok/s | 29-36 tok/s | 5.7 s, ~1450 tok/s | 1450-1620 tok/s | 1.5 -> 1.5 s | 124k |

The EXL3 rows ([how-it-works.md](how-it-works.md#exl3-checkpoints)) were measured the same way, one server per row
started by `rdna3/serve.sh` (on two cards it lowers `--memory-ratio` to 0.77 / 0.76 for EXL3), MTP head on, the same
scripts: the 12 sampled and 3 greedy answers, `quick_bench.py` for the
cold read, the depth sweep. The checkpoints answer differently, so the MTP head keeps a different share of its
guesses: read the decode columns across checkpoints as an order of magnitude.

With the head, one request decodes 35 % faster on two cards at the sampling above (+35 to +60 % greedy along the
sweep) and 6-13 % faster on one card (+9 to +16 % greedy); reading a block takes from 5 % longer (`xt`) to 1 % less. With more
requests decoding than `FREETOKEN_SPEC_BS_MAX` (4 in `xtx-xt` and its K/V-in-RAM variants, 1 in the other profiles),
the head only writes its KV: see the tables further down.

Without the head, decode loses 11 % from 9k to 260k on two cards and 16 % from 9k to 124k on one card (with it,
the text of each step decides how many drafts are kept). Where the one-card drop
comes from (`xtx`, one pass with `FT_STATS_EVERY=64 FT_MOE_STATS=1`, on an earlier build): the decode step takes ~27 ms
of GPU time at 9-16k and ~33 ms from 25k to 124k, flat beyond 25k, while the expert cache misses ~2.2 then 3.0-3.6 of
the 10 experts a layer reads: the longer context spreads the routing over more experts (the misses do not fall during
an answer, so it is not the prefill evicting the cache), and attention itself adds little.

## Depth sweep, `xtx-xt`

| Depth | TG with the head / without | PP new, with / without | Agent turn, with / without |
|---:|---:|---:|---:|
| 8.8k | 88.2 / 64.8 | 2191 / 2172 | 0.88 / 0.89 s |
| 16.0k | 91.6 / 64.7 | 2108 / 2093 | 0.90 / 0.90 s |
| 25.0k | 96.3 / 61.8 | 2163 / 2146 | 0.93 / 0.93 s |
| 40.0k | 88.6 / 58.2 | 2184 / 2169 | 0.80 / 0.94 s |
| 60.0k | 94.1 / 59.0 | 2105 / 2160 | 0.87 / 0.89 s |
| 86.9k | 83.8 / 59.3 | 2121 / 2170 | 0.85 / 0.87 s |
| 122.9k | 92.7 / 59.2 | 2058 / 2125 | 0.86 / 0.86 s |
| 164.9k | 80.1 / 58.2 | 2034 / 2093 | 0.87 / 0.87 s |
| 206.9k | 77.7 / 57.5 | 1971 / 2032 | 0.90 / 0.91 s |
| 259.8k | 86.7 / 57.7 | 1909 / 1975 | 0.90 / 0.89 s |

Each sweep ran after the decode answers and `quick_bench.py` on the same server. The ~1k-token agent turn takes
0.8-0.9 s at every depth: such a turn reads only its routed experts through the expert cache and reuses the previous
prompt's tokens ([how-it-works.md](how-it-works.md#short-chunks-agent-turns)).

## Depth sweep, one card

| Depth | `xtx` TG with / without the head | `xtx` PP new | `xtx` turn | `xt` TG with / without | `xt` PP new | `xt` turn |
|---:|---:|---:|---:|---:|---:|---:|
| 8.8k | 43.7 / 38.0 | 1518 / 1557 | 1.50 / 1.54 s | 34.5 / 31.1 | 1356 / 1394 | 1.67 / 1.65 s |
| 16.0k | 39.8 / 35.2 | 1597 / 1583 | 1.54 / 1.55 s | 32.1 / 29.2 | 1410 / 1414 | 1.73 / 1.72 s |
| 25.0k | 38.4 / 33.4 | 1672 / 1652 | 1.58 / 1.59 s | 30.5 / 27.5 | 1483 / 1483 | 1.75 / 1.76 s |
| 40.0k | 37.5 / 33.1 | 1649 / 1680 | 1.47 / 1.50 s | 29.5 / 27.1 | 1457 / 1508 | 1.65 / 1.66 s |
| 60.0k | 38.1 / 32.9 | 1565 / 1611 | 1.51 / 1.55 s | 29.9 / 26.7 | 1397 / 1456 | 1.68 / 1.70 s |
| 86.9k | 37.9 / 32.7 | 1517 / 1558 | 1.47 / 1.51 s | 29.8 / 26.4 | 1346 / 1408 | 1.64 / 1.67 s |
| 123.9k | 36.5 / 31.9 | 1591 / 1641 | 1.51 / 1.55 s | 28.9 / 26.0 | 1406 / 1469 | 1.68 / 1.71 s |

PP is the same with and without the head within 5 %; one card reads 8k-token chunks (16k on two). Without the head
the servers fill the PLE rows before each launch (`FREETOKEN_PLE_SYNC` in [options.md](options.md)).

## Depth sweep, EXL3 checkpoints

Median of two passes, MTP head on:

| Depth | `xtx-xt` 3.05 bpw TG / PP new / turn | `xtx-xt` 4.05 bpw TG / PP new / turn |
|---:|---:|---:|
| 8.8k | 97.0 / 2315 / 0.70 s | 87.1 / 2300 / 0.79 s |
| 16.0k | 109.2 / 2221 / 0.72 s | 86.2 / 2181 / 0.83 s |
| 25.0k | 114.3 / 2270 / 0.72 s | 92.5 / 2245 / 0.85 s |
| 40.0k | 104.0 / 2284 / 0.71 s | 90.3 / 2272 / 0.88 s |
| 60.0k | 114.5 / 1977 / 0.91 s | 94.0 / 2178 / 0.89 s |
| 86.9k | 102.0 / 2063 / 0.71 s | 91.1 / 2205 / 0.90 s |
| 122.9k | 117.2 / 2167 / 0.73 s | 92.9 / 2128 / 0.91 s |
| 164.9k | 94.9 / 1973 / 0.92 s | 91.2 / 2105 / 0.94 s |
| 206.9k | 98.5 / 1945 / 0.75 s | 90.0 / 2044 / 0.97 s |
| 259.8k | 111.1 / 1986 / 0.94 s | 93.4 / 1967 / 0.97 s |

| Depth | `xtx` 3.05 bpw | `xtx` 4.05 bpw | `xt` 3.05 bpw | `xt` 4.05 bpw |
|---:|---:|---:|---:|---:|
| 8.8k | 58.2 / 1715 / 1.02 s | 46.9 / 1642 / 1.31 s | 45.8 / 1514 / 1.18 s | 36.0 / 1471 / 1.48 s |
| 16.0k | 57.2 / 1806 / 1.07 s | 44.0 / 1788 / 1.38 s | 45.5 / 1595 / 1.23 s | 34.1 / 1577 / 1.54 s |
| 25.0k | 56.9 / 1821 / 1.11 s | 41.0 / 1819 / 1.42 s | 44.7 / 1519 / 1.28 s | 32.0 / 1616 / 1.58 s |
| 40.0k | 54.1 / 1802 / 1.13 s | 38.8 / 1793 / 1.41 s | 42.9 / 1601 / 1.29 s | 30.5 / 1588 / 1.56 s |
| 60.0k | 54.6 / 1760 / 1.14 s | 40.0 / 1689 / 1.45 s | 43.2 / 1552 / 1.30 s | 31.7 / 1509 / 1.61 s |
| 86.9k | 54.3 / 1700 / 1.13 s | 37.6 / 1638 / 1.37 s | 42.9 / 1500 / 1.30 s | 29.9 / 1448 / 1.53 s |
| 123.9k | 52.2 / 1740 / 1.18 s | 36.7 / 1718 / 1.36 s | 41.0 / 1534 / 1.34 s | 29.4 / 1509 / 1.53 s |

Each cell: TG (tok/s) / PP of the new block (tok/s) / agent turn. On two cards the agent turns take 0.70-0.94 s at
3.05 bpw against 0.80-0.93 s with NVFP4: such a turn copies the routed experts the cache lacks
([how-it-works.md](how-it-works.md#short-chunks-agent-turns)), and a 3.05 bpw expert weighs 1.86 MB against 2.76 MB (not measured
apart).

## Agent turns on real code (`xtx-xt`)

Conversations built from this repository's own source and docs (77-113k tokens), then 8 turns each: a ~2k-token tool
result and a ~250-token answer, temperature 0.6 (what the reference machine's agent client sends). An earlier build,
when `xtx-xt` verified drafts for at most two requests (`FREETOKEN_SPEC_BS_MAX=2`; now 4):

| Per request | With the head | Without |
|---|---:|---:|
| One conversation: decode | 71.1 tok/s | 55.8 tok/s |
| One conversation: time to the first token of a turn | 1.75 s | 1.73 s |
| Two conversations at once (sub-agents), `FREETOKEN_SPEC_BS_MAX=1`: decode | 31.1 tok/s | 33.2 tok/s |
| Same, `FREETOKEN_SPEC_BS_MAX=2`: decode | 33.6-34.4 tok/s | |
| Two conversations at once: time to the first token | 1.96 s | 1.95 s |

Two requests share each decode step. With `FREETOKEN_SPEC_BS_MAX=1` the head then only writes its KV, so each gets
~31-33 tok/s at ~100k of context, with or without it; with `2` it keeps verifying drafts for both. On the reference
machine's own launcher settings (image input on): 36.4 against 31.6 tok/s each, and with thinking on 38.8
against 34.2 on the turns the two decode together; two conversations of 115k and 121k tokens decoded together with no
out-of-memory retry.

## Several requests and agents at once (`xtx-xt`)

| Workload | Result |
|---|---|
| Decode, 1 / 2 / 3 / 4 requests at once, greedy 512-token answers, counted while every request decodes (median of two passes) | with the head 72.3 / 81.9 / 82.2 / 89.2 tok/s in total (72.3 / 40.9 / 27.4 / 22.3 per request); without it 59.5 / 79.2 / 85.6 / 92.3 (59.5 / 39.6 / 28.6 / 23.1) |
| 1 / 2 / 3 / 4 requests at once, greedy answers of 93-200 tokens, time from the send to the last token (two rounds) | with the head 3.6 / 5.9 / 8.4-8.6 / 10.1 s; without it 4.1-4.2 / 5.9-6.0 / 8.2-8.3 / 9.8 s |
| 5 requests with different sampling settings (greedy, T=1, top-p only, top-k 1000 + top-p decoding together; top-k 20 + top-p queued behind them), 256-token answers | ~27-28 tok/s for each of the 4 decoding together |

In `xtx-xt` and its K/V-in-RAM variants three and four requests verify too, 2 rows each (`FREETOKEN_SPEC_BS_MAX=4`,
`FREETOKEN_SPEC_BS_DEEP=2`), against the setting before (`2` / `2`: 3-4 requests run one row each). On `xtx-xt-4x262k`, greedy 768-token
answers unless noted, tok/s per request, two runs each, a 300k-token pool, the MTP head drafting under each request's
sampling:

| Requests | NVFP4 before -> now | EXL3 3.05 bpw before -> now |
|---|---:|---:|
| 3 at once | 38.4 -> 40.8-40.9 | 44.4 -> 48.3 |
| 4 started 1.5 s apart | 34.2-34.3 -> 35.3-35.4 | 41.0-41.9 -> 43.6-43.7 |
| 4 at once (512 tokens) | 31.3-32.8 -> 33.8-33.9 | 38.3 -> 40.2-40.3 |
| 4 started apart, T 0.6 | 32.4-33.3 -> 32.0-33.5 | 40.5-41.2 -> 41.8-42.0 |
| 1 (four 768-token answers in turn) | 95.9-96.0 -> 95.8-96.0 | 97.0 -> 96.9-97.1 |

With 2 to 4 rows each at four requests (NVFP4, every step at that count): 35.2 tok/s at 2 rows, 31.7 at 3, 30.4 at 4,
31.3 with one row.

With `FREETOKEN_SPEC_BS_MAX=1` (an earlier setting of `xtx-xt`; the untested `xtx-xtx` and `xt-xt` keep it), only a
request decoding alone verifies drafts: with two or more, steps run one row per request and the head only writes its KV,
an eager pass that cost 3-4 % at 3-4 requests on `xtx-xt` (an earlier build); a lone request finishes 13 % sooner.

A request's decode is computed the same way alone or batched (per-row kernels); prompts prefilled in the same batch can
differ at rounding level, as they already do with prefix caching.

## Speculative decoding (MTP head)

`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4` (adaptive depth), each profile without the head against the same plus the
head (one server per configuration; the odd half of the 120-item set). Greedy answers identical with and without the
head on every profile (5 chat prompts, the 60 items, the 108k-token turns, the functional checks). The
[summary](#summary) and the depth sweeps give one request's decode with and without the head; by kind of work:

| One request, decode tok/s (or time) | `xtx-xt` | `xtx` (2 rows) | `xt` (2 rows) |
|---|---:|---:|---:|
| 3 chat prompts x 2, median | 57.7 -> 74.8 | 38.0 -> 40.3 | 31.2 -> 32.7 |
| 3 agent turns on a 3.5k-token prompt | 51.5 / 58.9 / 54.3 -> 71.8 / 65.4 / 66.5 | 34.9 / 38.5 / 32.4 -> 37.6 / 40.3 / 34.7 | 28.8 / 30.7 / 27.4 -> 29.8 / 31.7 / 28.5 |
| 108k-token prompt, then a follow-up turn | 56.4 / 55.4 -> 71.0 / 76.2 | 35.1 / 35.1 -> 38.0 / 39.0 | 29.1 / 28.8 -> 31.5 / 31.7 |
| Time to the first token of that prompt | 56.8 -> 59.2 s | 78.7 -> 82.4 s | 84.5 -> 88.4 s |
| One 1500-token answer | 59.8 -> 73.9 | - | - |
| 60 code / math items (time) | 296 -> 215 s | 516 -> 473 s | 613 -> 563 s |

On `xtx-xt` a request keeps 1.9-3.9 tokens per step (512-step averages). The one-card columns ran at 2 rows, their
answers identical to plain decode (-: not measured at 2 rows), except their times to the first token, which come from
the run with the two-card step costs below.

On one card the extra rows cost more: its expert cache holds ~2.3k (XT) or ~3.5k (XTX) of the 24.6k experts against
~7k per card on `xtx-xt`, and a 4-row step on the XT misses 8-10 experts per layer against 2-3 in plain decode (all
copied over PCIe): 93-98 ms against ~31. With the two-card step costs (mostly 4 rows) the head made one card slower
(chat prompts: `xtx` 38.0 -> 34.6 tok/s, `xt` 31.2 -> 27.6; the other figures of that run in
[journey.md](journey.md#15-speculative-decoding-with-the-mtp-head-2026-10-01)). At 2 rows (53 ms) the head pays; the
one-card step costs make the adaptive depth keep to 2 rows (`xt`: chat 32.6 / 29.8 / 36.3 tok/s against 30.9 / 28.9 /
32.2 without the head).

Step costs (`xtx-xt`, NVFP4, `FREETOKEN_SPEC_TIMING=64` over decode, concurrency and prompt-reading runs, contexts up
to ~41k tokens; GPU ms per step, phases summed):

| Requests x rows | 1 x 2 | 1 x 3 | 1 x 4 | 2 x 2 | 2 x 3 | 2 x 4 | 3 x 1 | 4 x 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Step (forward + verify + head + draft chain) | 25.3 | 28.8 | 36.4 | 39.7 | 53.1 | 65.0 | 36.3 | 44.5 |
| Of which the forward | 21.4 | 25.2 | 31.7 | 35.5 | 48.6 | 59.1 | 35.6 | 43.9 |
| Steps measured | 1640 | 1745 | 1434 | 25 | 68 | 272 | 1022 | 1022 |

These runs used `FREETOKEN_SPEC_BS_MAX=2`: with three or four requests a step ran one row each and the head only wrote
its KV (now 2 rows each in `xtx-xt` and its K/V-in-RAM variants, [above](#several-requests-and-agents-at-once-xtx-xt)).

## Precision

- **Scored benchmark** (671 items: HumanEval 164, MBPP 257, GSM8K 250; greedy, thinking off): the community ROCm port
  (bf16) 636, the ROCm port this build started from (bf16) 632, the int8-dense builds 635. Two bf16 builds already
  differ by 4 items: no measurable loss.
- **Agentic benchmark** (the author's, private: a coding agent in OpenCode fixing 29 bugs planted in a ~50k-line
  codebase, long sessions, single runs): builds with the int8 attention KV cache fixed 8-9 bugs, builds with the bf16 KV
  12-14. The int8 KV is out of every profile ([options.md](options.md)).
- **What the agentic benchmark covered**: the two-card builds that scored 12-13 had the uneven split, int8 dense layers,
  one MoE all-reduce per block, the split-K GEMVs, the top-k-first sampler, 16k chunks with blocked PLE, expert reuse in
  prefill, the host-memory all-reduce and the tuned prefill tiles (swept again since, on upstream's rewritten prefill
  GEMM, whose NVFP4 answers differ). Added after the last scored build, and checked for exactness instead: the tuned
  decode tiles, up to 4 concurrent requests (a request's decode is identical alone or batched; prompts prefilled
  together can differ at rounding level), the RAM tier, the sampler rework for requests without a small `top_k`, rank
  0's token broadcast, the MTP head's drafts drawn under each request's sampling (exact speculative sampling) and three
  or four requests verifying 2 rows each on `xtx-xt-4x262k` (greedy answers unchanged). The one-card profiles were not
  run on it. A build on ROCm 10 with the MTP head (`xtx-xt`) scored 12 and 18 in two runs; the same build with the EXL3
  checkpoints ([how-it-works.md](how-it-works.md#exl3-checkpoints)) scored 13 at 3.05 bpw and 12 at 4.05 bpw, one run
  each.
- **Exactness checks**: a fixed set of 120 greedy prompts (not published) gave identical answers for the host-memory
  all-reduce, the tuned NVFP4 tiles (prefill and decode), expert reuse in prefill, a single request on the 4-request
  build, and the review-fix branch against the swept one. Kernel checks prove the blocked prefill MoE and batched decode
  rows identical (`moe_block_check.py`, `int8_rows_check.py`); four conversations pushed out to the RAM tier and brought
  back give the same greedy answers (and `host_kv_check.py` checks the copies).

## Images (`--vision`, `xtx-xt`)

Measured with `rdna3/serve.sh --vision`, images capped at 1024 tokens unless stated otherwise.

**Text with the vision build.** Four sessions ran interleaved (text-only, vision, text-only, vision), each with three
`quick_bench.py` runs; the ranges below span both sessions of each mode:

| | text-only | `--vision` |
|---|---:|---:|
| decode, 512 tokens | 53.1-55.1 tok/s | 53.3-54.3 tok/s |
| cold read, 8.3k tokens | 4.28-4.40 s | 4.37-4.40 s |
| agent turn, ~1.5k new tokens | 1.27-1.29 s | 1.27-1.28 s |
| expert cache (both ranks take the smaller plan) | 7,228 | 7,217 |
| greedy answers, 34 prompts of 20 to 20k tokens (thinking off) | reference | identical in every session |

With the tower on both ranks (a first version), rank 1 planned 351 fewer experts (6,877): rank 1 never encodes, so
it holds no tower.

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
  The first request of any kind after a start takes 1.3-1.9 s, the JIT modules being kept in the kernel-cache volume.

## Memory while serving (`xtx-xt`)

| | Used (MTP head on) |
|---|---|
| XTX VRAM | ~21.3 GiB of 24.0 while decoding (the desktop's share included), up to ~23.7 GiB while a 16k-token chunk is read deep in the context |
| XT VRAM | ~18.0 GiB of 20.0 while decoding, up to ~19.9 GiB while a 16k-token chunk is read deep in the context |
| Host RAM | 82 GiB resident over the two processes, all of it locked (`FREETOKEN_MLOCK`): ~63 GiB of experts (each rank pins its own share of every expert), the RAM tier (4.6 + 4.9 GiB with the snapshots), the rest the processes |

VRAM is the driver's count (`mem_info_vram_used`) sampled twice a second over a depth sweep to 248k: the ~2 GiB left
free per card while decoding is the prefill's working memory, which a deep 16k-token chunk uses almost entirely (56 MiB
left on the XT). Host RAM is the kernel's `Mlocked` count after start.

**Every profile and checkpoint** (one server each, `rdna3/serve.sh`): RAM is the growth of the kernel's `Mlocked`
count from before the start to after it, and the expert cache the planned size, of 24,576 experts (the same on both
cards: each rank plans on its own free memory, then both take the smaller plan), both from the servers of the summary
above (`xtx-xt` at 262,144 tokens of context; an earlier build: a later one locks 1.2 GiB more RAM and caches 7 % more
experts with NVFP4, 6,977 -> 7,464 slots); VRAM, from an earlier build, is the highest `mem_info_vram_used` sampled
every 50 ms over an agent-like first turn (a title request decoding while a ~21k-token prompt is read), the decode
answers, `quick_bench.py` and the depth sweep, the desktop's share included.

| | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| RAM locked, `xtx-xt` / `xtx` / `xt` | 82 / 73 / 73 GiB | 60 / 51 / 51 GiB | 75 / 66 / 66 GiB |
| Expert cache, `xtx-xt` / `xtx` / `xt` | 6,990 / 3,567 / 2,328 | 9,800 / 4,902 / 3,057 | 7,217 / 3,686 / 2,298 |
| VRAM peak, `xtx-xt` (XTX / XT of 24,560 / 20,464 MiB) | 23,524 / 20,291 MiB | 24,284 / 19,228 MiB | 24,035 / 19,197 MiB |
| VRAM peak, `xtx` / `xt` | 23,506 / 19,367 MiB | 22,943 / 19,072 MiB | 22,868 / 19,081 MiB |

No GPU queue was stopped by a full VRAM in those VRAM runs; the two 4.05 bpw one-card servers stopped their queues
once, for 47-59 ms, on their first request (the VRAM was 1.7 GiB from full; the cause was not traced).

How to reproduce or check a change: [testing.md](testing.md).
