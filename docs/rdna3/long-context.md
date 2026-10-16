# Long contexts with the K/V in host RAM (experimental)

Qwen3.8-Flash-Next is trained to 262,144 positions, and Qwen's model card extends it to 1,000,000 with YaRN
(`rope_type: yarn`, `factor: 4`, `original_max_position_embeddings: 262144`). With the attention K/V in host RAM, two
cards hold either one conversation of up to 1M tokens (`xtx-xt-1m`) or four conversations of 262,144 tokens at once
(`xtx-xt-4x262k`), where `xtx-xt` shares 262,144 tokens between its requests. This page reports what was measured from
2026-10-10 to 2026-10-12 on the reference machine (RX 7900 XTX + RX 7900 XT, MTP head on), mostly with the EXL3 3.05
bpw checkpoint.
Neither profile is agent-validated yet.

## What it takes

- **The attention K/V does not fit in VRAM at 1M.** It costs 14,144 bytes per token on each card (13 attention layers
  with the MTP head's, one KV head of 256 per rank, plus the compressed index keys), so 1M tokens is 13.2 GiB per card.
  On `xtx-xt` the engine refuses that plan: the 7900 XT's budget for experts and KV is 10.9 GB at
  `--memory-ratio 0.77`, and 1M of KV plus the minimum 1024 expert slots needs 14.9 GB.
- **`FREETOKEN_QSA_KV_HOST=1`** keeps the K/V slab in pinned, device-mapped host memory, handed to torch as a GPU tensor
  (`kernel/host_mapped.py`): the decode and verify kernels read the ~2051 selected tokens of each row in place across
  PCIe, and a prefill first copies the pages of the batch's history (since 2026-10-14 only the K/V rows its tokens select,
`FREETOKEN_QSA_STAGE_SELECTED`), one layer at a time, to a VRAM buffer
  (`kernel/triton/qsa/stage.py`, 28 GB/s), so its rows do not each cross PCIe. The index tiers stay in VRAM. At 1M the
  engine then plans 11,261 expert slots per card (9,932 for the `xtx-xt` profile when it served 250k with the K/V in VRAM, 2026-10-10; 9,800 at 262k on 2026-10-16) and
  locks ~75 GiB of host RAM; the 7900 XTX peaked at 23.20 of 23.98 GiB (sysfs) during a 1M read with the desktop on it. The
  answers are bit-identical to the K/V in VRAM (the same text at 29k and 118k).
- **Rope.** Two ways past 262,144 positions:
  - YaRN from `rope_parameters`, the model card's setting. Upstream's `--hf-overrides` (`235e201`) sets it without
    editing `config.json`; the section is replaced whole, so it carries every key:
    `rdna3/serve.sh xtx-xt-1m ... -- --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
    "original_max_position_embeddings": 262144, "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    "mrope_interleaved": true, "mrope_section": [11, 11, 10]}}}'` (the config it builds is checked on the CPU by
    `tests/models/qwen4_exp/test_config.py`; on the GPU, EXL3 3.05 at 497k, the same text as the `config.json` edit,
    12/12, read at 2036 tok/s, decode 85 tok/s). The checkpoints carry a vision tower, so the
    rope is the multi-axis one; YaRN composes with it and the table covers `original x factor` positions, as SGLang /
    vLLM size it (upstream's code since the merge of `ee9789f`, written here first). Bit-identical to HF transformers
    5.16.1's rotary embedding (`tests/layers/test_rotary_yarn.py`).
  - **`FREETOKEN_ROPE_MAX_POSITION=1048576`**: the plain rope with a longer table, no YaRN. Below 262,144 the rows are
    the same bits as without it, so short contexts are unchanged.
- `rdna3/profiles/xtx-xt-1m.env` puts it together for one long conversation (plain rope, K/V on the host,
  `FREETOKEN_HOST_KV=0`); `rdna3/profiles/xtx-xt-4x262k.env` keeps each request to the native 262,144 positions and
  sizes the K/V pool for four of them (`KV_TOKENS=1050000`, see [Four conversations at once](#four-conversations-of-262144-tokens-at-once)).

## Speed (EXL3 3.05 bpw, `xtx-xt`, one request)

Cold reads of contexts that share no prefix (stdlib code, below), then a 700-token greedy answer:

| Context | `xtx-xt` at 250k (2026-10-10), K/V in VRAM: PP / TG | 1M, K/V on the host: PP / TG | 1M, both decode fixes below |
|---|---|---|---|
| 29k | 2268 / 93.2 tok/s | (first request after a start) | |
| 118k | 2185 / 90.6 | 2376 / 84.3 | (non-coherent only: TG 84.2) |
| 237k | 2128 / 94.1 | 2293 / 86.6 (YaRN) | (non-coherent only: TG 88.5) |
| 498k | - | 2048 / 85.8 | 1973 / 91.9 (first request after a start) |
| 699k | - | 1891 / 75.1 | |
| 988k | - | 1711 / 72.1 (9.6 min to read) | 1775 / 83.6 (9.3 min) |

Without the prefill staging, reading across PCIe ran at 789 tok/s at 118k. An agent conversation on a 498k-token
context: first turn 245 s, the next two turns 4.2 and 2.3 s to the first token (the prefix is reused).

## Quality

- **Retrieval** (Python stdlib files, default values replaced by new random ones so that memory does not answer, plus
  notes inserted between files; one request with every question): 100 % from 29k to 988k with the plain rope and with
  YaRN. Harder set (30 code questions, 6 notes, 3 two-hop notes): YaRN 19/19 at 237k, 25/25 at 498k, 33/36 at 988k;
  plain rope 32/36 at 988k. The two-hop questions fail at 988k with both (the answer is the pointer note), and pass at
  237k and 498k. With the indexer budget doubled to 4096 (config edit) the 988k run misses the same four, reads at
  1551 tok/s and decodes at 75.5: the sparse selection is not what loses the two-hop questions.
- **Short texts** (12 texts of 2.6-5.2k tokens from this repository's code and docs, teacher-forced, raw completions):
  next-token NLL 1.1298 without YaRN, 1.1339 with factor 4 (+0.4 %), 1.1307 with factor 2; KL to no-YaRN 0.032
  (factor 4) and 0.018 (factor 2), top-1 agreement 93.4 % and 95.1 %.
- **A 988k-token text** (this repository's code and docs, raw completion): the plain rope's NLL stays at 0.46-0.79
  nats per token at every depth (no rise past 262k); YaRN factor 4 is 0.2-1.3 % worse up to ~650k and 1-3 % better
  beyond.
- Measurement traps met on the way: teacher-forcing a text sent as a chat *user* turn measures mostly the
  probability of `<|im_end|>` (~94 % at every position); the stdlib is memorized (raw NLL 0.06, top-1 98.5 %), so it
  only serves retrieval with planted values.

## Decode with the K/V on the host

Measured in the afternoon of 2026-10-10, after the tables above:

- **Where the decode time went.** QSA decode attention, 4 verify rows of one request (2051 selected tokens each, 90 %
  shared), one layer on the 7900 XT: 122 us with the K/V in VRAM, 313 us from host memory allocated
  `Portable | Mapped` (8 rows: 565 us); 1 row: the same as VRAM. A mapped allocation is coherent, so the GPU's L2 does
  not keep its lines and each row crosses PCIe again. Allocated `hipHostMallocNonCoherent` (`kernel/host_mapped.py`,
  now the default on ROCm): 129 us (8 rows: 220 us), the same bits. In the server it gave less than the kernel time
  suggests: decode 84.2 tok/s at 118k (84.3 before), 88.5 at 237k, 87.9 at 498k (85.8 before); the MoE weights going
  through the same L2 probably evict the K/V lines between rows (not measured).
- **The indexer's top-k fell back to one program per row on a 1M-token table.** Its split path holds at most 16
  chunks of 8192 columns; a 1M-token context's page table is 250,000 block columns wide, so `_split_plan` gave up and
  each row was scanned by a single program: 504 us per decode layer at 988k visible tokens against 77-83 us split
  (4 rows, the same winners, ties included). The split now takes more chunks past 16; tables up to 131,072 columns
  (contexts up to 524,288 tokens, so the 262k profiles) keep their plan. This tree first merged the extra chunks in one
  spilled tile; since the merge of upstream `ee9789f` it runs upstream's own fix (`5031e3d`): reduction levels, each
  program keeping the top-k of 8,192 candidates, until the survivors fit one merge, with no length limit. On the 7900
  XT, one decode layer's top-k at 247,000 visible columns (988k tokens): 53 / 64 / 96 us for 1 / 4 / 8 rows against
  65 / 75 / 101 with the spilled tile, the same winners; 4-7 us more than the spilled tile at 30,000 visible columns,
  the same as before on the 65,536-column tables of the 262k profiles (`rdna3/bench/qsa_topk_bench.py`).
  Below ~100k visible tokens the wide table's split costs ~40 us per layer more than the one-program scan did. With
  both changes, measured with the spilled tile (a cold read, then 700 tokens): 498k read at 1973 tok/s, decode 91.9
  tok/s (85.8 before); 988k read at 1775 tok/s in 9.3 min (1711), decode 83.6 tok/s (72.1); the same text as before
  both changes. With upstream's levels (the image of the merge): the same text again at 498k (read 2014 tok/s, decode
  93.0) and 988k (1780, 84.7), XTX peak 22.76 GiB.
- **A/B at 118k** (decode tok/s, EXL3 3.05): K/V in VRAM, 262k table (the `xtx-xt` profile as it was then: 250k, 9,932 expert slots) 90.6;
  K/V on the host, 262k table (12,667 slots) 86.6; K/V on the host, 1M table (11,261 slots) 84.2. Both the host K/V
  and the wide table cost a few percent there.
- **Tried and dropped: the first 262,144 tokens' pages also in VRAM** (decode reads them there, the allocator hands
  them out first). Bit-identical, but those 3.5 GB per card leave 7,400 expert slots: decode 84.4 tok/s at 237k, below
  the host K/V alone (88.5). The prefix cache also keeps the low pages of finished conversations.
- **Two cold requests at once** (498k + 237k): both 12/12 with the same answers as alone; the 7900 XTX peaked at 23.09
  of 23.98 GiB (sysfs, desktop included), the 7900 XT at 17.44 of 19.98. The 498k request decoded at 6.3 tok/s while
  the other one was read: prefill goes first (upstream policy, every profile; `FREETOKEN_DECODE_INTERLEAVE` in
  [options.md](options.md)).

## Four conversations of 262,144 tokens at once

The K/V pool is shared by the running requests: `rdna3/serve.sh` passes a profile's `CTX` as the cap of one request
(`--max-seq-len-override`) and its `KV_TOKENS` (default `CTX`) as the pool (`--kv-reserve-tokens`). A request is admitted
when the pool can take its prompt plus its `max_tokens`, which the server lowers to what is left under the cap, so a
request never reserves more than 262,144 tokens and four of them fit in 4 x 262,144. On `xtx-xt` the pool is the
262,144 tokens of one context, and the conversations that leave it go to the RAM tier (`FREETOKEN_HOST_KV`).

`xtx-xt-4x262k` (cap 262,144, native rope, pool 1,050,240 tokens, 11,345 expert slots per card), four conversations
on distinct Python stdlib contexts (no shared prefix, planted values as in the retrieval set above), 2026-10-10:

| Step | Result |
|---|---|
| 4 cold prompts of 255.7-257.9k tokens sent together | read one after another (prefill goes first): 1,027,138 tokens in 463 s (2218 tok/s), every answer right (13/13, 13/13, 12/12, 8/8) |
| 4 next turns together (257-260k, ~800 new tokens each) | first token after 3.0-4.2 s, decode 29.4-30.1 tok/s each (no drafts past two requests) |
| 2 next turns together (259-261k) | first token 1.8-2.9 s, decode 57.2 and 60.6 tok/s |
| 1 next turn alone (261k) | first token 1.8 s, decode 91.3 tok/s (94.1 on `xtx-xt` at 237k, K/V in VRAM) |
| 4 turns together to the cap | first token 3.0-4.6 s; each conversation ends at exactly 262,144 tokens (1,048,576 in the pool at once), no error |
| 1 more turn | refused at once: `prompt is too long: 262188 tokens > 262144 maximum` (`context_length_exceeded`) |
| Peaks | 7900 XTX 23.24 of 23.98 GiB (sysfs, desktop included), 7900 XT 17.10 of 19.98; 77.1 GiB of host RAM locked, 31.9 GiB left available |

## A decode step on `xtx-xt-4x262k`, kernel by kernel (night of 2026-10-11)

EXL3 3.05 bpw, `offload`, the profile's defaults. `rdna3/bench/depth_sweep.py` (one request, greedy 256-token answers,
one pass): TG 107-110 tok/s at 25-60k tokens, 101 at 248k; PP of the new block 2340-2430 tok/s at 25-60k, 2020 at 248k;
an agent turn (~1k new tokens on the cached conversation) 0.87 s at 25k, 1.76 s at 248k. Four agents on distinct ~257k
contexts: the cold reads one after another (the last first token after 442 s), then 28.2-30.2 tok/s each decoding
together (a step ~32 ms, 96 % of it GPU time; ~33 experts active per layer, 3-6 % of them missing the cache), 92.0
alone at 259k. `--hybrid` (fetch fraction 0.6) on this profile, same runs: -8 % at 8.8k, +3 % at 60k and 123k, within
2 % elsewhere; four agents 28.6-30.3 tok/s each, one alone 88.5.

**Per kernel** (`rocprofv3 --kernel-trace`, a 20 s window of sampled one-request decode). ROCm's `torch.profiler` in
this image records no GPU kernel; `rocprofv3` does, run with a writable `-w` directory and the GPU workers stopped with
SIGTERM before the container (killed, they write nothing). GPU time per decode step on rank 0 (7900 XTX), 22.9 ms:

| Kernel | ms / step | What |
|---|---:|---|
| `_gemv_splitk_int8` + `_gemv_int8_row_group` | 6.7 | the int8 dense layers |
| `_exl3_gemv_kernel` | 3.4 | the routed experts' EXL3 GEMVs |
| `host_allreduce_kernel` | 2.4 | 99 calls (two a layer), 24.6 us each on average, mostly waiting for the other card |
| `fast_index_copy_multi` | 2.0 | the expert misses copied from host RAM (one call a layer) |
| `__amd_rocclr_batchMemOp` | 0.6 | the graph waiting at layer 1 for the PLE rows the host reads from disk (mostly the tracer's doing: below) |
| `ncclDevKernel_Generic_4` | 0.5 | RCCL, 4-6 a step, ~92 us each: the LM head's all-gathers (the verify step's and each draft's, ~60 % of them on rank 1) and the broadcasts of rank 0's sampled tokens |

- **Between the kernels**: every kernel of the replayed graph starts 3.2-3.5 us after the previous one ends (under the
  tracer); a GDN layer is ~28 kernels, an attention layer ~40. The median layer: attention part 164 us of wall time
  for 127 us of kernels, MoE part 262 us for 197. A chain of tiny kernels replayed from a graph takes 2.7-2.8 us per
  kernel on either card, the same with `HIP_FORCE_DEV_KERNARG=1` or `0` (in serving too, on the final code: one
  request 90.3 / 90.4 tok/s against 90.5 unset, four requests 35.3 / 35.4 against 35.4): no runtime setting found;
  fewer kernels would help.
- **The two cards**: rank 1's trace covers another stretch of the run (212 steps in its window, 404 on rank 0), so they
  compare per step only roughly. Their kernels take about the same GPU time per step (20.5 and 21.1 ms without the
  all-reduces); at the median all-reduce rank 0 waits ~10 us longer than rank 1 (17-18 against 8 us), so rank 1 is the
  one that arrives late. On rank 1, `_exl3_out_kernel` took 14.6 us against 6.9 on rank 0 at the same grid (found below).
- **On one card** (`xtx`, EXL3 3.05, one request) the expert misses dominate: `fast_index_copy_multi` 12.0 of 32.4 ms per
  step (37 %), 251 us a layer, which is the time `--hybrid` takes back ([cpu-experts.md](cpu-experts.md)).

**What came of it** (changes bit-exact unless said; the serving A/B at the end):

- **Rank 1's EXL3 kernels.** Rank 1's 256-row slice gives the down projection a single K split, and Triton compiles an
  argument equal to 1 as a constant: that variant of `_exl3_out_kernel` ran ~2x slower on both cards (cold, the 7900 XT
  13.6 -> 7.8 us at 1 token, 20.2 -> 6.8 at 4; the XTX 12.0 -> 8.5 and 17.7 -> 6.0). `SPLITS` is no longer specialized,
  the same bits. Rank 1's gate|up GEMV takes 4 N tiles a program (the same sums; the K tile, the split and some warp
  counts change them): cold 33.8 / 62.2 / 69.7 -> 29.4 / 53.0 / 53.5 us at 2 verify rows / 4 rows / 4 requests. Rank
  0's shapes have no faster config with the same bits. Together, against the code without them (same session): one
  request 85.8 -> 87.4 tok/s (+1.8 %, 12 of 12 answers faster), four requests +0.4 %, forward pass -1.6 to -2.3 % at 2-4
  rows, greedy answers identical.
- **Rank 0's sampled tokens through the host all-reduce** instead of RCCL broadcasts (rank 1 adds zeros, exact in fp32
  for token ids; the LM head's all-gathers stay on RCCL): those broadcasts bought nothing back (probably time waiting
  for the other card).
  Measured again on the final code the next morning: one request -0.3 % (EXL3, 0 of 12 answers faster) and -0.2 %
  (NVFP4); four requests +2.2 % on EXL3 but -0.5 % on NVFP4, and the host path for the plain decode step only (four
  requests) +0.2 %: within the noise of that test. Dropped.
- **The PLE fill.** The disk backend reads every row with O_DIRECT (the 30.4 GiB table does not stay in the page
  cache: a buffered read took ~70 us a row): a fill of 4 tokens (64 rows, 16 hash heads) took 415 us alone in a
  microbenchmark, 500-560 with both ranks reading, 0.84-0.90 ms in serving at 4 rows, while only layer 0 runs before
  the graph waits for it (a pread pool of 32 threads instead of io_uring: 360 us). A RAM cache of the rows read (512
  MiB per rank, direct-mapped; in the last tenth of a 258k-token code context 84 % of the 2-gram and 68 % of the
  3-gram windows occurred before; in the benchmarks' short conversations 21-60 % of the rows hit) brought the fill at
  four requests from 0.88 to 0.57 ms and decode nothing (one request +0.0 %, four 34.7 -> 34.3 tok/s, -1.0 %): the
  fill is not on the step's critical path. NVFP4 `xtx-xt` on the final code: +0.1 %, four requests 0.0 %. Dropped.
- **int8 GEMV configs for the uneven split at 2-4 rows.** Keeping the bits leaves only BLOCK_N free: a 4-wide tile for
  the GDN in_proj (9270 / 7210 x 2560) took 3-11 % less per call at 2-4 rows cold, and nothing in serving: first
  measured with the slower fusions below in place, then again on the final code, one request -0.3 % (EXL3, forward
  +0.1 to +0.4 %) and -0.2 % (NVFP4), four requests +0.5 % and +1.6 % (within that test's noise). Dropped. K blocks of
  512 for three shapes (and a wider HC up tile) are faster (forward pass -1.5 / -2.6 / -2.9 % at 2 / 3 / 4 rows, -2.3 % at four requests, 34.0
  -> 34.8 tok/s each) but sum each row in another order: greedy answers changed (6 of 14), so under Rule 0
  ([decisions.md](decisions.md)) they are opt-in (`FREETOKEN_INT8_TP_TILES=1`).
- **Two small kernels computed inside the next GEMV** (`hc_silu` before the hyper-connection's up projection, SiLU-and-mul
  before the shared expert's down projection), the same bits: 10 % slower in serving (87.3 -> 78.4 tok/s, forward
  +8 to +15 %). Likely reason (not profiled): every program of the consuming GEMV recomputes the activation of the whole
  row it reads, 10,240 programs for the up projection. Dropped.
- **`hc_silu` in the producing GEMV instead**: the hyper-connection's int8 down GEMV stores its low-rank columns through
  `hc_silu` (once per value; the inject columns as they were) and the up GEMV reads that strided slice in place
  (`FREETOKEN_HC_SILU_EPILOGUE`), the same bits, two kernels less per layer: one request 87.4 -> 88.5 tok/s (+1.2 %, 12
  of 12 answers faster), four requests 34.3 -> 35.0 (+1.9 %), forward pass -1.3 % at 2-4 rows and -1.9 % at four
  requests, greedy answers identical. NVFP4 `xtx-xt`: 78.0 -> 78.8 (+1.1 %, 12 of 12), four requests +1.1 %, the same
  greedy answers.
- **Two more kernels folded into a GEMV, each value computed once**: the up projection mixes its gates into the
  streams as it computes them (`hc_gate_mix`, `FREETOKEN_HC_MIX_IN_GEMV`; a program owns one hidden index, loads each
  stream's weight row once and computes that gate for every row), and the shared expert's gate | up projection stores
  SiLU(gate) x up (`FREETOKEN_SHARED_ACT_IN_GEMV`; a program owns its gate columns and the matching up columns,
  sharing each weight tile across all the rows when K fits one tile and across 2 rows when it does not; past 2 rows a
  K-blocked shape keeps the two kernels, which were as fast or faster there). The first version of the mix accumulated
  `sigmoid(g) * x` through another expression than `_hc_gate_mix_kernel`: a single-seed test matched, but 53 of 120
  random cases had one to four values one bf16 step apart, and in serving 7 of 14 greedy answers changed; with the
  kernel's own expression, 0 of 240 cases (and 0 of 960 for the activation), and the tests draw several seeds. The
  first versions also reloaded the weights for every row: the mix was then slower from 4 rows on NVFP4 (four requests
  -1.8 %) and capped at 3 rows, and the activation cost NVFP4 0.8 %, so it was dropped; with the weights loaded once,
  measured the next morning against the code without each (same session; four requests over two passes): the mix at
  every row count +0.6 % at one request on `xtx-xt-4x262k` (12 of 12 answers faster, forward -1.2 % at 4 rows) and
  +1.4 % at four requests, NVFP4 `xtx-xt` +0.5 % and +1.1 %; the activation +0.8 % (12 of 12, forward -0.7 to -1.3 %)
  and +1.3 %, NVFP4 +0.1 % and -0.1 % (its shapes are K-blocked); greedy answers identical. Both kept: together, in
  the code kept, against the night's final code (same session): `xtx-xt-4x262k` 89.0 -> 90.6 tok/s (+1.8 %, 12 of 12)
  and +2.1 % at four requests, NVFP4 `xtx-xt` 79.2 -> 79.8 (+0.8 %, 12 of 12) and +1.0 %, greedy answers identical; a
  256k-token prompt in `check` mode read in 114 s, 13/13 answers, 92.6 tok/s, no disagreement between the ranks.
- **Grouping the verify rows that share an expert in one GEMV** does not pay. At one request the 4 verify rows route
  40 times to ~22 experts per layer, and the EXL3 decode GEMV runs one program per route, so a shared expert's trellis
  is decoded once per row. A GEMV that decodes each tile once for up to four routes on a slot (bit-identical) took rank
  0's shapes from 104 to 51 us (gate|up) and 54 to 30 us (down) on the 7900 XTX, but that microbenchmark called the
  same 22 experts over and over, which then sat in the Infinity Cache. As a step reads them (48 layers, each on its own
  experts out of a 4096-slot bank) the expert GEMVs of a step took 7.95-8.09 ms with or without the grouping; in serving
  the forward pass went 26.4 -> 26.8 ms at 4 rows (+2.2 / +3.4 % at 2 / 3), EXL3 on `xtx` -1.5 % tok/s. NVFP4 (marlin
  decode): grouped slower in 17 of 18 shapes, by up to 63 %. Dropped.
- **The MTP cost table, measured again** (`FREETOKEN_SPEC_TIMING`, GPU ms per step at one request): EXL3
  `xtx-xt-4x262k` 24.3 / 26.2 / 30.8 at 2 / 3 / 4 rows, NVFP4 `xtx-xt` (`--hybrid`) 25.2 / 27.5 / 32.9, against
  25.3 / 30.7 / 35.5 in `spec_decode.py` (2026-10-01). With the measured table (`FREETOKEN_SPEC_COSTS`), same session:
  EXL3 85.8 -> 86.4 tok/s sampled, greedy texts identical; NVFP4 80.8 -> 81.6. Within the noise: the dynamic depth
  already ran 3-4 rows most of the time (1202 + 1465 of 3300 steps). On one card (`xtx`, `--hybrid`) the depth stays at
  2 rows (4593 of 4593 steps), and forcing 3 or 4 rows was slower (NVFP4 50.6 / 44.5 tok/s, EXL3 47.4 / 40.4, against
  56.1 and 52.8 at 2 rows): the one-card table is right.

Also measured the next morning: `num_stages` 1-4 on the decode GEMVs (EXL3 per-route, the five largest int8 shapes;
load pipelining only, the same bits) moved no shape beyond the ~2 % spread of the cold timings; the router's top-k in
one packed int64 reduction per pick instead of a max and a min (the same picks and weights, ties, -0.0, NaN and -inf
included) saved ~0.5 us a call; several hidden indices per program of the fused mix kept the bits and gained nothing.
None kept. Run-to-run noise, the same code in two sessions: one request +-0.1 % (forward +-0.1-0.2 %), four requests
+-1 %.

The PLE wait, seen again in a trace of the final code (rank 1): its median is 24 us, but it comes in runs of
consecutive steps where it lasts 1.6 ms (median of those, up to 3 ms), 40 % of the steps, 0.68 ms a step on average.
Those runs are the tracer's doing. They sit in the steps of 2 rows (median 1.5 ms there, 4 us at 3 and 4 rows): after
the verify step the host builds the next step while the GPU runs the MTP head and its chained drafts, which take the
least time at 2 rows (the host's wait for the step's tokens after the launch, `FREETOKEN_PLE_FILL_TIMING`, averages
0.4 ms at 2 rows, 1.4 at 3, 2.4 at 4). With the HIP calls traced as well (`--hip-runtime-trace`), `hipGraphLaunch`
held the host 1.3-1.9 ms (p10-p90), the host then found the tokens long there (its wait 1.5 us at the median of the
2-row steps), and the GPU reached the PLE layer ~0.44 ms (median) before the launch returned, then waited for the fill
(~0.45 ms) that only starts after it. Without the tracer the launch takes 0.05 ms and the replay's input copies before
it 0.15 ms (p50 on both ranks, a timing probe), and at 2 rows the host waits for the tokens (0.2 ms at the median): it
is early, so the GPU waits at most for the part of the fill (0.56 ms) that outlasts layer 0. A fill on a helper thread
from the start of the dispatch (the launch releases the GIL; the same rows, the same flag) changed nothing measurable:
forward pass -0.1 / -0.3 / -0.4 % at 2 / 3 / 4 rows, one request +0.2 % (10 of 12 answers faster), four requests
+0.5 %, greedy answers identical. Not kept; with the fill off the critical path, the RAM cache above could not pay
either. A longer active wait of the HIP runtime before it sleeps (`ROC_ACTIVE_WAIT_TIMEOUT=5000`, -0.1 %) and a
shorter GIL switch interval (0.5 ms, +0.1 %) changed nothing either.

**Serving A/B** (`xtx-xt-4x262k`; 12 sampled 512-token answers at one request, 4 requests x 3 rounds, 14 greedy
prompts; each change against the code without it, same session, `FREETOKEN_SPEC_TIMING` for the forward pass):

| Change | One request | Four requests, each | Forward pass, 2 / 3 / 4 rows; 4 requests | Greedy answers |
|---|---:|---:|---|---|
| Rank 1's EXL3 kernels (kept) | 85.8 -> 87.4 tok/s (+1.8 %) | +0.4 % | -2.3 / -2.2 / -1.6 %; -0.4 % | identical |
| `hc_silu` in the down GEMV's store (kept) | 87.4 -> 88.5 (+1.2 %) | 34.3 -> 35.0 (+1.9 %) | -1.3 / -1.3 / -1.3 %; -1.9 % | identical |
| Bit-exact int8 tiles (dropped; measured again on the final code) | -0.3 % | +0.5 % | +0.4 / +0.1 / +0.2 %; -0.6 % | identical |
| Tokens through the host all-reduce (dropped; final code) | -0.3 % | +2.2 % (plain decode step only, two passes: +0.2 %) | +0.1 / +0.1 / +0.3 %; -1.8 % | identical |
| PLE row cache (dropped) | 0.0 % | 34.7 -> 34.3 (-1.0 %) | +0.2 / 0.0 / 0.0 %; +1.1 % | identical |
| `hc_silu` / SiLU-and-mul inside the consuming GEMV (dropped) | 87.3 -> 78.4 (-10 %) | | +8 / +12 / +15 % | identical |
| `hc_gate_mix` inside the up GEMV, up to 3 rows (first version) | 88.4 -> 89.1 (+0.8 %) | (off at 4 rows) | -1.8 / -1.2 / -0.3 % | identical |
| `hc_gate_mix` at every row count, weights loaded once (kept) | 89.4 -> 89.9 (+0.6 %) | 34.6 -> 35.1 (+1.4 %) | -0.1 / -0.1 / -1.2 %; -1.5 % | identical |
| SiLU-and-mul in the shared gate \| up GEMV's store (first version, dropped: NVFP4 -0.8 %) | 89.0 -> 89.3 (+0.4 %) | +0.6 % | -0.5 / -0.9 / -0.3 %; -0.7 % | identical |
| SiLU-and-mul with weight tiles shared across rows (kept) | 89.4 -> 90.0 (+0.8 %) | 34.6 -> 35.0 (+1.3 %) | -1.3 / -1.1 / -0.7 %; -1.4 % | identical |
| All kept changes, against the code before the night (8.4 h apart) | 85.5 -> 90.6 (+5.9 %, 12 of 12 answers faster) | 33.9 -> 35.4 (+4.5 %) | | identical (14 / 14) |

NVFP4 `xtx-xt` (the EXL3 kernels do not apply): `hc_silu` in the store 78.0 -> 78.8 tok/s (+1.1 %, 12 of 12 answers
faster), four requests 24.2 -> 24.5 (+1.1 %), forward pass -1.8 / -1.2 / -0.9 % at 2 / 3 / 4 rows, -1.2 % at four
requests, greedy answers identical (14 / 14).
With everything kept (the EXL3 kernels do not apply there), against the code before the night (8.6 h apart):
77.8 -> 79.8 tok/s (+2.6 %, 12 of 12 answers faster), four requests 24.5 -> 24.7 (+0.7 %), greedy answers identical
(14 / 14).

The four agents of the start of this section again, on the final code (one pass each, the first in the evening): the
same scores (13/13, 13/13, 12/12, 8/8), the cold reads in 445 s (442), then 29.6-31.6 tok/s each decoding together
(28.2-30.2) and 95.1 alone at 259k (92.0).

## Decode glue in kernels (afternoon of 2026-10-12)

A 4-row step on `xtx-xt-4x262k` launched ~1750 kernels, ~400 of them small torch ops (rocprofv3, rank 1), each costing
its own run plus the ~2.8 us between two kernels. The eager part of a verify step (verify, rollback, the MTP head and
its chained guesses, the next step's inputs) ran ~300 of them; inside the graph the first QSA layer's index plan ran
~25 and the PLE layer's 4-row conv ~25. Kept, each giving the bits of the torch ops it replaces (bitwise tests over
random batches, `tests/kernels/test_decode_glue.py`; greedy answers identical, 14 / 14, also against the morning's
code):

- **The QSA index-write plan** in two kernels instead of ~30 ops (`FREETOKEN_QSA_PLAN_KERNEL`): it runs once per
  forward, so four times a 4-row step (the graph, the MTP head, two chained guesses).
- **The PLE's m-row decode conv** in one kernel (`FREETOKEN_PLE_CONV_KERNEL`). Triton contracts `a * b + c` into an
  FMA unless told not to (`enable_fp_fusion=False`): with it, 90-121 of 150 random cases differed whatever the sum
  order; without it, the order torch takes for `sum(-1)` over 4 values on gfx1100, (p0 + p1) + (p2 + p3), matched in
  150 of 150 (the two other orders: 81 and 97 differed).
- **The MTP head's KV-only pass** (more requests decoding than `FREETOKEN_SPEC_BS_MAX`) stops once its K/V and index
  keys are written instead of attending and combining (`FREETOKEN_MTP_KV_ONLY_STOP`).
- **A verify step's glue** (`FREETOKEN_SPEC_GLUE_KERNELS`): the rollback copies one kernel per state (four, instead of
  ~24 ops), the token-pool writes one kernel, the kept counts of greedy rows, the MTP head's rows and each chained
  guess's addressing one kernel each, the acceptance log summed on the host.

Measured with each change against the same tree with it switched off (same session; "greedy" is the mean over 9
greedy 512-token answers whose texts do not depend on the drafts, so they compare answer by answer; "sampled" the 12
sampled answers of the earlier A/Bs; four requests over two passes):

| Change | Greedy, one request | Sampled, one request | Four requests, each | Step time (spec timer) |
|---|---:|---:|---:|---|
| plan + PLE conv + KV-only stop, `xtx-xt-4x262k` | 103.4 -> 105.3 tok/s (+1.8 %, 9 of 9 faster) | 90.7 -> 92.3 (+1.8 %, 12 of 12) | 35.48 -> 36.15 (+1.9 %) | -1.6 to -1.9 % at 2-4 rows and four requests |
| same, NVFP4 `xtx-xt` | 91.0 -> 92.5 (+1.6 %, 9 of 9) | 80.2 -> 81.4 (+1.5 %, 12 of 12) | 24.77 -> 25.02 (+1.0 %) | -1.0 to -1.6 % |
| verify glue, `xtx-xt-4x262k` | 105.5 -> 107.1 (+1.5 %, 9 of 9) | 92.4 -> 93.7 (+1.4 %, 11 of 12) | 35.83 -> 36.02 (+0.5 %) | -1.0 / -1.2 % at 3 / 4 rows |
| same, NVFP4 `xtx-xt` | 92.4 -> 93.9 (+1.6 %, 9 of 9) | 81.3 -> 82.5 (+1.5 %, 11 of 12, one equal) | 25.23 -> 25.15 (-0.3 %) | -1.2 to -1.5 % at 2-4 rows |

Measured and dropped the same afternoon:

- **Verifying drafts with three or four requests** (`FREETOKEN_SPEC_BS_MAX=4`; the 2026-10-03 decision predated the
  adaptive depth): four requests 26.3 tok/s each against 35.35. The depth, costed for one request, picks 4 rows for
  each of the four (a forward pass of 94 ms against 27 for one row each, ~2.55 tokens kept per request and step); the
  only 2-row step seen cost 64 ms. Greedy answers of four at once changed in 3 of 12.
- **MTP drafts over part of the vocabulary** (the draft's argmax over 38-75k frequent token ids, the LM head's rows for
  them copied and split across the cards; the target still verifies every draft, so answers do not change): the head
  and the chained guesses took 1.36 -> 1.0-1.07 and 2.25 -> 1.5-1.65 ms of GPU time at 4 rows, but fewer drafts were
  kept (greedy on fixed texts 103.7 against 103.5 and 104.1 tok/s with 75k and 52k ids) and the copied rows (~92 MiB a
  card with 75k ids) came off the expert cache: four requests 35.9 -> 34.8 tok/s (one pass each, -3 %; forward pass
  +2.8 %). A first set built from code only missed 4.6 % of English prose tokens; with man pages, READMEs and word
  lists, 0.5-1.5 % of held-out ones.
- **Narrower V tiles in the GDN recurrent kernel** (the same bits; other warp counts change them): 1-3 us a call,
  ~0.25 % of a step (estimated). Not run in serving.

The code as kept, measured once more the same afternoon (one pass): `xtx-xt-4x262k` greedy 107.5 tok/s, sampled 93.9
(90.6 in the morning's A/B, 85.5 before the night; other sessions), four requests 35.9; a ~256k-token conversation
read in 111 s, 13/13 answers, then 96.0 and 101.6 tok/s; NVFP4 `xtx-xt` greedy 93.8, sampled 82.5 (79.8 and 77.8),
four requests 24.5 (25.0-25.2 in the two-pass runs above). Greedy answers identical to the morning's code on both
profiles (14 / 14).
