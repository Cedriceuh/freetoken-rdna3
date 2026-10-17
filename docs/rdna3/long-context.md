# Long contexts with the K/V in host RAM

Qwen3.8-Flash-Next is trained to 262,144 positions, and Qwen's model card extends it to 1,000,000 with YaRN
(`rope_type: yarn`, `factor: 4`, `original_max_position_embeddings: 262144`). With the attention K/V in host RAM, two
cards hold either one conversation of up to 1M tokens (`xtx-xt-1m`) or four conversations of 262,144 tokens at once
(`xtx-xt-4x262k`), where `xtx-xt` shares 262,144 tokens between its requests. The numbers below were measured on the
reference machine (RX 7900 XTX + RX 7900 XT, MTP head on), mostly with the EXL3 3.05 bpw checkpoint, on earlier builds
than the summary of [benchmarks.md](benchmarks.md).
Neither profile is agent-validated yet.

## What it takes

- **The attention K/V does not fit in VRAM at 1M.** It costs 14,144 bytes per token on each card (13 attention layers
  with the MTP head's, one KV head of 256 per rank, plus the compressed index keys), so 1M tokens is 13.2 GiB per card.
  On `xtx-xt` the engine refuses that plan: the 7900 XT's budget for experts and KV is 10.9 GB at
  `--memory-ratio 0.77`, and 1M of KV plus the minimum 1024 expert slots needs 14.9 GB.
- **`FREETOKEN_QSA_KV_HOST=1`** keeps the K/V slab in pinned, device-mapped host memory, handed to torch as a GPU tensor
  (`kernel/host_mapped.py`): the decode and verify kernels read the ~2051 selected tokens of each row in place across
  PCIe, and a prefill first copies the K/V rows its tokens select from the batch's history
  (`FREETOKEN_QSA_STAGE_SELECTED`), one layer at a time, to a VRAM buffer (`kernel/triton/qsa/stage.py`, 28 GB/s), so
  its rows do not each cross PCIe. The index tiers stay in VRAM. At 1M the engine then plans 11,261 expert slots per
  card (9,800 for `xtx-xt` at 262k with the K/V in VRAM) and locks ~75 GiB of host RAM; the 7900 XTX peaked at 23.20 of
  23.98 GiB (sysfs) during a 1M read with the desktop on it. The answers are bit-identical to the K/V in VRAM (the same
  text at 29k and 118k).
- **Rope.** Two ways past 262,144 positions:
  - YaRN from `rope_parameters`, the model card's setting. Upstream's `--hf-overrides` sets it without
    editing `config.json`; the section is replaced whole, so it carries every key:
    `rdna3/serve.sh xtx-xt-1m ... -- --hf-overrides '{"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
    "original_max_position_embeddings": 262144, "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    "mrope_interleaved": true, "mrope_section": [11, 11, 10]}}}'` (the config it builds is checked on the CPU by
    `tests/models/qwen4_exp/test_config.py`; on the GPU, EXL3 3.05 at 497k, the same text as the `config.json` edit,
    12/12, read at 2036 tok/s, decode 85 tok/s). The checkpoints carry a vision tower, so the
    rope is the multi-axis one; YaRN composes with it and the table covers `original x factor` positions, as SGLang /
    vLLM size it (the code is upstream's). Bit-identical to HF transformers 5.16.1's rotary embedding, within two
    float32 ulps of 5.18's, which rounds a few rows differently (`tests/layers/test_rotary_yarn.py`).
  - **`FREETOKEN_ROPE_MAX_POSITION=1048576`**: the plain rope with a longer table, no YaRN. Below 262,144 the rows are
    the same bits as without it, so short contexts are unchanged.
- `rdna3/profiles/xtx-xt-1m.env` puts it together for one long conversation (plain rope, K/V on the host,
  `FREETOKEN_HOST_KV=0`); `rdna3/profiles/xtx-xt-4x262k.env` keeps each request to the native 262,144 positions and
  sizes the K/V pool for four of them (`KV_TOKENS=1050000`, see [Four conversations at once](#four-conversations-of-262144-tokens-at-once)).

## Speed (EXL3 3.05 bpw, one request)

Cold reads of contexts that share no prefix (stdlib code, below), then a 700-token greedy answer:

| Context | `xtx-xt-1m`, K/V on the host: PP | TG | `xtx-xt` (250k context), K/V in VRAM: PP / TG |
|---|---:|---:|---|
| 118k | 2376 tok/s | 84.2 tok/s | 2185 / 90.6 tok/s |
| 237k | 2293 (YaRN) | 88.5 | 2128 / 94.1 |
| 498k | 2014 | 93.0 | - |
| 699k | 1891 | 75.1 | - |
| 988k | 1780 | 84.7 | - |

The 498k and 988k rows were measured with both decode changes of [the next section](#decode-with-the-kv-on-the-host)
(the non-coherent host memory and the indexer's split top-k), the 118k and 237k decode with the first only, and the
118k, 237k and 699k reads and the 699k decode with neither. An agent conversation on a 498k-token context: first turn
245 s, the next two turns 4.2 and 2.3 s to the first token (the prefix is reused).

`xtx-xt-4x262k`, the profile's defaults (`rdna3/bench/depth_sweep.py`, one request, greedy 256-token answers, one
pass): TG 107-110 tok/s at 25-60k tokens, 101 at 248k; PP of the new block 2340-2430 tok/s at 25-60k, 2020 at 248k.
Agent turns on this profile: [how-it-works.md](how-it-works.md#prefill). `--hybrid` (fetch fraction 0.6) on this
profile, the same runs: -8 % at 8.8k, +3 % at 60k and 123k, within 2 % elsewhere; four agents on distinct ~257k
contexts 28.6-30.3 tok/s each against 28.2-30.2 without it, one alone 88.5 against 92.0 at 259k.

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

## Decode with the K/V on the host

- **Host memory the GPU's L2 can keep.** On ROCm the K/V is allocated `hipHostMallocNonCoherent`
  (`kernel/host_mapped.py`): unlike a coherent mapped allocation, its lines can stay in L2, where the 4 verify rows of
  a request share 90 % of their ~2051 selected tokens. One layer's QSA decode attention, 4 verify rows, on the 7900 XT:
  129 us from host memory (8 rows: 220 us) against 313 us from a coherent `Portable | Mapped` allocation (8 rows:
  565 us), 122 us with the K/V in VRAM. In serving the gain is smaller than these kernel times suggest (decode 84.2
  tok/s at 118k against 84.3 coherent, 87.9 at 498k against 85.8): the MoE weights going through the same L2 probably
  evict the K/V lines between rows (not measured).
- **The indexer's top-k on a wide table.** A 1M-token context's page table is 250,000 block columns wide. The top-k
  reduces it in levels, each program keeping the top-k of 8,192 candidates, until the survivors fit one merge, with no
  length limit (upstream's code). One decode layer's top-k on the 7900 XT at 988k tokens (247,000 visible columns): 53 /
  64 / 96 us for 1 / 4 / 8 rows (`rdna3/bench/qsa_topk_bench.py`).
- **What the host K/V and the wide table cost** (decode tok/s at 118k, EXL3 3.05): K/V in VRAM, 262k table (`xtx-xt`
  at 250k of context, 9,932 expert slots) 90.6; K/V on the host, 262k table (12,667 slots) 86.6; K/V on the host, 1M
  table (11,261 slots) 84.2. Both the host K/V and the wide table cost a few percent there.
- **Two cold requests at once** (498k + 237k): both 12/12 with the same answers as alone; the 7900 XTX peaked at 23.09
  of 23.98 GiB (sysfs, desktop included), the 7900 XT at 17.44 of 19.98. The 498k request decoded at 6.3 tok/s while
  the other one was read: prefill goes first (upstream policy, every profile; `FREETOKEN_DECODE_INTERLEAVE` in
  [options.md](options.md)).

How these were found, what was tried and dropped, and a decode step of `xtx-xt-4x262k` traced kernel by kernel:
[journey.md](journey.md#22-four-conversations-of-262k-or-one-of-1m-the-kv-in-host-ram-2026-10-10) and
[journey.md](journey.md#25-a-decode-step-kernel-by-kernel-2026-10-11-and-12).

## Four conversations of 262,144 tokens at once

The K/V pool is shared by the running requests: `rdna3/serve.sh` passes a profile's `CTX` as the cap of one request
(`--max-seq-len-override`) and its `KV_TOKENS` (default `CTX`) as the pool (`--kv-reserve-tokens`). A request is admitted
when the pool can take its prompt plus its `max_tokens`, which the server lowers to what is left under the cap, so a
request never reserves more than 262,144 tokens and four of them fit in 4 x 262,144. On `xtx-xt` the pool is the
262,144 tokens of one context, and the conversations that leave it go to the RAM tier (`FREETOKEN_HOST_KV`).

`xtx-xt-4x262k` (cap 262,144, native rope, pool 1,050,240 tokens, 11,345 expert slots per card), four conversations
on distinct Python stdlib contexts (no shared prefix, planted values as in the retrieval set above):

| Step | Result |
|---|---|
| 4 cold prompts of 255.7-257.9k tokens sent together | read one after another (prefill goes first): 1,027,138 tokens in 463 s (2218 tok/s), every answer right (13/13, 13/13, 12/12, 8/8) |
| 4 next turns together (257-260k, ~800 new tokens each) | first token after 3.0-4.2 s, decode 29.4-30.1 tok/s each (no drafts past two requests) |
| 2 next turns together (259-261k) | first token 1.8-2.9 s, decode 57.2 and 60.6 tok/s |
| 1 next turn alone (261k) | first token 1.8 s, decode 91.3 tok/s (94.1 on `xtx-xt` at 237k, K/V in VRAM) |
| 4 turns together to the cap | first token 3.0-4.6 s; each conversation ends at exactly 262,144 tokens (1,048,576 in the pool at once), no error |
| 1 more turn | refused at once: `prompt is too long: 262188 tokens > 262144 maximum` (`context_length_exceeded`) |
| Peaks | 7900 XTX 23.24 of 23.98 GiB (sysfs, desktop included), 7900 XT 17.10 of 19.98; 77.1 GiB of host RAM locked, 31.9 GiB left available |

The same four conversations on a later build (one pass): the same scores, the cold reads in 445 s, then 29.6-31.6
tok/s each decoding together and 95.1 alone at 259k.
