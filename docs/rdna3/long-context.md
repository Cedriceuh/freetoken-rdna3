# Long contexts with the K/V in host RAM (experimental)

Qwen3.8-Flash-Next is trained to 262,144 positions, and Qwen's model card extends it to 1,000,000 with YaRN
(`rope_type: yarn`, `factor: 4`, `original_max_position_embeddings: 262144`). With the attention K/V in host RAM, two
cards hold either one conversation of up to 1M tokens (`xtx-xt-1m`) or four conversations of 262,144 tokens at once
(`xtx-xt-4x262k`), where `xtx-xt` shares 250,000 tokens between its requests. This page reports what was measured on
2026-10-10 on the reference machine (RX 7900 XTX + RX 7900 XT, MTP head on) with the EXL3 3.05 bpw checkpoint only.
Neither profile is agent-validated yet.

## What it takes

- **The attention K/V does not fit in VRAM at 1M.** It costs 14,144 bytes per token on each card (13 attention layers
  with the MTP head's, one KV head of 256 per rank, plus the compressed index keys), so 1M tokens is 13.2 GiB per card.
  On `xtx-xt` the engine refuses that plan: the 7900 XT's budget for experts and KV is 10.9 GB at
  `--memory-ratio 0.77`, and 1M of KV plus the minimum 1024 expert slots needs 14.9 GB.
- **`FREETOKEN_QSA_KV_HOST=1`** keeps the K/V slab in pinned, device-mapped host memory, handed to torch as a GPU tensor
  (`kernel/host_mapped.py`): the decode and verify kernels read the ~2051 selected tokens of each row in place across
  PCIe, and a prefill first copies the pages of the batch's history, one layer at a time, to a VRAM buffer
  (`kernel/triton/qsa/stage.py`, 28 GB/s), so its rows do not each cross PCIe. The index tiers stay in VRAM. At 1M the
  engine then plans 11,261 expert slots per card (9,932 for the `xtx-xt` profile at 250k with the K/V in VRAM) and
  locks ~75 GiB of host RAM; the 7900 XTX peaked at 23.20 of 23.98 GiB (sysfs) during a 1M read with the desktop on it. The
  answers are bit-identical to the K/V in VRAM (the same text at 29k and 118k).
- **Rope.** Two ways past 262,144 positions:
  - YaRN from `rope_parameters` (the model card's edit): the checkpoints carry a vision tower, so the rope is the
    multi-axis one; YaRN now composes with it, and its table covers `original x factor` positions as SGLang / vLLM
    size it. Bit-identical to HF transformers 5.16.1's rotary embedding (`tests/layers/test_rotary_yarn.py`).
  - **`FREETOKEN_ROPE_MAX_POSITION=1048576`**: the plain rope with a longer table, no YaRN. Below 262,144 the rows are
    the same bits as without it, so short contexts are unchanged.
- `rdna3/profiles/xtx-xt-1m.env` puts it together for one long conversation (plain rope, K/V on the host,
  `FREETOKEN_HOST_KV=0`); `rdna3/profiles/xtx-xt-4x262k.env` keeps each request to the native 262,144 positions and
  sizes the K/V pool for four of them (`KV_TOKENS=1050000`, see [Four conversations at once](#four-conversations-of-262144-tokens-at-once)).

## Speed (EXL3 3.05 bpw, `xtx-xt`, one request)

Cold reads of contexts that share no prefix (stdlib code, below), then a 700-token greedy answer:

| Context | 250k profile, K/V in VRAM: PP / TG | 1M, K/V on the host: PP / TG | 1M, both decode fixes below |
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
  (4 rows, the same winners, ties included). The split now takes more chunks past 16, and only a row with more live
  chunks than one resident tile holds merges in a spilled tile; tables up to 131,072 columns (contexts up to 524,288
  tokens, so the 250k profiles) keep their plan, and a plan merges at most 16,384 candidates (32 chunks: contexts up to
  1,048,576 tokens; a longer table takes the one-program path again). Below ~100k visible tokens the wide table's split costs ~40 us per
  layer more than the one-program scan did. With both changes (a cold read, then 700 tokens): 498k read at 1973 tok/s,
  decode 91.9 tok/s (85.8 before); 988k read at 1775 tok/s in 9.3 min (1711), decode 83.6 tok/s (72.1); the same text
  as before both changes.
- **A/B at 118k** (decode tok/s, EXL3 3.05): K/V in VRAM, 262k table (the `xtx-xt` profile, 9,932 expert slots) 90.6;
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
250,000 tokens of one context, and the conversations that leave it go to the RAM tier (`FREETOKEN_HOST_KV`).

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
