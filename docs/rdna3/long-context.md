# Long contexts: the K/V in host RAM

Two independent settings decide how much context a server holds:

1. **Where the attention K/V lives** sets how many tokens all the requests hold together. In VRAM (`xtx-xt`) the pool
   is 262,144 tokens, shared by every request running. In pinned host RAM (`FREETOKEN_QSA_KV_HOST=1`) it is as large
   as your RAM allows: about a million tokens on the 128 GB reference machine with EXL3 3.05 bpw, three 262k contexts
   with NVFP4.
2. **How far the positions go** sets how long one request can be. The model is trained to 262,144 positions, and a
   longer prompt is refused. `FREETOKEN_ROPE_MAX_POSITION=1048576` extends its rope table to 1,048,576 positions; below
   262,144 the table is the same, so shorter contexts are unchanged.

The two profiles with the K/V in RAM combine them differently:

| Profile | K/V | Pool, all requests together | One request, up to | Positions |
|---|---|---:|---:|---|
| `xtx-xt` | VRAM | 262,144 | 262,144 | native |
| `xtx-xt-4x262k` | host RAM | 1,050,000 (4 x 262,144 and a spare page each) | 262,144 | native |
| `xtx-xt-1m` | host RAM | 1,000,000 | 1,000,000 | extended to 1,048,576 |

- **Several agents that each need the full context**: `xtx-xt-4x262k`. Every request stays within the positions the
  model was trained on.
- **One conversation longer than 262,144 tokens**: `xtx-xt-1m`. Past 262,144 the model runs beyond its trained
  positions: see [Quality](#quality-past-262144-tokens).

Qwen's model card reaches 1M positions with YaRN instead. This build extends the plain rope: it leaves the model as it
is below 262,144 tokens, where YaRN changes its outputs on short texts too, and both kept their retrieval to 988k tokens
([journey.md](journey.md#22-four-conversations-of-262k-or-one-of-1m-the-kv-in-host-ram-2026-10-10)).

Neither profile is agent-validated yet. Both were measured on the reference machine (RX 7900 XTX + RX 7900 XT, MTP head
on), mostly with the EXL3 3.05 bpw checkpoint, on earlier builds than the summary of [benchmarks.md](benchmarks.md).

## Running them

```bash
rdna3/serve.sh xtx-xt-4x262k --model ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw
rdna3/serve.sh xtx-xt-1m     --model ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw
```

Set your client's context window to 262144 (`xtx-xt-4x262k`) or 1000000 (`xtx-xt-1m`). `rdna3/serve.sh --check` says
which pool fits in your RAM.

## What they cost

- **RAM.** The K/V takes 14,144 bytes per token on each card, so 6.9 GiB per 262k conversation on two cards and ~26 GiB
  for 1M tokens, all of it locked. Measured with EXL3 3.05 bpw: ~77 GiB of RAM locked by `xtx-xt-4x262k`, ~75 GiB by
  `xtx-xt-1m`. The NVFP4 checkpoint's experts alone take 63 GiB: on the 126 GiB reference machine three conversations
  fit (`rdna3/serve.sh xtx-xt-4x262k --model DIR -- --kv-reserve-tokens 787500`, 94 GiB locked after the start, ~97 in
  use, with little else running), or two to keep room (`525000`); four, with benchmark tools beside them, ran it out
  of memory. `xtx-xt-1m` is untested with NVFP4. How to count for your machine:
  [limits.md](limits.md#how-much-ram-do-i-need).
- **The RAM tier is off** (`FREETOKEN_HOST_KV=0`): the pool already lives in RAM.
- **Speed.** A request alone decodes a few percent slower than with the K/V in VRAM: at 118k, EXL3 3.05 bpw, 90.6
  tok/s in VRAM, 86.6 on the host with a 262k page table, 84.2 with the 1M one (the kernels read the K/V across PCIe,
  and the wider table costs a few percent more). It reads prompts a little faster: the VRAM the K/V no longer takes goes to
  the expert cache (~11.3k experts per card instead of 9.8k on `xtx-xt`, earlier builds).
- **Prefill goes first.** While a long prompt is read, the other requests barely decode: a 498k-token request decoded
  at 6.3 tok/s while a 237k-token one was read. That is upstream's scheduling, the same on every profile.
  `FREETOKEN_DECODE_INTERLEAVE=8` with `FREETOKEN_DECODE_INTERLEAVE_LONG=1` gives the decoding requests eight steps
  after each 16k-token chunk of a long read, at ~3 % of the read's speed, without touching agent turns
  ([options.md](options.md#scheduling)).

## `xtx-xt-4x262k`: four conversations of 262,144 tokens

`rdna3/serve.sh` passes a profile's `CTX` as the cap of one request (`--max-seq-len-override`) and its `KV_TOKENS` as the
pool (`--kv-reserve-tokens`; `CTX` when unset). A request is admitted when the pool can take its prompt plus its
`max_tokens`, which the server lowers to what is left under the cap, so no request reserves more than 262,144 tokens
and four of them fit. On `xtx-xt` the pool is one context, and the conversations that leave it go to the RAM tier.

Four conversations on distinct Python stdlib contexts (no shared prefix, planted values to retrieve), EXL3 3.05 bpw,
an earlier build on which three and four requests did not verify drafts yet:

| Step | Result |
|---|---|
| 4 cold prompts of 255.7-257.9k tokens sent together | read one after another: 1,027,138 tokens in 463 s (2218 tok/s), every answer right |
| 4 next turns together (257-260k, ~800 new tokens each) | first token after 3.0-4.2 s, decode 29.4-30.1 tok/s each |
| 2 next turns together (259-261k) | first token 1.8-2.9 s, decode 57.2 and 60.6 tok/s |
| 1 next turn alone (261k) | first token 1.8 s, decode 91.3 tok/s (94.1 on `xtx-xt` at 237k, K/V in VRAM) |
| 4 turns together to the cap | each conversation ends at exactly 262,144 tokens (1,048,576 in the pool at once), no error |
| 1 more turn | refused at once: `prompt is too long: 262188 tokens > 262144 maximum` (`context_length_exceeded`) |
| Peaks | 7900 XTX 23.24 of 23.98 GiB, 7900 XT 17.10 of 19.98 GiB; 77.1 GiB of RAM locked, 31.9 GiB left available |

The same four on a later build (one pass): the same scores, the cold reads in 445 s, 29.6-31.6 tok/s each decoding
together, 95.1 alone at 259k. Three and four requests now verify drafts, 2 rows each: +6-9 % at three, +3-6 % at four
greedy, -0.3 to +2.5 % at T 0.6 (a 300k-token pool, 768-token answers:
[benchmarks.md](benchmarks.md#several-requests-and-agents-at-once-xtx-xt)).

One request, EXL3 3.05 bpw (`rdna3/bench/depth_sweep.py`, greedy, one pass): decode 107-110 tok/s at 25-60k tokens and 101 at 248k;
reading a new block 2340-2430 tok/s at 25-60k and 2020 at 248k. An agent turn of 142-360 tokens takes 0.46-0.59 s to
its first token at 224k. NVFP4 with three contexts: 130-340-token turns at 25k 0.58-0.70 s, a 259,728-token prompt
read in 120 s. `--hybrid` brings no clear gain here: the same sweep -8 % at 8.8k, +3 % at 60k and 123k, within 2 %
elsewhere; four agents at ~257k 28.6-30.3 tok/s each against 28.2-30.2 without it ([cpu-experts.md](cpu-experts.md)).

## `xtx-xt-1m`: one conversation of up to 1M tokens

EXL3 3.05 bpw, cold reads of stdlib code with nothing shared, then a 700-token greedy answer:

| Context | Prompt reading | Decode | `xtx-xt` (K/V in VRAM, 250k context) |
|---|---:|---:|---|
| 118k | 2376 tok/s | 84.2 tok/s | 2185 / 90.6 tok/s |
| 237k | - | 88.5 tok/s | 2128 / 94.1 tok/s |
| 498k | 2014 tok/s | 93.0 tok/s | - |
| 988k | 1780 tok/s (9.3 min) | 84.7 tok/s | - |

The rows come from successive builds: the 498k and 988k ones have both decode fixes of
[journey.md](journey.md#22-four-conversations-of-262k-or-one-of-1m-the-kv-in-host-ram-2026-10-10) (the non-coherent host
memory and the indexer's split top-k), the 118k and 237k decode only the first, the 118k read neither. The 7900 XTX
peaked at 23.20 of 23.98 GiB (sysfs, desktop included) during a 1M read.

An agent conversation on a 498k-token context: first turn 245 s, the next two 4.2 and 2.3 s to the first token (the
prefix is reused). The pool is shared: up to four requests run at once, together within 1M tokens.

### Quality past 262,144 tokens

- **Retrieval** (Python stdlib files with their default values replaced by random ones, so that memory cannot answer,
  plus notes inserted between files; one request asking every question): 100 % from 29k to 988k tokens.
- **A harder set at 988k** (30 code questions, 6 notes, 3 two-hop notes that point to another): 32 of 36. The three
  two-hop questions fail, the answer quoting the pointer note. With the indexer's budget doubled to 4096 (a config
  edit) the same four miss (read at 1551 tok/s, decode 75.5): the sparse attention's selection is not the cause.
- **Next-token loss** on a 988k-token text (this repository's code and docs): 0.46-0.79 nats per token at every depth,
  with no rise past 262k.

## How the K/V lives in RAM

- **The slab** is pinned, device-mapped host memory handed to torch as a GPU tensor (`kernel/host_mapped.py`). The
  decode and verify kernels read the ~2051 tokens each row selects in place, across PCIe. On ROCm it is allocated
  non-coherent, so the GPU's L2 can keep the lines the verify rows of a request share (90 % of their tokens): 2.4x
  faster in the attention kernel, but only 0-2.4 % in serving, where the MoE weights going through the same L2
  probably evict them (not measured). The index tiers stay in VRAM.
- **A prefill** first copies to a VRAM buffer the K/V rows its tokens select, one layer at a time
  (`FREETOKEN_QSA_STAGE_SELECTED`, 28 GB/s), instead of reading each row across PCIe.
- **The answers are bit-identical** to the K/V in VRAM (the same text at 29k and 118k).
- **At 1M tokens** the indexer's page table is 250,000 block columns wide; its top-k reduces it in levels (53 / 64 /
  96 us a layer for 1 / 4 / 8 rows at 988k, on the 7900 XT).

How it was built, what was tried and dropped, and a decode step of `xtx-xt-4x262k` traced kernel by kernel:
[journey.md](journey.md#22-four-conversations-of-262k-or-one-of-1m-the-kv-in-host-ram-2026-10-10) and
[journey.md](journey.md#25-a-decode-step-kernel-by-kernel-2026-10-11-and-12).
