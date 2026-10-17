# Limits and FAQ

## How much RAM do I need?

The server locks the RAM it uses: nothing of it can be swapped or given back while it runs. The system, the desktop and
your other programs live in what is left. The rule `rdna3/serve.sh --check` applies, from the reference machine (125.7
GiB as Linux reports it, a desktop on):

**what the server locks + ~3 GiB it grows by in use + ~30 GiB for everything else <= your RAM (`free -g`)**

- With ~31 GiB left beside it after the start (NVFP4, three conversations of 262k in RAM: 94.4 GiB locked), it served
  751 requests with 16 GB still free; hours later it held ~97 GiB.
- With ~29 GiB left and 13-15 GiB of other programs beside the desktop, the host ran out of memory
  ([journey.md](journey.md#29-a-rank-killed-by-the-hosts-oom-killer-2026-10-17)); with ~25 GiB left (four
  conversations, 100.3 GiB locked) and benchmark tools running beside it, too.

So between 25 and 30 GiB left, a setup only fits with little else running; below 25 it does not. A machine without a
desktop needs less (untested). `rdna3/serve.sh --check` does this arithmetic for your machine (with `--model DIR`,
for your checkpoint) and gives the command for what fits; `serve.sh` warns at start when a server would not fit.

What the server locks after its start, measured:

| | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| two cards, the K/V in VRAM (`xtx-xt`) | 82 GiB | 60 GiB | 75 GiB |
| one card (`xtx`, `xt`) | 73 GiB | 51 GiB | 66 GiB |

Where it goes: the experts first (the engine keeps **every** expert in pinned RAM, each rank its own share of each one,
and the GPUs cache the most used ones: ~63 GiB in NVFP4, ~43 GiB in EXL3 3.05 bpw), then the RAM tier that keeps
conversations pushed off the GPUs (9.5 GiB on two cards, ~5 GiB on one; `FREETOKEN_HOST_KV=0` removes it), and a few
GiB for the processes. The n-gram tables (51 GB; 33 GB in EXL3 3.05 bpw) stay on disk and only use the free page cache.
The untested two-card profiles (`xtx-xtx`, `xt-xt`) should lock about what `xtx-xt` does, `gre` ~70 GiB (estimates).

**Conversations kept in RAM** (`xtx-xt-4x262k`, `xtx-xt-1m`): the attention K/V moves to RAM in place of the RAM tier,
14,144 bytes per token on each card, so **6.9 GiB per 262k conversation** on two cards and 26 GiB for 1M tokens
(estimates within 2 GiB of the measured layouts). On the 125.7 GiB reference machine:

- EXL3 3.05 bpw: 60 - 9.5 + 4 x 6.9 = ~78 GiB for four conversations of 262k (77 measured): ~45 GiB left, fits.
- NVFP4: two conversations, ~86 GiB: fits. Three, ~93 GiB (94.4 measured, ~97 in use): fits with little else running;
  `rdna3/serve.sh xtx-xt-4x262k --model DIR -- --kv-reserve-tokens 787500` sizes the pool for three (262,500 tokens
  each). Four, ~100 GiB: does not.

## Can it run with 64 GB of RAM?

Not with the NVFP4 checkpoint: its experts alone need ~63 GiB of RAM, and the engine has no mode that reads experts
from disk. An **EXL3 checkpoint at 3 bits per weight** ([how-it-works.md](how-it-works.md#exl3-checkpoints))
needs 42.6 GiB for its experts (63.3 in NVFP4; 43.5 and ~64.6 with the MTP head's experts, which an NVFP4 checkpoint stores in bf16, 4.7 GiB,
and the engine quantizes to NVFP4 at load): on `xtx-xt`, with the RAM
tier off and the experts loaded one file at a time, the server ran under a 56 GiB container limit at 52.3 GiB used
(`FREETOKEN_HOST_KV=0` in the profile file, then `rdna3/serve.sh xtx-xt --model DIR --memory 56g -- --expert-load
serial`; the parallel loader was killed under a 52 GiB limit). A real 64 GB machine is untested. On the agentic benchmark the 3.05 bpw checkpoint fixed 13 of 29 bugs (one run; NVFP4: 12 and 18 in two runs).

## How much VRAM?

20 GB per card runs the full profiles (the XT is the reference small card: its VRAM peaked at 19,367 of 20,464 MiB over a sweep to 124k tokens). 16 GB (RX
7900 GRE) is expected to work with a shorter context and a small expert cache (`gre`, untested). More VRAM means more
cached experts, which is where decode speed comes from.

## Which GPUs?

| GPU | Status |
|---|---|
| RX 7900 XTX, RX 7900 XT (gfx1100) | tested, alone and together |
| RX 7900 GRE (gfx1100, 16 GB) | untested; uses this image; profile `gre` |
| RX 7800 XT (gfx1101, 16 GB) | untested; needs `--build-arg GPU_ARCH=gfx1101`; start from `gre` (same VRAM) |
| RX 7700 XT (gfx1101, 12 GB), RX 7600 (gfx1102, 8 GB) | untested; less VRAM than any profile plans for |
| Radeon PRO W7900 / W7800 (gfx1100) | untested; should behave like a larger XTX |
| RDNA4 (RX 9070, gfx1201) | untested; needs `--build-arg GPU_ARCH=gfx1201`. The bf16 GEMV tiles are gfx11-only (generic path there); the int8 GEMV, tuned NVFP4 tiles, samplers and host all-reduce are ROCm-wide but were tuned on RDNA3. The community port this work started from targets gfx1201 |
| NVIDIA | untested: see below |

## Does it work on NVIDIA GPUs?

Probably, but nothing was run on NVIDIA. The upstream engine targets NVIDIA first, and this build keeps its CUDA paths:
the ROCm pieces (sampling routes, host-memory all-reduce, RDNA3 GEMVs, ROCm copy engine) are only active on ROCm, and
the other tuning options are off unless set or bit-exact. Some changes are active everywhere: tensor parallelism for
Qwen3.8-Flash-Next itself (with its one all-reduce per MoE block and blocked PLE and prefill MoE), the TP>1 relay
handshake, a GDN boundary fix in the prefix cache, CUDA graphs captured for every batch size up to `--cuda-graph-max-bs`
(when it is 8 or less), rank 0's token broadcast on the torch.distributed path (with `--disable-pynccl`), the
tokenizer's prompt prefix cache (the same tokens), short prefills through the expert cache with the Triton NVFP4 / EXL3
expert kernels (the same bits), and the Qwen VL vision tower never split (Qwen3.8 builds it on rank 0 only), with rank
0's image embeddings broadcast at TP>1. On NVIDIA, upstream FreeToken is the better-tested choice.

## More than two GPUs?

Not for Qwen3.8-Flash-Next in this build. With 2 KV heads, TP=4 stops at an assertion on the attention projection
width (the heads would have to be replicated, which the model code does not do yet), and TP=3 divides neither the KV
nor the GDN heads. The uneven split itself accepts any number of ranks, and the host-memory all-reduce is two-ranks
only (more ranks would use RCCL). Nothing beyond two GPUs has been run.

## Other models?

The profiles are tuned for Qwen3.8-Flash-Next NVFP4 (the tuned kernel tiles, the chunk sizes and the memory ratios are
this model's). Other models FreeToken supports ([models.md](../models.md)) should load with upstream's behavior plus
the ROCm fixes (untested here); the tuning options may or may not apply to them.

## Images (vision)?

Off by default: `rdna3/serve.sh` passes `--text-model-only` unless it is started with `--vision`. Measured on
`xtx-xt`, `xtx` and `xt`, up to each profile's full context ([benchmarks.md](benchmarks.md#images---vision-xtx-xt)).
With `--vision`:

- **Rank 0 holds the tower and encodes.** The vision tower (27 ViT blocks, ~0.45 B parameters, 856 MiB in bf16) is
  built whole on rank 0 only, never split, so encoding adds no all-reduce. Rank 0 broadcasts the embeddings, so every
  rank feeds the text model the same numbers. On the reference pair both cards encode bit-identically anyway, but
  other cards may round the tower's GEMMs differently, and a per-rank difference would stay in the residual stream.
- **The tower stays bf16.** `FREETOKEN_INT8_DENSE` skips it.
- **Weights streamed from RAM.** Its blocks stream from pinned RAM two at a time (`--mm-encoder-weights host`, the
  default). It keeps 294 MiB of VRAM instead of 1034 MiB resident, at the same encode speed.
- **Images are capped at 1024 tokens.** Images are scaled down to 1024 tokens (one per 32x32 pixels, about one
  megapixel). `-- --image-max-tokens N` changes it; the checkpoint allows 16384. The tower attends over the whole
  image, so its cost grows faster than the image: 0.1 s at 1024 tokens, 0.9 s at 4096 and 12 s at 16384 on the XTX.
  Every image token is also a prompt token.
- **Text is unchanged.** The text model switches to the 3-axis rope (mrope). The greedy answers were identical to the
  text-only build.
- **Speed on two cards.** Decode, prompt reading and the agent turn stayed within the spread between sessions at
  every depth up to 255k.
- **Speed on one card.** The tower shares the only card, so the expert cache holds ~165 fewer experts and decode is
  ~1.5-2 % slower. Use `--vision` there only if you send images.

## Speculative decoding (MTP)?

On in the profiles except `gre` (`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4`); `rdna3/serve.sh --no-mtp` serves without it
([options.md](options.md#speculative-decoding-with-the-mtp-head-qwen38-flash-next)). Measured on `xtx-xt`,
`xtx` and `xt` with Qwen3.8-Flash-Next ([benchmarks.md](benchmarks.md#speculative-decoding-mtp-head)):

- **One request at a time gains (two in `xtx-xt` and its K/V-in-RAM variants), more do not.** On `xtx-xt`, a request
  alone decodes 35 % faster at the sampling `rdna3/serve.sh` gives and 35-60 % faster greedy along a sweep to 260k of
  context. With more decoding at once than
  `FREETOKEN_SPEC_BS_MAX` (2 in `xtx-xt` and its K/V-in-RAM variants, 1 in the other profiles), steps run one row per request: from the send to the
  last token, 3-4 requests take 3-4 % longer; two agents at ~100k of context decode 8-15 % faster each with `2` than
  with `1`.
- **One card gains less**: +6-13 % at `rdna3/serve.sh`'s sampling and +9-16 % greedy on `xtx` or `xt`. Its step costs
  keep the verify steps mostly at 2 rows: the extra rows' missing experts cross PCIe there.
- **Greedy answers are unchanged**: identical to plain decode on every profile (chat prompts, 60 code and math items,
  108k-token turns, the functional checks), with the default fp32 GDN state (`FREETOKEN_MAMBA_SSM_DTYPE`). Sampled requests use exact speculative sampling: the same
  distribution, not the same draws.
- **Reading a prompt is up to 5 % slower** (the head reads it too), and the head takes room in the expert cache (6,990
  experts per card on `xtx-xt` instead of 7,268).
- **Agentic benchmark** (an earlier build): 12 and 18 of 29 in two runs on `xtx-xt`, in line with the builds before
  it (without the head: 12-14).
- **Not covered**: other cards (`FREETOKEN_SPEC_COSTS` holds the step costs measured on `xtx-xt` and on `xt`), images.
- **Tool-call snapshots**: the GDN state saved at a tool-call opener for the next turn's prefix reuse is skipped when a
  verify step jumps over that token; the next turn then reuses less of its prefix. Likewise the state at the end of an
  answer is not kept when the step that wrote its last token kept rows after it (then the next turn reads that answer
  again); over 6 turns ending on their own, the reuse was the same as without the head.

## Anything else to know?

- **No authentication by default**: the API accepts any request. Keep `--host 127.0.0.1`, or require a key with
  upstream's `--api-key` (`rdna3/serve.sh <profile> ... -- --api-key KEY`; clients then send `Authorization: Bearer
  KEY`, every route but `/health` answers 401 without it; the key shows in `docker inspect` and `ps`), or put an
  authenticating proxy in front before serving a network. `rdna3/bench/` scripts send no key.
- **Runtime cache rebuild** (`/v1/cache/rebuild`) is refused under an uneven split (the ranks could disagree) and
  with the MTP head (its rollback holds the state pools): change the sizes by restarting the server.
- **Host settings matter**: unless `vm.compact_unevictable_allowed=0` keeps memory compaction off the engine's locked
  memory, the GPU queues stop for 5-22 s each time it moves registered host memory; with the default power profile,
  the first request after a pause waits ~10 s for the VRAM clock ([troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run)).
- **Linux only** (the ROCm container needs `/dev/kfd`); no Windows or WSL support.
