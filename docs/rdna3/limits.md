# Limits and FAQ

## How much RAM do I need?

| Profile | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw | Machine (NVFP4) |
|---|---:|---:|---:|---|
| `xtx-xt` | 82 GiB | 59 GiB | 74 GiB | 128 GB |
| `xtx-xtx`, `xt-xt` *(untested)* | ~82 GiB (estimated) | ~59 GiB (estimated) | ~74 GiB (estimated) | 128 GB |
| `xtx`, `xt` | 72 GiB | 51 GiB | 65 GiB | 96 GB is tight, 128 GB comfortable |
| `gre` *(untested)* | ~70 GiB (estimated) | | | 96 GB |

Where it goes, for Qwen3.8-Flash-Next: ~63 GiB of experts (the engine keeps **every** expert in pinned RAM, each rank
its own share of each one; the GPUs cache the most used ones), the RAM tier for conversations (9.5 GiB on two cards,
~5 GiB on one; `FREETOKEN_HOST_KV=0` removes it), and a few GiB for the processes. The n-gram tables (51 GB; 33 GB
in EXL3 3.05 bpw) stay on disk and only use the free page cache.

## Can it run with 64 GB of RAM?

Not with the NVFP4 checkpoint: its experts alone need ~63 GiB of RAM, and the engine has no mode that reads experts
from disk. An **EXL3 checkpoint at 3 bits per weight** (experimental, [how-it-works.md](how-it-works.md#exl3-checkpoints))
needs 42.6 GiB for its experts (63.3 in NVFP4; 43.5 and 68.0 with the MTP head's experts): on `xtx-xt`, with the RAM
tier off and the experts loaded one file at a time, the server ran under a 56 GiB container limit at 52.3 GiB used
(`FREETOKEN_HOST_KV=0` in the profile file, then `rdna3/serve.sh xtx-xt --model DIR --memory 56g -- --expert-load
serial`; the parallel loader was killed under a 52 GiB limit). A real 64 GB machine is untested. On the agentic benchmark the 3.05 bpw checkpoint fixed 13 of 29 bugs (one run; NVFP4: 12 and 18 in two runs).

## How much VRAM?

20 GB per card runs the full profiles (the XT is the reference small card, 0.8 GiB left at 124k tokens). 16 GB (RX
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
most tuning options are off unless set. Some changes are active everywhere: tensor parallelism for Qwen3.8-Flash-Next
itself (with its one all-reduce per MoE block and blocked PLE and prefill MoE), the TP>1 relay handshake, a GDN
boundary fix in the prefix cache, CUDA graphs captured for every batch size up to `--cuda-graph-max-bs` (when it is 8
or less), rank 0's token broadcast on the torch.distributed path (with `--disable-pynccl`), and the Qwen VL vision
tower never split (Qwen3.8 builds it on rank 0 only), with rank 0's image embeddings broadcast at TP>1. On NVIDIA,
upstream FreeToken is the better-tested choice.

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

On in the profiles (`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4`); `rdna3/serve.sh --no-mtp` serves without it
([options.md](options.md#speculative-decoding-with-the-mtp-head-qwen38-flash-next-experimental)). Measured on `xtx-xt`,
`xtx` and `xt` with Qwen3.8-Flash-Next ([benchmarks.md](benchmarks.md#speculative-decoding-mtp-head)):

- **One request at a time gains (two in `xtx-xt`), more do not.** On `xtx-xt`, a request alone decodes 34 % faster at the model's
  sampling and 30-61 % faster greedy along a sweep to 248k of context (earlier runs: +11-39 % on chat, agent turns, a
  108k-token context and a 1500-token answer; x1.38 on 60 code and math items). With more decoding at once than
  `FREETOKEN_SPEC_BS_MAX` (2 in `xtx-xt`, 1 in the other profiles), steps run one row per request: from the send to the
  last token, 3-4 requests take 3-4 % longer; two agents at ~100k of context decode 8-15 % faster each with `2` than
  with `1`.
- **One card gains less**: +7-8 % at the model's sampling and +6-12 % greedy on `xtx` or `xt`, -8 % time on code and
  math; one of six sampled test prompts (a long story) is 3-4 % slower. Its step costs keep the verify steps mostly at 2
  rows: the extra rows' missing experts cross PCIe there.
- **Greedy answers are unchanged**: identical to plain decode on every profile (chat prompts, 60 code and math items,
  108k-token turns, the functional checks), with the default fp32 GDN state (`FREETOKEN_MAMBA_SSM_DTYPE`). Sampled requests use exact speculative sampling: the same
  distribution, not the same draws.
- **Reading a prompt is 0-4 % slower** (the head reads it too; 4-5 % on a 108k-token prompt), and the head takes room
  in the expert cache (7,127 experts per card on `xtx-xt` instead of 7,395).
- **Agentic benchmark**: 12 and 18 of 29 in two runs on `xtx-xt` (ROCm 10), in line with the builds before it (12-14).
- **Not covered**: other cards (`FREETOKEN_SPEC_COSTS` holds the step costs measured on `xtx-xt` and on `xt`), images.
- **Tool-call snapshots**: the GDN state saved at a tool-call opener for the next turn's prefix reuse is skipped when a
  verify step jumps over that token; the next turn then reuses less of its prefix. Likewise the state at the end of an
  answer is not kept when the step that wrote its last token kept rows after it (then the next turn reads that answer
  again); over 6 turns ending on their own, the reuse was the same as without the head.

## Anything else to know?

- **No authentication**: the API accepts any request. Keep `--host 127.0.0.1`, or put an authenticating proxy in front
  before serving a network.
- **Runtime cache rebuild** (`/v1/cache/rebuild`) is refused under an uneven split (the ranks could disagree) and
  with the MTP head (its rollback holds the state pools): change the sizes by restarting the server.
- **Host settings matter**: unless `vm.compact_unevictable_allowed=0` keeps memory compaction off the engine's locked
  memory, the GPU queues stop for 5-22 s each time it moves registered host memory; with the default power profile,
  the first request after a pause waits ~10 s for the VRAM clock ([troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run)).
- **Linux only** (the ROCm container needs `/dev/kfd`); no Windows or WSL support.
