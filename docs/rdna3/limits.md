# Limits and FAQ

## How much RAM do I need?

| Profile | RAM used while serving | Machine |
|---|---:|---|
| `xtx-xt` | 81 GiB (measured) | 128 GB |
| `xtx-xtx`, `xt-xt` *(untested)* | ~81 GiB (estimated) | 128 GB |
| `xtx`, `xt` | ~75 GiB (estimated) | 96 GB is tight, 128 GB comfortable |
| `gre` *(untested)* | ~72 GiB (estimated) | 96 GB |

Where it goes, for Qwen3.8-Flash-Next: ~63 GiB of experts (the engine keeps **every** expert in pinned RAM, each rank
its own share of each one; the GPUs cache the most used ones), the RAM tier for conversations (9.0 GiB on two cards,
~5 GiB on one; `FREETOKEN_HOST_KV=0` removes it), and a few GiB for the processes. The 51 GB of n-gram tables stay on
disk and only use the free page cache.

## Can it run with 64 GB of RAM?

Not with Qwen3.8-Flash-Next. The experts alone need ~63 GiB of RAM: NVFP4 (4.5 bits per weight) is the smallest
format this engine runs for this model (MXFP4 would save ~4 GB but has no checkpoint), and the engine has no mode that
reads experts from disk. Getting there would take engine work (not keeping a RAM copy of the experts the GPUs already
hold, plus a 3-bit expert format, or streaming experts from an SSD), with a precision cost to measure.

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

## Anything else to know?

- **No authentication**: the API accepts any request. Keep `--host 127.0.0.1`, or put an authenticating proxy in front
  before serving a network.
- **Runtime cache rebuild** (`/v1/cache/rebuild`) is refused under an uneven split (the ranks could disagree).
- **An occasional decode stall**, from a few seconds up to ~21 s, seen on the reference machine in about one run in
  three on every build measured since 2026-09-28; not understood yet.
- **Linux only** (the ROCm container needs `/dev/kfd`); no Windows or WSL support.
