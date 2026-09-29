# Profiles

A profile is one file in `rdna3/profiles/` holding every setting of a tested hardware setup. `rdna3/serve.sh <profile>`
turns it into a `docker run` + `ft serve` command (`--dry-run` prints it).

| Profile | GPUs | Context | Decode (short / deep) | 8.4k prompt | Agent turn | At once |
|---|---|---:|---:|---:|---:|---:|
| `xtx-xt` | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | 262,144 | 55 / 48-53 tok/s | 4.5 s | 1.2-1.9 s | 4 requests |
| `xtx` | RX 7900 XTX 24 GB | 131,072 | 36.5 / 32.5-34.3 tok/s | 10.5 s | 2.3-2.7 s | 1 request |
| `xt` | RX 7900 XT 20 GB | 131,072 | 28.5 / 27-28 tok/s | 8.9 s | 2.4-2.7 s | 1 request |
| `xtx-xtx` *(untested)* | 2x RX 7900 XTX 24 GB | 262,144 | expected >= xtx-xt | – | – | 4 requests |
| `xt-xt` *(untested)* | 2x RX 7900 XT 20 GB | 262,144 | expected ~50 tok/s | – | – | 4 requests |
| `gre` *(untested)* | RX 7900 GRE 16 GB | 65,536 | expected ~20 tok/s | – | – | 1 request |

"Deep" is the decode speed measured at every context step up to the profile's maximum (255k for `xtx-xt`, 124k for
one card); "agent turn" is the time to the first token when 1-2k new tokens are added to a cached conversation,
from 10k to the maximum depth. All numbers: [benchmarks.md](benchmarks.md).

## File format

```bash
GPUS=2                  # how many cards (--gpus overrides; the automatic choice is described below)
VRAM_GIB="24 20"        # minimum VRAM per rank, used for the automatic choice
TESTED=0                # derived, not measured: serve.sh says so at start (default 1)
CTX=262144              # context length (--ctx overrides); passed as --max-seq-len-override and --kv-reserve-tokens
MEMORY=110g             # container RAM limit (--memory overrides)
TUNABLEOP=1             # mount rdna3/tunableop (tuned GEMM choices), read-only
FREETOKEN_...=...       # any other KEY=value line becomes an environment variable of the container
FT_ARGS="..."           # flags passed to `ft serve`
```

A comment may follow a value on the same line. Copy a file to make your own profile; it shows up in
`rdna3/serve.sh --list`.

## `xtx-xt`: two cards, tensor parallel

| Setting | Value | Why |
|---|---|---|
| `--tp-size 2` | | every layer is split over both cards, which work at the same time |
| `FREETOKEN_TP_SPLIT` | `0.55` | the XTX (rank 0) takes 55 % of the split work: it has more VRAM and ~14 % more compute. Measured optimum; 57.5 % and 60 % overload it during prefill |
| `FREETOKEN_TP_ALLOW_IMBALANCE` | `1` | plan memory on the smaller card instead of refusing unequal cards |
| `FREETOKEN_INT8_DENSE` | `1` | dense layers in int8 (weights only, routers stay bf16): +28 % decode, frees ~2.3 GiB per card for the expert cache |
| `FREETOKEN_HOST_ALLREDUCE` | `1` | small all-reduces through shared host memory instead of RCCL's proxies (no GPU peer-to-peer on consumer boards): +2.5-4.8 % decode, bit-exact |
| `FREETOKEN_NVFP4_PREFILL_TUNED` / `_DECODE_TUNED` | `1` | expert GEMM tiles swept on these cards for this model: long prompts -11 %, decode +1.3 %, bit-exact |
| `FREETOKEN_HOST_KV` | `1` | conversations evicted from the GPUs go to RAM (~9 GiB locked) and come back in ~2 s |
| `--moe-strategy offload --moe-cache-auto` | | experts live in RAM; every byte of VRAM left after the dense weights and the KV is an expert cache (8.3k of the 24,576 experts on the XTX, 7.2k on the XT) |
| `--expert-load parallel` | | reads the expert files into RAM with parallel readers; `auto` (the engine's default) does the same but falls back to a slower serial read when free RAM is short: use it on a machine with little RAM to spare |
| `--quant-backend moe.nvfp4=triton` | | the Triton NVFP4 expert kernels (the other backends need NVIDIA hardware) |
| `--ple-backend disk` | | the 51 GB n-gram embedding tables are read from the checkpoint files on demand |
| `--max-running-requests 4 --cuda-graph-max-bs 4` | | up to 4 requests decoded together, each computed row by row as if alone (prompts prefilled in the same batch can differ at rounding level): 57 / 82 / 113 tok/s in total at 1 / 2 / 4 |
| `--max-prefill-length 16384` | | 16k-token prompt chunks: each chunk streams every expert once, so bigger chunks read long prompts ~1.5x faster |
| `--memory-ratio 0.80` | | VRAM budget; 0.80 leaves the headroom 16k chunks need on the 20 GB card |
| `--moe-prefill-hit-d2d` | | experts already cached on the GPU are reused during prefill instead of crossing PCIe: agent turn 1.55 -> 1.18 s |
| `--disable-pynccl` | | ROCm uses RCCL through torch.distributed |
| `--tool-call-parser qwen3_coder --reasoning-parser qwen3` | | tool calls and thinking in the model's format |
| context | 262,144 | the model's native maximum |

RAM: ~81 GiB while serving (64.7 GiB of experts, ~9 GiB for the RAM tier, the rest the two processes).

## `xtx` and `xt`: one card

Same settings minus the two-card ones (split, host all-reduce), plus:

| Setting | Value | Why |
|---|---|---|
| `--tp-size 1` | | one card |
| `--max-running-requests 1` | | the GDN state of 4 requests would take ~3 GiB at TP=1 |
| `--max-prefill-length 8192` | | with 16k chunks a single card runs out of working memory: a 60k-token prompt took 166 s and decode fell to 1.9 tok/s |
| `FREETOKEN_HOST_KV_TOKENS` / `_SNAPSHOTS` | `131072` / `16` | the RAM tier sized to the context (~5 GiB locked) |
| context | 131,072 | what fits: the XT ends a 124k-token conversation with 0.8 GiB of VRAM left |

The cards are picked by VRAM: `xt` takes the smallest card with at least 20 GiB, `xtx` the smallest with 24 GiB, so a
machine with both keeps the other card free.

## The untested profiles

Derived from the measured profiles and from the memory plans the engine logged on the tested cards; `serve.sh` prints
a note when you start one (`TESTED=0` in the file).

- **`xtx-xtx`**: `xtx-xt` with an even split. `xtx-xt` plans every pool on its smaller card, so two XTX cards should
  get ~3.7 GiB more expert cache each (~8.5k instead of 7.2k cached experts): decode at least as fast as `xtx-xt`. The
  tuned NVFP4 tiles are keyed to the 0.55 split, so the even split's shapes use default tiles.
- **`xt-xt`**: `xtx-xt` with an even split. The expert cache should be about the same as `xtx-xt` (already sized by
  its XT); each XT does 50 % of the split work instead of 45 %: expect decode close to `xtx-xt`, ~50 tok/s.
- **`gre`**: `xt` on 16 GB. From the XT's plan (int8 dense weights 5.7 GiB, KV 3.1 GiB at 131k tokens, 20 % headroom)
  a 16 GB card keeps only ~3-4 GiB for the expert cache (4-7 % of the experts cached, most expert reads cross PCIe),
  with 576 GB/s of VRAM bandwidth instead of 800: ~20 tok/s at best. It starts at 64k tokens of context with 4k
  prefill chunks; raise `CTX` only after checking the free memory printed at start-up.

## Adapting a profile to other cards (untested)

Nothing below has been measured; treat it as a starting point and check with the benchmarks in
[testing.md](testing.md).

- **Two identical cards** (2x XTX, 2x XT): start from `xtx-xt` and delete `FREETOKEN_TP_SPLIT` (even split). Keep
  `FREETOKEN_TP_ALLOW_IMBALANCE=1`: two "identical" cards rarely report the same free memory.
- **Another unequal pair**: set `FREETOKEN_TP_SPLIT` to about rank 0's share of the total VRAM, then try +/-2.5 %;
  `FREETOKEN_TP_SPLIT=11:9` style weights also work. The tuned NVFP4 tiles are keyed to this model's shapes at a
  0.55 split: other splits fall back to the default tiles (a log line names each untuned shape once).
- **Smaller RDNA3 cards**: the RX 7900 GRE (16 GB) is also gfx1100 and runs this image; expect a shorter context and
  fewer cached experts: lower `CTX` first, then `--max-prefill-length`. The RX 7800 XT / 7700 XT (gfx1101) and 7600
  (gfx1102) need their own image (`--build-arg GPU_ARCH=gfx1101`) and are not tested at all.
- **Less RAM**: `FREETOKEN_HOST_KV=0` saves the ~9 GiB of the RAM tier (evicted conversations are re-read instead).
  The experts themselves cannot go below ~65 GiB for this model: [limits.md](limits.md).
- **The TunableOp files** in `rdna3/tunableop/` are tied to this image's PyTorch / HIP / hipBLASLt versions (their
  header records them). With a different base image set `TUNABLEOP=0`, or tune once and freeze (see
  [options.md](options.md#pytorch-tunableop)).

Every option in these files is explained in [options.md](options.md).
