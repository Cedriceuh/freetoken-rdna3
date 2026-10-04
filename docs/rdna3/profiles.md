# Profiles

A profile is one file in `rdna3/profiles/` holding every setting of one hardware setup (measured, or derived and
marked `TESTED=0`). `rdna3/serve.sh <profile>` turns it into a `docker run` + `ft serve` command (`--dry-run` prints
it).

| Profile | GPUs | Context | Decode, MTP head on (off) | Deep decode, greedy | Cold 8.3k prompt | Agent turn | At once |
|---|---|---:|---:|---:|---:|---:|---:|
| `xtx-xt` | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | 250,000 | 77 (57.5) tok/s | 72-86 (52-57) tok/s | 4.1 s | 1.3-2.0 s | 4 requests |
| `xtx` | RX 7900 XTX 24 GB | 131,072 | 39 (36) tok/s | 33-42 (29.5-39) tok/s | 7.7 s | 2.4-2.7 s | 1 request |
| `xt` | RX 7900 XT 20 GB | 131,072 | 32 (29.5) tok/s | 27-34 (25-32) tok/s | 8.3 s | 2.5-2.8 s | 1 request |
| `xtx-xtx` *(untested)* | 2x RX 7900 XTX 24 GB | 250,000 | expected >= `xtx-xt` | – | – | – | 4 requests |
| `xt-xt` *(untested)* | 2x RX 7900 XT 20 GB | 250,000 | expected a little below `xtx-xt` | – | – | – | 4 requests |
| `gre` *(untested)* | RX 7900 GRE 16 GB | 65,536 | expected ~20 tok/s at best (head off) | – | – | – | 1 request |

"Decode" is one request's 512-token answers at the model's default sampling (median of 12); "deep decode" is greedy
decode at every context step from 9k to the profile's maximum (248k on two cards, 124k on one); "agent turn" is the
time to the first token when ~1k new tokens are added to a cached conversation, over the same steps. The profiles turn
the MTP head on; `rdna3/serve.sh <profile> --no-mtp` turns it off (numbers in brackets). All numbers:
[benchmarks.md](benchmarks.md).

## File format

```bash
GPUS=2                  # how many cards the automatic choice picks (--gpus picks them yourself)
VRAM_GIB="24 20"        # minimum VRAM per rank, used for the automatic choice
TESTED=0                # derived, not measured: serve.sh says so at start (default 1)
CTX=250000              # context length (--ctx overrides); passed as --max-seq-len-override and --kv-reserve-tokens
MEMORY=110g             # container RAM limit (--memory overrides)
TUNABLEOP=0             # 1: mount rdna3/tunableop (GEMM choices tuned for the ROCm 7.14 image), read-only
FREETOKEN_...=...       # any other KEY=value line becomes an environment variable of the container
FT_ARGS="..."           # flags passed to `ft serve` (its --tp-size must match GPUS)
```

A comment may follow a value on the same line. Copy a file to make your own profile; it shows up in
`rdna3/serve.sh --list`.

## `xtx-xt`: two cards, tensor parallel

| Setting | Value | Why |
|---|---|---|
| `--tp-size 2` | | every layer is split over both cards, which work at the same time |
| `FREETOKEN_TP_SPLIT` | `0.55` | the XTX (rank 0) takes 55 % of the split work: it has more VRAM and ~14 % more compute. Measured optimum; 57.5 % and 60 % overload it during prefill |
| `FREETOKEN_TP_ALLOW_IMBALANCE` | `1` | plan memory on the smaller card (upstream refuses ranks whose free memory differs by more than 2 GiB) |
| `FREETOKEN_INT8_DENSE` | `1` | dense layers in int8 (weights only, routers stay bf16): +28 % decode, frees ~2.4 GiB per card for the expert cache; lossy, agent-validated |
| `FREETOKEN_HOST_ALLREDUCE` | `1` | small all-reduces through shared host memory instead of RCCL's proxies (no GPU peer-to-peer here): +2.5-4.8 % decode, bit-exact |
| `FREETOKEN_NVFP4_PREFILL_TUNED` / `_DECODE_TUNED` | `1` | expert GEMM tiles swept on these cards for this model (the decode ones for this split's shapes): long prompts -11 %, decode GEMMs ~13 % faster in the kernel sweep, bit-exact |
| `FREETOKEN_HOST_KV` | `1` | conversations evicted from the GPUs go to RAM (9.5 GiB locked) and come back in ~2 s; exact |
| `FREETOKEN_MTP` / `FREETOKEN_SPEC_VERIFY_M` | `1` / `4` | the checkpoint's MTP head drafts up to 3 tokens a step for a request decoding alone (two at once in `xtx-xt`, `FREETOKEN_SPEC_BS_MAX=2`: +8-15 % each): +34 % decode at the model's sampling (+7-8 % on one card), greedy answers unchanged; `rdna3/serve.sh --no-mtp` turns it off ([limits.md](limits.md#speculative-decoding-mtp)) |
| `--moe-strategy offload --moe-cache-auto` | | experts live in RAM; every byte of VRAM left after the dense weights and the KV is an expert cache (7.1k of the 24,576 experts, on each card) |
| `--expert-load parallel` | | reads the expert files into RAM with parallel readers; `auto` (the engine's default) does the same but falls back to a slower serial read when free RAM is short: use `auto` on a machine with little RAM to spare |
| `--quant-backend moe.nvfp4=triton` | | the Triton NVFP4 expert kernels (the other backends need NVIDIA hardware) |
| `--ple-backend disk` | | the 51 GB n-gram embedding tables are read from the checkpoint files on demand |
| `--max-running-requests 4 --cuda-graph-max-bs 4` | | up to 4 requests decoded together (55 / 81 / 98 / 105 tok/s in total at 1 / 2 / 3 / 4 on the ROCm 7.14 release build, without the MTP head) |
| `--max-prefill-length 16384` | | 16k-token prompt chunks: each chunk streams every expert once, so bigger chunks read long prompts faster (an 8.4k prompt in 4.3 s instead of 5.5 s with 4k chunks) |
| `--memory-ratio 0.80` | | VRAM budget; 0.80 leaves the headroom 16k chunks need on the 20 GB card |
| `--moe-prefill-hit-d2d` | | experts already cached on the GPU are reused during prefill instead of crossing PCIe: agent turn 1.55 -> 1.18 s |
| `--disable-pynccl` | | ROCm uses RCCL through torch.distributed |
| `--tool-call-parser qwen3_coder --reasoning-parser qwen3` | | tool calls and thinking in the model's format |
| context | 250,000 | the model's native maximum (262,144) rounded down: the last 12k tokens would only take KV memory from the expert cache |

With several requests, each one's decode is computed row by row as if it were alone; prompts prefilled in the same
batch can differ at rounding level. The 4-request setting, the tuned decode tiles and the RAM tier were added after
the agentic benchmark runs and are checked for exactness instead ([benchmarks.md](benchmarks.md#precision)).

RAM: 82 GiB while serving, all of it locked (measured: ~63 GiB of experts, 9.5 GiB for the RAM tier, the rest the two processes); 59 GiB with the EXL3 3.05 bpw checkpoint, 74 GiB with 4.05 bpw.

With an experimental EXL3 checkpoint
([how-it-works.md](how-it-works.md#exl3-checkpoints)) the same profile applies: the
experts go through the EXL3 kernels (the start log says `MoE experts: exl3 via triton`), so `--quant-backend
moe.nvfp4=triton` and the `FREETOKEN_NVFP4_*_TUNED` tiles do nothing; the split rounds to whole 128-wide blocks (384 /
256, i.e. 60 / 40, which NVFP4 avoids; the EXL3 numbers are measured at it); the n-gram tables take 33 GB (3.05 bpw)
instead of 51. On two cards `serve.sh` lowers `--memory-ratio` to 0.77 (0.76 from 4 bpw): the card holding the
384-wide slice bounds the expert cache, and at 0.80 the 7900 XTX peaked 200 MiB under full VRAM
([journey.md](journey.md#16-exl3-checkpoints-2026-10-04)); `-- --memory-ratio X` overrides it.

## `xtx` and `xt`: one card

Same settings minus the two-card ones (split, imbalance, host all-reduce, 4 requests). `FREETOKEN_NVFP4_DECODE_TUNED`
stays set but its table has no one-card shapes: it only logs a note. The MTP head gains less on one card (+7-8 %, +6-12 % greedy: its
verify steps mostly stay at 2 rows, the extra rows' missing experts crossing PCIe). Plus:

| Setting | Value | Why |
|---|---|---|
| `--tp-size 1` | | one card |
| `--max-running-requests 1` | | the GDN state of 4 requests would take ~2 GiB more at TP=1 (estimated) |
| `--max-prefill-length 8192` | | with 16k chunks a single card runs out of working memory: a 60k-token prompt took 166 s and decode fell to 1.9 tok/s |
| `FREETOKEN_HOST_KV_TOKENS` / `_SNAPSHOTS` | `131072` / `16` | the RAM tier sized to the context (~5 GiB locked, estimated) |
| context | 131,072 | what fits: the XT ends a 124k-token conversation with 0.8 GiB of VRAM left |

RAM: 72 GiB while serving, locked (measured on each card; EXL3: 51 GiB at 3.05 bpw, 65 GiB at 4.05 bpw).

The cards are picked by VRAM: `xt` takes the smallest card with at least 20 GiB, `xtx` the smallest with 24 GiB, so a
machine with both keeps the other card free. The one-card profiles were measured for speed, not run on the agentic
benchmark.

## The untested profiles

Derived from the measured profiles and from the memory plans the engine logged on the tested cards; `serve.sh` prints
a note when you start one (`TESTED=0` in the file).

- **`xtx-xtx`**: `xtx-xt` with an even split. Both cards plan on 24 GB instead of the XT's 20 GB, each holding half of
  every expert: ~9k cached experts instead of 7.1k (estimate), so decode at least as fast as `xtx-xt`. The tuned NVFP4
  decode tiles are keyed to the 0.55 split, so the even split's decode shapes use default tiles. An EXL3 checkpoint
  refuses the even split (its expert slices must be whole 128-wide blocks): add `FREETOKEN_TP_SPLIT=0.6` to the file
  for it (untested, as for `xt-xt`).
- **`xt-xt`**: `xtx-xt` with an even split. Each XT holds half of every expert instead of 45 %: slightly fewer cached
  experts (~6.5k, estimate), and each XT does 50 % of the split work: expect decode a little below `xtx-xt`.
- **`gre`**: `xt` on 16 GB. From the XT's plan (int8 dense weights 5.7 GiB, KV 3.1 GiB at 131k tokens, 20 % headroom)
  a 16 GB card keeps only ~3-4 GiB for the expert cache (4-7 % of the experts cached, most expert reads cross PCIe),
  with 576 GB/s of VRAM bandwidth instead of 800: ~20 tok/s at best. It starts at 65,536 tokens of context with 4k
  prefill chunks; raise `CTX` only after checking the free memory printed at start-up.

## Adapting a profile to other cards (untested)

Nothing below has been measured; treat it as a starting point and check with the benchmarks in
[testing.md](testing.md).

- **Two identical cards**: use `xtx-xtx` or `xt-xt` (`xtx-xt` without `FREETOKEN_TP_SPLIT`); for another identical pair
  copy one and set `VRAM_GIB`. `FREETOKEN_TP_ALLOW_IMBALANCE=1` is harmless there and needed when one card drives a
  display or runs other work (upstream refuses a free-memory difference above 2 GiB).
- **Another unequal pair**: set `FREETOKEN_TP_SPLIT` to about rank 0's share of the total VRAM, then try +/-2.5 %;
  `FREETOKEN_TP_SPLIT=11:9` style weights also work (an EXL3 checkpoint moves in steps of 128 of the 640-wide
  intermediate, 20 %). The tuned NVFP4 decode tiles are keyed to this model's shapes at a 0.55 split: other splits
  fall back to the default tiles (a log line names each untuned shape once).
- **Smaller RDNA3 cards**: the RX 7900 GRE (16 GB) uses this gfx1100 image: start from `gre`. The RX 7800 XT (16 GB,
  gfx1101) needs its own image (`--build-arg GPU_ARCH=gfx1101`) and can start from `gre`; the 7700 XT (12 GB,
  gfx1101) and 7600 (8 GB, gfx1102) have less VRAM than any profile plans for. None of them is tested.
- **Less RAM**: `FREETOKEN_HOST_KV=0` saves the RAM tier (9.5 GiB on two cards, ~5 GiB on one; evicted conversations
  are re-read instead). In NVFP4 the experts themselves cannot go below ~63 GiB; the experimental EXL3 3 bpw
  checkpoint needs 42.6 GiB: [limits.md](limits.md).
- **The TunableOp files** in `rdna3/tunableop/` are ROCm 7.14's (their header records PyTorch / HIP / hipBLASLt), so
  the profiles set `TUNABLEOP=0` on the ROCm 10 image; to use TunableOp, tune for your image once and freeze (see
  [options.md](options.md#pytorch-tunableop)).

Every environment variable in these files is explained in [options.md](options.md); the `ft serve` flags in the tables
above and in upstream's [CLI reference](../cli.md).
