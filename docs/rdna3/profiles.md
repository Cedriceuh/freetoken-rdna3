# Profiles

A profile is one file in `rdna3/profiles/` holding every setting of one hardware setup (measured, or derived and
marked `TESTED=0`). `rdna3/serve.sh <profile>` turns it into a `docker run` + `ft serve` command (`--dry-run` prints
it); `rdna3/serve.sh --check` says which ones fit your GPUs. Speeds below are with the NVFP4 checkpoint unless a cell
names another (all three checkpoints: [README](../../README.md#which-checkpoint)).

| Profile | GPUs | Context | Decode, MTP head on (off) | Deep decode, greedy | Cold 8.3k prompt | Agent turn | At once |
|---|---|---:|---:|---:|---:|---:|---:|
| `xtx-xt` | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | 262,144 | 88 (65.5) tok/s | 78-96 (58-65) tok/s | 3.8 s | 0.8-0.9 s | 4 requests |
| `xtx` | RX 7900 XTX 24 GB | 131,072 | 46 (41) tok/s | 36-44 (32-38) tok/s | 5.6 s | 1.5-1.6 s | 1 request |
| `xt` | RX 7900 XT 20 GB | 131,072 | 36 (34) tok/s | 29-34 (26-31) tok/s | 6.2 s | 1.6-1.8 s | 1 request |
| `xtx-xtx` *(untested)* | 2x RX 7900 XTX 24 GB | 262,144 | expected >= `xtx-xt` | – | – | – | 4 requests |
| `xt-xt` *(untested)* | 2x RX 7900 XT 20 GB | 262,144 | expected a little below `xtx-xt` | – | – | – | 4 requests |
| `gre` *(untested)* | RX 7900 GRE 16 GB | 65,536 | expected ~20 tok/s at best (head off) | – | – | – | 1 request |
| `xtx-xt-4x262k` | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | 262,144 per request, 4 at once | EXL3 3.05 (one pass, earlier build): 95.1 tok/s alone at 259k, 29.6-31.6 each with 4 | – | – | EXL3 3.05: 142-360-token turns 0.46-0.59 s alone at 224k | 4 requests |
| `xtx-xt-1m` | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | 1,000,000 | EXL3 3.05, greedy, earlier builds: 84.2-93.0 tok/s at 118k, 498k and 988k | – | – | – | 4 requests, sharing 1M |

"Decode" is one request's 512-token answers at the sampling `rdna3/serve.sh` gives (temperature 0.6; median of 12);
"deep decode" is greedy decode at every context step from 9k to 260k on two cards and 124k on one; "agent turn" is the
time to the first token when ~1k new tokens are added to a cached conversation, over the same steps. The profiles turn
the MTP head on; `rdna3/serve.sh <profile> --no-mtp` turns it off (numbers in brackets). All numbers:
[benchmarks.md](benchmarks.md). `--hybrid` (off by default) has the CPU compute part of the expert misses: NVFP4 decode +46 % on
`xtx`, +51 % on `xt`, +2 % on `xtx-xt`, measured on an earlier build than this table ([cpu-experts.md](cpu-experts.md)).

## File format

```bash
GPUS=2                  # how many cards the automatic choice picks (--gpus picks them yourself)
VRAM_GIB="24 20"        # minimum VRAM per rank, used for the automatic choice
TESTED=0                # derived, not measured: serve.sh says so at start (default 1)
CTX=262144              # context length of one request (--ctx overrides); passed as --max-seq-len-override
KV_TOKENS=1050000       # the K/V pool all requests share, --kv-reserve-tokens (default: CTX, never less)
MEMORY=110g             # container RAM limit (--memory overrides)
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
| `FREETOKEN_NVFP4_PREFILL_TUNED` / `_DECODE_TUNED` | `1` | expert GEMM tiles swept on these cards for this model (the decode ones for this split's shapes): long prompts -11 %, the down decode GEMV x1.06-1.19 in the kernel sweep, bit-exact against the default tiles |
| `FREETOKEN_NVFP4_PREFILL_BLOCK` | `8192` | the NVFP4 prefill MoE 8192 tokens at a time instead of 4096: cold reads +7.9 % (arms alternated), the same bits; 16384 left the XT 0.65 GiB of VRAM at the peak of a 255k-token read ([options.md](options.md)) |
| `FREETOKEN_HOST_KV` | `1` | conversations evicted from the GPUs go to RAM (9.5 GiB locked) and come back in ~2 s; exact |
| `FREETOKEN_MTP` / `FREETOKEN_SPEC_VERIFY_M` | `1` / `4` | the checkpoint's MTP head drafts up to 3 tokens a step for a request decoding alone (two at once in `xtx-xt`, `FREETOKEN_SPEC_BS_MAX=2`: +8-15 % each): +35 % decode at `rdna3/serve.sh`'s sampling (+6-13 % on one card), greedy answers unchanged; `rdna3/serve.sh --no-mtp` turns it off ([limits.md](limits.md#speculative-decoding-mtp)) |
| `--moe-strategy offload --moe-cache-auto` | | experts live in RAM; every byte of VRAM left after the dense weights and the KV is an expert cache (7.0k of the 24,576 experts, on each card) |
| `--expert-load parallel` | | reads the expert files into RAM with parallel readers; `auto` (the engine's default) does the same but falls back to a slower serial read when free RAM is short: use `auto` on a machine with little RAM to spare |
| `--quant-backend moe.nvfp4=triton` | | the Triton NVFP4 expert kernels (the other backends need NVIDIA hardware) |
| `--ple-backend disk` | | the 51 GB n-gram embedding tables are read from the checkpoint files on demand |
| `--max-running-requests 4 --cuda-graph-max-bs 4` | | up to 4 requests decoded together (1 / 2 / 3 / 4 greedy answers sent at once finish in 3.6 / 5.9 / 8.4-8.6 / 10.1 s) |
| `--default-temperature 0.6` | | added by `rdna3/serve.sh` to every profile: requests that send no temperature get 0.6 instead of the checkpoint's 1.0 (`-- --default-temperature T` changes it) |
| `--max-prefill-length 16384` | | 16k-token prompt chunks: each chunk streams every expert once (chunks of up to 1280 tokens read only their routed experts through the cache), so bigger chunks read long prompts faster (an 8.4k prompt in 4.3 s instead of 5.5 s with 4k chunks, in the A/B that chose them) |
| `--memory-ratio 0.80` | | VRAM budget; 0.80 leaves the headroom 16k chunks need on the 20 GB card |
| `--moe-prefill-hit-d2d` | | in a chunk that streams whole layers (over 1280 tokens), experts already cached on the GPU are copied on the GPU instead of crossing PCIe again |
| `--disable-pynccl` | | ROCm uses RCCL through torch.distributed |
| `--tool-call-parser qwen3_coder --reasoning-parser qwen3` | | tool calls and thinking in the model's format |
| context | 262,144 | the model's native maximum; a longer prompt is refused (`context_length_exceeded`). A 259,728-token prompt reads in 121.9 / 116.7 / 117.4 s (NVFP4 / EXL3 3.05 / 4.05 bpw) and is answered at 75.8 / 71.3 / 68.1 tok/s. The K/V of the last 12k tokens (250,000 -> 262,144) costs 134-152 expert slots per card (1.5-1.9 %) and 0.2-1.7 % of agent-turn decode, the same answers: `--ctx 250000` takes them back if you never need the full context |

With several requests, each one's decode is computed row by row as if it were alone; prompts prefilled in the same
batch can differ at rounding level. The 4-request setting, the tuned decode tiles and the RAM tier are checked for
exactness, not on the agentic benchmark ([benchmarks.md](benchmarks.md#precision)).

RAM: 82 GiB while serving, all of it locked (measured: ~63 GiB of experts, 9.5 GiB for the RAM tier, the rest the two processes); 60 GiB with the EXL3 3.05 bpw checkpoint, 75 GiB with 4.05 bpw.

With an EXL3 checkpoint
([how-it-works.md](how-it-works.md#exl3-checkpoints)) the same profile applies: the
experts go through the EXL3 kernels (the start log says `MoE experts: exl3 via triton`), so `--quant-backend
moe.nvfp4=triton` and the `FREETOKEN_NVFP4_*_TUNED` tiles do nothing; the split rounds to whole 128-wide blocks (384 /
256, i.e. 60 / 40, which NVFP4 avoids; the EXL3 numbers are measured at it); the n-gram tables take 33 GB (3.05 bpw)
instead of 51. On two cards `serve.sh` lowers `--memory-ratio` to 0.77 (0.76 from 4 bpw): the card holding the
384-wide slice bounds the expert cache, and at 0.80 the 7900 XTX peaked 200 MiB under full VRAM
([journey.md](journey.md#16-exl3-checkpoints-2026-10-04)); `-- --memory-ratio X` overrides it.

## `xtx-xt-4x262k` and `xtx-xt-1m`: the K/V in host RAM

The `xtx-xt` settings with the attention K/V in pinned host memory (`FREETOKEN_QSA_KV_HOST=1`), so the K/V pool is no
longer bounded by VRAM: `xtx-xt-4x262k` keeps each request to the model's native 262,144 positions and holds four of
them at once (`KV_TOKENS=1050000`), `xtx-xt-1m` serves one conversation of up to 1M tokens (the plain rope past
262,144, `FREETOKEN_ROPE_MAX_POSITION`). The RAM tier is off (`FREETOKEN_HOST_KV=0`): the pool already lives in RAM.
Measured with EXL3 3.05 bpw (~77 GiB of host RAM locked by `xtx-xt-4x262k`, ~75 GiB by `xtx-xt-1m`), and NVFP4 on
`xtx-xt-4x262k` with a pool of three contexts (`-- --kv-reserve-tokens 787500`: agent turns, a 259,728-token
prompt, four agents at once; 94 GiB locked after the start, ~97 GiB in use). On the 126 GiB reference machine that
fits with little else running (13-15 GiB of other programs beside the desktop ran it out of memory), and four NVFP4
contexts (~100 GiB, with benchmark tools beside them) did not fit: with NVFP4 on 128 GB,
`rdna3/serve.sh xtx-xt-4x262k --model DIR -- --kv-reserve-tokens 787500`, or two conversations (525000) to keep room.
How to count for your RAM: [limits.md](limits.md#how-much-ram-do-i-need). A
request alone decodes a few percent slower than on `xtx-xt` (the kernels read the K/V across PCIe), its prompt reads
faster (more expert slots). Measurements: [long-context.md](long-context.md).

## `xtx` and `xt`: one card

Same settings minus the two-card ones (split, imbalance, host all-reduce, 4 requests). `FREETOKEN_NVFP4_DECODE_TUNED`
stays set but its table has no one-card shapes: it only logs a note. The MTP head gains less on one card (+6-13 %, +9-16 % greedy: its
verify steps mostly stay at 2 rows, the extra rows' missing experts crossing PCIe). Plus:

| Setting | Value | Why |
|---|---|---|
| `--tp-size 1` | | one card |
| `--max-running-requests 1` | | the GDN state of 4 requests would take ~2 GiB more at TP=1 (estimated) |
| `--max-prefill-length 8192` | | with 16k chunks a single card runs out of working memory: a 60k-token prompt took 166 s and decode fell to 1.9 tok/s |
| `FREETOKEN_HOST_KV_TOKENS` / `_SNAPSHOTS` | `131072` / `16` | the RAM tier sized to the context (~5 GiB locked, estimated) |
| context | 131,072 | what fits: the XT's VRAM peaked at 19,367 of 20,464 MiB over a sweep to 124k tokens |

RAM: 73 GiB while serving, locked (measured on each card; EXL3: 51 GiB at 3.05 bpw, 66 GiB at 4.05 bpw).

The cards are picked by VRAM: `xt` takes the smallest card with at least 20 GiB, `xtx` the smallest with 24 GiB, so a
machine with both keeps the other card free. The one-card profiles were measured for speed, not run on the agentic
benchmark.

## The untested profiles

Derived from the measured profiles and from the memory plans the engine logged on the tested cards; `serve.sh` prints
a note when you start one (`TESTED=0` in the file).

- **`xtx-xtx`**: `xtx-xt` with an even split, the NVFP4 prefill MoE 4096 tokens at a time and drafts verified for one
  request at a time (`FREETOKEN_NVFP4_PREFILL_BLOCK`, `FREETOKEN_SPEC_BS_MAX` unset: untested on these pairs). Both cards plan on 24 GB instead of the XT's 20 GB, each holding half of
  every expert: ~9k cached experts instead of 7.0k (estimate), so decode at least as fast as `xtx-xt`. The tuned NVFP4
  decode tiles are keyed to the 0.55 split, so the even split's decode shapes use default tiles. An EXL3 checkpoint
  refuses the even split (its expert slices must be whole 128-wide blocks): add `FREETOKEN_TP_SPLIT=0.6` to the file
  for it (untested, as for `xt-xt`).
- **`xt-xt`**: `xtx-xt` with an even split, 4096-token prefill blocks and drafts for one request at a time (as
  `xtx-xtx`). Each XT holds half of every expert instead of 45 %: slightly fewer cached
  experts (~6.5k, estimate), and each XT does 50 % of the split work: expect decode a little below `xtx-xt`.
- **`gre`**: `xt` on 16 GB. From the XT's plan (int8 dense weights 5.7 GiB, KV 3.1 GiB at 131k tokens, 20 % headroom)
  a 16 GB card keeps only ~3-4 GiB for the expert cache (4-7 % of the experts cached, most expert reads cross PCIe),
  with 576 GB/s of VRAM bandwidth instead of 800: ~20 tok/s at best. It starts at 65,536 tokens of context with 4k
  prefill chunks; raise `CTX` only after checking the free memory printed at start-up.

## Adapting a profile to other cards (untested)

Nothing below has been measured; treat it as a starting point and check with the benchmarks in
[testing.md](testing.md).

- **Two identical cards**: use `xtx-xtx` or `xt-xt` (`xtx-xt` without `FREETOKEN_TP_SPLIT`, `FREETOKEN_NVFP4_PREFILL_BLOCK` and
  `FREETOKEN_SPEC_BS_MAX`); for another identical pair
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
  are re-read instead). In NVFP4 the experts themselves cannot go below ~63 GiB; the EXL3 3 bpw
  checkpoint needs 42.6 GiB: [limits.md](limits.md).

Every environment variable in these files is explained in [options.md](options.md); the `ft serve` flags in the tables
above and in upstream's [CLI reference](../cli.md).
