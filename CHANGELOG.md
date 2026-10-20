# Changelog

## Unreleased

- The input embedding in RAM, the same bits. Qwen3.8-Flash-Next's input table (248,320 x 2560 bf16, not tied to the LM
  head) stays in pinned, device-mapped host memory and the same lookup kernel reads its rows across PCIe
  (`FREETOKEN_EMBED_HOST`); the 606 MiB a rank it held in VRAM go to the expert cache. `xtx-xt-4x262k` with a 3 x 262k
  pool, two runs each: 8,768 -> 9,255 slots with NVFP4, 11,554 -> 12,095 with EXL3 3.05 bpw; NVFP4 agent turns
  0.728-0.729 -> 0.707-0.711 s, decode 89.7-89.9 -> 91.2-91.7 tok/s, four requests 29.0-29.3 -> 30.5 tok/s each; EXL3
  agent turns 0.485-0.487 -> 0.472 s, decode 93.5-93.6 -> 94.3-94.5 tok/s, four requests 36.2 -> 36.8-37.1; cold reads
  unchanged. One card (`xtx`, EXL3, 131k): 4,783-4,793 -> 5,459-5,468 slots, decode 52.3-52.7 -> 55.4 tok/s, agent turns
  1.120-1.126 -> 1.083-1.086 s. The server locks 1.2 GiB more RAM (`serve.sh --check` counts it).
- Faster PLE row reads and long prefills, the same bits. The disk row store splits a large fill over four io_uring
  rings, one thread each (`FREETOKEN_ROW_STORE_RINGS`: one thread drove 240-290k reads a second, the disk takes ~490k),
  and reads a row that crosses a 4 KiB page as its two pages, apart (on the reference NVMe, reads spanning two 4 KiB
  units, even 2.5 % of a batch, halved the whole batch's rate); the store that reads the prefills keeps the rows it read
  in a RAM cache (`FREETOKEN_PLE_ROW_CACHE_ROWS`, 1M rows: 176 MiB with NVFP4, 118 MiB with EXL3), as an agent session's
  n-grams recur across its turns. A long NVFP4 prefill block runs each expert's full 128-row tiles in one launch and its
  remaining rows in 32-row tiles in a second (`FREETOKEN_NVFP4_PREFILL_TAIL_TILES`), EXL3's gate-up GEMM full 64-row
  tiles and 32-row remainders (`FREETOKEN_EXL3_PREFILL_TAIL_TILES`). On `xtx-xt-4x262k` with the reference machine's
  launcher settings and a 3 x 262k pool, the previous build -> this one without `FREETOKEN_EMBED_HOST`, two runs each:
  NVFP4 cold reads of 113k tokens 36.2-36.3 -> 34.1-34.2 s, 159k 51.5-51.6 -> 48.6-48.8 s, 210k 68.7 -> 64.9-65.0 s,
  agent turns at 45k (~520 tokens) 0.741 -> 0.730 s, ~2,600-token turns 1.21 -> 1.16 s (the cache); EXL3 3.05 bpw agent
  turns 0.523-0.526 -> 0.494-0.495 s (0.485-0.488 with the cache), ~2,600-token turns 0.99 -> 0.94 s (the cache), cold
  reads of 40k 12.5 -> 11.9 s, 113k 34.0-34.1 -> 33.2-33.3 s, 210k 64.4 -> 63.1-63.2 s; one card (`xtx`, EXL3) cold
  reads of 40k / 113k 21.0-21.1 / 59.1-59.2 -> 20.5-20.6 / 58.1-58.2 s. Through the depth sweep to 260k (without the
  cache): the new blocks read 5-12 % faster (NVFP4) and 4-10 % (EXL3), the following agent turns 1.5-5 % and 4-8 %
  shorter. The same answers; decode, four requests and one-word turns unchanged.
- Short agent turns and cold reads, the same bits. A prefill's per-layer-embedding rows are read from the checkpoint
  on a thread, by rank 0 only, which broadcasts them (`FREETOKEN_PLE_PREFILL_PREFETCH`, `FREETOKEN_PLE_PREFILL_RANK0`:
  the GPUs idled ~50 ms an agent turn and ~1.1 s a 16k-token chunk while every rank read them), and the next chunk's
  rows of a long prompt are read while a chunk computes; a slot-cache prefill reads its predicted experts' copy plan
  once the layer's GEMMs are queued instead of in a sync at every layer (`FREETOKEN_PREFILL_PREFETCH_DEFER`); Triton
  kernels launch through a shorter host path that calls the same compiled kernel (`FREETOKEN_TRITON_FAST_LAUNCH`); a
  prefill dequantizes its int8 dense weights in one Triton pass with `torch.mul`'s bits
  (`FREETOKEN_INT8_PREFILL_DEQUANT`); `FREETOKEN_PREFILL_COPY_OVERLAP=auto` now covers EXL3 (with the deferred plan);
  with NVFP4 experts a short prefill copies ahead only the experts at least two tokens predict, PCIe being saturated
  (`FREETOKEN_PREFILL_PREDICT_MIN_VOTES`); and on two cards each block of a chunk's routed experts and of its
  GatedDeltaNet output all-reduces while the next block computes (`FREETOKEN_PREFILL_AR_OVERLAP`,
  `FREETOKEN_GDN_AR_OVERLAP`, the latter after a check, at the first chunk of each length, that hipBLASLt gives a
  block's rows the bits of the whole call), the engine synchronizing once before the next decode step (without it,
  four NVFP4 requests decoded a third slower after such a read). The QSA indexer scores 16 query rows a program for a
  prefill of one request (`FREETOKEN_QSA_SCORE_ROWS`, the padded heads' sum order kept). On `xtx-xt-4x262k` with the
  reference machine's launcher settings, the previous build -> this one, two runs each: EXL3 3.05 bpw one-word turns
  at 34k (40-90 tokens to read) 0.237-0.243 -> 0.185-0.197 s, agent turns at 45k (500-750 tokens) 0.509-0.512 ->
  0.432-0.434 s, a 45k-token document 17.2 -> 13.6 s, a 164k-token prompt 66.8-67.0 -> 49.5-49.9 s; NVFP4 (3 x 262k
  pool) 0.256-0.261 -> 0.227-0.228 s, 0.718 -> 0.621-0.622 s, 18.2 -> 14.5 s, 70.2-70.3 -> 52.7-52.8 s. The same
  answers (15 greedy prompts, an agent conversation, one-word turns, 164k and 96k-token prompts); decode and four
  requests unchanged.
- Short agent turns, the same bits: the GatedDeltaNet prefill no longer reads the longest request back from the GPU in
  every GDN layer (36 host syncs a forward), NVFP4 prefill blocks of up to 1536 tokens take 4-warp expert tiles
  (`FREETOKEN_NVFP4_PREFILL_TUNED`, set in the profiles) and EXL3 blocks of up to 512 tokens 16-row tiles. On
  `xtx-xt-4x262k` with the reference machine's launcher settings, time to the first token, the previous build -> this
  one: NVFP4 (3 x 262k pool) 200-250-token turns of an agent conversation at 34k 0.448 -> 0.398-0.399 s, one-word turns
  at 26k 0.394 -> 0.227 s; EXL3 3.05 bpw 0.328 -> 0.287 s and 0.230 -> 0.218 s. The same answers (15 greedy prompts, the
  agent conversation, the one-word turns); decode, four requests and a 133k-token cold read unchanged.
- NVFP4 short turns, the same bits: the routed experts a short prefill's prediction missed are copied on the prefill
  copy stream while the shared expert runs (`FREETOKEN_PREFILL_COPY_OVERLAP`, default `auto`: NVFP4 experts decoded on
  the GPU, not `--hybrid`). On `xtx-xt-4x262k` with the launcher's settings: one-word turns 0.239 -> 0.225-0.228 s,
  agent turns 0.407 -> 0.398-0.399 s, the decode unchanged. Off with EXL3, where the turns gained 3-4 % but the decode
  after them ran 1-2 % slower.
- A rank killed by the host's OOM killer no longer leaves a half server: the API server exits with code 1 even while
  streams are open (it used to wait for them forever, the other rank holding its GPU; with a rank 0 killed under a
  stream on `xtx-xt-4x262k`, the container exited 37 s later, both cards free, and the service restarted), and
  `rdna3/serve.sh` starts the container with `--oom-score-adj -999` (not with rootless Docker), so the kernel picks
  other processes first.
- `rdna3/serve.sh --check [--model DIR] [profile]` looks at the machine without changing it: the amdgpu driver,
  Docker, the commit the image was built from, the GPUs (named when VRAM and compute units match an RX 7900) and the
  profiles they fit, the host settings, a container of the image already running, and a downloaded checkpoint (its
  type, missing weight files); it prints the fix for each and the command to start. It also says what fits in the
  machine's RAM: what each checkpoint locks with the K/V in VRAM and, with the K/V in RAM, how many 262k conversations
  or how long a single one, with the `--kv-reserve-tokens` / `--ctx` to start it. The arithmetic: the measured totals,
  6.9 GiB per 262k conversation on two cards, ~3 GiB of growth in use, and ~30 GiB kept for everything else (25 at the
  least), from the reference machine's runs and out-of-memory events. `rdna3/serve.sh` warns at start when a server
  would not fit. `--list-gpus` names the cards and their gfx arch. `rdna3/download.sh nvfp4|exl3-3.05|exl3-4.05 [DIR]`
  downloads a checkpoint with the image's Hugging Face CLI after checking the disk space. The README and
  `docs/rdna3/getting-started.md` are rewritten as a path for newcomers (what you need, which profile and checkpoint,
  clients set up, updating), and the docs describe the current build without dates: history stays in
  `docs/rdna3/journey.md`.
- Agent turns: a new prompt of up to 1280 tokens on a cached conversation reads only its routed experts, through
  the GPU expert cache (`FREETOKEN_PREFILL_CACHED_MAX`, 1280 for Qwen3.8-Flash-Next), those the next layer's router
  predicts from this layer's input copied while this layer computes; the experts it brings in become the next victims,
  so the decode after the turn keeps its cache. The tokenizer reuses the previous prompt's tokens up to its last turn
  start (`FREETOKEN_TOKENIZE_PREFIX_CACHE`, the same tokens; ~0.5 s -> 1-3 ms at 224k), and with the K/V in RAM a
  prefill stages only the K/V rows its tokens select (`FREETOKEN_QSA_STAGE_SELECTED`). The same answers; under
  `--hybrid` they move with the cache contents, as they always did there. Time to the first token at 25k of context,
  the previous `main` -> this build (arms alternated, 9/9 answers identical): 130-340-token turns NVFP4
  `xtx-xt-4x262k` (3 x 262k pool) 1.22-1.24 -> 0.58-0.70 s and EXL3 3.05 bpw 0.86-0.89 -> 0.40-0.48 s, ~1k-token turns 1.33-1.34 ->
  0.93-0.98 s and 0.94-0.99 -> 0.69-0.74 s, 3k-token turns -2 to -8 %; decode unchanged (-2 to +4 %). At 173k / 224k
  on EXL3 3.05 bpw `xtx-xt-4x262k`: 1.13-1.25 -> 0.43-0.50 s and 1.31-1.43 -> 0.46-0.59 s.
- Cold reads, the same bits: the GatedDeltaNet prefill conv runs token-major (`FREETOKEN_GDN_CONV_ROWS`) and, on two
  cards, the NVFP4 prefill MoE 8192 tokens at a time (`FREETOKEN_NVFP4_PREFILL_BLOCK=8192` in `xtx-xt` and
  `xtx-xt-4x262k`); the EXL3 prefill GEMMs run two pipeline stages. The previous `main` -> this build, the same texts,
  arms alternated: NVFP4 `xtx-xt-4x262k` (3 x 262k pool) 98k tokens 44.8-44.9 -> 40.2-40.3 s, 116k 52.8 -> 46.8-46.9 s (+11 to +13 %);
  EXL3 3.05 bpw 98k 40.2 -> 38.2 s, 116k 47.0-47.1 -> 44.3-45.5 s (+3 to +6 %).
- The two-card profiles (`xtx-xt`; `xtx-xtx`, `xt-xt` untested) keep the model's whole 262,144-token context, as in 0.1.0 and like `xtx-xt-4x262k`. At 262,144 against 250,000 (the last 12k tokens' K/V given to the expert cache instead), on `xtx-xt` 134-152 fewer expert slots per card (1.5-1.9 %), the decode of agent turns 0.2-1.7 % slower (turns 1-8 of a 25k conversation, the same answers); a 259,728-token prompt reads in 117-122 s and is answered at 68-76 tok/s; a longer one is refused (`context_length_exceeded`).
- `--default-temperature T`: the temperature a request that sends none gets, in place of the checkpoint's
  (`generation_config.json`, 1.0 for Qwen3.8-Flash-Next); requests that send one keep it. `rdna3/serve.sh` passes 0.6
  (not to an image older than the flag, picked with `--image`).
- Fixes: the EXL3 prefill's mid step needs 572 B of scratch per lane instead of 2.7 KB (`FREETOKEN_EXL3_MID_COLS`,
  the same bits, 1.7-3.3x faster): a dispatch of the old kernel was the head of a stopped queue when a rank held an
  all-reduce until the watchdog stopped the server (seen with a profiler slowing the host, in a short turn after a
  196k-token read; waiting for its scratch is the likely cause, not proven). The JIT modules stay in the kernel-cache volume (`TVM_FFI_CACHE_DIR`): first request after a start
  ~10 s -> 1.3-1.9 s; the batched-DMA probe no longer races its own fill (a rank fell back to one copy per expert at
  random once the modules came from a cache).
- One card without the MTP head (`--no-mtp`): a decode step's PLE disk rows are filled before its launch
  (`FREETOKEN_PLE_SYNC=auto` takes launch-gating on one GPU when the scheduler overlaps the steps): there the captured
  wait cost ~11 ms a step. NVFP4 `xtx --no-mtp`: 26.6 -> 38.3 tok/s greedy, the same tokens. Two cards, and one card
  with the head, keep the wait-sync, 2-4 % faster there.
- Decode, bit-exact and on by default (greedy answers identical): the EXL3 output kernel no longer runs Triton's
  constant-1 variant for a single K split, and rank 1's gate|up GEMV takes 4 N tiles a program; the hyper-connection's
  down GEMV applies `hc_silu` as it stores (`FREETOKEN_HC_SILU_EPILOGUE`) and its up GEMV mixes the gates into the
  streams (`FREETOKEN_HC_MIX_IN_GEMV`); the shared expert's gate | up GEMV stores SiLU-and-mul
  (`FREETOKEN_SHARED_ACT_IN_GEMV`). Then the small torch ops around them in kernels: the QSA index-write plan
  (`FREETOKEN_QSA_PLAN_KERNEL`), the PLE's m-row decode conv (`FREETOKEN_PLE_CONV_KERNEL`), the MTP head's KV-only
  pass stopped after its writes (`FREETOKEN_MTP_KV_ONLY_STOP`), a verify step's rollback, token writes and draft
  addressing (`FREETOKEN_SPEC_GLUE_KERNELS`): +1.8 % then +1.4 % at one request on top. `xtx-xt-4x262k`: 85.5 -> 93.9
  tok/s at one request (sampled), 33.9 -> 36.0 at four; NVFP4 `xtx-xt` 77.8 -> 82.5 (runs of different sessions; each
  change's same-session A/B in [docs/rdna3/journey.md](docs/rdna3/journey.md#25-a-decode-step-kernel-by-kernel-2026-10-11-and-12)). K blocks of 512 for the
  uneven split's int8 shapes are faster at 2-4 rows but not bit-exact: opt-in (`FREETOKEN_INT8_TP_TILES=1`). Measured
  and dropped: bit-exact int8 tiles, the sampled tokens through the host all-reduce, a RAM cache of the PLE rows, the
  PLE fill on a helper thread (the PLE waits in the traces come from the tracer), activations computed in the
  consuming GEMV's prologue (10 % slower), `num_stages` on the decode GEMVs, a packed router top-k, a grouped decode
  GEMV, the MTP cost table, verifying drafts with four requests (-26 %), MTP drafts over part of the vocabulary (fewer
  kept, VRAM off the expert cache), `HIP_FORCE_DEV_KERNARG`; with the decode step kernel by kernel (rocprofv3) and the
  gaps between kernels.
- `rdna3/serve.sh --hybrid` (off by default): the CPU computes part of each decode step's expert misses from RAM
  (upstream's `--moe-strategy hybrid`, which could not run here). The executor signals with HIP's stream memops (it fell
  back to two host callbacks per layer on ROCm; upstream's executor has its own now), sizes its scratch for the MTP verify
  rows up front, runs under tensor parallelism with NVFP4 (each rank on half the cores), reads EXL3 experts (AVX2), and
  reads (NVFP4) or decodes (EXL3) an expert once for the rows that share it. The hybrid layer's GPU ends are one kernel
  each and carry the handshake themselves (the step's task word on two pinned flag words, system-scope atomics: no
  stream memop left), and the kernel that picks which misses to fetch reads the routes as
  vectors (same decisions); the GPU decode kernels skip the routes the CPU computes instead of running them at weight 0
  (+1-3 %, `FREETOKEN_HYBRID_SKIP_CPU_ROUTES`). Cores by slice size, SMT siblings and EXL3 column blocks were measured
  too (off: no gain, or slower in serving). NVFP4: `xtx` 42.7 -> 62.3 tok/s, `xt` 34.6 -> 52.1, `xtx-xt` +2 % alone and
  +15 % each with four; EXL3 3.05 bpw +5 % on `xtx`, even on `xtx-xt`, a loss on `xtx-xt-4x262k`
  ([docs/rdna3/cpu-experts.md](docs/rdna3/cpu-experts.md)).
- Upstream FreeToken merged up to `9b68e00` (15 commits); the authors' commits are kept
  (five cherry-picked first, the rest merged). On this tree's path: transformers 5.18 (the Qwen3.8-Flash-Next config,
  the chat-template tokens, tool calls and a ~105k-token text, and the image preprocessing read the same as with 5.16,
  bit for bit); upstream's CPU MoE executor split into weight formats, now carrying this tree's EXL3 experts, its
  grouping of routes by expert and the hybrid publish / merge kernels on upstream's task-word handshake; upstream's
  slot pools for the expert cache (one pool for this model, so agent turns keep the prefill through the cache); the
  hybrid route split inside upstream's ensure kernel; presence / frequency penalties, also on MTP verify steps;
  upstream's vendored ggml kernels for GGUF checkpoints (they build for gfx1100) in place of this tree's ROCm build of
  the old ones; flashlib 0.4.0, numpy 2.5.3 in the image (numba and the NVIDIA cutlass / cuda-bindings packages gone).
  ROCm keeps `kernel/triton/moe_align.py` over upstream's staged align. Against the build before the merge
  (`xtx-xt-4x262k`, NVFP4 with three 262k contexts and EXL3 3.05 bpw with four):
  greedy with the MTP head 14/14 and a 25k-token agent conversation 9/9, the same answers on both; decode per turn
  -0.8 to +0.9 %; agent-turn time to the first token 0.59-1.69 s against 0.59-1.66 (NVFP4), 0.40-1.46 against
  0.40-1.45 (EXL3); cold reads of 36k / 46k tokens 2375 / 2512 tok/s against 2387 / 2525 (NVFP4), 2565 / 2680 against
  2552 / 2661 (EXL3); four 20k-token requests decoding together (NVFP4) 29.0-29.9 tok/s each against 28.9-29.9; a
  259,728-token prompt read at 2338 tok/s, 400 tokens at 74.7 tok/s, a longer one refused. One card without
  `--hybrid`: greedy 14/14 the same. `--hybrid` was not compared: two runs of the build before the merge
  already gave different greedy answers (7 of 14 the same; not investigated).
  The CPU suite fails only where it failed before, plus one test that fails on upstream as well (two GGUF architectures
  without an AOT entry); the GPU tests of the touched paths as before.
- Upstream FreeToken merged up to `ee9789f` (23 commits); the authors' commits are kept.
  On this tree's path: a race in the disk PLE staging fixed (the overlap scheduler could rewrite the pinned rows of a
  batch whose copy was still queued); upstream's NVFP4 marlin decode kernel with the decode
  tiles swept again: decode GEMVs 8-16 % less time at one request, NVFP4 greedy answers changed (5 of 15 prompts, all
  still right), decode with the MTP head 76.4 tok/s against 73.8 (`xtx-xt`, 12 sampled answers); the YaRN table, YaRN
  under the multi-axis rope and the split top-k past 16 chunks are upstream's own now (the
  top-k of one layer at 988k tokens 53-64 us against 65-75); `--hf-overrides` (YaRN without editing `config.json`) and
  `--api-key` (off unless given); a NaN in the sampling softmax, empty batches in the JIT index / store kernels, radix
  keys that kept request buffers alive. EXL3: greedy answers identical (15/15, also on `xtx-xt-4x262k`) and decode
  identical; `xtx-xt-1m` gives the same text as before at 498k and 988k, and YaRN through `--hf-overrides` the same
  text as the `config.json` edit at 497k. Upstream PR #134 left this tree's stack (merged upstream). Anthropic `/v1/messages` clients: a system
  message in the middle of a conversation now stays in place, which changes their prompt tokens.

- Lower prefill VRAM peaks, all bit-exact and on by default: the hyper-connection combines write in place
  (`FREETOKEN_HC_INPLACE`, which also adds the PLE output in place), the GDN delta rule of a prefill runs 4096 tokens
  of a request at a time (`FREETOKEN_GDN_PREFILL_BLOCK`), the MTP head frees the prompt's streams once read: -0.6 GiB
  at the 7900 XT's peak, -0.9 GiB at the XTX's. On ROCm the torch allocator is capped 512 MiB short of the card
  (`FREETOKEN_VRAM_MARGIN_MIB`), the card's use read from sysfs so that a desktop's buffers count: GPU page faults
  and a hung collective came with a card within tens of MiB of full. The profiles keep `--memory-ratio` 0.80 (0.77 for
  EXL3 on two cards): a fuller expert cache (0.83: +3.8 % decode) left the 7900 XTX, which drives the display, within
  ~0.1 GiB of full once the desktop held 1.4 GiB; the reductions are headroom instead.
- Prefill kernels that spilled registers on RDNA3 run more warps, bit-exact: the QSA sparse attention 4 instead of 2
  (`FREETOKEN_QSA_MIN_WARPS`, 39-45 % faster), two GDN prefill kernels 8 instead of 4 (`FREETOKEN_GDN_PREFILL_WARPS`,
  a GDN prefill 37 % faster). Prompt reading +2.6 to +3.5 % at every depth to 248k tokens.
- The image moves to ROCm 10.1 and PyTorch 2.14 (HIP 7.16, RCCL 2.30.7, its Triton 3.8). Its base has no virtual
  environment and no git: the build uses `python3` and installs git; PyTorch 2.14 needs C++20, so the C++ extensions
  build with `-std=c++20`. Greedy answers identical between PyTorch 2.13 and 2.14 on 10.1. NVFP4 answers are identical to 10.0; EXL3 ones
  changed, because the dense EXL3 weights were rebuilt at load with fp32 BLAS products whose kernel 10.1 changed: the
  rebuild now runs them in fp64 (`FREETOKEN_EXL3_RECON_FP64`, on), which gives the same weights on 10.0 and 10.1 (and a
  one-time change from the fp32 ones, 0.02-0.03 % of the weights by one unit in the last place). Profiling with
  rocprofiler (rocprofv3, torch.profiler) on 10.1 hung RCCL's all-reduce twice on two cards; serving did not.
- Fewer copies, bit-exact (on by default): the attention's q / k, its output gate, the GDN gate and the QSA indexer
  rows are read where their projections wrote them (`FREETOKEN_FEWER_COPIES`), and the router GEMV reduces in its last
  program (`FREETOKEN_GEMV_LAST_REDUCES`). `xtx-xt` with the MTP head: NVFP4 +2.3 %, EXL3 3.05 bpw +2.9 % decode;
  identical answers, same prompt reading.
- Fewer decode kernels, bit-exact (on by default): the MoE epilogue (routed sum, shared-expert gate, mul-add) in one
  kernel (`FREETOKEN_FUSED_MOE_EPILOGUE`), each hyper-connection combine fused with the next grouped RMSNorm
  (`FREETOKEN_FUSED_HC_NORM`), one split-K reduce launch for all the rows of a verify step. `xtx-xt` greedy decode:
  NVFP4 +3.5 % (+3.3 % with the MTP head), EXL3 +2.1-2.2 % (+2.1-3.2 %); identical answers, same prompt reading.
- EXL3 checkpoints: Qwen3.8-Flash-Next from `turboderp/Qwen3.8-Flash-Next-exl3` (3.05 and 4.05 bpw) serves with the
  routed experts kept in EXL3 (Triton kernels for gfx1100: split-K decode GEMVs, grouped prefill GEMMs, Hadamard
  rotations) and the other linears decoded to bf16 at load; nothing to switch on. At 3.05 bpw on `xtx-xt`: decode
  83-107 tok/s greedy up to 248k tokens (NVFP4 72-86), agent turns 0.9-1.6 s (1.3-2.0), 59 GiB of RAM locked (82);
  measured on `xtx` and `xt` too. Agentic benchmark, one run each: 13 of 29 at 3.05 bpw, 12 at 4.05 (NVFP4: 12 and 18).
  An expert kernel can now declare its prefill temporaries, which the cache planner keeps free (EXL3: ~0.6 GiB per card
  at 16k-token chunks; without it, the first long prompt after a start filled the 7900 XTX), and `rdna3/serve.sh`
  lowers `--memory-ratio` to 0.77 / 0.76 for EXL3 on two cards.
- Speculative decoding with the checkpoint's MTP head, on in the profiles (`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4`;
  `rdna3/serve.sh --no-mtp` turns it off): a decode step of one request verifies up to 3 drafted tokens, with an
  adaptive depth from step costs measured per card count; greedy answers identical to plain decode, exact speculative
  sampling. One request on `xtx-xt`: +34 % decode at the model's sampling, +30-61 % greedy up to 248k of context; one
  card: +7-8 % (+6-12 % greedy). The `xtx-xt` profile verifies drafts for two requests at once too
  (`FREETOKEN_SPEC_BS_MAX=2`): two agents at ~100k of context decode 8-15 % faster each. Several requests at once:
  3-4 % slower at 3-4 requests. With the ROCm 10 base, 12 and
  18 of 29 on the agentic benchmark (two runs).
- The image moves to ROCm 10.0 / PyTorch 2.13 and its own Triton 3.8. Triton 3.8's AMD range analysis gives
  `tl.histogram` an empty range and folds comparisons on its counts to false, which broke `moe_align_block_size`
  (an illegal memory access in the first prefill, then a hang): the kernel no longer compares those counts. HIP stream memops make the PLE wait-sync work there (`FREETOKEN_PLE_SYNC`, +3.5-4 % decode). TunableOp and its
  files are gone (< 0.2 % on prompt reading once tuned for this image).
- Optional switches: `FREETOKEN_MOE_COPY_OVERLAP` (expert copies on a side stream; measured 33 % slower end to end, leave
  it off), `FREETOKEN_HOST_ALLREDUCE_WAITLOG`
  (debug), `FREETOKEN_INT8_ROW_GROUPS` (on: up to 4 rows share each int8 weight tile).
- The long decode pauses seen on the reference machine: the kernel's memory compaction moving host memory the GPU
  driver maps, which stops the GPU queues for 5-22 s (once 140 s, and the server died). The engine now locks its memory
  (`FREETOKEN_MLOCK`) and the host needs `vm.compact_unevictable_allowed=0`; the VRAM clock after a pause needs the
  `COMPUTE` power profile (docs/rdna3/troubleshooting.md).
- `serve.sh` runs PyTorch's allocator with expandable segments: without them, every 14-16k-token prefill chunk past ~26k
  of context hit an out-of-memory retry.
- `rdna3/bench/depth_sweep.py`; the banner's source, `docs/rdna3/assets/social-preview.html`; `quick_bench.py` warms up
  with a prompt of the cold read's size (with the head, a server's first prompt of a new size took 1-5 s more).
- Image input on Qwen3.8-Flash-Next, off by default (`rdna3/serve.sh --vision`, images scaled down to 1024 tokens).
  The vision tower is never split: rank 0 encodes and broadcasts the embeddings, and the tower stays bf16 under
  `FREETOKEN_INT8_DENSE`. Measured on `xtx-xt`, `xtx` and `xt` to the full context: text answers are identical, there
  is no speed change on two cards, and decode is ~1.5-2 % slower on one.
- Fix: the TP>1 check for the vision tower never fired (it looked for `model.visual.` after the loader had renamed the
  keys to `visual.`), so such a start failed on a bare shape assertion.
- `rdna3/bench/vision_tower_bench.py`; `rdna3/tests/shape_check.py --vision`.

## 0.1.0 (2026-09-29)

First public release, on top of FreeToken `0d652e7` (the engine reports version 0.1.3).

- Qwen3.8-Flash-Next NVFP4 on RX 7900 XTX / XT: tensor parallelism over two unequal cards (uneven split, memory planned
  on the smaller card), int8 dense layers, RDNA3 GEMVs, host-memory all-reduce, tuned NVFP4 expert tiles, 16k prefill
  chunks with blocked PLE / MoE, expert reuse in prefill on ROCm.
- Up to 4 concurrent requests with batch-invariant decode kernels; conversations evicted from the GPUs kept in RAM.
- Exact sampling on ROCm: top-k-first for small `top_k`, a sorted threshold otherwise; rank 0's tokens broadcast.
- Profiles `xtx-xt`, `xtx`, `xt` (measured) and `xtx-xtx`, `xt-xt`, `gre` (untested) for Qwen3.8-Flash-Next;
  `rdna3/serve.sh` launcher; `rdna3/bench/quick_bench.py`.
