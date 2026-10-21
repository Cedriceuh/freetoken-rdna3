# Changelog

## Unreleased

What each change measured, and how it was found: [docs/rdna3/journey.md](docs/rdna3/journey.md); every switch:
[docs/rdna3/options.md](docs/rdna3/options.md).

### New

- EXL3 checkpoints of Qwen3.8-Flash-Next (`turboderp/Qwen3.8-Flash-Next-exl3`, 3.05 and 4.05 bpw), nothing to switch
  on: the routed experts stay in EXL3 behind Triton kernels for gfx1100, the other linears are decoded to bf16 at load.
  Agentic benchmark, one run each: 13 of 29 at 3.05 bpw, 12 at 4.05 (NVFP4: 12 and 18).
- Speculative decoding with the checkpoint's MTP head, on in the profiles except `gre` (`rdna3/serve.sh --no-mtp` turns
  it off): an adaptive 2-4 rows per step, greedy answers identical to plain decode, exact speculative sampling with the
  drafts drawn under each request's sampling. One request on `xtx-xt` +35 % decode; in `xtx-xt`, `xtx-xt-1m` and
  `xtx-xt-4x262k` up to four requests at once verify drafts, 2 rows each past two.
- Long contexts with the attention K/V in host RAM (`FREETOKEN_QSA_KV_HOST`): `xtx-xt-4x262k` holds four conversations
  of 262,144 tokens at once, `xtx-xt-1m` one of up to 1M tokens (the rope extended past 262,144 with
  `FREETOKEN_ROPE_MAX_POSITION`). EXL3 3.05 bpw: four cold 256k-token prompts read in 445 s, then 29.6-31.6 tok/s each
  decoding together; a 988k-token prompt read in 9.3 min, decoded at 84.7 tok/s.
- `rdna3/serve.sh --hybrid` (off by default): the CPU computes part of each decode step's expert misses from RAM
  (upstream's `--moe-strategy hybrid`, made to run on ROCm, on two cards and with EXL3 experts). NVFP4 decode on `xtx`
  42.7 -> 62.3 tok/s, on `xt` 34.6 -> 52.1.
- Image input, off by default (`rdna3/serve.sh --vision`): the vision tower whole on rank 0, its embeddings broadcast,
  images scaled down to 1024 tokens; text answers identical.
- `rdna3/serve.sh --check`: what the machine has, which profile fits and what fits in its RAM, changing nothing;
  `rdna3/download.sh` for the three checkpoints; `--default-temperature` (`serve.sh` passes 0.6); the README and
  getting-started rewritten as a path for newcomers.
- `rdna3/bench/depth_sweep.py`, `rdna3/bench/vision_tower_bench.py`, `rdna3/tests/shape_check.py --vision`, the
  banner's source (`docs/rdna3/assets/social-preview.html`).

### Faster, the same bits

- Agent turns: a prompt of up to 1280 tokens on a cached conversation reads its routed experts through the expert
  cache, the next layer's predicted and copied ahead; the tokenizer reuses the previous prompt's tokens; with the K/V
  in RAM a prefill copies only the K/V rows it selects; the host no longer syncs at every layer, Triton launches take a
  shorter path. EXL3 3.05 bpw `xtx-xt-4x262k`: 130-340-token turns at 25k 0.86-0.89 -> 0.40-0.48 s to the first token,
  then 500-750-token turns at 45k 0.51 -> 0.43 s.
- Cold reads: the PLE rows read on a thread by rank 0 only, over four io_uring rings, with a RAM cache; the all-reduces
  of each prefill block overlapping the next block; the QSA indexer scoring 16 rows a program; the GatedDeltaNet conv
  token-major; NVFP4 prefill blocks of 8192 tokens on two cards; tail tiles for long expert blocks. EXL3 3.05 bpw
  `xtx-xt-4x262k`: a 164k-token prompt 66.8-67.0 -> 49.5-49.9 s in one of these steps.
- Decode: the MoE epilogue and the hyper-connection norms fused, fewer copies, activations inside their GEMVs, the
  verify step's glue in kernels, the PLE rows waited for inside the captured step (`FREETOKEN_PLE_SYNC`), the input
  embedding in RAM (its VRAM to the expert cache). One of these steps: EXL3 3.05 bpw `xtx-xt-4x262k` 85.5 -> 93.9 tok/s at
  one request (sampled).
- Lower prefill VRAM peaks (in-place hyper-connections, blocked GatedDeltaNet prefill), and kernels that spilled
  registers on RDNA3 run more warps (prompt reading +2.6 to +3.5 %).

### Changed

- The two-card profiles keep the model's whole 262,144-token context.
- The image: ROCm 10.1, PyTorch 2.14 and its Triton 3.8 (with a fix of Triton 3.8's `tl.histogram` range in
  `moe_align`); EXL3 dense weights rebuilt in fp64 at load, the same weights on ROCm 10.0 and 10.1.
- Upstream FreeToken merged up to `ee9789f`, `9b68e00` and `cc56b6b`, the authors' commits kept: its NVFP4 decode
  kernel (5 of 15 greedy answers changed, all still right), fp32 NVFP4 global scales and its rewritten Triton NVFP4
  prefill GEMM, with this tree's tiles swept again (NVFP4 answers differ from the build before, EXL3 ones do not; NVFP4
  cold reads -12 %); `--hf-overrides`, `--api-key`, presence / frequency penalties; GGUF experts through upstream's
  kernels; a system message in the middle of an Anthropic `/v1/messages` conversation stays in place.
- Opt-in, not bit-exact: `FREETOKEN_INT8_TP_TILES`. Measured and left off: `FREETOKEN_MOE_COPY_OVERLAP` (33 % slower).

### Fixed

- Long decode pauses (5-22 s, once 140 s): the kernel's memory compaction moved host memory the GPU driver maps. The
  engine locks its memory (`FREETOKEN_MLOCK`) and the host needs `vm.compact_unevictable_allowed=0`; the first request
  after an idle pause waits for the VRAM clock unless the `COMPUTE` power profile is set.
- A rank killed by the host's OOM killer no longer leaves a half server: the API server exits with code 1, and
  `serve.sh` starts the container with `--oom-score-adj -999`.
- GPU page faults and a hung collective with a card within tens of MiB of full: the torch allocator is capped 512 MiB
  short of the card on ROCm (`FREETOKEN_VRAM_MARGIN_MIB`).
- Out-of-memory retries of every prefill chunk past ~26k of context: expandable segments in the allocator.
- One card without the MTP head: the PLE rows are filled before each launch (NVFP4 `xtx --no-mtp` 26.6 -> 38.3 tok/s).
- The JIT modules were rebuilt at every start (first request ~10 s, now 1.3-1.9 s); the batched-DMA probe raced its own
  fill; a rank held an all-reduce until the watchdog stopped the server (seen with a profiler slowing the host),
  with a dispatch of the EXL3 prefill's mid kernel at the head of its stopped queue, likely waiting for its 2.7 KB of
  scratch per lane (not proven; the kernel now needs 572 B); the CUDA-graph captures could free the verify step's rollback buffers when a capture with
  more requests followed one with fewer; a race in the disk PLE staging (upstream's fix); the TP>1 check of the vision
  tower never fired.

## 0.1.0 (2026-09-29)

First public release, on top of FreeToken `0d652e7` (the engine reports version 0.1.3).

- Qwen3.8-Flash-Next NVFP4 on RX 7900 XTX / XT: tensor parallelism over two unequal cards (uneven split, memory planned
  on the smaller card), int8 dense layers, RDNA3 GEMVs, host-memory all-reduce, tuned NVFP4 expert tiles, 16k prefill
  chunks with blocked PLE / MoE, expert reuse in prefill on ROCm.
- Up to 4 concurrent requests with batch-invariant decode kernels; conversations evicted from the GPUs kept in RAM.
- Exact sampling on ROCm: top-k-first for small `top_k`, a sorted threshold otherwise; rank 0's tokens broadcast.
- Profiles `xtx-xt`, `xtx`, `xt` (measured) and `xtx-xtx`, `xt-xt`, `gre` (untested) for Qwen3.8-Flash-Next;
  `rdna3/serve.sh` launcher; `rdna3/bench/quick_bench.py`.
