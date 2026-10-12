# Changelog

## Unreleased

- Decode, bit-exact and on by default (greedy answers identical): the EXL3 output kernel no longer runs Triton's
  constant-1 variant for a single K split, and rank 1's gate|up GEMV takes 4 N tiles a program; the hyper-connection's
  down GEMV applies `hc_silu` as it stores (`FREETOKEN_HC_SILU_EPILOGUE`) and its up GEMV mixes the gates into the
  streams (`FREETOKEN_HC_MIX_IN_GEMV`); the shared expert's gate | up GEMV stores SiLU-and-mul
  (`FREETOKEN_SHARED_ACT_IN_GEMV`). Then the small torch ops around them in kernels: the QSA index-write plan
  (`FREETOKEN_QSA_PLAN_KERNEL`), the PLE's m-row decode conv (`FREETOKEN_PLE_CONV_KERNEL`), the MTP head's KV-only
  pass stopped after its writes (`FREETOKEN_MTP_KV_ONLY_STOP`), a verify step's rollback, token writes and draft
  addressing (`FREETOKEN_SPEC_GLUE_KERNELS`): +1.8 % then +1.4 % at one request on top. `xtx-xt-4x262k`: 85.5 -> 93.9
  tok/s at one request (sampled), 33.9 -> 36.0 at four; NVFP4 `xtx-xt` 77.8 -> 82.5 (runs of different sessions; each
  change's same-session A/B in [docs/rdna3/long-context.md](docs/rdna3/long-context.md)). K blocks of 512 for the
  uneven split's int8 shapes are faster at 2-4 rows but not bit-exact: opt-in (`FREETOKEN_INT8_TP_TILES=1`). Measured
  and dropped: bit-exact int8 tiles, the sampled tokens through the host all-reduce, a RAM cache of the PLE rows, the
  PLE fill on a helper thread (the PLE waits in the traces come from the tracer), activations computed in the
  consuming GEMV's prologue (10 % slower), `num_stages` on the decode GEMVs, a packed router top-k, a grouped decode
  GEMV, the MTP cost table, verifying drafts with four requests (-26 %), MTP drafts over part of the vocabulary (fewer
  kept, VRAM off the expert cache), `HIP_FORCE_DEV_KERNARG`; with the decode step kernel by kernel (rocprofv3) and the
  gaps between kernels.
- `rdna3/serve.sh --hybrid` (experimental): the CPU computes part of each decode step's expert misses from RAM
  (upstream's `--moe-strategy hybrid`, which could not run here). The executor signals with HIP's stream memops (it
  fell back to two host callbacks per layer on ROCm), sizes its scratch for the MTP verify rows up front, runs under
  tensor parallelism with NVFP4 (each rank on half the cores), reads EXL3 experts (AVX2), and reads (NVFP4) or decodes
  (EXL3) an expert once for the rows that share it. The hybrid layer's GPU ends are one kernel each and carry the
  handshake themselves (step numbers, system-scope atomics: no stream memop left), and the kernel that picks which
  misses to fetch reads the routes as vectors (same decisions); the GPU decode kernels skip the routes the CPU
  computes instead of running them at weight 0 (+1-3 %, `FREETOKEN_HYBRID_SKIP_CPU_ROUTES`). Cores by slice size, SMT
  siblings and EXL3 column blocks were measured too (off: no gain, or slower in serving). NVFP4: `xtx` 39.4 -> 56.2
  tok/s, `xt` 31.4 -> 46.6, `xtx-xt` one request 78.2 -> 81.2-82.7, 4 requests 24.4 -> 28.2-28.6 each; EXL3: `xtx`
  +6 %, a loss on two cards ([docs/rdna3/cpu-experts.md](docs/rdna3/cpu-experts.md)).
- Upstream FreeToken merged up to `ee9789f` (23 commits, 2026-09-29 to 2026-10-10); the authors' commits are kept.
  On this tree's path: a race in the disk PLE staging fixed (the overlap scheduler could rewrite the pinned rows of a
  batch whose copy was still queued, `d3512b4`); upstream's NVFP4 marlin decode kernel (`7644984`) with the decode
  tiles swept again: decode GEMVs 8-16 % less time at one request, NVFP4 greedy answers changed (5 of 15 prompts, all
  still right), decode with the MTP head 76.4 tok/s against 73.8 (`xtx-xt`, 12 sampled answers); the YaRN table, YaRN
  under the multi-axis rope and the split top-k past 16 chunks are upstream's own now (`235e201`, `5031e3d`; the
  top-k of one layer at 988k tokens 53-64 us against 65-75); `--hf-overrides` (YaRN without editing `config.json`) and
  `--api-key` (off unless given); a NaN in the sampling softmax, empty batches in the JIT index / store kernels, radix
  keys that kept request buffers alive. EXL3: greedy answers identical (15/15, also on `xtx-xt-4x262k`) and decode
  identical; `xtx-xt-1m` gives the same text as before at 498k and 988k, and YaRN through `--hf-overrides` the same
  text as the `config.json` edit at 497k. Upstream PR #134 left this tree's stack (merged upstream). Anthropic `/v1/messages` clients: a system
  message in the middle of a conversation now stays in place (`93b0a59`), which changes their prompt tokens.

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
- The README and the banner show every checkpoint on every tested profile, over the whole context; the comparison
  with FreeToken as ported to ROCm is gone from them and from the benchmarks' summary.
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
- The GGUF kernels build on ROCm.
- Optional switches: `FREETOKEN_MOE_COPY_OVERLAP` (expert copies on a side stream; measured 33 % slower end to end, leave
  it off), `FREETOKEN_HOST_ALLREDUCE_WAITLOG`
  (debug), `FREETOKEN_INT8_ROW_GROUPS` (on: up to 4 rows share each int8 weight tile).
- The long decode pauses seen on the reference machine: the kernel's memory compaction moving host memory the GPU
  driver maps, which stops the GPU queues for 5-22 s (once 140 s, and the server died). The engine now locks its memory
  (`FREETOKEN_MLOCK`) and the host needs `vm.compact_unevictable_allowed=0`; the VRAM clock after a pause needs the
  `COMPUTE` power profile (docs/rdna3/troubleshooting.md).
- The two-card profiles serve 250,000 tokens of context (262,144 before), the same figure everywhere.
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
