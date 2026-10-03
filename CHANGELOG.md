# Changelog

## Unreleased

- Speculative decoding with the checkpoint's MTP head, on in the profiles (`FREETOKEN_MTP=1 FREETOKEN_SPEC_VERIFY_M=4`;
  `rdna3/serve.sh --no-mtp` turns it off): a decode step of one request verifies up to 3 drafted tokens, with an
  adaptive depth from step costs measured per card count; greedy answers identical to plain decode, exact speculative
  sampling. One request on `xtx-xt`: +34 % decode at the model's sampling, +30-61 % greedy up to 248k of context; one
  card: +7-8 % (+6-12 % greedy). Several requests at once: 3-4 % slower at 3-4 requests. With the ROCm 10 base, 12 and
  18 of 29 on the agentic benchmark (two runs).
- The image moves to ROCm 10.0 / PyTorch 2.13, with the ROCm 7.14 image's Triton 3.7.1 (3.8 miscompiles these
  kernels). HIP stream memops make the PLE wait-sync work there (`FREETOKEN_PLE_SYNC`, +3.5-4 % decode). TunableOp is
  off in the profiles (its files are 7.14's).
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
