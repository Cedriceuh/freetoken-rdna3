# Changelog

## Unreleased

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
