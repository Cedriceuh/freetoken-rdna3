# Maintaining

How this repository relates to upstream FreeToken and how to keep it current. For maintainers.

## What this tree is

Upstream [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) at `cc56b6b` ("feat(gguf): serve
GLM-5.3-Flash GGUF checkpoints on glm5-next (#721)"), plus the changes listed file by file in
[changes-from-upstream.md](changes-from-upstream.md). The history is upstream's own up to `0d652e7`, then this
project's commits on top (the first one holds everything up to 0.1.0), a merge of upstream up
to `ee9789f`, five upstream commits cherry-picked (`74a3877`, `9c635a1`, `7d6490c`, `9b68e00`, `b6193e8`, authors
kept), a merge of the rest up to `9b68e00` and a merge up to `cc56b6b`. Once the `upstream` remote is set up (below),
`git diff $(git merge-base HEAD upstream/main)` shows every change against the upstream version this tree is based on.

## The stack on top of upstream (oldest first)

| Piece | Origin | Leaves the stack when |
|---|---|---|
| RCCL for tensor-parallel execution | upstream PR #135 (zihaomu) | #135 merges |
| LDS clamp of the QSA attend kernel for RDNA's 64 KiB | community ROCm port (lukascechovic), upstream issue #349 | upstream's clamp (`7d6490c`, #587) sizes the tiles with the K/V cache's itemsize, this one with q's (the same tile unless K/V is 1 byte): kept ahead of it until they agree |
| TP>1 relay handshake before the first request | community ROCm port (lukascechovic), upstream issue #364 | upstream fixes #364 |
| GDN boundary carried across the prefill chunk seam | community ROCm port (lukascechovic) | upstream carries it |
| qwen4_exp tensor parallelism | rewritten from the community port's | upstream supports qwen4_exp at TP>1 |
| unequal cards, uneven split, int8 dense, GEMVs, host all-reduce, sampler, tuned tiles, blocked prefill, multi-request, RAM tier, review fixes | this repository | upstream merges it (to be proposed piece by piece) |
| MTP head, EXL3 checkpoints, K/V in host RAM, short prefills through the expert cache, `--hybrid` on ROCm | this repository | upstream merges it |
| `kernel/triton/moe_align.py` on ROCm, over upstream's staged align (`0bca88f`) | this repository (Triton 3.8 histogram fix) | upstream's align is checked on ROCm's Triton 3.8, or Triton fixes the histogram range |
| hybrid publish / merge kernels over pinned flag words | this repository | upstream drops its copies around the doorbell on ROCm |
| HIP stream memops for the PLE wait-sync (`row_store_ext.cpp`) | this repository | upstream's row store finds HIP's memops |
| `LaunchKernel::with_attr` refuses `use_pdl=True` on ROCm | this repository | upstream refuses it too |
| `Dockerfile.rdna3`, `rdna3/`, `docs/rdna3/` | this repository | never |

Left the stack at the merge of `ee9789f`: upstream merged #134 (CUDA-only backends gated off on ROCm, `9b585b7`) and
carries #133's HIP-portable JIT kernels since `803a3b4` (#133 itself was closed). Of both, this tree keeps one line: on
ROCm, `LaunchKernel::with_attr` refuses `use_pdl=True` where upstream ignores it. The YaRN rope table and the split
top-k past 16 chunks, written here first, are upstream's own since `235e201` and `5031e3d`.

At the merge of `9b68e00`: upstream's vendored ggml-cuda kernels (`e5d3bd3`, they build for gfx1100)
replaced this tree's ROCm build of the old GGUF kernels; upstream's hybrid ensure kernel splits the routes (`f02fbb1`),
which this tree's split kernel did, and its executor speaks HIP's memops (`2010271`), which this tree added: both left
the stack. Kept over upstream: `kernel/triton/moe_align.py` on ROCm instead of the staged align of `0bca88f` (the same
Triton 3.8 bug, fixed in the kernel here); the hybrid publish / merge kernels instead of upstream's copies around the
doorbell, which it keeps on ROCm. The prefill through the slot cache needs one slot pool (`b416f85`'s default here).

At the merge of `cc56b6b`: upstream's CPU MoE executor serves a format per MoE layer (`e8c6bf1`) and computes GGUF
experts with ggml-cpu instead of its Q4_0 kernels (`c8559fd`); this tree's EXL3 experts (with their own fp16 decode,
which upstream dropped from the shared header in `0db5023`), route grouping and hybrid flag words sit on it, and
`FREETOKEN_HYBRID_FETCH_FRACTION` fills its per-layer fractions. Upstream moved the NVFP4 launches to
`kernel/nvfp4_moe.py` (`ac6a15d`) and rewrote the Triton prefill GEMM (`03c2c9c`): the RDNA3 decode tiles and the skip
of the CPU's routes moved with the launches, the RDNA3 prefill tiles were swept again on the new GEMM. NVFP4 global
scales are fp32 (`0db5023`); with the rewritten GEMM, NVFP4 answers differ from the build before, EXL3 ones do not. The
qwen4_exp dense reader is upstream's scheme-driven one (`163d1de`), with this tree's tensor-parallel sharding, EXL3
reconstruction and MTP head on top; the expert-piece hook takes upstream's `layers` argument, and the TP slicing keeps
the experts' static activation scales (`e4f69d9`) whole on every rank.

## Keeping current

```bash
git remote add upstream https://github.com/FlashML-org/FreeToken.git     # once
git remote add luka https://github.com/lukascechovic/FreeToken.git       # optional: the community ROCm port
rdna3/tools/upstream-status.sh        # read-only: new upstream commits, and which stack pieces they touch
git checkout -b sync-upstream && git merge upstream/main              # on a branch, merged into main once validated
```

Upstream is merged, not rebased onto: the published history is never rewritten, so clones and forks keep working.
To take only some upstream commits, `git cherry-pick -x <commit>` keeps their author and records where they came
from; `upstream-status.sh` leaves the commits so recorded out of the new ones.
`upstream-status.sh` also reads an optional `luka` remote (the community ROCm port) when it exists. When upstream lands
one of the pieces of the table above, resolve the conflict in upstream's favour, then update the table and
[changes-from-upstream.md](changes-from-upstream.md).

After a merge, update the base commit named in `NOTICE`, this file and
[changes-from-upstream.md](changes-from-upstream.md), and its line counts
(`git diff --numstat $(git merge-base HEAD upstream/main)`). Then, in this order, each step gating the next:

1. CPU tests ([testing.md](testing.md)): only the environment failures listed there.
2. `docker build -f Dockerfile.rdna3 -t freetoken-rdna3:<tag> .` (its content checks must pass).
3. GPU validation ([testing.md](testing.md), the procedure): smoke test with sampling and a tool call, greedy answers
   identical to the previous build, decode / prefill windows interleaved against it, the depth sweep, several agents.
4. Release only if speed is at least equal and every check passes; keep the previous image as a fallback.

## Releasing

1. Measure on the image being released, then update every place the numbers appear: [benchmarks.md](benchmarks.md),
   `README.md`, `llms.txt`, [profiles.md](profiles.md), the headers in `rdna3/profiles/`, the banner
   (`docs/rdna3/assets/social-preview.png`, rendered from `social-preview.html`) and `CHANGELOG.md`; then run
   `rdna3/tools/make-llms-full.sh`. `rdna3/bench/quick_bench.py` and `depth_sweep.py` take the headline numbers.
2. `rdna3/tools/privacy_scan.sh` (machine names as extra patterns): no personal paths, e-mail addresses, secrets or AI co-author lines in the
   tree or the history being published.
3. Tag the release `rdna3-vX.Y.Z` (upstream's own tags are `vX.Y.Z`, so a fetch from upstream never clashes), with
   an annotated tag; the image records the commit it was built from (`/opt/FreeToken-BUILD-PROVENANCE.txt`).

## Tuned tiles

The NVFP4 tile tables (`FREETOKEN_NVFP4_*_TUNED`) were produced with `rdna3/bench/nvfp4_*_sweep.py`; rerun them for a
new model, split or card, keeping only tiles that preserve the summation order (M / N tiles, warps, stages). The EXL3
decode GEMV table (`_GEMV_TUNED` in `kernel/triton/exl3.py`) came from `rdna3/bench/exl3_gemv_bench.py` and is keyed
to this model's 384 / 256 shapes; there the N tiles per program keep the sums, the K tile and split change them, and
some warp counts do too (check the outputs bit for bit). The int8 GEMV configs (`_CONFIGS` in
`kernel/triton/int8_gemv.py`): only BLOCK_N keeps the sums; the K block, split and warps change them (checked on the
uneven split's shapes). Sweep cold (48 distinct matrices or 48 layers' own slots, as a decode step reads
them) with a warm-up pass, and time the current config first and last: a warm microbenchmark keeps the weights in the
Infinity Cache and promises gains that serving does not see.
