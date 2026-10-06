# Maintaining

How this repository relates to upstream FreeToken and how to keep it current. For maintainers.

## What this tree is

Upstream [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) at `0d652e7` (2026-09-26, "feat(rocm): add
RDNA3 and RDNA4 runtime foundation (#132)"), plus the changes listed file by file in
[changes-from-upstream.md](changes-from-upstream.md). The history is upstream's own up to that commit, then this
project's commits on top (the first one holds everything up to 0.1.0). Once the `upstream` remote is set up (below),
`git diff $(git merge-base HEAD upstream/main)` shows every change against the upstream version this tree is based on.

## The stack on top of upstream (oldest first)

| Piece | Origin | Leaves the stack when |
|---|---|---|
| TVM-FFI JIT kernels portable to HIP | upstream PR #133 (zihaomu) | #133 merges |
| CUDA-only backends gated off on ROCm | upstream PR #134 (zihaomu) | #134 merges |
| RCCL for tensor-parallel execution | upstream PR #135 (zihaomu) | #135 merges |
| LDS clamp of the QSA attend kernel for RDNA's 64 KiB | community ROCm port (lukascechovic), upstream issue #349 | upstream fixes #349 |
| TP>1 relay handshake before the first request | community ROCm port (lukascechovic), upstream issue #364 | upstream fixes #364 |
| GDN boundary carried across the prefill chunk seam | community ROCm port (lukascechovic) | upstream carries it |
| qwen4_exp tensor parallelism | rewritten from the community port's | upstream supports qwen4_exp at TP>1 |
| unequal cards, uneven split, int8 dense, GEMVs, host all-reduce, sampler, tuned tiles, blocked prefill, multi-request, RAM tier, review fixes | this repository | upstream merges it (to be proposed piece by piece) |
| `Dockerfile.rdna3`, `rdna3/`, `docs/rdna3/` | this repository | never |

## Keeping current

```bash
git remote add upstream https://github.com/FlashML-org/FreeToken.git     # once
git remote add luka https://github.com/lukascechovic/FreeToken.git       # optional: the community ROCm port
rdna3/tools/upstream-status.sh        # read-only: new upstream commits, and which stack pieces they touch
git checkout -b sync-upstream && git merge upstream/main              # on a branch, merged into main once validated
```

Upstream is merged, not rebased onto: the published history is never rewritten, so clones and forks keep working.
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
new model, split or card, keeping only tiles that preserve the summation order (M / N tiles, warps, stages).
The EXL3 decode GEMV table (`_GEMV_TUNED` in `kernel/triton/exl3.py`) came from `rdna3/bench/exl3_gemv_bench.py`
and is keyed to this model's 384 / 256 shapes.
