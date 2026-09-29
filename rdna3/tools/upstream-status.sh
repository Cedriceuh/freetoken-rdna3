#!/usr/bin/env bash
# What changed upstream since it was last merged here, and which pieces of the stack upstream now carries.
# Read-only (fetches only). Needs a remote named `upstream` (FlashML-org/FreeToken); a `luka` remote (the community
# ROCm port) is fetched too when it exists; the PR / issue states need the GitHub CLI (`gh`). The merge itself is
# done by hand, see docs/rdna3/maintaining.md.
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
git fetch -q upstream
git remote | grep -qx luka && git fetch -q luka || true

base=$(git merge-base HEAD upstream/main)
echo "this branch is based on: $(git log -1 --format='%h %cd %s' --date=short "$base")"
echo "upstream/main:      $(git log -1 --format='%h %cd %s' --date=short upstream/main)"
echo "new upstream commits: $(git rev-list --count HEAD..upstream/main)"
git log --date=short --format='  %h %cd %s' HEAD..upstream/main | head -40
echo
echo "touching what we serve (qwen4_exp, rocm/hip, moe/nvfp4, sampling, scheduler, engine):"
git log --date=short --format='  %h %cd %s' HEAD..upstream/main -- \
  python/freetoken/models/qwen4_exp python/freetoken/kernel/triton/sampling.py python/freetoken/engine \
  python/freetoken/moe python/freetoken/layers/quantization/moe python/freetoken/scheduler \
  python/freetoken/kernel/backend.py python/freetoken/kernel/csrc | head -30
echo
if git rev-parse -q --verify luka/rocm-gfx1201 >/dev/null; then
  echo "community ROCm port (luka/rocm-gfx1201), commits not in this tree: $(git rev-list --count HEAD..luka/rocm-gfx1201)"
  git log --date=short --format='  %h %cd %s' HEAD..luka/rocm-gfx1201 | head -15
  echo
fi
echo "stack pieces upstream may have absorbed:"
for pr in 133 134 135; do
  printf '  PR #%s: %s\n' "$pr" "$(gh pr view "$pr" -R FlashML-org/FreeToken --json state,mergedAt -q '.state + " " + (.mergedAt // "")' 2>/dev/null || echo '?')"
done
for issue in 349 364; do
  printf '  issue #%s: %s\n' "$issue" "$(gh issue view "$issue" -R FlashML-org/FreeToken --json state,title -q '.state + " - " + .title' 2>/dev/null || echo '?')"
done
# grep a file, not a pipe: with pipefail, `git show | grep -q` can fail on SIGPIPE
if grep -q 'supports TP=1 only' <(git show upstream/main:python/freetoken/models/qwen4_exp/weight.py); then
  echo "  qwen4_exp TP>1: still refused upstream (our port stays)"
else
  echo "  qwen4_exp TP>1: upstream no longer refuses it -> compare with our port before merging"
fi
if grep -q '_row_barrier' <(git show upstream/main:python/freetoken/kernel/triton/sampling.py); then
  echo "  sampling: upstream kernels still spin on _row_barrier (ROCm keeps the sorted-threshold path)"
else
  echo "  sampling: _row_barrier gone upstream -> re-test with rdna3/tests/run_sampling_tests.sh and rdna3/bench/sampling_topp_bench.py"
fi
