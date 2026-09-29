#!/usr/bin/env bash
# each mode in its own detached container, killed after 90 s if it hangs
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MOD="${1:?usage: run_sampling_tests.sh <module> (e.g. freetoken.kernel.triton.sampling)}"
IMAGE="${FREETOKEN_IMAGE:-freetoken-rdna3:latest}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE (build it, or set FREETOKEN_IMAGE)" >&2; exit 1; }
for mode in plain topp topk topktopp; do
  docker rm -f samp-test >/dev/null 2>&1
  docker run -d --name samp-test --device=/dev/kfd --device=/dev/dri --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" -e "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}" \
    -e SAMPLING_MOD="$MOD" -e PYTHONPATH=/src/python -v "$REPO":/src:ro \
    -v "$REPO"/rdna3/tests:/dbg:ro "$IMAGE" python /dbg/sampling_test.py $mode >/dev/null
  if timeout 90 docker wait samp-test >/dev/null; then st=done; else st=HUNG; docker kill samp-test >/dev/null; fi
  echo "[$mode $st] $(docker logs samp-test 2>&1 | grep -E ' ok|device' | tr '\n' ' ')"
  docker rm -f samp-test >/dev/null 2>&1
done
