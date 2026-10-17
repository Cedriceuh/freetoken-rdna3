#!/usr/bin/env bash
# Download a Qwen3.8-Flash-Next checkpoint with the image's own Hugging Face CLI: nothing to install on the host, the
# files belong to your user. Run it again after an interruption: files already complete are skipped.
#
#   rdna3/download.sh <checkpoint> [DIR] [-- extra `hf download` flags]
#
# checkpoints (docs/rdna3/getting-started.md#which-checkpoint), with their default DIR:
#   nvfp4       RadixArk/Qwen3.8-Flash-Next-NVFP4                            135 GB  ~/models/Qwen3.8-Flash-Next-NVFP4
#   exl3-3.05   turboderp/Qwen3.8-Flash-Next-exl3, revision 3.05bpw_h5_ng5    85 GB  ~/models/Qwen3.8-Flash-Next-exl3-3.05bpw
#   exl3-4.05   turboderp/Qwen3.8-Flash-Next-exl3, revision 4.05bpw_h6_ng6   108 GB  ~/models/Qwen3.8-Flash-Next-exl3-4.05bpw
#
# Give the same DIR to `rdna3/serve.sh --model`.
# HF_TOKEN, when set, is passed on (higher rate limits). --image REF picks another image than freetoken-rdna3:latest.
set -euo pipefail
DOCKER="$(command -v docker || echo docker)"
image="freetoken-rdna3:latest" which="" dir="" extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --image) [ $# -ge 2 ] || { echo "--image needs a value" >&2; exit 2; }; image="$2"; shift 2 ;;
    -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    --) shift; extra=("$@"); break ;;
    -*) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    *) if [ -z "$which" ]; then which="$1"; elif [ -z "$dir" ]; then dir="$1"; else echo "too many arguments" >&2; exit 2; fi
       shift ;;
  esac
done
case "$which" in
  # bytes: the size of a complete download (du -sb), for the disk check
  nvfp4) repo=RadixArk/Qwen3.8-Flash-Next-NVFP4 rev="" gb=135 bytes=135253765159 name=Qwen3.8-Flash-Next-NVFP4 ;;
  exl3-3.05) repo=turboderp/Qwen3.8-Flash-Next-exl3 rev=3.05bpw_h5_ng5 gb=85 bytes=85139450503
             name=Qwen3.8-Flash-Next-exl3-3.05bpw ;;
  exl3-4.05) repo=turboderp/Qwen3.8-Flash-Next-exl3 rev=4.05bpw_h6_ng6 gb=108 bytes=107463609955
             name=Qwen3.8-Flash-Next-exl3-4.05bpw ;;
  *) echo "usage: rdna3/download.sh nvfp4|exl3-3.05|exl3-4.05 [DIR]   (see --help)" >&2; exit 2 ;;
esac
dir="${dir:-$HOME/models/$name}"

"$DOCKER" image inspect "$image" >/dev/null 2>&1 ||
  { echo "no image $image: build it first (docker build -f Dockerfile.rdna3 -t $image .)" >&2; exit 2; }

# room left on that disk (the nearest directory that exists), counting what an earlier, interrupted run already
# wrote, plus 1 GB to spare; skipped when --include / --exclude download part of it
part=""
for a in "${extra[@]}"; do case "$a" in --include|--include=*|--exclude|--exclude=*) part=1 ;; esac; done
have=0 there="$dir"
[ ! -d "$dir" ] || have=$(du -sb "$dir" | cut -f1)
while [ ! -d "$there" ]; do there="$(dirname "$there")"; done
free=$(df -B1 --output=avail "$there" | tail -1 | tr -d ' ')
if [ -z "$part" ] && [ $((free + have)) -lt $((bytes + 1000000000)) ]; then
  echo "not enough disk space for $dir: the checkpoint takes $gb GB, $(( (free + have) / 1000000000 )) GB available" >&2
  exit 2
fi
mkdir -p "$dir"
dir="$(cd "$dir" && pwd)"

echo "$repo${rev:+ ($rev)}, $gb GB -> $dir" >&2
args=(download "$repo" --local-dir /out)
[ -z "$rev" ] || args+=(--revision "$rev")
envs=(-e HF_HOME=/out/.cache/huggingface)
[ -z "${HF_TOKEN:-}" ] || envs+=(-e HF_TOKEN)
"$DOCKER" run --rm --user "$(id -u):$(id -g)" "${envs[@]}" -v "$dir:/out" --entrypoint hf "$image" "${args[@]}" "${extra[@]}"
echo "done: rdna3/serve.sh --check --model $dir" >&2
