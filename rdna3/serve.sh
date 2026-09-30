#!/usr/bin/env bash
# Serve a model with the freetoken-rdna3 image and one of the profiles in rdna3/profiles/.
#
#   rdna3/serve.sh <profile> --model DIR [options] [-- extra ft serve flags]
#   rdna3/serve.sh --list              the profiles, with their first comment line
#   rdna3/serve.sh --list-gpus         the GPUs, as HIP numbers them (index, PCI address, VRAM)
#
# options:
#   --model DIR        checkpoint directory (e.g. .../Qwen3.8-Flash-Next-NVFP4), mounted read-only
#   --port N           HTTP port (default 1919): OpenAI-compatible API at http://HOST:N/v1
#   --host ADDR        bind address (default 127.0.0.1; 0.0.0.0 to serve your LAN)
#   --ctx N            context length (default: the profile's CTX)
#   --gpus LIST        HIP_VISIBLE_DEVICES, rank 0 first (default: a one-card profile takes the smallest card with the
#                      profile's VRAM_GIB, a multi-card one the largest cards, largest first)
#   --image REF        image to run (default freetoken-rdna3:latest, built with Dockerfile.rdna3)
#   --name NAME        container name (default freetoken-rdna3)
#   --served-name ID   model id the API reports (default qwen3.8-flash-next)
#   --memory SIZE      container RAM limit (default: the profile's MEMORY)
#   --vision           accept image input: builds the vision tower on rank 0 (measured on xtx-xt, xtx and xt:
#                      docs/rdna3/benchmarks.md); images are scaled down to 1024 tokens (one per 32x32 pixels),
#                      `-- --image-max-tokens N` changes it
#   --dry-run          print the docker command instead of running it
#
# The kernel cache (JIT + autotune results) lives in a docker volume named after the image id, so it is built once
# per image. See docs/rdna3/getting-started.md and docs/rdna3/profiles.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER="$(command -v docker || echo docker)"
image="freetoken-rdna3:latest" name="freetoken-rdna3" served="qwen3.8-flash-next" host=127.0.0.1 port=1919
model="" ctx="" gpus="" memory="" dry="" profile="" extra=() mm_args=(--text-model-only)

list_gpus() {  # "hip_index vram_bytes pci_address" per GPU, from the kernel's KFD topology (no GPU context opened):
  # HIP numbers the GPU nodes in topology order
  local i=0 n p loc dom mem s b
  for n in $(ls -d /sys/class/kfd/kfd/topology/nodes/* | sort -t/ -k8,8n); do
    p="$n/properties"
    [ "$(awk '/^simd_count/{print $2}' "$p")" = 0 ] && continue
    loc=$(awk '/^location_id/{print $2}' "$p"); dom=$(awk '/^domain/{print $2}' "$p"); mem=0
    for b in "$n"/mem_banks/*/properties; do s=$(awk '/^size_in_bytes/{print $2}' "$b"); [ "$s" -gt "$mem" ] && mem=$s; done
    printf '%s %s %04x:%02x:%02x.%x\n' "$i" "$mem" "$dom" $((loc >> 8)) $(((loc >> 3) & 31)) $((loc & 7))
    i=$((i + 1))
  done
}

need() { [ $# -ge 2 ] || { echo "option $1 needs a value (see --help)" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --model) need "$@"; model="$2"; shift 2 ;;
    --port) need "$@"; port="$2"; shift 2 ;;
    --host) need "$@"; host="$2"; shift 2 ;;
    --ctx) need "$@"; ctx="$2"; shift 2 ;;
    --gpus) need "$@"; gpus="$2"; shift 2 ;;
    --image) need "$@"; image="$2"; shift 2 ;;
    --name) need "$@"; name="$2"; shift 2 ;;
    --served-name) need "$@"; served="$2"; shift 2 ;;
    --memory) need "$@"; memory="$2"; shift 2 ;;
    --vision) mm_args=(--image-max-tokens 1024); shift ;;
    --dry-run) dry=1; shift ;;
    --list)
      for f in "$HERE"/profiles/*.env; do printf '%-9s %s\n' "$(basename "$f" .env)" "$(sed -n '1s/^# //p' "$f")"; done
      exit 0 ;;
    --list-gpus) list_gpus | LC_ALL=C awk '{printf "HIP %s  PCI %s  %.1f GiB VRAM\n", $1, $3, $2 / 1073741824}'; exit 0 ;;
    -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    --) shift; extra=("$@"); break ;;
    -*) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    *) [ -z "$profile" ] || { echo "one profile only" >&2; exit 2; }; profile="$1"; shift ;;
  esac
done
[ -n "$profile" ] || { echo "usage: rdna3/serve.sh <profile> --model DIR   (profiles: rdna3/serve.sh --list)" >&2; exit 2; }
pfile="$HERE/profiles/$profile.env"
[ -f "$pfile" ] || { echo "no profile '$profile' (rdna3/serve.sh --list)" >&2; exit 2; }
[ -d "$model" ] || { echo "--model must be the checkpoint directory" >&2; exit 2; }
model="$(cd "$model" && pwd)"

# the profile: GPUS / VRAM_GIB / CTX / MEMORY / TUNABLEOP / TESTED / FT_ARGS drive this script, every other KEY=value is
# container env
envs=() ft_args=() p_gpus=1 p_vram="" p_ctx=131072 p_memory=110g p_tunableop=0 p_tested=1
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue ;; esac
  key="${line%%=*}" val="${line#*=}"
  case "$val" in  # a quoted value ends at its closing quote; an unquoted one at a " # comment"
    \"*) val="${val#\"}" val="${val%%\"*}" ;;
    *) val="${val%%[[:space:]]#*}" val="${val%"${val##*[![:space:]]}"}" ;;
  esac
  case "$key" in
    GPUS) p_gpus="$val" ;;
    VRAM_GIB) p_vram="$val" ;;
    TESTED) p_tested="$val" ;;
    CTX) p_ctx="$val" ;;
    MEMORY) p_memory="$val" ;;
    TUNABLEOP) p_tunableop="$val" ;;
    FT_ARGS) read -r -a ft_args <<< "$val" ;;
    *) envs+=(-e "$key=$val") ;;
  esac
done < "$pfile"
ctx="${ctx:-$p_ctx}" memory="${memory:-$p_memory}"
[ "$p_tested" = 1 ] || echo "note: profile $profile is UNTESTED (derived from measured profiles, see its header); please report what you measure" >&2

if [ -z "$gpus" ]; then
  # one GPU: the smallest card with the profile's VRAM (keeps a bigger card free); several: the largest ones, largest
  # first (the uneven split gives rank 0 the bigger share)
  read -r -a need <<< "${p_vram:-0}"
  mapfile -t found < <(list_gpus | sort -k2,2nr)
  if [ "$p_gpus" = 1 ]; then
    mapfile -t found < <(printf '%s\n' "${found[@]}" | awk -v m="${need[0]}" '$2 >= m * 1073741824 * 0.97' | sort -k2,2n)
  fi
  [ "${#found[@]}" -ge "$p_gpus" ] || { echo "profile $profile needs $p_gpus GPU(s) with ${p_vram:-any} GiB; found: $(list_gpus | wc -l) GPU(s) (rdna3/serve.sh --list-gpus)" >&2; exit 2; }
  pick=("${found[@]:0:$p_gpus}")
  for r in "${!pick[@]}"; do
    m="${need[$r]:-${need[0]}}"
    awk -v v="$(echo "${pick[$r]}" | cut -d' ' -f2)" -v m="$m" 'BEGIN { exit !(v < m * 1073741824 * 0.97) }' &&
      echo "warning: rank $r has less than the profile's $m GiB; lower --ctx or pick the cards with --gpus" >&2
  done
  gpus="$(printf '%s\n' "${pick[@]}" | cut -d' ' -f1 | paste -sd,)"
  echo "GPUs: $(printf '%s\n' "${pick[@]}" | LC_ALL=C awk '{printf "%sHIP %s (PCI %s, %.0f GiB)", (NR > 1 ? ", " : ""), $1, $3, $2 / 1073741824}')" >&2
fi
mounts=(-v "$model:/models/m:ro")
if [ "$p_tunableop" = 1 ]; then  # GEMM choices tuned for this image on 7900 XTX / XT (read-only, never tuned online)
  mounts+=(-v "$HERE/tunableop:/tunableop:ro")
  envs+=(-e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=0 -e PYTORCH_TUNABLEOP_FILENAME=/tunableop/tunableop_results%d.csv)
fi
if [ -z "$dry" ] && ! "$DOCKER" info >/dev/null 2>&1; then
  echo "cannot reach the Docker daemon: is it running, and is your user in the 'docker' group (log in again after adding it)?" >&2
  exit 2
fi
if image_id="$("$DOCKER" image inspect "$image" --format '{{.Id}}' 2>/dev/null)"; then
  short="${image_id#sha256:}" short="${short:0:12}"
elif [ -n "$dry" ]; then
  image_id="$image" short="IMAGE-ID"
  echo "note: no image $image yet; once built, its id replaces the tag and names the kernel-cache volume" >&2
else
  echo "no image $image: build it first (docs/rdna3/getting-started.md), or pick one with --image" >&2; exit 2
fi
video_gid="$(getent group video | cut -d: -f3 || true)" render_gid="$(getent group render | cut -d: -f3 || true)"
[ -n "$video_gid" ] && [ -n "$render_gid" ] ||
  { echo "the 'video' and 'render' groups are missing: they own /dev/kfd and /dev/dri (amdgpu driver)" >&2; exit 2; }
gpu_flags=(--device=/dev/kfd --device=/dev/dri --group-add "$video_gid" --group-add "$render_gid"
           --security-opt seccomp=unconfined)
mounts+=(-v "freetoken-rdna3-kcache-$short:/root/.cache/freetoken-rdna3")

cmd=("$DOCKER" run --rm --init --name "$name" --network host --ipc=host "${gpu_flags[@]}"
     --ulimit memlock=-1 --memory "$memory"
     -e PYTORCH_ALLOC_CONF=expandable_segments:False -e OMP_WAIT_POLICY=PASSIVE -e "HIP_VISIBLE_DEVICES=$gpus"
     "${envs[@]}" "${mounts[@]}"
     "$image_id" ft serve --model /models/m --host "$host" --port "$port" --served-model-name "$served" "${mm_args[@]}"
     --max-seq-len-override "$ctx" --kv-reserve-tokens "$ctx" "${ft_args[@]}" "${extra[@]}")
if [ -n "$dry" ]; then
  echo "# first: $DOCKER rm -f $name  (a container left behind would hold the GPUs and the port)"
  for a in "${cmd[@]}"; do
    if [[ "$a" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then printf '%s ' "$a"; else printf '%q ' "$a"; fi
  done
  echo
  exit 0
fi
"$DOCKER" rm -f "$name" >/dev/null 2>&1 || true
exec "${cmd[@]}"
