#!/usr/bin/env bash
# Serve a model with the freetoken-rdna3 image and one of the profiles in rdna3/profiles/.
#
#   rdna3/serve.sh <profile> --model DIR [options] [-- extra ft serve flags]
#   rdna3/serve.sh --list              the profiles: GPUs, context, K/V pool, measured or not, first comment line
#   rdna3/serve.sh --list-gpus         the GPUs, as HIP numbers them (index, PCI address, VRAM, model)
#   rdna3/serve.sh --check [--model DIR] [profile]
#                                      check this machine (driver, Docker, image, GPUs, host settings, the
#                                      checkpoint), suggest a profile and say what fits in its RAM; changes nothing
#
# options:
#   --model DIR        checkpoint directory (e.g. .../Qwen3.8-Flash-Next-NVFP4), mounted read-only
#   --port N           HTTP port (default 1919): OpenAI-compatible API at http://HOST:N/v1
#   --host ADDR        bind address (default 127.0.0.1; 0.0.0.0 to serve your LAN)
#   --ctx N            context length of one request (default: the profile's CTX)
#   --gpus LIST        HIP_VISIBLE_DEVICES, rank 0 first (default: a one-card profile takes the smallest card with the
#                      profile's VRAM_GIB, a multi-card one the largest cards, largest first)
#   --image REF        image to run (default freetoken-rdna3:latest, built with Dockerfile.rdna3)
#   --name NAME        container name (default freetoken-rdna3)
#   --served-name ID   model id the API reports (default qwen3.8-flash-next)
#   --memory SIZE      container RAM limit (default: the profile's MEMORY)
#   --vision           accept image input: builds the vision tower on rank 0 (measured on xtx-xt, xtx and xt:
#                      docs/rdna3/benchmarks.md); images are scaled down to 1024 tokens (one per 32x32 pixels),
#                      `-- --image-max-tokens N` changes it
#   --no-mtp           decode without the MTP draft head the profiles except gre turn on (it speeds up one request at a time;
#                      docs/rdna3/limits.md)
#   --hybrid           compute part of each decode step's expert misses on the CPU instead of copying them (off by
#                      default, NVFP4: +46 % on xtx, +51 % on xt, on xtx-xt +2 % for one request and +15 % with 4;
#                      EXL3: +5 % on xtx, even on xtx-xt; docs/rdna3/cpu-experts.md)
#   --dry-run          print the docker command instead of running it
#
# Requests that send no temperature get 0.6 (`ft serve --default-temperature 0.6`; the checkpoint's generation_config
# says 1.0): `-- --default-temperature T` changes it.
#
# The kernel cache (JIT + autotune results) lives in a docker volume named after the image id, so it is built once
# per image. See docs/rdna3/getting-started.md and docs/rdna3/profiles.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER="$(command -v docker || echo docker)"
# where the machine is read from; tests/rdna3/test_serve_sh.py points these at a fake one
MEMINFO="${SERVE_SH_MEMINFO:-/proc/meminfo}" KFD_NODES="${SERVE_SH_KFD_NODES:-/sys/class/kfd/kfd/topology/nodes}"
image="freetoken-rdna3:latest" name="freetoken-rdna3" served="qwen3.8-flash-next" host=127.0.0.1 port=1919
model="" ctx="" gpus="" memory="" dry="" no_mtp="" hybrid="" profile="" check="" extra=() mm_args=(--text-model-only)

list_gpus() {  # "hip_index vram_bytes pci_address gfx_arch simd_count" per GPU, from the kernel's KFD topology (no GPU
  # context opened): HIP numbers the GPU nodes in topology order
  local i=0 n p loc dom mem s b simd gv
  [ -d "$KFD_NODES" ] || return 0
  for n in $(ls "$KFD_NODES" | sort -n); do
    case "$n" in *[!0-9]*) continue ;; esac
    n="$KFD_NODES/$n" p="$n/properties"
    simd=$(awk '/^simd_count/{print $2}' "$p")
    [ "$simd" = 0 ] && continue
    loc=$(awk '/^location_id/{print $2}' "$p"); dom=$(awk '/^domain/{print $2}' "$p"); mem=0
    gv=$(awk '/^gfx_target_version/{print $2}' "$p")
    for b in "$n"/mem_banks/*/properties; do s=$(awk '/^size_in_bytes/{print $2}' "$b"); [ "$s" -gt "$mem" ] && mem=$s; done
    printf '%s %s %04x:%02x:%02x.%x gfx%d%x%x %s\n' "$i" "$mem" "$dom" $((loc >> 8)) $(((loc >> 3) & 31)) $((loc & 7)) \
      $((gv / 10000)) $((gv / 100 % 100)) $((gv % 100)) "$simd"
    i=$((i + 1))
  done
}

card_name() {  # $1 = one list_gpus line: its gfx arch, after the RX 7900 model when VRAM and compute units match one
  LC_ALL=C awk '{ g = int($2 / 1073741824 + 0.5); n = ""
    if ($4 == "gfx1100" && $5 == 192 && g == 24) n = "RX 7900 XTX "
    if ($4 == "gfx1100" && $5 == 168 && g == 20) n = "RX 7900 XT "
    if ($4 == "gfx1100" && $5 == 160 && g == 16) n = "RX 7900 GRE "
    print n $4 }' <<< "$1"
}

pval() {  # value of KEY $2 in profile file $1, without quotes or trailing comment (default $3)
  awk -F= -v k="$2" -v d="$3" '$1 == k { v = $2; sub(/[ \t]*#.*/, "", v); gsub(/"/, "", v); print v; f = 1; exit }
                                 END { if (!f) print d }' "$1"
}

profile_fits() {  # $1 = profile file: true if this machine's GPUs meet its GPUS and VRAM_GIB (the automatic choice's rule)
  local g r m
  local -a vram cards
  g=$(pval "$1" GPUS 1); read -r -a vram <<< "$(pval "$1" VRAM_GIB 0)"
  mapfile -t cards < <(list_gpus | sort -k2,2nr)
  [ "${#cards[@]}" -ge "$g" ] || return 1
  for ((r = 0; r < g; r++)); do
    m="${vram[$r]:-${vram[0]}}"
    LC_ALL=C awk -v v="$(cut -d' ' -f2 <<< "${cards[$r]}")" -v m="$m" 'BEGIN { exit !(v >= m * 1073741824 * 0.97) }' || return 1
  done
}

ckpt_kind() {  # $1 = checkpoint directory: nvfp4, exl3-<bits> or unknown, from its config.json (nothing without one)
  local c b
  [ -f "$1/config.json" ] || return 0
  c="$(tr -d ' \t\n' < "$1/config.json")"
  if [[ "$c" == *'"quant_method":"exl3"'* ]]; then
    b="$(grep -o '"bits":[0-9.]*' <<< "$c" | head -1 | cut -d: -f2)"; echo "exl3-${b:-?}"
  elif [[ "$c" == *'"quant_algo":"NVFP4"'* ]]; then echo nvfp4
  else echo unknown; fi
}

ckpt_label() { case "$1" in nvfp4) echo NVFP4 ;; exl3-*) echo "EXL3 ${1#exl3-} bpw" ;; *) echo "this checkpoint" ;; esac; }

# What a server locks in host RAM (all of it is locked), in GiB, right after its start: the experts and the processes
# (the measured totals of docs/rdna3/limits.md minus the RAM tier, plus 1.2 for what this build keeps in RAM beyond
# them: the input embedding, FREETOKEN_EMBED_HOST, and the PLE row cache; +1.18 measured on xtx-xt with NVFP4, +1.19
# on one card with EXL3), plus either the RAM tier (FREETOKEN_HOST_KV: 9.5 GiB on two cards, ~5 per 131,072 tokens on
# one) or, with the K/V in host RAM (FREETOKEN_QSA_KV_HOST), the K/V pool at 14,144 bytes per token and card. Within 2
# GiB of the four measured layouts with the K/V in RAM (EXL3 3.05 bpw 4 x 262k and 1M, NVFP4 3 and 4 x 262k). One 262k
# conversation takes 262,500 tokens of pool (the decode's spare page included): 6.9 GiB on two cards.
ram_core() {
  case "$1:$2" in
    2:nvfp4) echo 73.7 ;; 2:exl3-3.05) echo 51.7 ;; 2:exl3-4.05) echo 66.7 ;;
    1:nvfp4) echo 69.2 ;; 1:exl3-3.05) echo 47.2 ;; 1:exl3-4.05) echo 62.2 ;;
  esac
}
ram_locked() {  # $1 GPUs, $2 checkpoint, $3 K/V pool in host RAM in tokens (0: K/V in VRAM), $4 the RAM tier's
  # tokens (FREETOKEN_HOST_KV_TOKENS, 262144 by default; 0 when the tier is off). Unrounded: the verdicts judge this
  # value and ram_pool_max solves for it, so a pool --check offers is judged the same when asked for
  local c
  c="$(ram_core "$1" "$2")"
  [ -n "$c" ] || return 0
  LC_ALL=C awk -v c="$c" -v g="$1" -v p="$3" -v h="${4:-262144}" 'BEGIN {
    t = (h == 0 ? 0 : (g == 2 ? 9.5 : 5 * h / 131072))
    printf "%.3f", c + (p > 0 ? p * 14144 * g / 1073741824 : t) }'
}
gib() { LC_ALL=C awk -v x="$1" 'BEGIN { printf "%.0f", x }'; }  # a ram_locked value as the messages show it
# What the machine needs beside the server, measured on the 126 GiB reference machine (journey.md, sections 22 and
# 30): a server grows by a few GiB in use (NVFP4 with 3 x 262k in RAM: 94.4 GiB locked after its start, ~97 GiB hours
# later); with ~31 GiB left it served for hours, with ~29 GiB left and 13-15 GiB of other programs beside the desktop
# the host ran out of memory, and with ~25 GiB left (and benchmark tools) it did too. So: GiB left after the server's
# use >= RAM_KEEP fits, >= RAM_KEEP_MIN only with little else running, below that does not.
RAM_GROWTH=3 RAM_KEEP=30 RAM_KEEP_MIN=25
ram_total() { LC_ALL=C awk '/^MemTotal/ { printf "%.1f", $2 / 1048576 }' "$MEMINFO"; }
ram_verdict() {  # $1 GiB locked after start: ok, tight or over
  LC_ALL=C awk -v t="$(ram_total)" -v l="$1" -v g="$RAM_GROWTH" -v k="$RAM_KEEP" -v m="$RAM_KEEP_MIN" \
    'BEGIN { left = t - l - g + 0.001; print (left >= k ? "ok" : (left >= m ? "tight" : "over")) }'
}
ram_pool_max() {  # $1 GPUs, $2 checkpoint, $3 GiB to leave beside the server: the largest K/V pool in host RAM, in tokens
  local c
  c="$(ram_core "$1" "$2")"
  [ -n "$c" ] || return 0
  LC_ALL=C awk -v c="$c" -v g="$1" -v t="$(ram_total)" -v gr="$RAM_GROWTH" -v k="$3" \
    'BEGIN { p = (t - gr - k - c) * 1073741824 / (14144 * g); printf "%d", (p > 0 ? p : 0) }'
}

extra_pool() {  # the --kv-reserve-tokens passed after -- (the last one, as ft serve's parser keeps), if any
  local i v=""
  for ((i = 0; i < ${#extra[@]}; i++)); do
    case "${extra[$i]}" in
      --kv-reserve-tokens) v="${extra[$((i + 1))]:-}" ;;
      --kv-reserve-tokens=*) v="${extra[$i]#*=}" ;;
    esac
  done
  echo "$v"
}

conv_line() {  # --check: $2 conversations of 262k with the K/V in RAM for checkpoint $1, $3 a condition
  printf '        K/V in RAM: %s, ~%s GiB%s%s: xtx-xt-4x262k%s\n' \
    "$( [ "$2" = 1 ] && echo "1 conversation of 262k" || echo "$2 conversations of 262k at once")" \
    "$(gib "$(ram_locked 2 "$1" $(($2 * 262500)))")" \
    "$( case "$1:$2" in exl3-3.05:*|nvfp4:[123]) ;; *) echo " (untested)" ;; esac)" "$3" \
    "$( [ "$2" = 4 ] || echo " -- --kv-reserve-tokens $(($2 * 262500))")"
}

next_command() {  # the command --check suggests: the arguments given, without --check, with $1 / $2 / $3 = the profile
  # to use, a --ctx and a --kv-reserve-tokens to set (empty: keep)
  local a skip="" after="" pre=() post=() prof="$1"
  for a in "${orig_args[@]}"; do
    if [ -n "$after" ]; then post+=("$a"); continue; fi
    if [ -n "$skip" ]; then pre+=("$a"); skip=""; continue; fi
    case "$a" in
      --check) ;;
      --) after=1 ;;
      --model|--port|--host|--ctx|--gpus|--image|--name|--served-name|--memory) pre+=("$a"); skip=1 ;;
      -*) pre+=("$a") ;;
      *) prof="$a" ;;
    esac
  done
  pre=("$prof" "${pre[@]}")
  [ -n "$model" ] || pre+=(--model DIR)
  if [ -n "$2" ]; then  # drop a --ctx given, then set it
    local i keep=()
    for ((i = 0; i < ${#pre[@]}; i++)); do
      if [ "${pre[$i]}" = --ctx ]; then i=$((i + 1)); else keep+=("${pre[$i]}"); fi
    done
    pre=("${keep[@]}" --ctx "$2")
  fi
  if [ -n "$3" ]; then
    local i keep=()
    for ((i = 0; i < ${#post[@]}; i++)); do
      case "${post[$i]}" in
        --kv-reserve-tokens) i=$((i + 1)) ;;
        --kv-reserve-tokens=*) ;;
        *) keep+=("${post[$i]}") ;;
      esac
    done
    post=("${keep[@]}" --kv-reserve-tokens "$3")
  fi
  printf 'rdna3/serve.sh'
  for a in "${pre[@]}"; do printf ' %q' "$a"; done
  if [ "${#post[@]}" -gt 0 ]; then printf ' --'; for a in "${post[@]}"; do printf ' %q' "$a"; done; fi
  echo
}

check_machine() {  # --check: what this machine has, what is missing, which profile fits; exit status 1 if a start would fail
  local bad=0 v f n best="" prov commit tree head htree dirty arch total g kind running fits=0 base hk kvhost pool locked
  local p c m t ct mt want next_ctx="" next_pool=""
  ok() { printf '  ok    %s\n' "$*"; }
  warn() { printf '  warn  %s\n' "$*"; }
  fail() { printf '  FAIL  %s\n' "$*"; bad=1; }

  echo "System"
  if [ -e /dev/kfd ]; then ok "$(uname -sr), amdgpu compute driver present (/dev/kfd)"
  else fail "no /dev/kfd: the amdgpu kernel driver is not loaded (Linux only; nothing else to install on the host)"; fi
  if ! "$DOCKER" info >/dev/null 2>&1; then
    fail "cannot reach Docker: install it, start it, and add your user to the 'docker' group (then log in again)"
  else
    ok "Docker reachable"
    if "$DOCKER" image inspect "$image" >/dev/null 2>&1; then
      # the build records the commit it was made from and the tree of the files it copied (Dockerfile.rdna3)
      prov="$("$DOCKER" run --rm --entrypoint cat "$image" /opt/FreeToken-BUILD-PROVENANCE.txt 2>/dev/null || true)"
      commit="$(awk '$1 == "commit" { print $2 }' <<< "$prov")" tree="$(awk '$1 == "tree" { print $2 }' <<< "$prov")"
      head="$(git -C "$HERE/.." rev-parse HEAD 2>/dev/null || true)"
      htree="$(git -C "$HERE/.." rev-parse 'HEAD^{tree}' 2>/dev/null || true)"
      dirty="$(git -C "$HERE/.." status --porcelain 2>/dev/null | head -1)"
      if [ -z "$commit" ] || [ -z "$head" ]; then ok "image $image"
      elif [ "$commit" = "$head" ] && { [ -n "$dirty" ] || [ -z "$tree" ] || [ "$tree" = "$htree" ]; }; then
        # a dirty checkout's files have no tree to compare without writing one into .git: only say they differ
        ok "image $image, built from this checkout's commit (${head:0:7})"
        [ -z "$dirty" ] || printf '        the checkout has local changes: an image built before them does not run them\n'
      elif [ "$commit" = "$head" ]; then
        warn "image $image was built from commit ${head:0:7} with changes not in it: rebuild to run the commit"
        printf '        docker build -f Dockerfile.rdna3 -t %s .\n' "$image"
      elif [ -n "$tree" ] && [ "$tree" = "$htree" ] && [ -z "$dirty" ]; then
        ok "image $image, built from the files of this checkout's commit (${head:0:7})"
      else
        warn "image $image was built from commit ${commit:0:7}, this checkout is at ${head:0:7}: rebuild to run it"
        printf '        docker build -f Dockerfile.rdna3 -t %s .\n' "$image"
      fi
      arch="$("$DOCKER" image inspect "$image" --format '{{range .Config.Env}}{{println .}}{{end}}' |
              sed -n 's/^FREETOKEN_ROCM_ARCH=//p')"
    else
      fail "no image $image: build it (once, ~25 GB download):"
      printf '        docker build -f Dockerfile.rdna3 -t %s .\n' "$image"
    fi
  fi
  [ -n "$(getent group video)" ] && [ -n "$(getent group render)" ] ||
    fail "the 'video' and 'render' groups are missing: they own /dev/kfd and /dev/dri (amdgpu driver)"

  echo "GPUs"
  mapfile -t found < <(list_gpus)
  [ "${#found[@]}" -gt 0 ] || fail "no GPU in the KFD topology"
  for v in "${found[@]}"; do
    LC_ALL=C awk -v c="$(card_name "$v")" '{ printf "        HIP %s  %-20s %5.1f GiB  PCI %s\n", $1, c, $2 / 1073741824, $3 }' <<< "$v"
    g="$(cut -d' ' -f4 <<< "$v")"
    # cards under the smallest profile's 16 GB (an integrated GPU) are never picked: their arch does not matter
    if [ -n "${arch:-}" ] && [ "$g" != "$arch" ] &&
       LC_ALL=C awk -v b="$(cut -d' ' -f2 <<< "$v")" 'BEGIN { exit !(b >= 16 * 1073741824 * 0.97) }'; then
      warn "HIP $(cut -d' ' -f1 <<< "$v") is $g, the image is built for $arch: rebuild with --build-arg GPU_ARCH=$g (untested)"
    fi
  done
  # the profiles these GPUs can run, the most specific first: more GPUs, more VRAM asked for, measured, shortest name
  mapfile -t fit < <(for f in "$HERE"/profiles/*.env; do
    profile_fits "$f" || continue
    n="$(basename "$f" .env)"
    printf '%s %s %s %s %s\n' "$(pval "$f" GPUS 1)" "$(pval "$f" VRAM_GIB 0 | awk '{ for (i = 1; i <= NF; i++) s += $i; print s + 0 }')" \
      "$(pval "$f" TESTED 1)" "${#n}" "$n"
  done | sort -k1,1nr -k2,2nr -k3,3nr -k4,4n | cut -d' ' -f5)
  if [ "${#fit[@]}" -eq 0 ]; then
    fail "no profile fits these GPUs (rdna3/serve.sh --list; docs/rdna3/profiles.md#adapting-a-profile-to-other-cards-untested)"
  else
    best="${profile:-${fit[0]}}"
    ok "profiles for these GPUs: $(for n in "${fit[@]}"; do
         [ "$(pval "$HERE/profiles/$n.env" TESTED 1)" = 0 ] && printf '%s (untested)\n' "$n" || printf '%s\n' "$n"; done |
         paste -sd, | sed 's/,/, /g')"
    if [ -n "$profile" ]; then
      [ -f "$HERE/profiles/$profile.env" ] || { fail "no profile '$profile'"; best=""; }
      [ -z "$best" ] || profile_fits "$HERE/profiles/$profile.env" ||
        fail "profile $profile needs $(pval "$HERE/profiles/$profile.env" GPUS 1) GPU(s) with $(pval "$HERE/profiles/$profile.env" VRAM_GIB 0) GiB"
    else
      printf '        suggested: %s\n' "$best"
    fi
  fi

  total="$(ram_total)"
  echo "RAM: $total GiB. A server locks its RAM; beside it, keep ~$RAM_KEEP GiB for the system and your programs" \
       "($RAM_KEEP_MIN at the least; docs/rdna3/limits.md#how-much-ram-do-i-need)"
  kind=""
  [ -z "$model" ] || kind="$(ckpt_kind "$model")"
  [ -z "$kind" ] || [ -n "$(ram_core 2 "$kind")" ] || warn "$(ckpt_label "$kind"): its RAM use is not known here"
  # with a checkpoint given and no profile asked for: the most specific profile whose RAM fits (two cards, else one)
  if [ -z "$profile" ] && [ -n "$kind" ] && [ -n "$(ram_core 2 "$kind")" ] && [ "${#fit[@]}" -gt 0 ]; then
    for n in "${fit[@]}"; do
      case "$n" in *-4x262k|*-1m) continue ;; esac
      [ "$n" = "$best" ] || [ "$(pval "$HERE/profiles/$n.env" TESTED 1)" = 1 ] || continue
      locked="$(ram_locked "$(pval "$HERE/profiles/$n.env" GPUS 1)" "$kind" 0 \
                "$(pval "$HERE/profiles/$n.env" FREETOKEN_HOST_KV_TOKENS 262144)")"
      if [ "$(ram_verdict "$locked")" != over ]; then
        [ "$n" = "$best" ] || printf '        for this RAM with %s: %s\n' "$(ckpt_label "$kind")" "$n"
        best="$n"; break
      fi
    done
  fi
  base="${best:-xtx-xt}"
  case "$base" in *-4x262k|*-1m) base=xtx-xt ;; esac   # the same cards with the K/V in VRAM
  g="$(pval "$HERE/profiles/$base.env" GPUS 1)" hk="$(pval "$HERE/profiles/$base.env" FREETOKEN_HOST_KV_TOKENS 262144)"
  kvhost=""
  profile_fits "$HERE/profiles/xtx-xt-4x262k.env" && kvhost=1
  pool="$(extra_pool)"
  for v in nvfp4 exl3-3.05 exl3-4.05; do
    [ -z "$kind" ] || [ "$kind" = "$v" ] || continue
    locked="$(ram_locked "$g" "$v" 0 "$hk")" n=""
    [ "$(pval "$HERE/profiles/$base.env" TESTED 1)" = 1 ] || n=" (estimated)"
    case "$(ram_verdict "$locked")" in
      ok) ok "$(ckpt_label "$v"), $base (K/V in VRAM): locks ~$(gib "$locked") GiB$n"; fits=1 ;;
      tight) warn "$(ckpt_label "$v"), $base (K/V in VRAM): locks ~$(gib "$locked") GiB$n: fits with little else running"; fits=1 ;;
      *) if [ "$best" = "$base" ] && [ -n "$kind" ]; then
           fail "$(ckpt_label "$v"), $base: locks ~$(gib "$locked") GiB$n, more than this machine can hold"
         else warn "$(ckpt_label "$v"), $base (K/V in VRAM): locks ~$(gib "$locked") GiB$n, more than this machine can hold"; fi ;;
    esac
    [ -n "$kvhost" ] && [ "$g" = 2 ] || continue
    # the K/V in host RAM: as many 262k conversations (262,500 tokens of pool each, at most 4) as leave RAM_KEEP GiB,
    # one more if it still leaves RAM_KEEP_MIN; or the longest single conversation (at most 1M)
    p="$(ram_pool_max 2 "$v" "$RAM_KEEP")" c=0 m=0 ct=0 mt=0
    c=$((p / 262500)) m=$((p / 1000 * 1000))
    p="$(ram_pool_max 2 "$v" "$RAM_KEEP_MIN")"
    ct=$((p / 262500)) mt=$((p / 1000 * 1000))
    [ "$c" -le 4 ] || c=4; [ "$ct" -le 4 ] || ct=4
    [ "$m" -le 1000000 ] || m=1000000; [ "$mt" -le 1000000 ] || mt=1000000
    [ "$c" -lt 1 ] || conv_line "$v" "$c" ""
    [ "$ct" -le "$c" ] || conv_line "$v" "$ct" ", with little else running"
    [ "$ct" -ge 1 ] || printf '        K/V in RAM: not one 262k conversation fits\n'
    if [ "$mt" -gt 262144 ] && profile_fits "$HERE/profiles/xtx-xt-1m.env"; then
      t="$m"; [ "$m" -gt 262144 ] || t="$mt"
      printf '        or one conversation of up to %s tokens, ~%s GiB%s%s: xtx-xt-1m%s\n' \
        "$( [ "$t" = 1000000 ] && echo 1M || echo "$((t / 1000))k")" "$(gib "$(ram_locked 2 "$v" "$t")")" \
        "$( [ "$v" = exl3-3.05 ] || echo " (untested)")" "$( [ "$t" = "$m" ] || echo ", with little else running")" \
        "$( [ "$t" = 1000000 ] || echo " --ctx $t")"
    fi
    # the K/V-in-RAM profile asked for, with the checkpoint given: does its pool fit?
    [ -n "$kind" ] || continue
    case "$best" in
      *-4x262k)
        want="${pool:-$(pval "$HERE/profiles/$best.env" KV_TOKENS 1050000)}" locked="$(ram_locked 2 "$v" "$want")"
        case "$(ram_verdict "$locked")" in
          tight) warn "$best with $(ckpt_label "$v"): its $want-token K/V pool locks ~$(gib "$locked") GiB: fits with little else running" ;;
          over)
            if [ "$ct" -ge 1 ]; then
              warn "$best with $(ckpt_label "$v"): its $want-token K/V pool locks ~$(gib "$locked") GiB, more than this machine can hold"
              next_pool=$(( (c >= 1 ? c : ct) * 262500 ))
            else fail "$best with $(ckpt_label "$v"): not one 262k conversation fits in RAM"; fi ;;
        esac ;;
      *-1m)
        want="${pool:-${ctx:-$(pval "$HERE/profiles/$best.env" CTX 1000000)}}" locked="$(ram_locked 2 "$v" "$want")"
        case "$(ram_verdict "$locked")" in
          tight) warn "$best with $(ckpt_label "$v"): a $want-token context locks ~$(gib "$locked") GiB: fits with little else running" ;;
          over)
            if [ "$mt" -gt 262144 ]; then
              warn "$best with $(ckpt_label "$v"): a $want-token context locks ~$(gib "$locked") GiB, more than this machine can hold"
              next_ctx=$(( m > 262144 ? m : mt ))
            else fail "$best with $(ckpt_label "$v"): no context past 262k fits in RAM"; fi ;;
        esac ;;
    esac
  done
  if [ -z "$kind" ] && [ "$fits" = 0 ]; then
    fail "no checkpoint fits this machine's RAM (docs/rdna3/limits.md#how-much-ram-do-i-need)"
  fi
  # the running containers with /dev/kfd: a server of any image (an older one after a rebuild) holds the GPUs
  running=""
  "$DOCKER" info >/dev/null 2>&1 &&
    running="$("$DOCKER" ps -q 2>/dev/null |
               xargs -r "$DOCKER" inspect --format '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' 2>/dev/null |
               awk '/ \/dev\/kfd / { sub(/^\//, "", $1); print $1 }' | paste -sd, || true)"
  if [ -n "$running" ]; then
    warn "a container with the GPUs is running ($running): a server holds the GPUs and its RAM until it stops"
  else
    n="$(awk '/^MemAvailable/ { printf "%d", $2 / 1048576 }' "$MEMINFO")"
    [ "$(LC_ALL=C awk -v t="$total" -v a="$n" 'BEGIN { printf "%d", t - a }')" -le 15 ] ||
      warn "only $n GiB available now: other programs hold $(LC_ALL=C awk -v t="$total" -v a="$n" 'BEGIN { printf "%d", t - a }') GiB (the desktop and system took ~15 on the reference machine)"
  fi

  echo "Host settings"
  if [ "$(cat /proc/sys/vm/compact_unevictable_allowed 2>/dev/null)" = 0 ] &&
     [ "$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null)" = 0 ]; then
    ok "memory compaction kept off locked memory"
  else
    warn "memory compaction can stop the GPUs for 5-22 s while serving; to fix it, once:"
    printf '%s\n' "        printf 'vm.compaction_proactiveness=0\\nvm.compact_unevictable_allowed=0\\n' | sudo tee /etc/sysctl.d/99-llm-gpu.conf && sudo sysctl --system"
  fi
  for v in "${found[@]}"; do
    f="/sys/bus/pci/devices/$(cut -d' ' -f3 <<< "$v")/pp_power_profile_mode"
    [ -r "$f" ] || continue
    if grep -q 'COMPUTE *\*' "$f"; then ok "HIP $(cut -d' ' -f1 <<< "$v"): COMPUTE power profile"
    else
      warn "HIP $(cut -d' ' -f1 <<< "$v"): not on the COMPUTE power profile: the first request after a pause waits ~10 s; until reboot:"
      printf '        echo 5 | sudo tee %s\n' "$f"
    fi
  done
  if [ -n "$(ss -ltnH "sport = :$port" 2>/dev/null)" ]; then
    warn "port $port is already in use (a server running? docker ps) - stop it or pick --port"
  fi

  if [ -n "$model" ]; then
    echo "Model: $model"
    if [ ! -f "$model/config.json" ]; then
      fail "no config.json: not a checkpoint directory, or the download has not started"
    else
      case "$kind" in
        nvfp4) ok "Qwen3.8-Flash-Next NVFP4 checkpoint" ;;
        exl3-*) ok "EXL3 checkpoint, ${kind#exl3-} bits per weight" ;;
        *) warn "not one of the checkpoints the profiles are tuned for (NVFP4, EXL3)" ;;
      esac
      grep -q '"qwen4_exp"' "$model/config.json" || warn "model_type is not qwen4_exp (Qwen3.8-Flash-Next): untested here"
      # every weight file the checkpoint's index names must be there, none half-written
      n=0
      [ ! -f "$model/model.safetensors.index.json" ] ||
        n="$( { grep -o '"[^"]*\.safetensors"' "$model/model.safetensors.index.json" || true; } | tr -d '"' | sort -u |
             while read -r f; do [ -f "$model/$f" ] || echo "$f"; done | wc -l)"
      if [ ! -f "$model/model.safetensors.index.json" ]; then
        fail "no model.safetensors.index.json: the download is not finished, run it again"
      elif [ "$n" != 0 ] || [ -n "$(find "$model/.cache/huggingface/download" -name '*.incomplete' 2>/dev/null | head -1)" ]; then
        fail "download not finished ($n weight files missing): run the download again"
      fi
    fi
  fi

  echo
  if [ "$bad" = 0 ] && [ -n "$best" ]; then
    echo "Next: $(next_command "$best" "$next_ctx" "$next_pool")"
  elif [ "$bad" != 0 ]; then
    echo "Fix the FAIL lines first (docs/rdna3/getting-started.md)."
  fi
  return "$bad"
}

need() { [ $# -ge 2 ] || { echo "option $1 needs a value (see --help)" >&2; exit 2; }; }
orig_args=("$@")
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
    --no-mtp) no_mtp=1; shift ;;
    --hybrid) hybrid=1; shift ;;
    --check) check=1; shift ;;
    --list)
      printf '%-14s %-4s %9s %9s  %-8s %s\n' profile GPUs context "K/V pool" status description
      for f in "$HERE"/profiles/*.env; do
        awk -v n="$(basename "$f" .env)" -F= '
          NR == 1 { sub(/^# /, ""); d = $0 }
          /^(GPUS|CTX|KV_TOKENS|TESTED)=/ { v = $2; sub(/[ \t]*#.*/, "", v); k[$1] = v }
          END { c = k["CTX"] ? k["CTX"] : 131072; s = (k["TESTED"] == "0") ? "untested" : "measured"
                printf "%-14s %-4s %9s %9s  %-8s %s\n", n, k["GPUS"] ? k["GPUS"] : 1, c, k["KV_TOKENS"] ? k["KV_TOKENS"] : c, s, d }' "$f"
      done
      exit 0 ;;
    --list-gpus) while read -r v; do
        LC_ALL=C awk -v c="$(card_name "$v")" '{ printf "HIP %s  PCI %s  %.1f GiB VRAM  %s\n", $1, $3, $2 / 1073741824, c }' <<< "$v"
      done < <(list_gpus); exit 0 ;;
    -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
    --) shift; extra=("$@"); break ;;
    -*) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    *) [ -z "$profile" ] || { echo "one profile only" >&2; exit 2; }; profile="$1"; shift ;;
  esac
done
if [ -n "$check" ]; then check_machine; exit $?; fi
[ -n "$profile" ] || { echo "usage: rdna3/serve.sh <profile> --model DIR   (rdna3/serve.sh --check suggests a profile for this machine)" >&2; exit 2; }
pfile="$HERE/profiles/$profile.env"
[ -f "$pfile" ] || { echo "no profile '$profile' (rdna3/serve.sh --list)" >&2; exit 2; }
[ -d "$model" ] || { echo "--model must be the checkpoint directory" >&2; exit 2; }
model="$(cd "$model" && pwd)"

# the profile: GPUS / VRAM_GIB / CTX / KV_TOKENS / MEMORY / TESTED / FT_ARGS drive this script, every other KEY=value
# is container env
envs=() ft_args=() p_gpus=1 p_vram="" p_ctx=131072 p_kv="" p_memory=110g p_tested=1
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
    KV_TOKENS) p_kv="$val" ;;
    MEMORY) p_memory="$val" ;;
    FT_ARGS) read -r -a ft_args <<< "$val" ;;
    FREETOKEN_MTP|FREETOKEN_SPEC_*) [ -n "$no_mtp" ] || envs+=(-e "$key=$val") ;;  # --no-mtp drops the head's settings
    *) envs+=(-e "$key=$val") ;;
  esac
done < "$pfile"
user_ctx="$ctx" ctx="${ctx:-$p_ctx}" memory="${memory:-$p_memory}"
# the K/V pool the requests share: the profile's KV_TOKENS, never less than one request's context
kv="${p_kv:-$ctx}"
[ "$kv" -ge "$ctx" ] || kv="$ctx"
[ "$p_tested" = 1 ] || echo "note: profile $profile is UNTESTED (derived from measured profiles, see its header); please report what you measure" >&2

# EXL3 checkpoints on two cards: the expert slices round to 384 / 256, so the card holding 384 bounds the
# expert cache, and at the profiles' 0.80 the 7900 XTX peaked 200 MiB under full VRAM (two agents at 50k tokens: 24,341
# of 24,560 MiB at 0.78 with 4.05 bpw); 0.77 / 0.76 keep ~0.5 GiB free (xtx-xt). `-- --memory-ratio X` wins.
cfg="$(tr -d ' \t\n' < "$model/config.json" 2>/dev/null || true)"
if [[ "$cfg" == *'"quant_method":"exl3"'* ]] && [ "$p_gpus" = 2 ]; then
  bits="$(grep -o '"bits":[0-9.]*' <<< "$cfg" | head -1 | cut -d: -f2)"
  ratio=0.77
  awk -v b="${bits:-0}" 'BEGIN { exit !(b >= 4) }' && ratio=0.76
  ft_args+=(--memory-ratio "$ratio")
  echo "note: EXL3 checkpoint (${bits:-?} bpw) on two cards: --memory-ratio $ratio instead of the profile's ('-- --memory-ratio X' overrides)" >&2
fi

# --hybrid (docs/rdna3/cpu-experts.md): the split measured best on the Threadripper 3970X + PCIe 4.0: for NVFP4 a quarter
# of each step's misses fetched on one card (as `ft bench bw` picks), 40 % on two, where each rank's pool has half the
# cores and the RAM's bandwidth but its own PCIe link; 60 % for EXL3, whose experts cost the CPU more to decode
if [ -n "$hybrid" ]; then
  ft_args+=(--moe-strategy hybrid)
  if [[ "$cfg" == *'"quant_method":"exl3"'* ]]; then
    envs+=(-e FREETOKEN_HYBRID_FETCH_FRACTION=0.6)
    [ "$p_gpus" = 1 ] || echo "note: --hybrid with EXL3 experts on two cards measured even to 2-4 % slower than without (docs/rdna3/cpu-experts.md)" >&2
  elif [ "$p_gpus" = 1 ]; then
    envs+=(-e FREETOKEN_HYBRID_FETCH_FRACTION=0.25)
  else
    envs+=(-e FREETOKEN_HYBRID_FETCH_FRACTION=0.4)
  fi
fi

# the RAM this server will lock (ram_locked) against what the machine can spare: a warning, `--check` says what fits
kind="$(ckpt_kind "$model")" kvpool=0 tier_tokens=262144
for e in "${envs[@]}"; do
  case "$e" in
    FREETOKEN_QSA_KV_HOST=1) kvpool="$(extra_pool)"; kvpool="${kvpool:-$kv}" ;;
    FREETOKEN_HOST_KV=0) tier_tokens=0 ;;
    FREETOKEN_HOST_KV_TOKENS=*) [ "$tier_tokens" = 0 ] || tier_tokens="${e#*=}" ;;
  esac
done
locked="$(ram_locked "$p_gpus" "$kind" "$kvpool" "$tier_tokens")"
if [ -n "$locked" ]; then
  chk=(rdna3/serve.sh --check --model "$model")  # the check of this very server: its --ctx and K/V pool
  [ -z "$user_ctx" ] || chk+=(--ctx "$user_ctx")
  chk+=("$profile")
  [ -z "$(extra_pool)" ] || chk+=(-- --kv-reserve-tokens "$(extra_pool)")
  chk="$(printf '%q ' "${chk[@]}")" chk="${chk% }"
  case "$(ram_verdict "$locked")" in
    over) echo "warning: this server will lock ~$(gib "$locked") GiB of RAM, more than this $(ram_total) GiB machine can hold" \
               "beside the system: it may run out of memory; '$chk' shows what fits" >&2 ;;
    tight) echo "note: this server will lock ~$(gib "$locked") GiB of RAM: on this $(ram_total) GiB machine that fits with" \
                "little else running ('$chk')" >&2 ;;
  esac
fi

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
if [ -z "$dry" ] && ! "$DOCKER" info >/dev/null 2>&1; then
  echo "cannot reach the Docker daemon: is it running, and is your user in the 'docker' group (log in again after adding it)?" >&2
  exit 2
fi
# rootless or not (a Docker-compatible CLI without SecurityOptions counts as not), for --dry-run too
security="$("$DOCKER" info --format '{{json .SecurityOptions}}' 2>/dev/null || true)"
have_image=""
if image_id="$("$DOCKER" image inspect "$image" --format '{{.Id}}' 2>/dev/null)"; then
  short="${image_id#sha256:}" short="${short:0:12}" have_image=1
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
# the JIT modules in that volume too (older images build them under ~/.cache/tvm-ffi, lost at each start)
envs+=(-e TVM_FFI_CACHE_DIR=/root/.cache/freetoken-rdna3/tvm-ffi)

# an image built before --default-temperature (an older one picked with --image) would refuse it: passed when known
# (no image yet, under --dry-run: docker would try to pull the tag)
temp_args=(--default-temperature 0.6)
if [ -n "$have_image" ] && ! "$DOCKER" run --rm --entrypoint grep "$image_id" -q -e --default-temperature \
    /opt/FreeToken/python/freetoken/server/args.py >/dev/null 2>&1; then
  temp_args=()
  echo "note: image $image predates --default-temperature: requests without a temperature get the checkpoint's" >&2
fi

# expandable segments (the engine also sets them when PYTORCH_ALLOC_CONF is unset): without them the allocator's
# cache filled the VRAM left free and every 14-16k-token prefill chunk past ~26k of context failed once and retried
# (xtx-xt; none with them, same speed)
# --oom-score-adj: the engine is the host's largest process by far, so when the desktop fills the rest of the RAM the
# kernel's OOM killer took one TP rank and left the other holding its GPU; -999 makes it pick the other processes first.
# Left out with rootless Docker, which cannot lower a score (that takes CAP_SYS_RESOURCE)
oom_args=(--oom-score-adj -999)
[[ "${security:-}" == *rootless* ]] && oom_args=()
cmd=("$DOCKER" run --rm --init --name "$name" --network host --ipc=host "${gpu_flags[@]}"
     --ulimit memlock=-1 --memory "$memory" "${oom_args[@]}"
     -e PYTORCH_ALLOC_CONF=expandable_segments:True -e OMP_WAIT_POLICY=PASSIVE -e "HIP_VISIBLE_DEVICES=$gpus"
     "${envs[@]}" "${mounts[@]}"
     "$image_id" ft serve --model /models/m --host "$host" --port "$port" --served-model-name "$served" "${mm_args[@]}"
     --max-seq-len-override "$ctx" --kv-reserve-tokens "$kv" "${temp_args[@]}" "${ft_args[@]}" "${extra[@]}")
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
