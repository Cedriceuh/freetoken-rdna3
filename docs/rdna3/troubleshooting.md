# Troubleshooting

First run `rdna3/serve.sh --check` (with `--model DIR` and your profile): it catches a missing driver, Docker
permissions, an image built from another commit than the checkout, too little free RAM, a server already running and the host settings
below, and prints the fix for each.

## The server takes minutes to start

Normal: ~2.5 minutes on the reference machine to read ~68 GB of experts into RAM and build the plan. The first start
of a new image also compiles and autotunes GPU kernels (a few more minutes); results are cached in the
`freetoken-rdna3-kcache-<image id>` Docker volume, so it happens once per image. `/health` answers `"loading"` until
the weights are in.

## Slow decode right after starting a new image

Kernels still compiling or autotuning on first use (a cold cache once made a 52 tok/s build measure 23.9; on a fresh
image the first long prompt and the first reuse of a cached conversation are also slower once). Send a few requests, then
measure. If it stays slow, check that the kernel-cache volume is mounted (`serve.sh` does it).

## My desktop session closed, or the screen went black

A GPU kernel faulted hard on the card that drives your monitor: the driver resets that card and the desktop dies with
it. Check with `journalctl -k | grep -i "device coredump"`. With this build's profiles this should not happen: report
the fault.

## `profile ... needs N GPU(s)` or the wrong card is picked

`rdna3/serve.sh --list-gpus` shows what HIP sees (index, PCI address, VRAM, model). Two-card profiles put the largest
card first; one-card profiles take the smallest card with enough VRAM. Force a choice with `--gpus 1,0` (rank 0
first).

## Out of memory at start-up or on long prompts

- On one card, keep 8k prefill chunks (`--max-prefill-length 8192`): with 16k chunks a 60k-token prompt took 166 s
  and decode fell to 1.9 tok/s (the allocator thrashes).
- Lower the context (`--ctx 131072` on two cards, `--ctx 65536` on one), then the memory ratio
  (`-- --memory-ratio 0.75`).
- On ROCm the torch allocator is capped 512 MiB short of the card (`FREETOKEN_VRAM_MARGIN_MIB`; the log's `VRAM at
  start` and `torch allocator capped at` lines give the numbers), so a peak over it ends in `torch.OutOfMemoryError`
  with a traceback that says where. Without a cap, a peak above the card's VRAM does not always fail an allocation: the
  driver evicts memory and stops the process's GPU queues for seconds, and a rank has died on an illegal memory access
  that way. `/sys/class/drm/card*/device/mem_info_vram_used`
  sampled during the first long prompt shows how close the peak comes; the card driving the display also holds the
  desktop's buffers. On `xtx-xt`, `--memory-ratio` 0.80 -> 0.78 lowered the 7900 XTX's peak by 350 MiB.
- The container needs `--ulimit memlock=-1` (pinned RAM for the experts and the RAM tier) and enough RAM under its
  `--memory` limit (the profile's `MEMORY`: 110g, 104g with the K/V in RAM); an OOM kill shows as exit code 137.

## Long pauses in the middle of a run

Two host settings, measured on the reference machine ([getting-started.md](getting-started.md#2-two-host-settings)
gives the commands):

- **Memory compaction.** The GPU driver maps the host memory the engine registers (the expert banks, the RAM tier, the
  runtime's pinned buffers) without pinning its pages. When the kernel compacts memory and moves one of them, the driver
  stops every GPU queue of the process while it revalidates the ranges: 5-22 s, on both cards (`echo 1 >
  /proc/sys/vm/compact_memory` reproduces it). Turning off proactive compaction is not enough: with RAM this full, the
  kernel's background reclaim wakes the compaction daemon on its own (three stops in nine minutes of benchmark, one cold
  read taking 17 s instead of 4; once 140 s, and the server died on a collective timeout). The engine locks its memory
  (`FREETOKEN_MLOCK`, on by default; it needs `--ulimit memlock=-1`, which `serve.sh` passes); this setting keeps
  compaction off locked pages:

  ```bash
  printf 'vm.compaction_proactiveness=0\nvm.compact_unevictable_allowed=0\n' | sudo tee /etc/sysctl.d/99-llm-gpu.conf
  sudo sysctl --system
  ```

  With it, on `xtx-xt`: the compaction daemon still ran (39 times) but moved none of the engine's pages, and a forced
  compaction during decoding moved 45k other pages with no stopped queue (longest gap between two tokens 75 ms). Until
  it is set, the server logs a warning at start. The lock comes once the model is loaded: heavy memory use on the host
  during the load (another big process starting) can still stop the GPUs for seconds then, which only delays the
  start. The driver's own count is `evicted_ms` in `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu id>/`.
- **The VRAM clock after a pause.** With the default power profile the memory clock stays at 96 MHz for ~10 s after
  the GPU wakes up: the first request after 15 s or more of idle waited 9-13 s for its first token. The `COMPUTE` power
  profile raises it at once (1.4-1.6 s after each of 16 pauses of 20-90 s) and still lets it drop when idle (13-27 W per
  card); pinning the clock at its maximum works too but costs ~40 W per card at idle. As root, on every card (LACT's
  power profile setting keeps it across reboots):

  ```bash
  echo 5 | sudo tee /sys/class/drm/card*/device/pp_power_profile_mode   # 5 = COMPUTE in that file's list
  ```

The first request after a start takes 1.3-1.9 s: the JIT modules come from the kernel-cache volume
(`TVM_FFI_CACHE_DIR`). On the first start of a new image it took 28.1-28.4 s, while the kernels compiled once. Undervolting,
GFXOFF and the display's memory power saving were ruled out as causes of pauses.

## A GPU fault or a hung collective with a full card

Seen a few times on `xtx-xt`, mostly while a card was within a few hundred MiB of full and the driver had just evicted the
process's queues (`/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms` growing outside start-up):

- `CUDA error: an illegal memory access` on rank 0, with `[gfxhub] page fault` for the 7900 XTX in `dmesg`: once in an
  agent session with the EXL3 4.05 bpw checkpoint (the XTX peaks ~0.5 GiB from full), and once on purpose, when another
  process took and released 0.5-1.5 GiB on the XTX (the card driving the display) during the first 16k-token chunk of a
  long prompt.
- A hung all-gather (`Watchdog caught collective operation timeout ... _ALLGATHER_BASE`, the server stops) once in a
  depth sweep, the XT at 37 MiB from full during a 16k-token chunk at ~41k tokens of context.
- Profiling with two cards (rocprofv3 `--kernel-trace`, or torch.profiler around a prefill): RCCL's all-reduce of a
  prefill chunk hung (60 s watchdog) twice out of two; without a profiler the same load ran (not tried: one card, or
  profiling decode only).
- The same page fault once at start-up, ~3 s after the CUDA graphs were captured, with 3.9 GiB still free on the XTX:
  VRAM pressure does not explain that one. The 16 starts that followed were clean; restarting is enough.
- On purpose, with `--memory-ratio 0.83` and two agents at ~120k tokens, before the prefill peak reductions
  ([journey.md](journey.md#21-prefill-peaks-spills-and-the-vram-outside-torch-2026-10-07)): a page
  fault on the XT at 20420 of 20464 MiB, on the first try. With the torch allocator capped
  (`FREETOKEN_VRAM_MARGIN_MIB`) the same load gave an out-of-memory error instead.

What fills a card besides torch (logged at start as `outside torch`): ~1 GiB, growing by 200-250 MiB while serving,
plus, on the card driving the display, the desktop's buffers (0.66 to 1.8 GiB on the 7900 XTX here), which the free
memory HIP reports leaves out (`/sys/class/drm/card*/device/mem_info_vram_used` counts them; the cap reads it). Each
Triton kernel module the HIP runtime loads takes 2 MiB of VRAM whatever its size, and new shapes load new ones; kernels
that spill registers take scratch per GPU queue; RCCL ~8 MiB. The cap is set at start, so that growth, and the
desktop's, eats into its 512 MiB.

Neither came back in the same runs repeated (hours of agent sessions, the sweep again, the same VRAM pressure in other
10-minute runs): the eviction has to land at a bad moment. Keep other GPU-heavy
programs off the card that drives the display while serving, and lower `--memory-ratio` (more headroom, fewer cached
experts) if it happens again. Before restarting, save the log (`docker logs <container>`): the container is removed
when it stops.

## The server stops answering and one card's VRAM empties

The host ran out of RAM and the kernel's OOM killer stopped one rank: `journalctl -k | grep -i "killed process"` names a
`python3`. The engine locks its experts and its K/V pool in RAM (~95 GiB for the two ranks with NVFP4 and a pool of
787,500 tokens), so the desktop and every other program share what is left. `serve.sh` starts the container with
`--oom-score-adj -999` (not with rootless Docker, which cannot lower it): the kernel kills other processes first, and
the engine only when nothing else is left (the container's own `--memory` limit, for instance). When a rank dies, the
API server stops the surviving ranks (SIGKILL after 5 s) and exits with code 1, streams still open included, so a
service with `Restart=on-failure` starts it again: with a rank 0 killed under a stream on `xtx-xt-4x262k`, the container
had exited 37 s later and both cards were free. Close what holds the RAM (`ps -eo rss,comm --sort=-rss | head`) or lower
the K/V kept in RAM (`--kv-reserve-tokens`). The kernel's report (`journalctl -k`, the table after "Tasks state") lists
every process at that moment: on the reference machine each out-of-memory event came right after 63-66 Node.js
workers started at once (12-17 GiB, within seconds, so a memory monitor never shows them), one of them under
`npm exec jest`, whose default pool is one worker per core but one; cap such pools (Jest: `maxWorkers`) on a machine
that serves, or run the tool that starts them under a memory limit (`systemd-run --user --scope -p MemoryMax=16G ...`).

## Less VRAM, or a start that never ends, after a server was killed while starting

Seen once: a server removed with `docker rm -f` while it loaded its weights left an amdgpu kernel worker blocked in
`svm_range_restore_work` (`journalctl -k`: `task kworker/... blocked for more than ... seconds`, waiting on a fence in
`svm_range_map_to_gpus`), and the dead process's VRAM held: `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu>` listed a pid
with no `/proc` entry and 4.4 GiB on the 7900 XTX. The next servers got that much less (fewer expert slots, slower
decode, an out-of-memory error in a prefill), and a later one never came up. Only a reboot frees it. Stop a server
with `docker stop -t 60` (or its service) and let it exit; do not kill one during start-up.

## Answers stop mid-sentence

Deep in a long conversation (70-110k tokens of context) the model sometimes ends an answer in the middle of a sentence,
or of a word, and the later answers of that conversation then often stop early too (the model follows its own history).
Measured on `xtx-xt`, turns on this repository's code with thinking off and temperature 0.6: 4 of 90 conversations
with the MTP head, 2 of 28 without it. At the stop the end-of-text token had 26-75 % of the probability (first or
second candidate), so no sampling setting avoids it: the model card's non-thinking settings (temperature 0.7, top-p
0.8) give it 29-83 % at the same places, its thinking ones (1.0, top-p 0.95) 35-56 %. Replayed token for token, one
stop kept a large end-of-text probability from a fresh prefill as from the prefix cache, with
`FREETOKEN_QSA_TORCH_TOPK=1` and with `FREETOKEN_INT8_DENSE=0`: 34-98 % depending on the path (that sensitive to
rounding). So it is not the MTP head, the prefix cache or the RAM tier, the attention's block selection kernel, the
int8 dense layers or the sampler (320k draws checked, and the speculative sampling's own tests); whether the
unquantized model does the same was not checked. With thinking on and the reasoning kept in the history (temperature
0.6), 15 conversations and 90 turns of ~1800 tokens had no such stop, too few to say it does not happen there. When it
happens, ask the agent to go on, or start a new conversation.

## Sampling hangs

Upstream's cooperative top-k / top-p kernels hang on gfx1100; this build never uses them on ROCm. If you run a
different build, test it on a card that drives no display: `HIP_VISIBLE_DEVICES=<index>
rdna3/tests/run_sampling_tests.sh freetoken.kernel.triton.sampling`, with the index `rdna3/serve.sh --list-gpus` shows
for that card (GPU 0 otherwise; each mode is killed after 90 s). Always smoke-test with a temperature above 0: greedy
decoding never reaches the sampler.

## `FREETOKEN_NVFP4_DECODE_TUNED: no tuned tiles for (N, K) = ...` in the log

Expected with a split or card combination other than `xtx-xt` (one card included): those shapes use the default
tiles. Harmless.

## `TP ranks sampled different tokens in ... row(s)` in the log

Only printed with `FREETOKEN_TP_SYNC_TOKENS=check`. Rank 0's tokens are what the client receives either way; please
report the lines with your setup.

## `image input is disabled on this server (--text-model-only)`

The server was started without `--vision`: `rdna3/serve.sh <profile> --model DIR --vision`. On one card, `--vision`
costs ~1.5-2 % of decode even without images ([limits.md](limits.md#images-vision)).

## Reporting a problem

Open an issue with: the output of `rdna3/serve.sh --check --model DIR <profile>`, the profile and any changed option
(`rdna3/serve.sh ... --dry-run` prints everything), the host kernel version, the image's commit (`docker run --rm
--entrypoint cat freetoken-rdna3:latest /opt/FreeToken-BUILD-PROVENANCE.txt`), and the server log (`docker logs
freetoken-rdna3`).
