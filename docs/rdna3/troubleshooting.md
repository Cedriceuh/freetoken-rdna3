# Troubleshooting

## The server takes minutes to start

Normal: ~2.5 minutes on the reference machine to read ~68 GB of experts into RAM and build the plan. The first start of a new image also compiles and
autotunes GPU kernels (a few more minutes); results are cached in the `freetoken-rdna3-kcache-<image id>` Docker
volume, so it happens once per image. `/health` answers `"loading"` until the weights are in.

## Slow decode right after starting a new image

Kernels still compiling or autotuning on first use (a cold cache once made a 52 tok/s build measure 23.9; on a fresh
image the first long prompt and the first reuse of a cached conversation are also slower once). Send a few requests,
then measure. If it stays slow, check that the kernel-cache volume is mounted (`serve.sh` does it).

## My desktop session closed, or the screen went black

A GPU kernel faulted hard on the card that drives your monitor: the driver resets that card and the desktop dies with
it. Check with `journalctl -k | grep -i "device coredump"`. With this build's profiles this should not happen; if you
experiment with kernels or options, do it on a card without a monitor attached (`--gpus`), and report the fault.

## `profile ... needs N GPU(s)` or the wrong card is picked

`rdna3/serve.sh --list-gpus` shows what HIP sees (index, PCI address, VRAM). Two-card profiles put the largest card
first; one-card profiles take the smallest card with enough VRAM. Force a choice with `--gpus 1,0` (rank 0 first).

## Out of memory at start-up or on long prompts

- On one card, keep 8k prefill chunks (`--max-prefill-length 8192`): with 16k chunks a 60k-token prompt took 166 s
  and decode fell to 1.9 tok/s (the allocator thrashes).
- Lower the context (`--ctx 131072` on two cards, `--ctx 65536` on one), then the memory ratio
  (`-- --memory-ratio 0.75`).
- A prefill peak above the card's VRAM does not always fail an allocation: the driver evicts memory and stops the
  process's GPU queues for seconds (the `evicted_ms` counter of [Long pauses](#long-pauses-in-the-middle-of-a-run)),
  and a rank has died on an illegal memory access that way. `/sys/class/drm/card*/device/mem_info_vram_used` sampled
  during the first long prompt shows how close the peak comes; the card driving the display also holds the
  desktop's buffers. On `xtx-xt`, `--memory-ratio` 0.80 -> 0.78 lowered the 7900 XTX's peak by 350 MiB.
- The container needs `--ulimit memlock=-1` (pinned RAM for the experts and the RAM tier) and enough RAM under its
  `--memory` limit (default 110g); an OOM kill shows as exit code 137.

## Long pauses in the middle of a run

Two host settings, both measured on the reference machine on 2026-10-03:

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

  Then, on `xtx-xt`: the compaction daemon still ran (39 times) but moved none of the engine's pages, and a forced
  compaction during decoding moved 45k other pages with no stopped queue (longest gap between two tokens 75 ms). Until
  it is set, the server logs a warning at start. The lock comes once the model is loaded: heavy memory use on the host
  during the load (another big process starting) can still stop the GPUs for seconds then, which only delays the start. The driver's own count is `evicted_ms` in
  `/sys/class/kfd/kfd/proc/<pid>/stats_<gpu id>/`.
- **The VRAM clock after a pause.** With the default power profile the memory clock stayed at 96 MHz for ~10 s after
  the GPU woke up: the first request after 15 s or more of idle waited 9-13 s for its first token. The `COMPUTE` power
  profile raises it at once (1.4-1.6 s after each of 16 pauses of 20-90 s) and still lets it drop when idle (13-27 W per
  card); pinning the clock at its maximum works too but costs ~40 W per card at idle. As root, on every card (LACT's
  power profile setting keeps it across reboots):

  ```bash
  echo 5 | sudo tee /sys/class/drm/card*/device/pp_power_profile_mode   # 5 = COMPUTE in that file's list
  ```

The first request after the server starts still takes ~11 s (host-side warm-up, once). Undervolting, GFXOFF and the
display's memory power saving were ruled out on 2026-10-02.

## A GPU fault or a hung collective with a full card

Seen on `xtx-xt` on 2026-10-05/06, each time while a card was within a few hundred MiB of full and the driver had
just evicted the process's queues (`/sys/class/kfd/kfd/proc/<pid>/stats_<gpu>/evicted_ms` growing outside start-up):

- `CUDA error: an illegal memory access` on rank 0, with `[gfxhub] page fault` for the 7900 XTX in `dmesg`: once in an
  agent session with the EXL3 4.05 bpw checkpoint (the XTX peaks ~0.5 GiB from full), and once on purpose, when another
  process took and released 0.5-1.5 GiB on the XTX (the card driving the display) during the first 16k-token chunk of a
  long prompt.
- A hung all-gather (`Watchdog caught collective operation timeout ... _ALLGATHER_BASE`, the server stops) once in a
  depth sweep, the XT at 37 MiB from full during a 16k-token chunk at ~41k tokens of context.

Neither came back in the same runs repeated (hours of agent sessions, the sweep again, the same VRAM pressure in other
10-minute runs): the eviction has to land at a bad moment. Keep other GPU-heavy programs off the card that drives the
display while serving, and lower `--memory-ratio` (more headroom, fewer cached experts) if it happens again. Before
restarting, save the log (`docker logs <container>`): the container is removed when it stops.

## Answers stop mid-sentence

Deep in a long conversation (70-110k tokens of context) the model sometimes ends an answer in the middle of a sentence,
or of a word, and the later answers of that conversation then often stop early too (the model follows its own history).
Measured on 2026-10-04 on `xtx-xt`, turns on this repository's code with thinking off and temperature 0.6: 4 of 90
conversations with the MTP head, 2 of 28 without it. At the stop the end-of-text token had 26-75 % of the probability
(first or second candidate), so no sampling setting avoids it: the model card's non-thinking settings (temperature 0.7,
top-p 0.8) give it 29-83 % at the same places, its thinking ones (1.0, top-p 0.95) 35-56 %. Replayed token for token,
one stop kept a large end-of-text probability from a fresh prefill as from the prefix cache, with
`FREETOKEN_QSA_TORCH_TOPK=1` and with `FREETOKEN_INT8_DENSE=0`: 34-98 % depending on the path (that sensitive to
rounding). So it is not the MTP head, the prefix cache or the RAM tier, the attention's block selection kernel, the int8
dense layers or the sampler (320k draws checked, and the speculative sampling's own tests); whether the unquantized
model does the same was not checked. With thinking on, as the reference machine's agent client sends it (temperature
0.6, the reasoning kept in the history), 15 conversations and 90 turns of ~1800 tokens had no such stop, too few to say
it does not happen there. When it happens, ask the agent to go on, or start a new conversation.

## Sampling hangs

Upstream's cooperative top-k / top-p kernels hang on gfx1100; this build never uses them on ROCm. If you run a
different build, test it on a card that drives no display: `HIP_VISIBLE_DEVICES=<index>
rdna3/tests/run_sampling_tests.sh freetoken.kernel.triton.sampling`, with the index `rdna3/serve.sh --list-gpus` shows
for that card (GPU 0 otherwise; each mode is killed after 90 s). Always smoke-test with a temperature above 0: greedy decoding never reaches the sampler.

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

Open an issue with: the profile and any changed option (`rdna3/serve.sh ... --dry-run` prints everything), the GPUs
(`rdna3/serve.sh --list-gpus`), the host kernel version, the image's commit (`docker run --rm --entrypoint cat
freetoken-rdna3:latest /opt/FreeToken-BUILD-PROVENANCE.txt`), and the server log (`docker logs freetoken-rdna3`).
