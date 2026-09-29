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
- The container needs `--ulimit memlock=-1` (pinned RAM for the experts and the RAM tier) and enough RAM under its
  `--memory` limit (default 110g); an OOM kill shows as exit code 137.

## Long pauses (a few seconds, up to ~21 s) in the middle of a run

A known, not yet understood decode stall, seen on the reference machine in about one run in three on every build
measured since 2026-09-28. If you catch one with `FT_STATS_EVERY=40` set, the log around it is very welcome in an
issue.

## Sampling hangs

Upstream's cooperative top-k / top-p kernels hang on gfx1100; this build never uses them on ROCm. If you run a
different build, test it on a card that drives no display: `HIP_VISIBLE_DEVICES=<index>
rdna3/tests/run_sampling_tests.sh freetoken.kernel.triton.sampling`, with the index `rdna3/serve.sh --list-gpus` shows
for that card (GPU 0 otherwise; each mode is killed after 90 s). Always smoke-test with a temperature above 0: greedy decoding never reaches the sampler.

## Warnings about TunableOp

The files in `rdna3/tunableop/` match this image's PyTorch / HIP / hipBLASLt versions. With another base image, set
`TUNABLEOP=0` in your profile (or tune once, see [options.md](options.md#pytorch-tunableop)).

## `FREETOKEN_NVFP4_DECODE_TUNED: no tuned tiles for (N, K) = ...` in the log

Expected with a split or card combination other than `xtx-xt` (one card included): those shapes use the default
tiles. Harmless.

## `TP ranks sampled different tokens in ... row(s)` in the log

Only printed with `FREETOKEN_TP_SYNC_TOKENS=check`. Rank 0's tokens are what the client receives either way; please
report the lines with your setup.

## Reporting a problem

Open an issue with: the profile and any changed option (`rdna3/serve.sh ... --dry-run` prints everything), the GPUs
(`rdna3/serve.sh --list-gpus`), the host kernel version, the image's commit (`docker run --rm --entrypoint cat
freetoken-rdna3:latest /opt/FreeToken-BUILD-PROVENANCE.txt`), and the server log (`docker logs freetoken-rdna3`).
