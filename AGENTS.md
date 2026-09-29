# Instructions for AI coding agents

This repository is freetoken-rdna3: upstream FreeToken plus RDNA3 (ROCm) work. Start with [llms.txt](llms.txt), which
indexes the documentation; [docs/rdna3/how-it-works.md](docs/rdna3/how-it-works.md) and
[docs/rdna3/decisions.md](docs/rdna3/decisions.md) explain what was changed and why, and
[docs/rdna3/journey.md](docs/rdna3/journey.md) lists what was already tried and dropped: read it before proposing an
optimization.

## Rules

- **Precision before speed.** A change is either bit-exact (identical greedy answers to the previous build on the same
  prompts) or validated on an agentic, multi-turn benchmark (the maintainers' is private: describe the workload you
  ran). Short tests (scored items, needles, probes) may reject a
  change, never accept it alone ([testing.md](docs/rdna3/testing.md)).
- **Everything new is an environment variable, off by default** unless it is bit-exact or agent-validated, so
  upstream's behaviour stays one setting away.
  Document it in [docs/rdna3/options.md](docs/rdna3/options.md) with its measured effect.
- **Report only what was run.** Never state a test or benchmark result that was not produced on real hardware in this
  session; say "untested" otherwise, as the `TESTED=0` profiles do.
- **Do not run a kernel that may fault on the GPU that drives the user's monitor**: a hard fault resets the card and
  closes the desktop session. Ask first, and use a card without a display (`HIP_VISIBLE_DEVICES`).
- **Do not push, open pull requests or issues, or publish anything on the user's behalf.** The human owns every line
  and must be able to explain it.
- **No AI attribution lines** in commits (`Co-authored-by`, `Assisted-by`): the user is the author.

## Repository layout

```
python/freetoken/      the engine (upstream layout: server/, scheduler/, kvcache/, moe/, models/, kernel/, layers/, engine/)
  distributed/split.py        uneven tensor-parallel split
  kernel/host_allreduce.py    host-memory all-reduce
  kernel/triton/              RDNA3 GEMVs, top-k-first sampler, sampling (ROCm sorted threshold)
  kvcache/host_kv_pool.py     conversations kept in RAM (with kvcache/hybrid_radix_cache.py)
  models/qwen4_exp/           Qwen3.8-Flash-Next, tensor-parallel port
tests/                 CPU tests, mirroring python/freetoken/
rdna3/                 profiles, serve.sh, GPU checks (tests/), micro-benchmarks (bench/), TunableOp files, tools/
docs/rdna3/            this repository's documentation; docs/*.md is upstream's
Dockerfile.rdna3       the ROCm image
```

## Development

Everything runs in the image (ROCm 7.14, PyTorch 2.11, Triton), with the working tree mounted over it:

```bash
docker build -f Dockerfile.rdna3 -t freetoken-rdna3:latest .
docker run --rm -w /opt/FreeToken --entrypoint python3 freetoken-rdna3:latest \
  -m pytest -q -p no:cacheprovider tests          # CPU suite, ~5 min, 3 known environment failures (testing.md)
```

GPU checks and benchmarks: [docs/rdna3/testing.md](docs/rdna3/testing.md). A bug fix comes with a test that fails
before and passes after; a performance change comes with interleaved A/B numbers on the whole model. After editing the
docs, regenerate `llms-full.txt` with `rdna3/tools/make-llms-full.sh`.

## Code comments and commits

Comments explain a non-obvious "why", in one or two lines, in ASCII (`-`, `->`). Commits follow Conventional Commits,
imperative and lowercase (`fix(kvcache): size the host tier in pages, not bytes`), with a body only when the why is not
in the diff. Commit only when the user asks.
