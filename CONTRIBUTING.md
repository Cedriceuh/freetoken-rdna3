# Contributing

Thanks for helping! The most useful contributions right now:

1. **Numbers from other hardware.** Run a profile (the `TESTED=0` ones especially: `xtx-xtx`, `xt-xt`, `gre`), then
   `python3 rdna3/bench/quick_bench.py --label "<your cards>"`, and open an issue with the table, your GPUs
   (`rdna3/serve.sh --list-gpus`), RAM, and the profile. A setup that works deserves a measured profile.
2. **Bug reports** with what [troubleshooting.md](docs/rdna3/troubleshooting.md#reporting-a-problem) lists: profile,
   `serve.sh --dry-run` output, GPUs, host kernel version, image commit, server log.
3. **Fixes and optimizations**, following the rules below.

## Pull requests

- One change per pull request, with the hardware and the exact commands it was tested with.
- **Precision first**: a change must be bit-exact (identical greedy answers to `main` on the same prompts) or come with
  evidence from an agentic, multi-turn workload (the maintainers' benchmark is private: describe what you ran, on both
  builds). See [testing.md](docs/rdna3/testing.md).
- **Performance**: interleaved A/B numbers (same model, prompts and settings, on `main` and on your branch), tokens/s
  and time to first token, with the commands.
- **New behavior behind an environment variable**, off by default unless it is bit-exact or validated on an agentic
  workload, documented in
  [options.md](docs/rdna3/options.md).
- **Bug fixes** come with a test that fails before and passes after, where the code allows it.
- Commit messages: [Conventional Commits](https://www.conventionalcommits.org/) (`fix(sampler): ...`), imperative,
  lowercase.
- Generic fixes that are not RDNA3-specific are also worth proposing to upstream
  [FreeToken](https://github.com/FlashML-org/FreeToken).

## AI-assisted contributions

Welcome, on the same terms as upstream FreeToken: you are responsible for everything in your pull request, however it
was produced. You must understand the change, have run it on real hardware, and be able to explain it without AI help.
Pull requests with invented results (tests or benchmarks that were not run) are closed. Fully autonomous agents must
not open pull requests here.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0 ([LICENSE](LICENSE)).
