# freetoken-rdna3

![freetoken-rdna3: Qwen3.8-Flash-Next on RX 7900 XTX / XT, NVFP4 and EXL3 checkpoints](docs/rdna3/assets/social-preview.png)

**Fast local inference of big Mixture-of-Experts models on AMD Radeon RX 7900 XTX / 7900 XT (RDNA3, ROCm), on one or
two consumer GPUs** (Linux; the server locks 51-82 GiB of system RAM, depending on the checkpoint and the cards). A
tuned build of the [FreeToken](https://github.com/FlashML-org/FreeToken) MoE engine: tensor parallelism across two
unequal cards, a GPU expert cache fed from system RAM, int8 dense layers, exact sampling on ROCm, up to 4 concurrent
requests (parallel sub-agents), and a RAM tier that keeps conversations pushed off the GPUs instead of recomputing
them.

Built for, and measured with, **Qwen3.8-Flash-Next** (125B-parameter MoE, ~6B active per token) behind a coding agent
(OpenCode), with an OpenAI- and Anthropic-compatible API, from three checkpoints: NVFP4
([RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)) and EXL3 at 3.05 or 4.05 bits per weight
([turboderp](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), smaller, read directly by this build). Image
input is optional (`rdna3/serve.sh --vision`): no measurable cost on two cards, ~1.5-2 % of decode on one
([limits](docs/rdna3/limits.md#images-vision)).

### Speed over the whole context

One request, MTP head on, greedy: one conversation grown from 9k tokens to the profile's maximum context with blocks
the server has never seen, like large tool results ([method](docs/rdna3/benchmarks.md#how-the-numbers-are-taken)).

| Decode (TG) | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| `xtx-xt`: RX 7900 XTX + XT, 9k to 248k tokens | 72-86 tok/s | 83-107 tok/s | 76-87 tok/s |
| `xtx`: one RX 7900 XTX, 9k to 124k tokens | 33-42 tok/s | 42-52 tok/s | 36-44 tok/s |
| `xt`: one RX 7900 XT, 9k to 124k tokens | 27-34 tok/s | 33-42 tok/s | 28-35 tok/s |

| Reading a new 6-41k-token block (PP); agent turn, time to first token | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| `xtx-xt`, 9k to 248k tokens | 1670-1990 tok/s; 1.3-2.0 s | 1730-2050 tok/s; 0.9-1.6 s | 1710-2050 tok/s; 1.2-1.9 s |
| `xtx`, 9k to 124k tokens | 1130-1510 tok/s; 2.4-2.7 s | 1280-1590 tok/s; 1.5-1.9 s | 1200-1590 tok/s; 2.1-2.4 s |
| `xt`, 9k to 124k tokens | 1040-1340 tok/s; 2.5-2.8 s | 1200-1420 tok/s; 1.6-2.0 s | 1120-1420 tok/s; 2.2-2.6 s |

| Size and memory | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| Download | 135 GB | 85 GB | 108 GB |
| RAM while serving (locked): two cards; one card | 82 GiB; 72 GiB | 59 GiB; 51 GiB | 74 GiB; 65 GiB |
| Experts cached per card, `xtx-xt` (of 24,576) | 7.1k | 9.9k | 7.3k |

NVFP4's speeds were measured on the release image (2026-10-03), its one-card RAM and every EXL3 figure on this build
(2026-10-04), same machine and method. The 3 bpw experts are smaller, so more of them stay in VRAM (9.9k instead of
7.1k per card on `xtx-xt`); the checkpoints also answer differently, which changes how many of the MTP head's guesses
are kept.

The MTP head (the model's own draft layer) guesses the next tokens and one step checks them; greedy answers are
identical with and without it. On `xtx-xt` with NVFP4 it takes one request from 57.5 to 77 tok/s at the model's
sampling; with 3-4 requests at once it costs 3-4 %; `rdna3/serve.sh --no-mtp` turns it off. On real agent turns
(~90k-token contexts of code, temperature 0.6): 71 tok/s for a request alone, ~34 tok/s each when two sub-agents
decode at once ([benchmarks](docs/rdna3/benchmarks.md#agent-turns-on-real-code-xtx-xt)).

Four agents with 82-117k-token conversations, 3 turns each: after the first reads, the three rounds of turns take
**35 s instead of 599 s** without the RAM tier, because a conversation pushed off the GPUs is kept in system RAM and
resumes in ~2 s instead of being re-read.

Full numbers, methods and the cold-read times: [docs/rdna3/benchmarks.md](docs/rdna3/benchmarks.md).

## Quick start

You need Linux with the `amdgpu` driver, Docker, an RX 7900 XTX and/or 7900 XT, and enough RAM for the checkpoint: the
server locks 51-82 GiB (a 128 GB machine for NVFP4; why: [limits](docs/rdna3/limits.md)). Everything runs in the Docker image below; the `install.sh` at the root is
upstream's NVIDIA / CUDA installer and is not used here.

```bash
git clone https://github.com/Cedriceuh/freetoken-rdna3 && cd freetoken-rdna3
docker build -f Dockerfile.rdna3 -t freetoken-rdna3:latest .          # pulls two ROCm + PyTorch bases, ~60 GB
mkdir -p ~/models      # model download (~135 GB), with the image's own `hf`
docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/models/.cache/huggingface -v ~/models:/models \
  --entrypoint hf freetoken-rdna3:latest download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir /models/Qwen3.8-Flash-Next-NVFP4
rdna3/serve.sh xtx-xt --model ~/models/Qwen3.8-Flash-Next-NVFP4       # or: xtx, xt (one card); stays in the foreground
```

For an EXL3 checkpoint, download `turboderp/Qwen3.8-Flash-Next-exl3` with `--revision 3.05bpw_h5_ng5` (85 GB) or
`--revision 4.05bpw_h6_ng6` (108 GB) and point `--model` at it: nothing else changes
([details](docs/rdna3/how-it-works.md#exl3-checkpoints)).

Loading takes ~2.5 minutes on the reference machine (the first start of a new image also compiles GPU kernels for a few minutes); the server
is ready when `curl -s http://127.0.0.1:1919/health` (from another terminal) reports `"ok"`. Then point any
OpenAI-compatible client at `http://127.0.0.1:1919/v1` (model `qwen3.8-flash-next`), or an Anthropic-compatible one at
`http://127.0.0.1:1919`. Step by step, with a client example and a systemd unit: [docs/rdna3/getting-started.md](docs/rdna3/getting-started.md).

## Profiles

One ready-made profile per hardware setup, measured for the tested ones and derived for those marked untested
([details](docs/rdna3/profiles.md)):

| Profile | GPUs | Context | Requests at once | Status |
|---|---|---:|---:|---|
| `xtx-xt` | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | 250k | 4 | measured (tables above) |
| `xtx` | one RX 7900 XTX 24 GB | 131k | 1 | measured |
| `xt` | one RX 7900 XT 20 GB | 131k | 1 | measured |
| `xtx-xtx` | two RX 7900 XTX 24 GB | 250k | 4 | untested: decode expected >= `xtx-xt`; EXL3 needs `FREETOKEN_TP_SPLIT=0.6` |
| `xt-xt` | two RX 7900 XT 20 GB | 250k | 4 | untested: decode expected a little below `xtx-xt`; same EXL3 note |
| `gre` | one RX 7900 GRE 16 GB | 65k | 1 | untested: ~20 tok/s at best expected (NVFP4, head off) |

Untested profiles are derived from the measured ones and the engine's memory plans; the script says so when you start
one. Measured something? Please open an issue with your numbers.

`rdna3/serve.sh` picks the right cards by itself; `--dry-run` shows the exact `docker run` it would start.

## Documentation

- [Getting started](docs/rdna3/getting-started.md): install, first run, clients, running it as a service
- [Profiles](docs/rdna3/profiles.md): what each profile sets and how to adapt one to other cards
- [Options](docs/rdna3/options.md): every setting this build adds, with its measured effect
- [Benchmarks](docs/rdna3/benchmarks.md): speed at every context depth, several requests and agents, precision
- [How it works](docs/rdna3/how-it-works.md): what was changed in the engine and why
- [Decisions](docs/rdna3/decisions.md): each design choice, its alternatives and trade-offs
- [Limits and FAQ](docs/rdna3/limits.md): RAM, other GPUs (RDNA4, NVIDIA), other models
- [Troubleshooting](docs/rdna3/troubleshooting.md)
- [The journey](docs/rdna3/journey.md): what was tried, measured, kept and dropped
- [Testing](docs/rdna3/testing.md) and [maintaining](docs/rdna3/maintaining.md): checking a change, following upstream

For LLM agents: [`llms.txt`](llms.txt) indexes everything, [`AGENTS.md`](AGENTS.md) explains how to work on the code.

## Status

Tested on one machine: RX 7900 XTX + RX 7900 XT (gfx1100), Threadripper 3970X, 128 GB DDR4, ROCm 10.1 in the
container. Other RDNA3 cards, RDNA4, more than two GPUs and NVIDIA are untested ([limits](docs/rdna3/limits.md)).
Reports from other setups are very welcome.

## Credits and license

A modified version of [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache License 2.0), with ROCm work from
the FreeToken community. What changed from upstream and who wrote what: [NOTICE](NOTICE),
[docs/rdna3/changes-from-upstream.md](docs/rdna3/changes-from-upstream.md), [docs/rdna3/credits.md](docs/rdna3/credits.md).
Upstream's own README: [docs/FREETOKEN_UPSTREAM_README.md](docs/FREETOKEN_UPSTREAM_README.md).
