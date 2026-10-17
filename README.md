# freetoken-rdna3

![freetoken-rdna3: Qwen3.8-Flash-Next on RX 7900 XTX / XT, NVFP4 and EXL3 checkpoints](docs/rdna3/assets/social-preview.png)

**Run Qwen3.8-Flash-Next, a 125B-parameter Mixture-of-Experts model, on your own PC with one or two AMD Radeon RX 7900
XTX / XT cards.** You get a local server with an OpenAI- and Anthropic-compatible API: chat apps, coding agents
(OpenCode, Claude Code) and your own scripts use it like a cloud API, with 262k tokens of context per conversation (up
to 1M with `xtx-xt-1m`), while the model runs on your hardware.

It is a tuned build of the [FreeToken](https://github.com/FlashML-org/FreeToken) MoE engine for RDNA3 (ROCm). The model
does not fit in VRAM: every expert lives in system RAM and the GPUs keep the most used ones; two unequal cards share
the work.

## What you need

| | |
|---|---|
| GPUs | one or two RX 7900 XTX (24 GB) / RX 7900 XT (20 GB); other cards: [which GPUs](docs/rdna3/limits.md#which-gpus) |
| RAM | 128 GB recommended: the server locks 51 to 82 GiB, depending on the checkpoint and the number of cards ([table](#which-checkpoint)), more with conversations kept in RAM, and the system and your programs need ~30 GiB beside it ([how to count](docs/rdna3/limits.md#how-much-ram-do-i-need)) |
| Disk | 85 to 135 GB for the model, ~27 GB for the image (~29 GB more while it builds) |
| System | Linux x86_64 with the in-kernel `amdgpu` driver, and Docker. ROCm, PyTorch and the rest come inside the image |

`rdna3/serve.sh --check` looks at your machine, says what is missing, which profile fits your GPUs and what fits in your
RAM. It changes nothing.

## Quick start

```bash
git clone https://github.com/Cedriceuh/freetoken-rdna3 && cd freetoken-rdna3
rdna3/serve.sh --check                                   # what this machine has, which profile fits (no image yet: expected)
docker build -f Dockerfile.rdna3 -t freetoken-rdna3:latest .    # once; pulls a 25 GB ROCm base image
rdna3/download.sh nvfp4                                  # 135 GB into ~/models (or exl3-3.05, exl3-4.05: see below)
rdna3/serve.sh xtx-xt --model ~/models/Qwen3.8-Flash-Next-NVFP4  # use the profile --check suggests
```

The server loads in ~2.5 minutes on the reference machine (the first start of a new image also compiles GPU kernels for
a few minutes). It is
ready when `curl -s http://127.0.0.1:1919/health` answers `"ok"`. Then, from another terminal:

```bash
curl -s http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "Hello! Who are you?"}],
  "chat_template_kwargs": {"enable_thinking": false}}'
```

Every step explained, with client set-ups and running it as a service:
[docs/rdna3/getting-started.md](docs/rdna3/getting-started.md).

## What you can do with it

- **Chat** from any app that speaks the OpenAI API: base URL `http://127.0.0.1:1919/v1`, model `qwen3.8-flash-next`,
  any API key.
- **Run coding agents**: OpenCode through the OpenAI API (the client every agent measurement here used), or a client
  of the Anthropic API such as Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:1919`; not measured here). Tool calls
  and thinking are supported. On two cards up to four
  requests run at once, so sub-agents work in parallel. Set-ups:
  [connect a client](docs/rdna3/getting-started.md#7-connect-a-client).
- **Work on long inputs**: 262,144 tokens per conversation on two cards, 131,072 on one. `xtx-xt-4x262k` keeps four
  full-length conversations at once, `xtx-xt-1m` one of up to 1M tokens, as many as your RAM holds: each 262k
  conversation takes 6.9 GiB of it ([long-context](docs/rdna3/long-context.md), `rdna3/serve.sh --check`).
- **Switch between conversations**: one pushed off the GPUs is kept in system RAM and resumes in ~2 s instead of being
  read again.
- **Send images** with `rdna3/serve.sh ... --vision` ([images](docs/rdna3/limits.md#images-vision)).
- **Script it** with any OpenAI or Anthropic SDK, or serve your network (`--host 0.0.0.0`, add an API key:
  [security](docs/rdna3/limits.md#anything-else-to-know)).

## Which profile?

A profile holds every setting for one GPU setup; `rdna3/serve.sh <profile>` picks the cards by itself.

| Your GPUs | Profile | Context | Requests at once | Status |
|---|---|---:|---:|---|
| RX 7900 XTX + RX 7900 XT | `xtx-xt` | 262k, shared by the requests | 4 | measured |
| the same, several long agent conversations | `xtx-xt-4x262k` | 262k each | up to 4, as your RAM allows | measured |
| the same, one very long document | `xtx-xt-1m` | 1M | 4, sharing 1M | measured with EXL3 3.05 bpw |
| two RX 7900 XTX | `xtx-xtx` | 262k | 4 | untested; EXL3 needs `FREETOKEN_TP_SPLIT=0.6` |
| two RX 7900 XT | `xt-xt` | 262k | 4 | untested; EXL3 needs `FREETOKEN_TP_SPLIT=0.6` |
| one RX 7900 XTX | `xtx` | 131k | 1 | measured |
| one RX 7900 XT | `xt` | 131k | 1 | measured |
| one RX 7900 GRE | `gre` | 65k | 1 | untested |

Untested profiles are derived from the measured ones; the script says so when you start one. Measured one? Please open
an issue with your numbers. Details and adapting a profile to other cards: [profiles](docs/rdna3/profiles.md).

## Which checkpoint?

All three run on `xtx-xt`, `xtx` and `xt` with nothing to change: download one and point `--model` at its directory.
`xtx-xt-1m` is measured with EXL3 3.05 bpw only, `xtx-xt-4x262k` with EXL3 3.05 bpw (four conversations) and NVFP4
(three) ([profiles](docs/rdna3/profiles.md#xtx-xt-4x262k-and-xtx-xt-1m-the-kv-in-host-ram)). "bpw" is bits per weight;
the other terms: [glossary](docs/rdna3/README.md#terms-used-in-these-pages).

| | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| `rdna3/download.sh` | `nvfp4` | `exl3-3.05` | `exl3-4.05` |
| Hugging Face | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) |
| Download | 135 GB | 85 GB | 108 GB |
| RAM locked: two cards; one card | 82 GiB; 73 GiB | 60 GiB; 51 GiB | 75 GiB; 66 GiB |
| Decode, `xtx-xt` | 78-96 tok/s | 95-117 tok/s | 86-94 tok/s |
| Decode, `xtx` | 36-44 tok/s | 52-58 tok/s | 37-47 tok/s |

EXL3 3.05 bpw is the smallest and decodes the fastest on every measured profile: its experts are smaller, so more of
them stay in VRAM (the checkpoints also differ in how many of the MTP head's drafts are kept). On the
maintainers' agentic benchmark (29 planted bugs to fix in long agent sessions, on an earlier build) NVFP4 fixed 12 and
18 in two runs, EXL3 3.05 bpw 13 and 4.05 bpw 12 in one run each: too few runs to rank them.

## Speed

One request, greedy, the model's MTP draft head on (as the profiles ship): one conversation grown from 9k tokens to the
profile's maximum context with blocks the server has never seen, like large tool results
([method](docs/rdna3/benchmarks.md#how-the-numbers-are-taken)).

| Decode (TG) | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| `xtx-xt`: RX 7900 XTX + XT, 9k to 260k tokens | 78-96 tok/s | 95-117 tok/s | 86-94 tok/s |
| `xtx`: one RX 7900 XTX, 9k to 124k tokens | 36-44 tok/s | 52-58 tok/s | 37-47 tok/s |
| `xt`: one RX 7900 XT, 9k to 124k tokens | 29-34 tok/s | 41-46 tok/s | 29-36 tok/s |

| Reading a new 6-52k-token block (PP); agent turn, time to first token | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| `xtx-xt`, 9k to 260k tokens | 1910-2180 tok/s; 0.8-0.9 s | 1940-2280 tok/s; 0.7-0.9 s | 1970-2270 tok/s; 0.8-1.0 s |
| `xtx`, 9k to 124k tokens | 1520-1670 tok/s; 1.5-1.6 s | 1700-1820 tok/s; 1.0-1.2 s | 1640-1820 tok/s; 1.3-1.5 s |
| `xt`, 9k to 124k tokens | 1350-1480 tok/s; 1.6-1.8 s | 1500-1600 tok/s; 1.2-1.3 s | 1450-1620 tok/s; 1.5-1.6 s |

An "agent turn" is ~1k new tokens (a tool result) added to a conversation the server already holds. Requests running
at once share the GPUs: on `xtx-xt-4x262k`, four agents at 257-260k tokens decode at 29.6-31.6 tok/s each, one alone at
95.1 (EXL3 3.05 bpw, one pass on an earlier build than the tables above).
All the numbers: [benchmarks](docs/rdna3/benchmarks.md).

## Documentation

| I want to | Read |
|---|---|
| install it and connect my tools | [getting started](docs/rdna3/getting-started.md) |
| fix something that goes wrong | [troubleshooting](docs/rdna3/troubleshooting.md) |
| know if my RAM, GPUs or model will work | [limits and FAQ](docs/rdna3/limits.md) |
| change or adapt a profile | [profiles](docs/rdna3/profiles.md), [options](docs/rdna3/options.md) (every setting this build adds) |
| serve very long contexts | [long context](docs/rdna3/long-context.md) |
| use the CPU for part of the experts (`--hybrid`) | [CPU experts](docs/rdna3/cpu-experts.md) |
| see every measurement | [benchmarks](docs/rdna3/benchmarks.md) |
| understand the engine | [how it works](docs/rdna3/how-it-works.md), [decisions](docs/rdna3/decisions.md), [the journey](docs/rdna3/journey.md) (what was tried, kept and dropped) |
| contribute | [CONTRIBUTING](CONTRIBUTING.md), [testing](docs/rdna3/testing.md), [maintaining](docs/rdna3/maintaining.md) |

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
