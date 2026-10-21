# Getting started

From a Linux machine with one or two RX 7900 cards to a running server and your tools connected to it. Each step says
how to check that it worked. The two downloads (a 25 GB base image, an 85-135 GB model) take most of the time.

## 1. Check your machine

| | Two cards (`xtx-xt`) | One card (`xtx` or `xt`) |
|---|---|---|
| GPUs | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | one RX 7900 XTX 24 GB or RX 7900 XT 20 GB |
| System RAM the server locks: NVFP4; EXL3 3.05 / 4.05 bpw (keep ~30 GiB more for the system) | ~83 GiB; ~61 / ~76 GiB: a **128 GB** machine | ~74 GiB; ~52 / ~67 GiB |
| Disk | 135 GB for the NVFP4 model (85 / 108 GB in EXL3 3.05 / 4.05 bpw), ~27 GB for the image, ~29 GB more while building | same |
| Software | Linux x86_64 with the in-kernel `amdgpu` driver (`/dev/kfd` present), Docker | same |
| Tested on | Ubuntu 26.04 LTS, kernel 7.0 (older kernels untested) | same |

Other setups (two XTX, two XT, a GRE) have untested profiles. How to count the RAM you need, other GPUs and other
models: [limits.md](limits.md#how-much-ram-do-i-need).

The ROCm user space, PyTorch and Triton are inside the image: nothing ROCm-related has to be installed on the host.
Your user must be allowed to run Docker (the `docker` group; log in again after adding it) and be in the `video` and
`render` groups.

Get the code and let the launcher look at the machine:

```bash
git clone https://github.com/Cedriceuh/freetoken-rdna3 && cd freetoken-rdna3
rdna3/serve.sh --check
```

It checks the driver, Docker, the image, the GPUs, the RAM each checkpoint needs and the host settings of step 2, and
prints `ok`, `warn` or `FAIL` for each, with the command that fixes it. It names the profile for your GPUs
(`suggested:`), says what fits in your RAM, ends with the command to start the server (`Next:`), and changes nothing. Run it again after each step; with `--model DIR` it also checks a downloaded checkpoint. The image
is missing until step 3: that `FAIL` is expected now.

## 2. Two host settings

Without them the server works but pauses: both are measured on the reference machine
([why](troubleshooting.md#long-pauses-in-the-middle-of-a-run)). The first one is set once; the second one lasts until
the next reboot.

- **Keep memory compaction off the engine's locked memory.** Otherwise the kernel moves memory the GPUs use and the
  driver stops them for 5-22 s while it does. Persistent:

  ```bash
  printf 'vm.compaction_proactiveness=0\nvm.compact_unevictable_allowed=0\n' | sudo tee /etc/sysctl.d/99-llm-gpu.conf
  sudo sysctl --system
  ```

- **The `COMPUTE` power profile on each card.** With the default one, the VRAM clock stays low for ~10 s after an idle
  pause, and the first request after it waits that long. Until the next reboot (a tool such as LACT can keep it across
  reboots):

  ```bash
  echo 5 | sudo tee /sys/class/drm/card*/device/pp_power_profile_mode   # 5 = COMPUTE in that file's list
  ```

## 3. Build the image

```bash
docker build -f Dockerfile.rdna3 -t freetoken-rdna3:latest .
```

The first build pulls the ROCm PyTorch base image (25 GB, pinned by digest); after that a rebuild takes a couple of
minutes. Build from a git clone, not from a downloaded archive: the build records the commit it was made from
(`/opt/FreeToken-BUILD-PROVENANCE.txt` in the image, which `serve.sh --check` compares with your checkout) and fails
without `.git`. `--build-arg GPU_ARCH=gfx1100` is the default and the only architecture tested. Do not use
`install.sh` or `scripts/`: they install and build upstream's NVIDIA / CUDA wheels.

## 4. Download the model

### Which checkpoint?

Three quantized versions of Qwen3.8-Flash-Next run on `xtx-xt`, `xtx` and `xt` with nothing else to change (the other
profiles: [which checkpoint](../../README.md#which-checkpoint)).

| | NVFP4 | EXL3 3.05 bpw | EXL3 4.05 bpw |
|---|---:|---:|---:|
| `rdna3/download.sh` name | `nvfp4` | `exl3-3.05` | `exl3-4.05` |
| Download | 135 GB | 85 GB | 108 GB |
| RAM locked: two cards; one card | ~83 GiB; ~74 GiB | ~61 GiB; ~52 GiB | ~76 GiB; ~67 GiB |
| Decode, `xtx-xt` / `xtx` / `xt` | 78-96 / 36-44 / 29-34 tok/s | 95-117 / 52-58 / 41-46 tok/s | 86-94 / 37-47 / 29-36 tok/s |

EXL3 3.05 bpw is the smallest, needs the least RAM and decodes the fastest on every measured profile (its experts are
smaller, so more stay in VRAM; the checkpoints also differ in how many of the MTP head's drafts are kept). On the
maintainers' agentic benchmark (29 planted bugs to fix in long agent sessions, on an earlier build) the EXL3
checkpoints fixed 13 (3.05 bpw) and 12 (4.05 bpw) of 29, one run each; NVFP4 12 and 18 in two runs: too few runs to
rank them. "bpw" is bits per weight; other terms: [glossary](README.md#terms-used-in-these-pages).

### Download

```bash
rdna3/download.sh nvfp4          # or exl3-3.05, exl3-4.05; prints the directory it fills (below)
```

The directories: `~/models/Qwen3.8-Flash-Next-NVFP4`, `~/models/Qwen3.8-Flash-Next-exl3-3.05bpw`,
`~/models/Qwen3.8-Flash-Next-exl3-4.05bpw`. It runs the image's own Hugging Face CLI (nothing to install on the host; the files belong to your user), checks the
disk space first, and skips the files already complete when run again after an interruption. A second argument picks
another directory; `HF_TOKEN`, when set, is passed on. With `hf` installed on the host, the same thing by hand:
`hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4`, or
`turboderp/Qwen3.8-Flash-Next-exl3` with `--revision 3.05bpw_h5_ng5` or `--revision 4.05bpw_h6_ng6`.

What the NVFP4 135 GB hold: 68 GB of experts (loaded into RAM), 51 GB of n-gram embedding tables (read from disk on
demand, they never have to fit in RAM), 16 GB of other weights and the tokenizer.

Check: `rdna3/serve.sh --check --model ~/models/Qwen3.8-Flash-Next-NVFP4` names the checkpoint and the RAM it needs.

## 5. Start the server

```bash
rdna3/serve.sh xtx-xt --model ~/models/Qwen3.8-Flash-Next-NVFP4       # two cards
rdna3/serve.sh xtx    --model ~/models/Qwen3.8-Flash-Next-NVFP4       # one XTX
rdna3/serve.sh xt     --model ~/models/Qwen3.8-Flash-Next-NVFP4       # one XT
```

Use the profile `serve.sh --check` suggested ([which profile](../../README.md#which-profile)); `rdna3/serve.sh --list`
prints them all. The script picks the cards by itself (for two cards the larger one becomes rank 0) and prints them;
`--gpus 1,0` forces a choice (`rdna3/serve.sh --list-gpus` shows the numbers), `--dry-run` prints the full
`docker run` command instead of running it.

Loading takes about 2.5 minutes on the reference machine (it reads ~68 GB of experts into RAM). The very first start of
a new image also compiles and autotunes GPU kernels for a few more minutes; the results are kept in a Docker volume
(`freetoken-rdna3-kcache-<image id>`), so later starts are fast. The server is ready when the log prints
`API server is ready to serve on 127.0.0.1:1919`, or:

```bash
curl -s http://127.0.0.1:1919/health          # {"status": "ok", ...} once loaded ("loading" before)
```

`serve.sh` stays in the foreground: run the next commands from another terminal. Stop it with Ctrl-C or
`docker stop -t 30 freetoken-rdna3` and let it exit (a clean shutdown takes ~9 s). Do not kill it while it is still
loading: that once left GPU memory held until a reboot
([troubleshooting](troubleshooting.md#less-vram-or-a-start-that-never-ends-after-a-server-was-killed-while-starting)).

### Options worth knowing

| Option | What it does |
|---|---|
| `--port N`, `--host ADDR` | where the API listens (default `127.0.0.1:1919`). `--host 0.0.0.0` serves your network: the API has no authentication unless started with `-- --api-key KEY` ([security](limits.md#anything-else-to-know)) |
| `--ctx N` | a shorter context than the profile's. Where the K/V lives in VRAM, the room it frees goes to the expert cache; with `xtx-xt-1m` it shrinks the K/V kept in RAM; with `xtx-xt-4x262k` it only caps each conversation (the RAM pool is `--kv-reserve-tokens`) |
| `--vision` | accept images ([limits](limits.md#images-vision)); off by default, no measurable speed cost on two cards, ~1.5-2 % of decode on one |
| `--no-mtp` | turn off the MTP draft head: the head speeds up a request decoding alone (+35 % on `xtx-xt`) and up to four at once on `xtx-xt` and its K/V-in-RAM variants; it costs 3-4 % where more requests decode than it verifies (the other profiles, an earlier build; [limits](limits.md#speculative-decoding-mtp)) |
| `--hybrid` | the CPU computes part of the experts missing from VRAM; off by default, not agent-validated. Measured on an earlier build: NVFP4 +46 % decode on `xtx`, +51 % on `xt`, on `xtx-xt` +2 % for one request and +15 % each with four; EXL3 3.05 bpw +5 % on `xtx`, even on `xtx-xt` ([cpu-experts](cpu-experts.md)) |
| `--served-name ID` | the model id `/v1/models` reports (default `qwen3.8-flash-next`) |
| `--name NAME`, `--image REF`, `--memory SIZE` | container name, image, container RAM limit |
| `-- <flags>` | anything after `--` goes to `ft serve` (upstream's [CLI reference](../cli.md)) |

`rdna3/serve.sh --help` lists them all.

## 6. First request

```bash
curl -s http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals."}],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 1024
}'
```

- **Sampling defaults** for any field the request leaves out: temperature 0.6 (`rdna3/serve.sh` passes
  `--default-temperature 0.6`; the checkpoint's `generation_config.json` says 1.0), top_k 20 and top_p 0.95 from that
  file. Keeping `top_k` at 20 (or anything up to 256) uses the fastest sampler.
- **Thinking** is on by default (the reasoning goes to `reasoning_content`); the request above turns it off with
  `"chat_template_kwargs": {"enable_thinking": false}`. With thinking on, give `max_tokens` room (several thousand): if
  the budget runs out while the model is still thinking, `content` comes back empty with `finish_reason: "length"`.
- **Tool calls** are parsed (Qwen3 coder format) and returned as OpenAI `tool_calls`.

## 7. Connect a client

Endpoints: OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`;
Anthropic-compatible `/v1/messages`. The server serves one model and answers whatever model name a request carries.
The API key is not checked (use any string) unless the server was started with `-- --api-key KEY`.

For a coding agent, set its context window to the profile's context (262144 for the two-card profiles and
`xtx-xt-4x262k`, 1000000 for `xtx-xt-1m`, 131072 for `xtx` / `xt`, 65536 for `gre`, or your `--ctx`), so it compacts
the conversation at the right time.

**Any OpenAI-compatible app** (chat UIs, editors, SDKs): base URL `http://127.0.0.1:1919/v1`, model
`qwen3.8-flash-next`, any API key.

**OpenCode**: a provider in `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "freetoken": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "freetoken-rdna3",
      "options": { "baseURL": "http://127.0.0.1:1919/v1", "timeout": false, "headerTimeout": false },
      "models": {
        "qwen3.8-flash-next": {
          "name": "Qwen3.8-Flash-Next (local)",
          "reasoning": true,
          "limit": { "context": 262144, "output": 32768 },
          "interleaved": { "field": "reasoning_content" }
        }
      }
    }
  },
  "model": "freetoken/qwen3.8-flash-next"
}
```

The two timeouts off keep OpenCode waiting through long first reads; `interleaved` gives the model its earlier
reasoning back in the history (field `reasoning_content`). With `--vision`, add `"attachment": true` and `"modalities": {"input": ["text", "image"], "output":
["text"]}` to the model.

**Claude Code** (or an Anthropic SDK; not measured here) adds `/v1/messages` itself:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:1919 ANTHROPIC_API_KEY=local claude
```

**Python**, with the `openai` package:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="local")
reply = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "Summarize the plot of Hamlet in three sentences."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(reply.choices[0].message.content)
```

## 8. Day to day

### Watching it

- `docker logs -f freetoken-rdna3`: one `Prefill batch` line per prompt chunk (new vs cached tokens) and a
  `Decode batch` line every few dozen steps (requests running, tokens in context, tokens/s).
- `curl -s localhost:1919/v1/stats`: throughput, cache use, request counts.
- `curl -s 'localhost:1919/v1/requests?since=0'`: the recent requests with their time to first token.

### Running it as a service

Stop the foreground server first (`serve.sh` removes any container named `freetoken-rdna3` before it starts). A systemd
user unit, `~/.config/systemd/user/freetoken-rdna3.service` (adjust the two paths to where you cloned the repository
and downloaded the model):

```ini
[Unit]
Description=freetoken-rdna3 (Qwen3.8-Flash-Next)

[Service]
# at boot the user manager can start before the Docker daemon: wait for it
ExecStartPre=/bin/sh -c 'until docker info >/dev/null 2>&1; do sleep 2; done'
ExecStart=%h/freetoken-rdna3/rdna3/serve.sh xtx-xt --model %h/models/Qwen3.8-Flash-Next-NVFP4
ExecStop=/usr/bin/env docker stop -t 30 freetoken-rdna3
Restart=on-failure
RestartSec=15
TimeoutStartSec=300
TimeoutStopSec=60

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now freetoken-rdna3
loginctl enable-linger "$USER"      # keep it running when you are logged out
```

### Updating

```bash
git pull
docker tag freetoken-rdna3:latest freetoken-rdna3:prev     # keep the current image to go back to
docker build -f Dockerfile.rdna3 -t freetoken-rdna3:latest .
```

Then restart the server. The first start of the new image compiles its kernels again (a few minutes, once).
`rdna3/serve.sh --check` says when the image was built from another commit than the checkout. To go back:
`docker tag freetoken-rdna3:prev freetoken-rdna3:latest`, or `--image freetoken-rdna3:prev` for one run.

Each image keeps its compiled kernels in its own volume; `docker volume ls --filter name=freetoken-rdna3-kcache` lists
them, and the volume of an image you deleted can go (`docker volume rm <name>`).

## Next

- Something goes wrong: [troubleshooting.md](troubleshooting.md).
- Long contexts (four 262k conversations at once, or one of 1M): [long-context.md](long-context.md).
- What each profile sets and how to adapt one: [profiles.md](profiles.md).
- RAM, other GPUs, other models: [limits.md](limits.md).
