# Getting started

From a fresh Linux machine to an OpenAI-compatible endpoint serving Qwen3.8-Flash-Next on RX 7900 cards.

## 1. What you need

| | Two cards (`xtx-xt`) | One card (`xtx` or `xt`) |
|---|---|---|
| GPUs | RX 7900 XTX 24 GB + RX 7900 XT 20 GB | one RX 7900 XTX 24 GB or RX 7900 XT 20 GB |
| System RAM (NVFP4; EXL3 3.05 / 4.05 bpw) | 82 GiB used while serving: **128 GB** machine (59 / 74 GiB) | 72 GiB used: **96 GB** is tight, 128 GB comfortable (51 / 65 GiB) |
| Disk | 135 GB for the NVFP4 model (85 / 108 GB in EXL3 3.05 / 4.05 bpw), ~27 GB for the image (its ROCm 10.1 base included), ~29 GB more while building | same |
| Software | Linux x86_64 with the in-kernel `amdgpu` driver (`/dev/kfd` present), Docker | same |
| Tested on | Ubuntu 26.04 LTS, kernel 7.0 (older kernels untested) | same |

The ROCm user space (10.1), PyTorch and Triton are inside the image: nothing ROCm-related has to be installed on the
host. Your user must be allowed to run Docker and be in the `video` and `render` groups. Why so much RAM, and what to
do with less: [limits.md](limits.md). Two host settings avoid pauses of several seconds (memory compaction kept off
the engine's locked memory, the `COMPUTE` power profile): [troubleshooting.md](troubleshooting.md#long-pauses-in-the-middle-of-a-run).

## 2. Build the image

```bash
git clone https://github.com/Cedriceuh/freetoken-rdna3 && cd freetoken-rdna3
docker build -f Dockerfile.rdna3 -t freetoken-rdna3:latest .
```

Build from a git clone, not from a downloaded archive: the build records the commit it was made from
(`/opt/FreeToken-BUILD-PROVENANCE.txt` in the image) and fails without `.git`. The first build pulls the ROCm 10.1 PyTorch
image (25 GB, pinned by digest); after that a rebuild takes a couple of minutes. `--build-arg GPU_ARCH=gfx1100`
is the default and the only architecture tested. Do not use `install.sh` or `scripts/`: they install and build
upstream's NVIDIA / CUDA wheels.

## 3. Download the model

```bash
mkdir -p ~/models
docker run --rm --user "$(id -u):$(id -g)" -e HF_HOME=/models/.cache/huggingface -v ~/models:/models \
  --entrypoint hf freetoken-rdna3:latest download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir /models/Qwen3.8-Flash-Next-NVFP4
```

The image ships the Hugging Face CLI, so nothing is installed on the host; the files belong to your user. With `hf`
already installed (`pipx install huggingface_hub`), `hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir
~/models/Qwen3.8-Flash-Next-NVFP4` does the same.

About 135 GB: 68 GB of NVFP4 experts, 51 GB of n-gram embedding tables (read from disk on demand, they never have to
fit in RAM), 16 GB of the other weights and the tokenizer files.

The EXL3 checkpoints load the same way, from `turboderp/Qwen3.8-Flash-Next-exl3` with `--revision 3.05bpw_h5_ng5` (85
GB) or `--revision 4.05bpw_h6_ng6` (108 GB): nothing to switch on
([how-it-works.md](how-it-works.md#exl3-checkpoints)). On the agentic benchmark they fixed 13 and 12 of 29 bugs (one
run each; NVFP4: 12 and 18 in two runs).

## 4. Start the server

```bash
rdna3/serve.sh --list-gpus                                             # what HIP sees
rdna3/serve.sh xtx-xt --model ~/models/Qwen3.8-Flash-Next-NVFP4       # two cards
rdna3/serve.sh xtx    --model ~/models/Qwen3.8-Flash-Next-NVFP4       # one XTX
rdna3/serve.sh xt     --model ~/models/Qwen3.8-Flash-Next-NVFP4       # one XT
```

The script picks the cards by itself (for `xtx-xt` the larger one becomes rank 0) and prints them; `--gpus 1,0` forces
a choice, `--dry-run` prints the full `docker run` command instead of running it. Other options: `--port`
(default 1919), `--host` (default 127.0.0.1; 0.0.0.0 serves your network, and the API has no authentication:
[limits.md](limits.md#anything-else-to-know)), `--ctx`, `--served-name`, `--name`,
`--image`, `--memory`, `--vision` (image input, off by default:
[limits.md](limits.md#images-vision)), `--no-mtp` (decode without the MTP draft head the profiles turn on:
[limits.md](limits.md#speculative-decoding-mtp)), and `-- <extra ft serve flags>`.

Loading takes about 2.5 minutes on the reference machine (it reads ~68 GB of experts). The very first start of a new image also compiles and autotunes GPU kernels for a few
more minutes; the results are kept in a Docker volume (`freetoken-rdna3-kcache-<image id>`), so later starts are fast.
The server is ready when the log prints `API server is ready to serve on 127.0.0.1:1919`, or:

```bash
curl -s http://127.0.0.1:1919/health          # {"status": "ok", ...} once loaded ("loading" before)
```

`serve.sh` stays in the foreground (run the next commands from another terminal). Stop it with Ctrl-C or
`docker stop -t 30 freetoken-rdna3` (a clean shutdown takes ~9 s, close to docker's 10 s default grace period, so
`-t 30` leaves room).

## 5. First request

```bash
curl -s http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals."}],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 1024
}'
```

- **Sampling defaults** come from the model's `generation_config.json` (temperature 1.0, top_k 20, top_p 0.95) for any
  field the request leaves out. Keeping `top_k` at 20 (or anything up to 256) uses the fastest sampler.
- **Thinking** is on by default (the reasoning goes to `reasoning_content`); the request above turns it off with
  `"chat_template_kwargs": {"enable_thinking": false}`. With thinking on, give `max_tokens` room (several thousand): if
  the budget runs out while the model is still thinking, `content` comes back empty with `finish_reason: "length"`.
- **Tool calls** are parsed (Qwen3 coder format) and returned as OpenAI `tool_calls`.

## 6. Connect a client

Endpoints: OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`;
Anthropic-compatible `/v1/messages`. The API key is not checked (use any string).

- **OpenAI-compatible clients** (OpenCode, Codex, most SDKs): base URL `http://127.0.0.1:1919/v1`, model
  `qwen3.8-flash-next`.
- **Anthropic-compatible clients** (Claude Code, Anthropic SDKs) add `/v1/messages` themselves: base URL
  `http://127.0.0.1:1919` (for Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:1919`, any API key).

For a coding agent, set the context window to the profile's context (250000 for the two-card profiles, 131072 for
`xtx` / `xt`, 65536 for `gre`, or your `--ctx`) so the agent compacts at the right time.

## 7. Run it as a service (optional)

Stop the foreground server from step 4 first (`serve.sh` removes any container named `freetoken-rdna3` before it
starts). A systemd user unit, `~/.config/systemd/user/freetoken-rdna3.service` (adjust the two paths to where you
cloned the repository and downloaded the model):

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

## 8. Watching it

- `docker logs -f freetoken-rdna3`: one `Prefill batch` line per prompt chunk (new vs cached tokens) and a
  `Decode batch` line every few dozen steps (requests running, tokens in context, tokens/s).
- `curl -s localhost:1919/v1/stats`: throughput, cache use, request counts.
- `curl -s 'localhost:1919/v1/requests?since=0'`: the recent requests with their time to first token.

Next: [profiles.md](profiles.md) to understand or adapt the settings, [troubleshooting.md](troubleshooting.md) if
something goes wrong.
