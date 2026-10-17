# freetoken-rdna3 documentation

New here? Start with [getting-started.md](getting-started.md): from a Linux machine to a running server and your tools
connected to it.

| I want to | Page |
|---|---|
| install it, run it, connect a client, run it as a service, update it | [getting-started.md](getting-started.md) |
| fix something that goes wrong, report a bug | [troubleshooting.md](troubleshooting.md) |
| know if my RAM, GPUs or model will work (FAQ) | [limits.md](limits.md) |
| understand or adapt a hardware profile | [profiles.md](profiles.md) |
| serve four 262k conversations at once, or one of 1M | [long-context.md](long-context.md) |
| let the CPU compute part of the experts (`--hybrid`, off by default) | [cpu-experts.md](cpu-experts.md) |
| look up an environment variable this build adds | [options.md](options.md) |
| see the measurements and how they were taken | [benchmarks.md](benchmarks.md) |
| understand what changed in the engine, and why | [how-it-works.md](how-it-works.md), [decisions.md](decisions.md) |
| know what was tried and dropped (dated history) | [journey.md](journey.md) |
| check a change, report numbers | [testing.md](testing.md) |
| sync with upstream, release | [maintaining.md](maintaining.md), [changes-from-upstream.md](changes-from-upstream.md) |
| see who wrote what | [credits.md](credits.md) |

Upstream FreeToken's own documentation is in the parent directory ([install](../install.md), [CLI](../cli.md),
[models](../models.md), [quickstart](../quickstart.md)); its install guides are not how this build is installed: use
[getting-started.md](getting-started.md).

## Terms used in these pages

| Term | Meaning |
|---|---|
| profile | one file in `rdna3/profiles/` with every setting for one GPU setup (`xtx-xt`, `xtx`, ...) |
| checkpoint | one quantized version of the model's weights: NVFP4, EXL3 3.05 bpw or EXL3 4.05 bpw |
| TP | tensor parallelism: every layer split over the GPUs, which work on every token together |
| TG / decode | generating tokens, in tokens/s |
| PP / prefill | reading the prompt (prompt processing), in tokens/s or as time to first token |
| agent turn | a short new message (a tool result) added to a conversation the server already holds; measured as time to first token |
| MoE, experts | Mixture of Experts: each layer holds 512 small networks, and a router sends each token to 10 of them (plus one shared expert) |
| expert cache | the VRAM left after the other weights and the K/V, filled with the most used experts; the others are copied from RAM when needed (a miss) |
| GDN | GatedDeltaNet, the linear-attention layers (36 of the model's 48) with a fixed-size recurrent state |
| QSA | the model's full-attention layers (12 of 48), which keep a KV cache |
| KV cache, K/V | the attention layers' keys and values for every token of a conversation |
| PLE | per-layer n-gram embeddings: big lookup tables (51 GB; 33 GB in EXL3 3.05 bpw) read from disk |
| NVFP4 | 4-bit floating-point weights with shared scales, the experts' format in the RadixArk checkpoint |
| EXL3 | exllamav3's trellis format (3 or 4 bits per weight here), read by this build for its EXL3 checkpoints ([how-it-works.md](how-it-works.md#exl3-checkpoints)) |
| MTP | multi-token prediction, the model's speculative-decoding head: it drafts tokens that one step checks (on in the profiles, `rdna3/serve.sh --no-mtp` turns it off: [options.md](options.md#speculative-decoding-with-the-mtp-head-qwen38-flash-next)) |
| greedy | decoding that always takes the most likely token (temperature 0): a request alone gives the same answer to the same prompt, which is how exactness is checked |
| bit-exact | a change that gives the same greedy answers as without it |
| agentic benchmark | the maintainers' private test: 29 planted bugs an agent must fix in long sessions; the reference for changes that are not bit-exact |
| LDS | the GPU's on-chip shared memory (64 KiB per work group on RDNA3) |
| FTW | FreeToken's own weight format |
| RAM tier | this build's copy of evicted conversations in system RAM (`FREETOKEN_HOST_KV`) |
