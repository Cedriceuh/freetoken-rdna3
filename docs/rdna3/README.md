# freetoken-rdna3 documentation

| Page | For |
|---|---|
| [getting-started.md](getting-started.md) | installing, the first run, connecting a client, running it as a service |
| [profiles.md](profiles.md) | what each hardware profile sets, and adapting one |
| [long-context.md](long-context.md) | the K/V in host RAM: four 262k conversations at once, or one of 1M (experimental) |
| [options.md](options.md) | every environment variable this build adds |
| [benchmarks.md](benchmarks.md) | the measurements, and how they were taken |
| [how-it-works.md](how-it-works.md) | what changed in the engine, and why |
| [decisions.md](decisions.md) | each design choice with its trade-offs |
| [journey.md](journey.md) | the history: what worked, what did not |
| [limits.md](limits.md) | RAM, GPUs, NVIDIA, other models: the FAQ |
| [troubleshooting.md](troubleshooting.md) | symptoms and fixes, bug reports |
| [testing.md](testing.md) | checking a change, reporting numbers |
| [maintaining.md](maintaining.md) | syncing with upstream, releasing |
| [changes-from-upstream.md](changes-from-upstream.md) | every modified file, with the reason |
| [credits.md](credits.md) | who wrote what |

Upstream FreeToken's own documentation is in the parent directory ([install](../install.md), [CLI](../cli.md),
[models](../models.md), [quickstart](../quickstart.md)); its install guides are not how this build is installed: use
[getting-started.md](getting-started.md).

## Terms used in these pages

| Term | Meaning |
|---|---|
| TP | tensor parallelism: every layer split over the GPUs, which work on every token together |
| TG / decode | generating tokens, in tokens/s |
| PP / prefill | reading the prompt (prompt processing), in tokens/s or as time to first token |
| MoE, experts | Mixture of Experts: each layer holds 512 small networks, and a router sends each token to 10 of them (plus one shared expert) |
| GDN | GatedDeltaNet, the linear-attention layers (36 of the model's 48) with a fixed-size recurrent state |
| QSA | the model's full-attention layers (12 of 48), which keep a KV cache |
| KV cache | the attention layers' keys and values for every token of a conversation |
| PLE | per-layer n-gram embeddings: big lookup tables (51 GB; 33 GB in EXL3 3.05 bpw) read from disk |
| NVFP4 | 4-bit floating-point weights with shared scales, the experts' format in the RadixArk checkpoint |
| EXL3 | exllamav3's trellis format (3 or 4 bits per weight here), read by this build for experimental checkpoints ([how-it-works.md](how-it-works.md#exl3-checkpoints)) |
| MTP | multi-token prediction, the model's speculative-decoding head (on in the profiles, `rdna3/serve.sh --no-mtp` turns it off: [options.md](options.md#speculative-decoding-with-the-mtp-head-qwen38-flash-next-experimental)) |
| LDS | the GPU's on-chip shared memory (64 KiB per work group on RDNA3) |
| FTW | FreeToken's own weight format |
| RAM tier | this build's copy of evicted conversations in system RAM (`FREETOKEN_HOST_KV`) |
