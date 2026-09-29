# Testing and measuring

How changes to this build were checked, and how to check yours. Run GPU checks on a card that does **not** drive
your monitor when you experiment: a hard kernel fault resets the card, and the desktop with it.

## Report numbers for your setup

With a server running (any profile):

```bash
python3 rdna3/bench/quick_bench.py --label "2x RX 7900 XT, xt-xt"
```

About a minute; it measures decode speed, a cold ~8k-token read, an agent turn on the cached conversation and a tool
call, on text the server has never seen, and prints a Markdown table for an issue. For comparison, `xtx-xt` gives
~55 tok/s, ~4.5 s and ~1.2 s.

## CPU test suite

```bash
docker run --rm -w /opt/FreeToken --entrypoint python3 freetoken-rdna3:latest -m pytest -q -p no:cacheprovider tests
```

About 5 minutes, no GPU needed, on the tree the image was built from: 1471 passed, 501 skipped and 3 failures that
come from the environment (they need flashinfer, an NVIDIA-only package) and fail the same way on upstream
(`test_cache_budget.py::test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend`,
`test_cache_budget.py::test_adjust_config_resolves_num_tokens_generic`,
`test_offload.py::test_adjust_config_converts_moe_cache_rate_to_cache_size`).

To test source changes without rebuilding, mount your tree instead (`-v "$PWD":/src:ro -w /src -e
PYTHONPATH=/src/python`); `test_swiglu_clamp.py::test_cpu_extension_supports_swiglu_clamp` then fails too, because the
compiled CPU extension lives in the image's tree, not in yours.

## GPU checks (`rdna3/tests/`)

Each is a standalone script run inside the image with one or two GPUs (on a two-card machine, prefer the card that
does not drive your display: a faulting kernel resets its card), e.g.

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --ipc=host \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/src/python -v "$PWD":/src:ro \
  --entrypoint python3 freetoken-rdna3:latest /src/rdna3/tests/int8_gemv_check.py
```

| Script | Checks |
|---|---|
| `batch_memcpy_check.py` | the ROCm batched copy used by `--moe-prefill-hit-d2d` |
| `gemv_check.py` | the bf16 split-K GEMVs against `F.linear`, eager and under CUDA-graph capture |
| `int8_gemv_check.py` | the int8 GEMV against a dequantized reference, eager and in a graph |
| `int8_rows_check.py` | several rows at once give each row exactly what a batch of one gives, for every tuned shape |
| `moe_block_check.py` | the token-blocked NVFP4 prefill MoE returns exactly what the one-shot call returns |
| `host_allreduce_check.py` (2 GPUs: `HIP_VISIBLE_DEVICES=0,1`) | the host-memory all-reduce against RCCL |
| `host_kv_check.py` | the RAM-tier copies (device -> RAM -> device) are exact |
| `qsa_kv_int8_check.py` | the experimental int8 attention KV against bf16 and an fp32 reference |
| `run_sampling_tests.sh <module>` + `sampling_test.py` | sampling kernels in four modes, each killed after 90 s (hang detector). Runs on the host and starts its own containers (image: `FREETOKEN_IMAGE`, default `freetoken-rdna3:latest`) |
| `shape_check.py` (CPU) | the loader's per-rank tensors match the buffers the model declares, per TP rank; needs the checkpoint mounted at `/models/m` (`-v <model dir>:/models/m:ro`) |

## Micro-benchmarks (`rdna3/bench/`)

| Script | Measures |
|---|---|
| `sampling_topp_bench.py` | the ROCm sorted-threshold sampler against float64 and upstream's kernels (kept sets, seeded draws, distribution, speed) |
| `sampling_topk_bench.py` | the top-k-first sampler against the full-vocabulary path |
| `nvfp4_decode_sweep.py`, `nvfp4_prefill_sweep.py` | tile sweeps of the NVFP4 expert GEMMs (how the tuned tables were made) |
| `dense_gemv_bench.py` | decode dense projections, bf16 vs int8 |
| `router_topk_bench.py` | the MoE router softmax + top-k |
| `allreduce_bench.py` | tiny all-reduce latency between two GPUs (`torchrun --nproc-per-node 2`) |

Microbenchmarks only locate a cost: the weights stay hot in the 80-96 MB Infinity Cache and the kernel mix is not a
real step's. Decide on the whole model.

## Before trusting a change: the procedure used here

1. **Bit-exactness**: a fixed set of greedy prompts (thinking off; 120 here, not published: any set of your own, the
   same on both builds) on the old and the new build must give identical answers. A
   change that is not bit-exact goes to step 5.
2. **Speed**: decode and 8.4k-prefill windows, sessions interleaved (A B A B); a difference counts only beyond the
   spread between sessions. Never build images or run anything else on the machine meanwhile.
3. **Depth**: the growing-conversation sweep to the profile's maximum context ([benchmarks.md](benchmarks.md)).
4. **Several requests and sampling**: four concurrent agents with sampling, one request per sampler path (greedy,
   top-p only, large and small top-k) alone and batched, with `FREETOKEN_TP_SYNC_TOKENS=check` (no disagreement lines
   expected), and a smoke test with a tool call.
5. **Anything not bit-exact**: a scored benchmark to reject it early, then an agentic, multi-turn benchmark to accept
   it (the one used here is private; a long agent workload of your own, run on both builds). Short tests (scored items, needles, probes) missed a change that cost a third of the agentic score.
