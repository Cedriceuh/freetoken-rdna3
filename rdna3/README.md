# rdna3/

| Path | What |
|---|---|
| `serve.sh` | the launcher: `rdna3/serve.sh <profile> --model DIR`; `--check` looks at the machine and suggests a profile ([getting started](../docs/rdna3/getting-started.md)) |
| `download.sh` | downloads a checkpoint with the image's Hugging Face CLI: `rdna3/download.sh nvfp4\|exl3-3.05\|exl3-4.05 [DIR]` |
| `profiles/` | one file per hardware setup ([profiles](../docs/rdna3/profiles.md)) |
| `tests/` | GPU checks of the kernels this build adds ([testing](../docs/rdna3/testing.md)) |
| `bench/` | micro-benchmarks, tile sweeps, and `quick_bench.py` for a running server |
| `tools/` | `upstream-status.sh` (sync helper), `privacy_scan.sh` (release check), `make-llms-full.sh`, `check_doc_links.py` (links, anchors, llms-full.txt up to date; run by CI) |
