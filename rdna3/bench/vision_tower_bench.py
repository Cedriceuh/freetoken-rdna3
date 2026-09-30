"""Qwen3.8-Flash-Next vision tower alone on one GPU: encode time, peak VRAM and the attention kernels per image size.

Runs inside the image with the tree at /src, the checkpoint at /models/m and one GPU (HIP_VISIBLE_DEVICES):

  python3 /src/rdna3/bench/vision_tower_bench.py [--weights host|gpu] [--tokens 256,1024,...] [--save emb.pt]

Images are seeded random pixels (the tower's cost depends on the size only). --save writes the embeddings of every
size, to compare two cards bit for bit.
"""
import argparse
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/src/python")
from freetoken.distributed import set_tp_info
from freetoken.layers.quantization import finalize_quant
from freetoken.models.qwen3_vl.config import parse_vision_config
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel
from freetoken.models.qwen4_exp.weight import iter_vision_weights
from freetoken.utils import cached_load_hf_config, torch_dtype

M = "/models/m"
MiB = 1 << 20


def grid(tokens: int, merge: int) -> tuple[int, int]:
    """(h, w) in patches for ``tokens`` merged tokens, as square as the count allows."""
    rows = max(d for d in range(1, int(tokens**0.5) + 1) if tokens % d == 0)
    return rows * merge, tokens // rows * merge


def pixels(h: int, w: int, patch_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(h * w, patch_dim, generator=g).to(torch.bfloat16)


def attention_kernels(dev: torch.device) -> str:
    """Names of the GPU kernels one default SDPA call launches at the tower's head shape."""
    q = torch.randn(1, 16, 1024, 72, device=dev, dtype=torch.bfloat16)
    F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        F.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA}
    return ", ".join(sorted(n[:60] for n in names)) or "(none recorded)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", choices=["host", "gpu"], default="host")
    ap.add_argument("--tokens", default="256,1024,2048,4096,8192,16384")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--save")
    a = ap.parse_args()

    set_tp_info(rank=0, size=1)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    vc = parse_vision_config(cached_load_hf_config(M))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        tower = Qwen3VLVisionModel(vc)
    free0 = torch.cuda.mem_get_info()[0]
    tower.load_state_dict({n.removeprefix("visual."): t for n, t in iter_vision_weights(M, dev)})
    finalize_quant(tower)
    tower.place_weights(a.weights)
    torch.cuda.empty_cache()
    print(f"device {torch.cuda.get_device_name(dev)}, {torch.cuda.get_device_properties(dev).multi_processor_count} CUs; "
          f"torch {torch.__version__}, HIP {torch.version.hip}")
    print(f"tower weights ({a.weights}): {(free0 - torch.cuda.mem_get_info()[0]) / MiB:.0f} MiB of VRAM")
    print(f"attention kernels: {attention_kernels(dev)}")

    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
    merge = vc.spatial_merge_size
    print("| tokens | patches | pixels (approx.) | encode ms (min of reps) | peak VRAM above weights MiB |")
    print("|---:|---:|---:|---:|---:|")
    saved = {}
    for tokens in (int(t) for t in a.tokens.split(",")):
        h, w = grid(tokens, merge)
        px = pixels(h, w, patch_dim, seed=tokens)
        try:
            tower.forward(px, [[1, h, w]])  # warm: first call of a shape compiles / picks kernels
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(a.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                emb = tower.forward(px, [[1, h, w]])
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
            peak = (torch.cuda.max_memory_allocated() - base) / MiB
            if a.save:
                saved[tokens] = emb.cpu()
            del emb
            print(f"| {tokens} | {h * w} | {h * w * vc.patch_size**2 / 1e6:.1f} MP | {min(times):.0f} | {peak:.0f} |",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"| {tokens} | {h * w} | {h * w * vc.patch_size**2 / 1e6:.1f} MP | OOM | - |", flush=True)
        torch.cuda.empty_cache()

    if a.save:
        torch.save(saved, a.save)
        print(f"saved the embeddings of {sorted(saved)} tokens to {a.save}")


if __name__ == "__main__":
    main()
