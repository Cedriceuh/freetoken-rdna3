"""Latency of tiny bf16 all-reduces between the two GPUs (the decode collectives: hidden 2560 per token).
Run with torchrun --nproc-per-node 2; RCCL/NCCL env vars under test are inherited."""
import os, time
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
res = {}
for n in (2560, 2560 * 4, 124160 // 1):  # hidden, a 4-token batch, one lm_head logits gather-sized buffer
    x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
    for _ in range(50):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    # eager loop (host launch each time)
    t = time.perf_counter()
    for _ in range(500):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    eager = (time.perf_counter() - t) / 500 * 1e6
    # CUDA-graph replay, like decode
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(100):
                dist.all_reduce(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    graph = (time.perf_counter() - t) / 1000 * 1e6
    res[n] = (eager, graph)
if rank == 0:
    tag = os.environ.get("BENCH_TAG", "default")
    print(tag + " | " + " | ".join(f"n={n}: eager {e:.1f} us, graph {g:.1f} us" for n, (e, g) in res.items()), flush=True)
dist.destroy_process_group()
