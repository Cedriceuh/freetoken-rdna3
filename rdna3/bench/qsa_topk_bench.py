"""QSA block top-k (the indexer's selection, kernel/triton/qsa/topk.py) per call, CUDA-graph replays, on the page
tables of the profiles: 65,536 block columns (the 250k / 262k contexts) and 250,000 (the 1M profile), full or short
visible rows, 1 / 4 / 8 rows. Checks the winners against torch.topk and prints a checksum that compares across builds
(same seeds).

  python rdna3/bench/qsa_topk_bench.py
"""
import torch
from freetoken.kernel.triton.qsa import qsa_block_topk, qsa_block_topk_scratch_width

torch.manual_seed(0)
dev = torch.device("cuda")
for columns, visible_n in ((65536, 59000), (250000, 247000), (250000, 30000)):
    for rows in (1, 4, 8):
        logits = torch.randn(rows, columns, device=dev)
        visible = torch.full((rows,), visible_n, dtype=torch.int32, device=dev)
        width = 512
        scratch = torch.empty((rows, max(1, qsa_block_topk_scratch_width(columns, width))), dtype=torch.int32, device=dev)
        out = torch.empty(rows, width, dtype=torch.int32, device=dev)
        for _ in range(3):
            qsa_block_topk(logits, visible, out, scratch=scratch)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                qsa_block_topk(logits, visible, out, scratch=scratch)
        best = 1e9
        for _ in range(10):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); g.replay(); e.record(); torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) * 1000 / 20)
        ref = torch.topk(logits[:, :visible_n], width, dim=1).indices.sort(dim=1).values
        got = out.sort(dim=1).values
        print(f"columns {columns} visible {visible_n} rows {rows}: {best:6.1f} us  winners==torch.topk: "
              f"{bool((ref == got).all())}  checksum {int(got.long().sum())}", flush=True)
