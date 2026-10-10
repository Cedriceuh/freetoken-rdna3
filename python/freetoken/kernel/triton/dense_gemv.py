"""Decode (M == 1) split-K GEMV for dense bf16 projections on RDNA3.

At batch 1 the dense projections are weight-streaming GEMVs. On gfx1100 the rocBLAS / hipBLASLt picks (even
TunableOp-tuned) leave bandwidth on the table for several of Qwen3.8-Flash-Next's per-rank shapes; a plain
split-K Triton GEMV reads them faster under CUDA-graph replay (rdna3/bench/dense_gemv_bench.py).

Configs are a static table per (N, K) -- no autotune at run time, because the first call happens inside CUDA
graph capture. Shapes not in the table keep F.linear. ``FREETOKEN_TRITON_GEMV=0`` disables the path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# (N, K) -> (BLOCK_N, BLOCK_K, SPLIT, num_warps); only shapes measured faster than F.linear + TunableOp
_BF16_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    # Qwen3.8-Flash-Next, per rank at TP=2; tuned on the 7900 XT (the slower rank), faster on both cards
    (8240, 2560): (16, 512, 1, 8),  # GDN in_proj            XT 93.9 -> 75.3 us
    (6656, 2560): (16, 512, 1, 8),  # attention qkv_proj     XT 74.1 -> 52.2 us
    (10240, 320): (16, 512, 1, 8),  # HC input_mix up        XT 17.9 -> 16.1 us
    (512, 2560): (16, 128, 4, 8),   # router                 XT 11.0 -> 10.1 us
    (640, 2560): (32, 128, 4, 8),   # shared gate_up, index  XT 11.5 -> 10.9 us
}

_ENABLED = os.environ.get("FREETOKEN_TRITON_GEMV", "1") != "0"
# FREETOKEN_GEMV_LAST_REDUCES (default 1): a split-K config's last-arriving program sums the partials (the router:
# 48 launches fewer per decode step); the same bits as the separate reduce kernel
_LAST_REDUCES = os.environ.get("FREETOKEN_GEMV_LAST_REDUCES", "1") == "1"


@triton.jit
def _gemv_splitk_bf16(a_ptr, w_ptr, part_ptr, N, K, stride_wn, stride_pk, k_per,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROWS: tl.constexpr = 1, stride_am=0,
                      stride_pm=0, out_ptr=None, cnt_ptr=None, stride_om=0, SPLIT: tl.constexpr = 1,
                      LAST_REDUCES: tl.constexpr = False):
    """ROWS > 1 (several decoding requests): the unchanged per-row body runs once per activation row (unrolled), so
    each request gets the bits it gets alone -- F.linear at M > 1 would pick another BLAS reduction order."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    k0 = pid_k * k_per
    for r in tl.static_range(ROWS):
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for kk in range(0, k_per, BLOCK_K):
            offs_k = k0 + kk + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + r * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * a[None, :], axis=1)
        tl.store(part_ptr + r * stride_pm + pid_k * stride_pk + offs_n, acc, mask=n_mask)
    if LAST_REDUCES:
        # the last of a tile's SPLIT programs to arrive reduces its partial sums, element by element in split
        # order as _reduce_to_bf16 does (the same bits, one launch fewer); acq_rel at device scope publishes the
        # partials before the count and makes the last program read them after it, then the count is reset for
        # the next call (stream order)
        arrived = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT - 1:
            for r in tl.static_range(ROWS):
                red = tl.zeros((BLOCK_N,), dtype=tl.float32)
                for k in tl.static_range(SPLIT):
                    red += tl.load(part_ptr + r * stride_pm + k * stride_pk + offs_n, mask=n_mask, other=0.0)
                tl.store(out_ptr + r * stride_om + offs_n, red.to(tl.bfloat16), mask=n_mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")


@triton.jit
def _reduce_to_bf16(part_ptr, out_ptr, N, stride_pk, stride_pm, stride_om, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    # program (block, row): every row in one launch, each element summed over the splits in order as before
    row = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + row * stride_pm + k * stride_pk + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * stride_om + offs, acc.to(tl.bfloat16), mask=mask)


def gemv_config(weight: torch.Tensor) -> tuple[int, int, int, int] | None:
    """The table config for this bf16 weight, or None to keep F.linear."""
    if not _ENABLED or weight.dtype is not torch.bfloat16 or weight.stride(1) != 1:
        return None
    return _BF16_CONFIGS.get(tuple(weight.shape))


def gemv_counters(weight: torch.Tensor, cfg: tuple[int, int, int, int] | None) -> torch.Tensor | None:
    """Zeroed arrival counters for the last-program reduce of a split-K config (one per N tile), or None (one
    split, or FREETOKEN_GEMV_LAST_REDUCES=0). Allocate once, outside graph capture (layer finalize)."""
    if cfg is None or cfg[2] <= 1 or not _LAST_REDUCES:
        return None
    return torch.zeros(triton.cdiv(weight.shape[0], cfg[0]), dtype=torch.int32, device=weight.device)


def gemv_bf16(x: torch.Tensor, weight: torch.Tensor, cfg: tuple[int, int, int, int],
              counters: torch.Tensor | None = None) -> torch.Tensor:
    """``x`` [M, K] bf16 (contiguous, M small), ``weight`` [N, K] bf16 -> [M, N] bf16, fp32 accumulation; every row
    is computed exactly as it would be alone. ``counters`` (gemv_counters): the reduce runs in the GEMV's last
    program instead of its own launch."""
    N, K = weight.shape
    M = x.shape[0] if x.dim() == 2 else 1
    x2 = x.reshape(M, K)
    block_n, block_k, split, warps = cfg
    k_per = triton.cdiv(triton.cdiv(K, split), block_k) * block_k
    part = torch.empty((M, split, N), dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    last = counters is not None and split > 1
    _gemv_splitk_bf16[(triton.cdiv(N, block_n), split)](
        x2, weight, part, N, K, weight.stride(0), part.stride(1), k_per,
        BLOCK_N=block_n, BLOCK_K=block_k, ROWS=M, stride_am=x2.stride(0), stride_pm=part.stride(0),
        out_ptr=out if last else None, cnt_ptr=counters if last else None, stride_om=out.stride(0), SPLIT=split,
        LAST_REDUCES=last, num_warps=warps,
    )
    if not last:
        _reduce_to_bf16[(triton.cdiv(N, 512), M)](part, out, N, part.stride(1), part.stride(0), out.stride(0),
                                                  SPLIT=split, BLOCK=512, num_warps=4)
    return out


__all__ = ["gemv_bf16", "gemv_config", "gemv_counters"]
