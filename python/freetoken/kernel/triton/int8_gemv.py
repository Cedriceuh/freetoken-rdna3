"""Batch-1 split-K GEMV over weight-only INT8 rows (per-output-row fp32 scale), bf16 activations.

Decode reads every dense weight once per token; int8 halves the bytes. The weight is stored ``[N, K]`` int8 with a
``[N]`` fp32 scale (w ~= q * scale); accumulation is fp32, the scale is applied once per row in the reduce. Configs
are static per (N, K) (swept on a 7900 XT under CUDA-graph replay): the first call happens inside
CUDA-graph capture, so there is no autotune at run time.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# (N, K) -> (BLOCK_N, BLOCK_K, SPLIT, num_warps)
_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    # tuned on the 7900 XT (the slower rank) under CUDA-graph replay; in parentheses: the first, untuned tiles
    (8240, 2560): (2, 4096, 1, 4),     # GDN in_proj            30.3 us (v1 33.8)
    (2560, 3072): (2, 4096, 1, 4),     # o_proj / GDN out_proj  14.7 us (v1 21.9)
    (6656, 2560): (4, 4096, 1, 8),     # attention qkv          26.9 us (v1 33.5)
    (336, 10240): (1, 2048, 1, 8),     # HC down + inject        8.1 us (v1 15.5)
    (320, 10240): (1, 2048, 1, 8),     # top-level HC down
    (10240, 320): (1, 512, 1, 1),      # HC up                   8.8 us (v1 14.8)
    (640, 2560): (2, 512, 1, 1),       # shared gate_up, index   6.0 us (v1 13.3)
    (2560, 320): (2, 512, 1, 1),       # shared down             4.8 us (v1 11.8)
    (10240, 2560): (2, 4096, 1, 4),    # PLE key               38.4 us (v1 44.2)
    (2560, 2560): (2, 4096, 1, 4),     # PLE value             13.1 us (v1 19.3)
    (124160, 2560): (16, 512, 1, 8),   # LM head: v2 tiles gave nothing (~430 us)
    (512, 2560): (2, 512, 1, 1),       # router, when not kept bf16
    # uneven TP split (FREETOKEN_TP_SPLIT=0.55): the shared expert's gate_up is 704 / 576 rows; untuned, the 640 row
    # config (the other new shapes -- GDN 9270/7210 x 2560, out_proj 2560 x 3456/2688, down 2560 x 352/288 -- already
    # get their even-split neighbour's config from _default_config)
    (704, 2560): (2, 512, 1, 1),
    (576, 2560): (2, 512, 1, 1),
}


def _default_config(N: int, K: int) -> tuple[int, int, int, int]:
    # v2 tuning favoured 1-2 rows per program and a K block covering the row (one iteration)
    block_k = min(4096, max(512, triton.next_power_of_2(K)))
    return (2 if N >= 1024 else 1, block_k, 1, 4 if block_k >= 2048 else 1)


@triton.jit
def _gemv_splitk_int8(a_ptr, w_ptr, scale_ptr, out_ptr, part_ptr, N, K, stride_wn, stride_pk, k_per,
                      SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      ROWS: tl.constexpr = 1, stride_am=0, stride_om=0, ONE_PASS: tl.constexpr = False):
    """2D fp32 accumulator, one cross-lane reduction at the end. SPLIT == 1 applies the row scale and stores bf16
    directly; otherwise each K split writes its partial row sums for the deterministic reduce kernel (no atomics:
    their order would make greedy decoding non-reproducible).

    ROWS > 1 (several decoding requests): the SAME per-row body runs for each activation row in turn (unrolled at
    compile time), so every request gets exactly the bits it would get alone; the weight tile the first row loaded is
    still in cache for the next rows, so the weights cross VRAM about once per launch."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    k0 = pid_k * k_per
    if ONE_PASS:
        # K fits one BLOCK_K (the big dense weights): load the weight tile ONCE and apply it to every row. For one
        # row the general path computes acc = 0 + w * a, then sums it: the same values in the same [BLOCK_N,
        # BLOCK_K] shape, so each row still gets exactly its batch-of-one bits.
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
        sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
        for r in tl.static_range(ROWS):
            a = tl.load(a_ptr + r * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            row = tl.sum(w * a[None, :], axis=1) * sc
            tl.store(out_ptr + r * stride_om + offs_n, row.to(tl.bfloat16), mask=n_mask)
        return
    for r in tl.static_range(ROWS):
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for kk in range(0, k_per, BLOCK_K):
            offs_k = k0 + kk + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + r * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                        mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
            acc += w * a[None, :]
        row = tl.sum(acc, axis=1)
        if SPLIT == 1:
            row = row * tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
            tl.store(out_ptr + r * stride_om + offs_n, row.to(tl.bfloat16), mask=n_mask)
        else:
            tl.store(part_ptr + r * stride_om + pid_k * stride_pk + offs_n, row, mask=n_mask)


@triton.jit
def _gemv_int8_row_group(a_ptr, w_ptr, scale_ptr, out_ptr, N, K, stride_wn, k_per, stride_am, stride_om,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROWS: tl.constexpr):
    """2..4 rows, SPLIT == 1, K wider than one BLOCK_K: the general path's arithmetic per row (acc = 0; acc += w * a
    over the K blocks in order; sum(acc) * scale), with each weight tile loaded once for all the rows instead of
    once per row. Bit-identical to a batch of one (GPU check: every shape of the model, both cards). Measured in a
    graph at 4 rows: HC down 336 x 10240 17.7 -> 12.1 us, LM head 124160 x 2560 704 -> 499 us (7900 XT)."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    acc0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kk in range(0, k_per, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0).to(tl.float32)
        acc0 += w * tl.load(a_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)[None, :]
        acc1 += w * tl.load(a_ptr + stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)[None, :]
        if ROWS > 2:
            acc2 += w * tl.load(a_ptr + 2 * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)[None, :]
        if ROWS > 3:
            acc3 += w * tl.load(a_ptr + 3 * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)[None, :]
    sc = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    tl.store(out_ptr + offs_n, (tl.sum(acc0, axis=1) * sc).to(tl.bfloat16), mask=n_mask)
    tl.store(out_ptr + stride_om + offs_n, (tl.sum(acc1, axis=1) * sc).to(tl.bfloat16), mask=n_mask)
    if ROWS > 2:
        tl.store(out_ptr + 2 * stride_om + offs_n, (tl.sum(acc2, axis=1) * sc).to(tl.bfloat16), mask=n_mask)
    if ROWS > 3:
        tl.store(out_ptr + 3 * stride_om + offs_n, (tl.sum(acc3, axis=1) * sc).to(tl.bfloat16), mask=n_mask)


@triton.jit
def _reduce_scale_bf16(part_ptr, scale_ptr, out_ptr, N, stride_pk, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + k * stride_pk + offs, mask=mask, other=0.0)
    acc = acc * tl.load(scale_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


# FREETOKEN_INT8_ROWS_MAX (default 8): decode batches up to this many rows (concurrent requests) use the row-looped
# GEMV below instead of dequantizing the whole weight per call; each row is bit-identical to a batch of one.
ROWS_MAX = max(1, int(os.environ.get("FREETOKEN_INT8_ROWS_MAX", "8")))
# FREETOKEN_INT8_ROW_GROUPS=1 (default): several rows of a K-blocked shape share each weight tile (_gemv_int8_row_group)
_ROW_GROUPS = os.environ.get("FREETOKEN_INT8_ROW_GROUPS", "1") == "1"


def gemv_int8(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``x`` [M, K] bf16 (M <= ROWS_MAX, unit inner stride), ``qweight`` [N, K] int8, ``scale`` [N] fp32 -> [M, N]
    bf16. Every row is computed exactly as it would be alone (M = 1)."""
    N, K = qweight.shape
    M = x.shape[0] if x.dim() == 2 else 1
    x2 = x.reshape(M, K) if x.dim() != 2 else x
    block_n, block_k, split, warps = _CONFIGS.get((N, K)) or _default_config(N, K)
    k_per = triton.cdiv(triton.cdiv(K, split), block_k) * block_k
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    one_pass = split == 1 and k_per >= K and block_k >= K
    if M > 1 and split == 1 and not one_pass and _ROW_GROUPS:
        # several rows, K in several blocks: the weight tile read once per group of up to 4 rows (one-pass shapes keep
        # the kernel below: holding 4 one-pass accumulators spills, measured 4-10x slower)
        for r0 in range(0, M, 4):
            rows = min(4, M - r0)
            if rows == 1:
                _gemv_splitk_int8[(triton.cdiv(N, block_n), 1)](
                    x2[r0:], qweight, scale, out[r0:], out[r0:], N, K, qweight.stride(0), 0, k_per, SPLIT=1,
                    BLOCK_N=block_n, BLOCK_K=block_k, ROWS=1, stride_am=x2.stride(0), stride_om=out.stride(0),
                    ONE_PASS=False, num_warps=warps)
            else:
                _gemv_int8_row_group[(triton.cdiv(N, block_n),)](
                    x2[r0:], qweight, scale, out[r0:], N, K, qweight.stride(0), k_per, x2.stride(0), out.stride(0),
                    BLOCK_N=block_n, BLOCK_K=block_k, ROWS=rows, num_warps=warps)
        return out
    part = torch.empty((M, split, N), dtype=torch.float32, device=x.device) if split > 1 else out
    _gemv_splitk_int8[(triton.cdiv(N, block_n), split)](
        x2, qweight, scale, out, part, N, K, qweight.stride(0), part.stride(1) if split > 1 else 0, k_per,
        SPLIT=split, BLOCK_N=block_n, BLOCK_K=block_k, ROWS=M, stride_am=x2.stride(0),
        stride_om=part.stride(0) if split > 1 else out.stride(0),
        ONE_PASS=M > 1 and one_pass, num_warps=warps,
    )
    if split > 1:
        for r in range(M):
            _reduce_scale_bf16[(triton.cdiv(N, 512),)](part[r], scale, out[r], N, part.stride(1), SPLIT=split,
                                                        BLOCK=512, num_warps=4)
    return out


__all__ = ["gemv_int8"]
