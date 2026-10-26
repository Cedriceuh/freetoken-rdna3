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

from freetoken.kernel.triton.activation import _LOG2E, _fast_ex2


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


# FREETOKEN_INT8_TP_TILES=1 (opt-in, not bit-exact): K blocks of 512 for the GDN in_proj and attention qkv of xtx-xt's
# uneven split, which then take several rows through the row-group kernel instead of the one-pass path, whose four
# accumulators spill; an 8-wide, 4-warp tile for the HC up projection (still one pass; it then keeps its separate gate
# mix, gemv_int8_hc_mix wanting the 1-wide config). Each row is summed in another order than with the shipped configs,
# so greedy answers change at rounding level (6 of 14 prompts), and Rule 0 (docs/rdna3/decisions.md) wants such a
# change validated on the agentic benchmark first. xtx-xt-4x262k, EXL3 3.05: forward pass -1.5 / -2.6 / -2.9 % at 2 /
# 3 / 4 rows, -2.3 % at four requests. Configs that keep the bits (other BLOCK_N only) took 3-11 % less per call cold
# but nothing in serving.
if os.environ.get("FREETOKEN_INT8_TP_TILES", "0") == "1":
    _CONFIGS.update({
        (9270, 2560): (1, 512, 1, 2),
        (7210, 2560): (1, 512, 1, 2),
        (6656, 2560): (4, 512, 1, 4),
        (10240, 320): (8, 512, 1, 4),
    })


# shapes that take another shape's tiles (``share_config``): an uneven LM head's shards (FREETOKEN_LMHEAD_SPLIT) take
# the even head's, so its K order and its bits
_SHARED: dict[tuple[int, int], tuple[int, int]] = {}


def share_config(shape: tuple[int, int], like: tuple[int, int]) -> None:
    """Run ``shape`` (N, K) with the tiles of ``like``."""
    _SHARED[tuple(shape)] = tuple(like)


def _config(N: int, K: int) -> tuple[int, int, int, int]:
    N, K = _SHARED.get((N, K), (N, K))
    return _CONFIGS.get((N, K)) or _default_config(N, K)


def _default_config(N: int, K: int) -> tuple[int, int, int, int]:
    # v2 tuning favoured 1-2 rows per program and a K block covering the row (one iteration)
    block_k = min(4096, max(512, triton.next_power_of_2(K)))
    return (2 if N >= 1024 else 1, block_k, 1, 4 if block_k >= 2048 else 1)


@triton.jit
def _out_bf16(row, offs_n, EPI_HC: tl.constexpr, EPI_COLS):
    """A row of results (fp32, scaled) as stored. EPI_HC > 0: columns below EPI_COLS get hc_silu (kernel/triton/hc.py)
    of their bf16 value, the bits that kernel stores (silu(x / EPI_HC) rounded to bf16), the others stay as they are."""
    y = row.to(tl.bfloat16)
    if EPI_HC > 0:
        x = y.to(tl.float32) / EPI_HC
        y = tl.where(offs_n < EPI_COLS, (x * tl.sigmoid(x)).to(tl.bfloat16), y)
    return y


@triton.jit
def _gemv_splitk_int8(a_ptr, w_ptr, scale_ptr, out_ptr, part_ptr, N, K, stride_wn, stride_pk, k_per,
                      SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      ROWS: tl.constexpr = 1, stride_am=0, stride_om=0, ONE_PASS: tl.constexpr = False,
                      EPI_HC: tl.constexpr = 0, EPI_COLS=0):
    """2D fp32 accumulator, one cross-lane reduction at the end. SPLIT == 1 applies the row scale and stores bf16
    directly; otherwise each K split writes its partial row sums for the deterministic reduce kernel (no atomics:
    their order would make greedy decoding non-reproducible).

    ROWS > 1 (several decoding requests): the SAME per-row body runs for each activation row in turn (unrolled at
    compile time), so every request gets exactly the bits it would get alone; the weight tile the first row loaded is
    still in cache for the next rows, so the weights cross VRAM about once per launch."""
    _gemv_splitk_int8_body(tl.program_id(0), tl.program_id(1), a_ptr, w_ptr, scale_ptr, out_ptr, part_ptr, N, K,
                           stride_wn, stride_pk, k_per, SPLIT, BLOCK_N, BLOCK_K, ROWS, stride_am, stride_om, ONE_PASS,
                           EPI_HC, EPI_COLS)


@triton.jit
def _gemv_splitk_int8_body(pid_n, pid_k, a_ptr, w_ptr, scale_ptr, out_ptr, part_ptr, N, K, stride_wn, stride_pk, k_per,
                           SPLIT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROWS: tl.constexpr,
                           stride_am, stride_om, ONE_PASS: tl.constexpr, EPI_HC: tl.constexpr, EPI_COLS):
    """``_gemv_splitk_int8``'s program (pid_n, pid_k), callable from a kernel that runs other work beside it."""
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
            tl.store(out_ptr + r * stride_om + offs_n, _out_bf16(row, offs_n, EPI_HC, EPI_COLS), mask=n_mask)
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
            tl.store(out_ptr + r * stride_om + offs_n, _out_bf16(row, offs_n, EPI_HC, EPI_COLS), mask=n_mask)
        else:
            tl.store(part_ptr + r * stride_om + pid_k * stride_pk + offs_n, row, mask=n_mask)


@triton.jit
def _gemv_int8_row_group(a_ptr, w_ptr, scale_ptr, out_ptr, N, K, stride_wn, k_per, stride_am, stride_om,
                         BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROWS: tl.constexpr,
                         EPI_HC: tl.constexpr = 0, EPI_COLS=0):
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
    tl.store(out_ptr + offs_n, _out_bf16(tl.sum(acc0, axis=1) * sc, offs_n, EPI_HC, EPI_COLS),
             mask=n_mask)
    tl.store(out_ptr + stride_om + offs_n, _out_bf16(tl.sum(acc1, axis=1) * sc, offs_n, EPI_HC, EPI_COLS),
             mask=n_mask)
    if ROWS > 2:
        tl.store(out_ptr + 2 * stride_om + offs_n, _out_bf16(tl.sum(acc2, axis=1) * sc, offs_n, EPI_HC, EPI_COLS),
             mask=n_mask)
    if ROWS > 3:
        tl.store(out_ptr + 3 * stride_om + offs_n, _out_bf16(tl.sum(acc3, axis=1) * sc, offs_n, EPI_HC, EPI_COLS),
             mask=n_mask)


@triton.jit
def _reduce_scale_bf16(part_ptr, scale_ptr, out_ptr, N, stride_pk, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + k * stride_pk + offs, mask=mask, other=0.0)
    acc = acc * tl.load(scale_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _hc_mix_row(acc, w, a_ptr, sc, rn_ptr, n, offs_k, k_mask, one):
    """One row's contribution of one stream to the mix: its gate column as the GEMV computes it (rounded to bf16), then
    _hc_gate_mix_kernel's own accumulation."""
    a = tl.load(a_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
    g = (tl.sum(w * a[None, :], axis=1) * sc).to(tl.bfloat16)
    x = tl.load(rn_ptr + n + one)
    acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    return acc


@triton.jit
def _gemv_int8_hc_mix(a_ptr, w_ptr, scale_ptr, rn_ptr, out_ptr, K, HD, stride_wn, stride_am, stride_rn, stride_om,
                      HC: tl.constexpr, BLOCK_K: tl.constexpr, ROWS: tl.constexpr):
    """The hyper-connection's up projection and hc_gate_mix in one pass, for a K that fits one BLOCK_K and up to 8 rows:
    the program of hidden index h loads each of the HC weight rows h, HD + h, ... once, computes that gate column of
    every row as the GEMV does (a [1, BLOCK_K] tile, one warp, rounded to bf16) and mixes it with _hc_gate_mix_kernel's
    own expression, in stream order (kernel/triton/hc.py): out[r, h] = sum_k sigmoid(gate[r, k * HD + h]) *
    rn[r, k * HD + h] / HC. The same bits as the two kernels."""
    h = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    one = tl.zeros((1,), dtype=tl.int32)
    acc0 = tl.zeros((1,), dtype=tl.float32)
    acc1 = tl.zeros((1,), dtype=tl.float32)
    acc2 = tl.zeros((1,), dtype=tl.float32)
    acc3 = tl.zeros((1,), dtype=tl.float32)
    acc4 = tl.zeros((1,), dtype=tl.float32)
    acc5 = tl.zeros((1,), dtype=tl.float32)
    acc6 = tl.zeros((1,), dtype=tl.float32)
    acc7 = tl.zeros((1,), dtype=tl.float32)
    for stream in tl.static_range(HC):
        n = stream * HD + h
        w = tl.load(w_ptr + n * stride_wn + offs_k[None, :], mask=k_mask[None, :], other=0).to(tl.float32)
        sc = tl.load(scale_ptr + n + one)
        acc0 = _hc_mix_row(acc0, w, a_ptr, sc, rn_ptr, n, offs_k, k_mask, one)
        if ROWS > 1:
            acc1 = _hc_mix_row(acc1, w, a_ptr + stride_am, sc, rn_ptr + stride_rn, n, offs_k, k_mask, one)
        if ROWS > 2:
            acc2 = _hc_mix_row(acc2, w, a_ptr + 2 * stride_am, sc, rn_ptr + 2 * stride_rn, n, offs_k, k_mask, one)
        if ROWS > 3:
            acc3 = _hc_mix_row(acc3, w, a_ptr + 3 * stride_am, sc, rn_ptr + 3 * stride_rn, n, offs_k, k_mask, one)
        if ROWS > 4:
            acc4 = _hc_mix_row(acc4, w, a_ptr + 4 * stride_am, sc, rn_ptr + 4 * stride_rn, n, offs_k, k_mask, one)
        if ROWS > 5:
            acc5 = _hc_mix_row(acc5, w, a_ptr + 5 * stride_am, sc, rn_ptr + 5 * stride_rn, n, offs_k, k_mask, one)
        if ROWS > 6:
            acc6 = _hc_mix_row(acc6, w, a_ptr + 6 * stride_am, sc, rn_ptr + 6 * stride_rn, n, offs_k, k_mask, one)
        if ROWS > 7:
            acc7 = _hc_mix_row(acc7, w, a_ptr + 7 * stride_am, sc, rn_ptr + 7 * stride_rn, n, offs_k, k_mask, one)
    tl.store(out_ptr + h + one, (acc0 / HC).to(tl.bfloat16))
    if ROWS > 1:
        tl.store(out_ptr + stride_om + h + one, (acc1 / HC).to(tl.bfloat16))
    if ROWS > 2:
        tl.store(out_ptr + 2 * stride_om + h + one, (acc2 / HC).to(tl.bfloat16))
    if ROWS > 3:
        tl.store(out_ptr + 3 * stride_om + h + one, (acc3 / HC).to(tl.bfloat16))
    if ROWS > 4:
        tl.store(out_ptr + 4 * stride_om + h + one, (acc4 / HC).to(tl.bfloat16))
    if ROWS > 5:
        tl.store(out_ptr + 5 * stride_om + h + one, (acc5 / HC).to(tl.bfloat16))
    if ROWS > 6:
        tl.store(out_ptr + 6 * stride_om + h + one, (acc6 / HC).to(tl.bfloat16))
    if ROWS > 7:
        tl.store(out_ptr + 7 * stride_om + h + one, (acc7 / HC).to(tl.bfloat16))


@triton.jit
def _act_pair_out(sum_g, sum_u, sg, su, out_ptr, offs_n, n_mask):
    """The gate and up columns as the GEMV stores them (scaled, rounded to bf16), then silu_and_mul's own expression
    (kernel/triton/activation.py), stored as bf16."""
    g = (sum_g * sg).to(tl.bfloat16).to(tl.float32)
    u = (sum_u * su).to(tl.bfloat16).to(tl.float32)
    y = g / (1.0 + _fast_ex2(-g * _LOG2E)) * u
    tl.store(out_ptr + offs_n, y.to(tl.bfloat16), mask=n_mask)


@triton.jit
def _gemv_int8_act_pair(a_ptr, w_ptr, scale_ptr, out_ptr, D, K, stride_wn, k_per, stride_am, stride_om,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROWS: tl.constexpr, ONE_PASS: tl.constexpr):
    """A merged gate | up projection (D gate rows, then D up rows) and silu_and_mul in one kernel: each program computes
    BLOCK_N gate columns and the BLOCK_N up columns D further with _gemv_splitk_int8's arithmetic (the same tile
    shapes, each weight tile loaded once for the program's rows: all of them when K fits one BLOCK_K, else up to 2), so
    every value gets the GEMV's bits; then silu(gate) * up as activation.py computes it. Output [ROWS, D]."""
    _act_pair_body(tl.program_id(0), a_ptr, w_ptr, scale_ptr, out_ptr, D, K, stride_wn, k_per, stride_am, stride_om,
                   BLOCK_N, BLOCK_K, ROWS, ONE_PASS)


@triton.jit
def _act_pair_body(pid_n, a_ptr, w_ptr, scale_ptr, out_ptr, D, K, stride_wn, k_per, stride_am, stride_om,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ROWS: tl.constexpr, ONE_PASS: tl.constexpr):
    """``_gemv_int8_act_pair``'s program ``pid_n``, callable from a kernel that runs other work beside it."""
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < D
    sg = tl.load(scale_ptr + offs_n, mask=n_mask, other=0.0)
    su = tl.load(scale_ptr + D + offs_n, mask=n_mask, other=0.0)
    if ONE_PASS:
        offs_k = tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        m2 = n_mask[:, None] & k_mask[None, :]
        wg = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=m2, other=0).to(tl.float32)
        wu = tl.load(w_ptr + (D + offs_n)[:, None] * stride_wn + offs_k[None, :], mask=m2, other=0).to(tl.float32)
        for r in tl.static_range(ROWS):
            a = tl.load(a_ptr + r * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            _act_pair_out(tl.sum(wg * a[None, :], axis=1), tl.sum(wu * a[None, :], axis=1), sg, su,
                          out_ptr + r * stride_om, offs_n, n_mask)
        return
    accg0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    accu0 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    accg1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    accu1 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    accg2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    accu2 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    accg3 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    accu3 = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for kk in range(0, k_per, BLOCK_K):
        offs_k = kk + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        m2 = n_mask[:, None] & k_mask[None, :]
        wg = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=m2, other=0).to(tl.float32)
        wu = tl.load(w_ptr + (D + offs_n)[:, None] * stride_wn + offs_k[None, :], mask=m2, other=0).to(tl.float32)
        a0 = tl.load(a_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        accg0 += wg * a0[None, :]
        accu0 += wu * a0[None, :]
        if ROWS > 1:
            a1 = tl.load(a_ptr + stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            accg1 += wg * a1[None, :]
            accu1 += wu * a1[None, :]
        if ROWS > 2:
            a2 = tl.load(a_ptr + 2 * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            accg2 += wg * a2[None, :]
            accu2 += wu * a2[None, :]
        if ROWS > 3:
            a3 = tl.load(a_ptr + 3 * stride_am + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            accg3 += wg * a3[None, :]
            accu3 += wu * a3[None, :]
    _act_pair_out(tl.sum(accg0, axis=1), tl.sum(accu0, axis=1), sg, su, out_ptr, offs_n, n_mask)
    if ROWS > 1:
        _act_pair_out(tl.sum(accg1, axis=1), tl.sum(accu1, axis=1), sg, su, out_ptr + stride_om, offs_n, n_mask)
    if ROWS > 2:
        _act_pair_out(tl.sum(accg2, axis=1), tl.sum(accu2, axis=1), sg, su, out_ptr + 2 * stride_om, offs_n, n_mask)
    if ROWS > 3:
        _act_pair_out(tl.sum(accg3, axis=1), tl.sum(accu3, axis=1), sg, su, out_ptr + 3 * stride_om, offs_n, n_mask)


# FREETOKEN_INT8_ROWS_MAX (default 8): decode batches up to this many rows (concurrent requests) use the row-looped
# GEMVs instead of dequantizing the whole weight per call; each row is bit-identical to a batch of one.
ROWS_MAX = max(1, int(os.environ.get("FREETOKEN_INT8_ROWS_MAX", "8")))
# FREETOKEN_INT8_ROW_GROUPS=1 (default): several rows of a K-blocked shape share each weight tile (_gemv_int8_row_group)
_ROW_GROUPS = os.environ.get("FREETOKEN_INT8_ROW_GROUPS", "1") == "1"
# FREETOKEN_INT8_ACT_PAIR4 (default 1): a K-blocked gate | up projection at 3-4 rows (a verify step) runs its GEMV and
# silu_and_mul in one launch (_gemv_int8_act_pair, one column pair a program) instead of the row-group GEMV then the
# activation; the same bits
_ACT_PAIR4 = os.environ.get("FREETOKEN_INT8_ACT_PAIR4", "1") == "1"


def gemv_int8(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, *, silu_hc: int = 0,
              silu_cols: int = 0) -> torch.Tensor:
    """``x`` [M, K] bf16 (M <= ROWS_MAX, unit inner stride), ``qweight`` [N, K] int8, ``scale`` [N] fp32 -> [M, N]
    bf16. Every row is computed exactly as it would be alone (M = 1). ``silu_hc`` > 0: the output columns below
    ``silu_cols`` come out as ``hc_silu(out[:, :silu_cols], silu_hc)`` (kernel/triton/hc.py), bit for bit."""
    N, K = qweight.shape
    M = x.shape[0] if x.dim() == 2 else 1
    x2 = x.reshape(M, K) if x.dim() != 2 else x
    block_n, block_k, split, warps = _config(N, K)
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
                    ONE_PASS=False, EPI_HC=silu_hc, EPI_COLS=silu_cols, num_warps=warps)
            else:
                _gemv_int8_row_group[(triton.cdiv(N, block_n),)](
                    x2[r0:], qweight, scale, out[r0:], N, K, qweight.stride(0), k_per, x2.stride(0), out.stride(0),
                    BLOCK_N=block_n, BLOCK_K=block_k, ROWS=rows, EPI_HC=silu_hc, EPI_COLS=silu_cols, num_warps=warps)
        return out
    part = torch.empty((M, split, N), dtype=torch.float32, device=x.device) if split > 1 else out
    _gemv_splitk_int8[(triton.cdiv(N, block_n), split)](
        x2, qweight, scale, out, part, N, K, qweight.stride(0), part.stride(1) if split > 1 else 0, k_per,
        SPLIT=split, BLOCK_N=block_n, BLOCK_K=block_k, ROWS=M, stride_am=x2.stride(0),
        stride_om=part.stride(0) if split > 1 else out.stride(0),
        ONE_PASS=M > 1 and one_pass, EPI_HC=silu_hc if split == 1 else 0, EPI_COLS=silu_cols, num_warps=warps,
    )
    if split > 1:
        for r in range(M):
            _reduce_scale_bf16[(triton.cdiv(N, 512),)](part[r], scale, out[r], N, part.stride(1), SPLIT=split,
                                                        BLOCK=512, num_warps=4)
        if silu_hc:
            from freetoken.kernel.triton.hc import hc_silu

            out[:, :silu_cols] = hc_silu(out[:, :silu_cols], silu_hc)
    return out


def gemv_int8_hc_mix(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, rn: torch.Tensor, hc: int):
    """``hc_gate_mix(rn, gemv_int8(x, qweight, scale), hc)`` (kernel/triton/hc.py) in one kernel, the same bits: ``x``
    [M, K] bf16 (unit inner stride, any row stride), ``qweight`` [hc * HD, K] int8 whose config is the one-pass
    (1, BLOCK_K >= K, 1, 1) one, ``rn`` [M, hc * HD] bf16 -> [M, HD] bf16. None when the config is another one."""
    N, K = qweight.shape
    M = x.shape[0]
    cfg = _config(N, K)
    if cfg[0] != 1 or cfg[2] != 1 or cfg[3] != 1 or cfg[1] < K or N % hc or M > min(ROWS_MAX, 8) or rn.stride(1) != 1:
        return None
    HD = N // hc
    out = torch.empty((M, HD), dtype=torch.bfloat16, device=x.device)
    _gemv_int8_hc_mix[(HD,)](x, qweight, scale, rn, out, K, HD, qweight.stride(0), x.stride(0), rn.stride(0),
                             out.stride(0), HC=hc, BLOCK_K=cfg[1], ROWS=M, num_warps=1)
    return out


def gemv_int8_act_mul(x: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor | None:
    """``silu_and_mul(gemv_int8(x, qweight, scale))`` (the Triton activation, kernel/triton/activation.py) in one
    kernel, the same bits: ``qweight`` [2D, K] (gate rows, then up rows) -> [M, D] bf16. None for a split-K config, and
    past 2 rows for a K-blocked one."""
    N, K = qweight.shape
    M = x.shape[0] if x.dim() == 2 else 1
    x2 = x.reshape(M, K) if x.dim() != 2 else x
    block_n, block_k, split, warps = _config(N, K)
    if split != 1 or N % 2 or M > ROWS_MAX:
        return None
    D = N // 2
    k_per = triton.cdiv(K, block_k) * block_k
    one_pass = k_per >= K and block_k >= K
    pair4 = _ACT_PAIR4 and not one_pass and 2 < M <= 4
    if not one_pass and M > 2 and not pair4:
        # K-blocked shapes: the pair kernel shares tiles across 2 rows, the row-group GEMV across 4; past 2 rows the
        # two kernels in a row are as fast or faster (cold, 48 matrices: 288-352 wide, 3-6 rows within 1 us, 7-8 rows
        # +2 to +6 us)
        return None
    if pair4:
        # FREETOKEN_INT8_ACT_PAIR4: 3-4 rows of a K-blocked shape in one launch, one gate and one up column a program
        # (BLOCK_N is the one tile size that keeps the sums), so its eight accumulators fit the registers
        block_n = 1
    out = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
    step = M if one_pass or pair4 else 2
    for r0 in range(0, M, step):
        rows = min(step, M - r0)
        _gemv_int8_act_pair[(triton.cdiv(D, block_n),)](
            x2[r0:], qweight, scale, out[r0:], D, K, qweight.stride(0), k_per, x2.stride(0), out.stride(0),
            BLOCK_N=block_n, BLOCK_K=block_k, ROWS=rows, ONE_PASS=one_pass, num_warps=warps)
    return out


__all__ = ["gemv_int8", "gemv_int8_act_mul", "gemv_int8_hc_mix"]

