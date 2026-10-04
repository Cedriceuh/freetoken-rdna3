"""EXL3 trellis kernels for RDNA3: tile dequant, the routed-expert GEMV (decode) and grouped GEMM (prefill),
and the Hadamard rotations around them.

The reference is :mod:`freetoken.layers.quantization.exl3_codec`. A tile is 8*K uint32 words; its
256 weights come in encoded order j = c0*32 + tq*8 + h*4 + i (NVIDIA's mma fragment order), at
tile row ``8*(i//2) + 2*tq + i%2`` and column ``8*h + c0``. The kernels decode in that order and
move the activations instead of the weights: a GEMV/GEMM reads its 16 activations of a tile row in
the encoded row order, the dequant stores each weight at its own (row, col).

G consecutive weights of a tile share one two-word window ((G-1)*K + 16 <= 32 bits): G = 4 up to
K = 5, 2 above.

An expert ``W = diag(suh) H Wq H diag(svh)`` (H: normalized 128-point Hadamard per block) runs as
rotate-in (``H(x * suh)``), the trellis product, then rotate-out (``H(y) * svh``); the Hadamards are
128x128 ``tl.dot``s on fp16 hi/lo halves, so the rotations keep ~fp32 precision.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.layers.quantization.exl3_codec import CB_3INST, CB_MCG, CB_MUL1


def group_size(K: int) -> int:
    return 4 if K <= 5 else 2


@functools.lru_cache(maxsize=None)
def _dot4(device: torch.device) -> bool:
    """RDNA3 (gfx11) has v_dot4_u32_u8; elsewhere the byte sum is plain integer ops."""
    if torch.version.hip is None:
        return False
    return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx11")


@functools.lru_cache(maxsize=None)
def _hadamard_pm1(device: torch.device) -> torch.Tensor:
    """128x128 Sylvester Hadamard, +-1 entries (exact in fp16); the 1/sqrt(128) is applied after the dot."""
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device=device, dtype=torch.float16).contiguous()


@triton.jit
def _exl3_states(t_base, KT: tl.constexpr, NT: tl.constexpr, stride_a, stride_b, K: tl.constexpr, G: tl.constexpr):
    """16-bit states of KT x NT tiles from ``t_base`` (int32 words), ``[KT, NT, 256]`` uint32 in encoded order."""
    W: tl.constexpr = 8 * K
    NG: tl.constexpr = 256 // G
    kt = tl.arange(0, KT)[:, None, None, None]
    nt = tl.arange(0, NT)[None, :, None, None]
    jg = tl.arange(0, NG)[None, None, :, None]
    iw = tl.arange(0, G)[None, None, None, :]
    b0 = (jg * G + 257) * K - 16  # first bit of the first window (ring offset keeps it positive)
    b2 = b0 + (G - 1) * K + 16  # end of the last window
    i0 = b0 // 32
    i2 = (b2 - 1) // 32
    s2 = ((i2 + 1) * 32 - b2).to(tl.uint32)
    tile = t_base + kt * stride_a + nt * stride_b
    wa = tl.load(tile + (i0 % W)).to(tl.uint32, bitcast=True)
    wb = tl.load(tile + (i2 % W)).to(tl.uint32, bitcast=True)
    # low 32 bits of (wa:wb) >> s2, which hold all G windows; the double shift keeps s2 = 0 legal
    lo = (wb >> s2) | ((wa << 1) << (31 - s2))
    states = (lo >> ((G - 1 - iw) * K).to(tl.uint32)) & 0xFFFF
    return tl.reshape(states, (KT, NT, 256))


@triton.jit
def _byte_sum(x, acc, DOT4: tl.constexpr):
    """acc + the sum of the four bytes of uint32 ``x`` (uint32)."""
    if DOT4:  # RDNA3 v_dot4_u32_u8: one instruction
        ones = tl.full(x.shape, 0x01010101, tl.uint32)
        return tl.inline_asm_elementwise(
            "v_dot4_u32_u8 $0, $1, $2, $3", "=v,v,v,v", [x, ones, acc], dtype=tl.uint32, is_pure=True, pack=1
        )
    else:
        pairs = (x & 0x00FF00FF) + ((x >> 8) & 0x00FF00FF)  # two 16-bit byte-pair sums
        return ((pairs * 0x10001) >> 16) + acc  # high half = both pairs (<= 1020, no carry out of the low half)


@triton.jit
def _exl3_codebook(s, CB: tl.constexpr, DOT4: tl.constexpr):
    """Codebook value of uint32 states ``s``: fp32 holding the fp16 value the CUDA kernels produce.

    mul1: fp16(0x6400 + bytesum) = 1024 + bytesum, then fma with fp16 0x1eee / 0xc931; that fma is exact
    in fp32 for these operands, so one cast to fp16 rounds like ``__hfma``."""
    if CB == 2:  # mul1
        x = s * 0x83DCD12D
        h = _byte_sum(x, tl.full(x.shape, 1024, tl.uint32), DOT4).to(tl.float32)
        return (h * 0.00676727294921875 - 10.3828125).to(tl.float16).to(tl.float32)
    else:
        if CB == 1:  # mcg
            x = s * 0xCBAC1FED
        else:  # 3inst
            x = s * 89226354 + 64248484
        x = (x & 0x8FFF8FFF) ^ 0x3B603B60
        lo = (x & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
        hi = (x >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
        return (lo + hi).to(tl.float32)


@triton.jit
def _load_had(h_ptr):
    i = tl.arange(0, 128)
    return tl.load(h_ptr + i[:, None] * 128 + i[None, :])


@triton.jit
def _had(x, h, HOIST: tl.constexpr, PRECISE: tl.constexpr):
    """_had128 on ``h``: the matrix already loaded (HOIST) or its pointer, loaded right before the dot. Keeping the
    32 KiB matrix live across a loop pays only with enough warps to hold it (decode: 12 -> 165 us when hoisted)."""
    if HOIST:
        return _had128(x, h, PRECISE)
    else:
        return _had128(x, _load_had(h), PRECISE)


@triton.jit
def _had128(x, hm, PRECISE: tl.constexpr):
    """``x [R, 128]`` fp32 times the normalized Hadamard (``hm``: the +-1 matrix, fp16, from _load_had), fp32: an fp16
    dot, plus a second dot on the fp16 rounding error when PRECISE (~fp32; prefill rounds its activations to fp16
    anyway). Callers load ``hm`` once and loop over their 128-blocks: reloading it per block cost more than the dots."""
    hi = x.to(tl.float16)
    y = tl.dot(hi, hm, out_dtype=tl.float32)
    if PRECISE:
        y += tl.dot((x - hi.to(tl.float32)).to(tl.float16), hm, out_dtype=tl.float32)
    return y * 0.08838834764831845


@triton.jit
def _had128_vec(x):
    """One 128-vector ``x`` (fp32) times the normalized Hadamard: H[i, j] = (-1)^popcount(i & j) built in registers, a
    broadcast product and one reduction (decode: one program per vector keeps many programs in flight)."""
    i = tl.arange(0, 128)[:, None]
    j = tl.arange(0, 128)[None, :]
    v = i & j
    v ^= v >> 4
    v ^= v >> 2
    v ^= v >> 1
    sign = 1.0 - 2.0 * (v & 1).to(tl.float32)
    return tl.sum(x[:, None] * sign, axis=0) * 0.08838834764831845


@triton.jit
def _encoded_rows(KT: tl.constexpr):
    """Activation offsets of KT tile rows in encoded order: ``[KT, 4(tq), 4(i)]`` -> 16*kt + 8*(i//2) + 2*tq + i%2."""
    kt = tl.arange(0, KT)[:, None, None]
    tq = tl.arange(0, 4)[None, :, None]
    i = tl.arange(0, 4)[None, None, :]
    return kt * 16 + 8 * (i // 2) + 2 * tq + (i % 2)


# ---------------------------------------------------------------------------------------------
# dequant
# ---------------------------------------------------------------------------------------------


@triton.jit
def _exl3_dequant_kernel(
    t_ptr, out_ptr,
    stride_ta, stride_tb, stride_or,
    K: tl.constexpr, CB: tl.constexpr, G: tl.constexpr, KT: tl.constexpr, NT: tl.constexpr, DOT4: tl.constexpr,
):
    """Wq[k, n] (fp16, no Hadamard / scales) for a KT x NT block of tiles."""
    ka0 = tl.program_id(0) * KT
    nb0 = tl.program_id(1) * NT
    s = _exl3_states(t_ptr + ka0 * stride_ta + nb0 * stride_tb, KT, NT, stride_ta, stride_tb, K, G)
    w = tl.reshape(_exl3_codebook(s, CB, DOT4), (KT, NT, 8, 4, 2, 4))
    kt = tl.arange(0, KT)[:, None, None, None, None, None]
    nt = tl.arange(0, NT)[None, :, None, None, None, None]
    c0 = tl.arange(0, 8)[None, None, :, None, None, None]
    tq = tl.arange(0, 4)[None, None, None, :, None, None]
    h = tl.arange(0, 2)[None, None, None, None, :, None]
    i = tl.arange(0, 4)[None, None, None, None, None, :]
    row = (ka0 + kt) * 16 + 8 * (i // 2) + 2 * tq + (i % 2)
    col = (nb0 + nt) * 16 + 8 * h + c0
    tl.store(out_ptr + row.to(tl.int64) * stride_or + col, w.to(tl.float16))


def _pow2_divisor(n: int, cap: int) -> int:
    d = 1
    while d * 2 <= cap and n % (d * 2) == 0:
        d *= 2
    return d


def _bits(words: int) -> int:
    if words % 16:
        raise NotImplementedError(f"EXL3 half-integer bitrate ({words / 16} bpw) is not supported")
    return words // 16


def exl3_dequant(trellis: torch.Tensor, codebook: int) -> torch.Tensor:
    """Decoded tiles Wq ``[in, out]`` fp16 of a ``[in/16, out/16, 16*K]`` int16 trellis on the GPU."""
    A, B, words = trellis.shape
    K = _bits(words)
    t = trellis.contiguous().view(torch.int32)
    out = torch.empty((A * 16, B * 16), dtype=torch.float16, device=trellis.device)
    KT, NT = _pow2_divisor(A, 2), _pow2_divisor(B, 4)
    _exl3_dequant_kernel[(A // KT, B // NT)](
        t, out, t.stride(0), t.stride(1), out.stride(0),
        K=K, CB=codebook, G=group_size(K), KT=KT, NT=NT, DOT4=_dot4(trellis.device), num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------------------------
# routed-expert products (rotation domain: no Hadamard, no scales)
# ---------------------------------------------------------------------------------------------


@triton.jit
def _exl3_gemv_kernel(
    a_ptr, t_ptr, slot_ptr, c_ptr,
    KSPAN, HALF_TILES,
    stride_ar, stride_ah,
    stride_ts, stride_ta, stride_tb,
    stride_cs, stride_cr,
    K: tl.constexpr, CB: tl.constexpr, G: tl.constexpr, KT: tl.constexpr, NT: tl.constexpr, DOT4: tl.constexpr,
):
    """c[split, r, n] = sum over this split's k of a[r, half(n), k] * Wq[slot[r]][k, n], NT column tiles of one route.

    ``a`` holds the rotated inputs, one row per (route, half): a gate|up bank has two halves whose inputs differ
    (each projection has its own suh), any other bank one."""
    r = tl.program_id(0)
    nb0 = tl.program_id(1) * NT
    split = tl.program_id(2)
    slot = tl.load(slot_ptr + r).to(tl.int64)
    half = nb0 // HALF_TILES
    a_base = a_ptr + r.to(tl.int64) * stride_ar + half * stride_ah
    t_base = t_ptr + slot * stride_ts + nb0 * stride_tb
    a_rows = _encoded_rows(KT)
    # products accumulate per encoded position; the cross-lane reduction runs once, after the K loop
    acc = tl.zeros((KT, NT, 8, 4, 2, 4), tl.float32)
    for ka0 in range(split * KSPAN, (split + 1) * KSPAN, KT):
        s = _exl3_states(t_base + ka0 * stride_ta, KT, NT, stride_ta, stride_tb, K, G)
        w = tl.reshape(_exl3_codebook(s, CB, DOT4), (KT, NT, 8, 4, 2, 4))
        a = tl.load(a_base + ka0 * 16 + a_rows).to(tl.float32)  # [KT, 4(tq), 4(i)]
        acc += w * tl.reshape(a, (KT, 1, 1, 4, 1, 4))
    acc = tl.sum(tl.sum(tl.sum(acc, axis=5), axis=3), axis=0)
    nt = tl.arange(0, NT)[:, None, None]
    c0 = tl.arange(0, 8)[None, :, None]
    h = tl.arange(0, 2)[None, None, :]
    col = (nb0 + nt) * 16 + 8 * h + c0
    tl.store(c_ptr + split * stride_cs + r.to(tl.int64) * stride_cr + col, acc)


# (KIN tiles, half N tiles) -> (KT, NT, split_k, warps), swept on the 7900 XT (rdna3/bench/exl3_gemv_bench.py); the
# intermediate-384 shapes (rank 0 = the XTX at a 0.55 / 0.6 split) re-checked on the XTX, 3 repeats: down 18.7-20.8 ->
# 17.2-17.5 us with (2, 2, 2, 2), gate|up unchanged (30.5-30.9 us, tied with the best)
_GEMV_TUNED: dict[tuple[int, int], tuple[int, int, int, int]] = {
    (160, 24): (4, 4, 2, 8),  # gate|up, intermediate 384 per rank
    (24, 160): (2, 2, 2, 2),  # down, 384
    (160, 16): (1, 1, 8, 2),  # gate|up, 256
    (16, 160): (1, 2, 1, 2),  # down, 256
}


def _gemv_config(kin_tiles: int, half_tiles: int) -> tuple[int, int, int, int]:
    tuned = _GEMV_TUNED.get((kin_tiles, half_tiles))
    if tuned is not None:
        return tuned
    split = _pow2_divisor(kin_tiles, 8 if kin_tiles >= 64 else 4)
    kt = _pow2_divisor(kin_tiles // split, 4)
    return kt, _pow2_divisor(half_tiles, 2), split, 4


def exl3_gemv(
    a: torch.Tensor,
    trellis_bank: torch.Tensor,
    slots: torch.Tensor,
    codebook: int,
    *,
    out: torch.Tensor | None = None,
    kt: int | None = None,
    nt: int | None = None,
    split_k: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Per-route GEMV over a slot bank of EXL3 trellises, in the rotation domain.

    ``a``: ``[R, halves, KIN]`` rotated inputs; ``trellis_bank``: ``[S, KIN/16, N/16, 16*K]`` int16; ``slots``:
    ``[R]`` bank row per route. Returns ``[split_k, R, N]`` fp32 partial sums over K (the caller adds them);
    with two halves, columns ``[0, N/2)`` read ``a[:, 0]`` and ``[N/2, N)`` read ``a[:, 1]``."""
    R, halves, kin = a.shape
    S, A, B, words = trellis_bank.shape
    assert kin == A * 16 and B % halves == 0, (tuple(a.shape), tuple(trellis_bank.shape))
    K = _bits(words)
    t = trellis_bank.view(torch.int32)
    half_tiles = B // halves
    cfg = _gemv_config(A, half_tiles)
    KT, NT = kt or cfg[0], nt or cfg[1]
    split_k, num_warps = split_k or cfg[2], num_warps or cfg[3]
    assert A % (KT * split_k) == 0 and half_tiles % NT == 0, (A, KT, split_k, half_tiles, NT)
    if out is None:
        out = torch.empty((split_k, R, B * 16), dtype=torch.float32, device=a.device)
    a = a.contiguous()
    _exl3_gemv_kernel[(R, B // NT, split_k)](
        a, t, slots, out,
        A // split_k, half_tiles,
        a.stride(0), a.stride(1),
        t.stride(0), t.stride(1), t.stride(2),
        out.stride(0), out.stride(1),
        K=K, CB=codebook, G=group_size(K), KT=KT, NT=NT, DOT4=_dot4(a.device), num_warps=num_warps,
    )
    return out


@triton.jit
def _exl3_moe_gemm_kernel(
    a_ptr, t_ptr, c_ptr, sorted_ptr, expert_ptr, ntpp_ptr,
    EM, R, N_TILES, HALF_TILES, KIN_TILES,
    stride_ar, stride_ah,
    stride_ts, stride_ta, stride_tb,
    stride_cr,
    K: tl.constexpr, CB: tl.constexpr, G: tl.constexpr, KT: tl.constexpr, NT: tl.constexpr, DOT4: tl.constexpr,
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
):
    """Grouped GEMM over expert-sorted routes: c[route, n] = sum_k a[route, half(n), k] * Wq[expert][k, n]."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = N_TILES // NT
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    route = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rmask = route < R
    slot = tl.load(expert_ptr + pid_m).to(tl.int64)
    nb0 = pid_n * NT
    half = nb0 // HALF_TILES
    a_base = a_ptr + route[:, None] * stride_ar + half * stride_ah
    a_rows = tl.reshape(_encoded_rows(KT), (KT * 16,))
    t_base = t_ptr + slot * stride_ts + nb0 * stride_tb
    acc = tl.zeros((BLOCK_M, NT * 16), tl.float32)
    for ka0 in range(0, KIN_TILES, KT):
        s = _exl3_states(t_base + ka0 * stride_ta, KT, NT, stride_ta, stride_tb, K, G)
        w = tl.reshape(_exl3_codebook(s, CB, DOT4).to(tl.float16), (KT, NT, 8, 4, 2, 4))
        # rows (kt, tq, i) in encoded order, columns (nt, h, c0) = the natural column order
        w = tl.reshape(tl.permute(w, (0, 3, 5, 1, 4, 2)), (KT * 16, NT * 16))
        a = tl.load(a_base + ka0 * 16 + a_rows[None, :], mask=rmask[:, None], other=0.0).to(tl.float16)
        acc += tl.dot(a, w, out_dtype=tl.float32)
    col = nb0 * 16 + tl.arange(0, NT * 16)
    tl.store(c_ptr + route[:, None] * stride_cr + col[None, :], acc, mask=rmask[:, None])


def exl3_moe_gemm(
    a: torch.Tensor,
    trellis_bank: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    codebook: int,
    *,
    block_m: int,
    out: torch.Tensor | None = None,
    kt: int = 2,
    nt: int = 8,
    num_warps: int = 4,
    num_stages: int = 1,
    group_m: int = 8,
) -> torch.Tensor:
    """Prefill counterpart of :func:`exl3_gemv`: ``a`` ``[R, halves, KIN]``, routes sorted by expert with
    ``moe_align_block_size`` at ``block_m``. Returns ``[1, R, N]`` fp32. The tile defaults (with block_m 64) won a sweep
    on the 7900 XT for Qwen3.8-Flash-Next's per-rank shapes (a one-layer sweep at 2048 tokens: -35..40 % vs 32 / 2 / 4)."""
    R, halves, kin = a.shape
    S, A, B, words = trellis_bank.shape
    assert kin == A * 16 and B % halves == 0
    K = _bits(words)
    t = trellis_bank.view(torch.int32)
    half_tiles = B // halves
    kt, nt = _pow2_divisor(A, kt), _pow2_divisor(half_tiles, nt)
    if out is None:
        out = torch.empty((1, R, B * 16), dtype=torch.float32, device=a.device)
    a = a.contiguous()
    EM = sorted_ids.shape[0]
    grid = (triton.cdiv(EM, block_m) * (B // nt),)
    _exl3_moe_gemm_kernel[grid](
        a, t, out, sorted_ids, expert_ids, num_tokens_post_padded,
        EM, R, B, half_tiles, A,
        a.stride(0), a.stride(1),
        t.stride(0), t.stride(1), t.stride(2),
        out.stride(1),
        K=K, CB=codebook, G=group_size(K), KT=kt, NT=nt, DOT4=_dot4(a.device),
        BLOCK_M=block_m, GROUP_M=group_m, num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _gemm_tile_loop(acc, a_base, t_base, rmask, a_rows, KIN_TILES, stride_ta, stride_tb,
                    K: tl.constexpr, CB: tl.constexpr, G: tl.constexpr, KT: tl.constexpr, NT: tl.constexpr,
                    DOT4: tl.constexpr):
    for ka0 in range(0, KIN_TILES, KT):
        s = _exl3_states(t_base + ka0 * stride_ta, KT, NT, stride_ta, stride_tb, K, G)
        w = tl.reshape(_exl3_codebook(s, CB, DOT4).to(tl.float16), (KT, NT, 8, 4, 2, 4))
        w = tl.reshape(tl.permute(w, (0, 3, 5, 1, 4, 2)), (KT * 16, NT * 16))
        a = tl.load(a_base + ka0 * 16 + a_rows[None, :], mask=rmask[:, None], other=0.0).to(tl.float16)
        acc += tl.dot(a, w, out_dtype=tl.float32)
    return acc


@triton.jit
def _exl3_gemm_down_out_kernel(
    a_ptr, t_ptr, svh_ptr, w_ptr, h_ptr, out_ptr, sorted_ptr, expert_ptr, ntpp_ptr,
    EM, R, N_BLOCKS, KIN_TILES,
    stride_ar, stride_ts, stride_ta, stride_tb, stride_vs, stride_or,
    K: tl.constexpr, CB: tl.constexpr, G: tl.constexpr, KT: tl.constexpr, DOT4: tl.constexpr,
    BLOCK_M: tl.constexpr, GROUP_M: tl.constexpr,
):
    """Prefill down product with the rotation out as its epilogue: H, svh and the route's router weight on a
    128-column block, stored per route (``out [R, H]``) for the per-token sum."""
    NT: tl.constexpr = 8
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_in_group = GROUP_M * N_BLOCKS
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m * BLOCK_M >= tl.load(ntpp_ptr):
        return
    route = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rmask = route < R
    slot = tl.load(expert_ptr + pid_m).to(tl.int64)
    a_rows = tl.reshape(_encoded_rows(KT), (KT * 16,))
    y = _gemm_tile_loop(tl.zeros((BLOCK_M, 128), tl.float32), a_ptr + route[:, None] * stride_ar,
                        t_ptr + slot * stride_ts + pid_n * NT * stride_tb, rmask, a_rows, KIN_TILES, stride_ta, stride_tb,
                        K, CB, G, KT, NT, DOT4)
    cols = pid_n * 128 + tl.arange(0, 128)
    y = _had128(y, _load_had(h_ptr), False) * tl.load(svh_ptr + slot * stride_vs + cols).to(tl.float32)[None, :]
    y = y * tl.load(w_ptr + route, mask=rmask, other=0.0).to(tl.float32)[:, None]
    tl.store(out_ptr + route[:, None] * stride_or + cols[None, :], y.to(out_ptr.dtype.element_ty), mask=rmask[:, None])


def exl3_gemm_down_out(a, down, down_svh, route_weights, sorted_ids, expert_ids, ntpp, codebook, *, block_m: int,
                       out_dtype: torch.dtype, kt: int = 2, num_warps: int = 4, group_m: int = 8) -> torch.Tensor:
    """Fused prefill down product and rotation out: ``[R, H]`` per-route outputs, already weighted by the router."""
    R, halves, kin = a.shape
    S, A, B, words = down.shape
    assert halves == 1 and kin == A * 16 and B % 8 == 0
    K = _bits(words)
    t = down.view(torch.int32)
    out = torch.empty((R, B * 16), dtype=out_dtype, device=a.device)
    kt = _pow2_divisor(A, kt)
    EM = sorted_ids.shape[0]
    _exl3_gemm_down_out_kernel[(triton.cdiv(EM, block_m) * (B // 8),)](
        a, t, down_svh, route_weights, _hadamard_pm1(a.device), out, sorted_ids, expert_ids, ntpp,
        EM, R, B // 8, A,
        a.stride(0), t.stride(0), t.stride(1), t.stride(2), down_svh.stride(0), out.stride(0),
        K=K, CB=codebook, G=group_size(K), KT=kt, DOT4=_dot4(a.device), BLOCK_M=block_m, GROUP_M=group_m,
        num_warps=num_warps, num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------------------------
# rotations around the products
# ---------------------------------------------------------------------------------------------


@triton.jit
def _exl3_rotate_in_kernel(
    x_ptr, s_ptr, slot_ptr, h_ptr, a_ptr,
    Q, stride_xt, stride_ss, stride_sh, stride_aq,
    NH: tl.constexpr, TOP_K: tl.constexpr, BR: tl.constexpr, CB: tl.constexpr, PRECISE: tl.constexpr, HOIST: tl.constexpr,
):
    """a[q, b] = H((x[token(q)] * suh[slot(q), half(q)])[b-block]) for BR (route, half) rows q and CB 128-blocks."""
    q = tl.program_id(0) * BR + tl.arange(0, BR)
    qm = q < Q
    r = q // NH
    half = q % NH
    t = (r // TOP_K).to(tl.int64)
    slot = tl.load(slot_ptr + r, mask=qm, other=0).to(tl.int64)
    if HOIST:
        hm = _load_had(h_ptr)
    else:
        hm = h_ptr
    for b in range(tl.program_id(1) * CB, (tl.program_id(1) + 1) * CB):
        cols = b * 128 + tl.arange(0, 128)
        x = tl.load(x_ptr + t[:, None] * stride_xt + cols[None, :], mask=qm[:, None], other=0.0).to(tl.float32)
        s = tl.load(s_ptr + slot[:, None] * stride_ss + half[:, None] * stride_sh + cols[None, :],
                    mask=qm[:, None], other=0.0).to(tl.float32)
        y = _had(x * s, hm, HOIST, PRECISE)
        tl.store(a_ptr + q[:, None].to(tl.int64) * stride_aq + cols[None, :], y.to(a_ptr.dtype.element_ty),
                 mask=qm[:, None])


@triton.jit
def _exl3_mid_kernel(
    c_ptr, slot_ptr, svh_ptr, suh_ptr, h_ptr, a_ptr,
    R, INTER, SPLITS,
    stride_cs, stride_cr, stride_vs, stride_us, stride_ar,
    BR: tl.constexpr, CB: tl.constexpr, PRECISE: tl.constexpr, HOIST: tl.constexpr,
):
    """Between the two products, per 128-block of the intermediate: rotate gate and up out (H, svh), silu(g) * u,
    then rotate into down's domain (suh_down, H)."""
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = r < R
    slot = tl.load(slot_ptr + r, mask=rm, other=0).to(tl.int64)
    if HOIST:
        hm = _load_had(h_ptr)
    else:
        hm = h_ptr
    vb = svh_ptr + slot[:, None] * stride_vs
    for b in range(tl.program_id(1) * CB, (tl.program_id(1) + 1) * CB):
        cols = b * 128 + tl.arange(0, 128)
        g = tl.zeros((BR, 128), tl.float32)
        u = tl.zeros((BR, 128), tl.float32)
        for s in range(SPLITS):
            base = c_ptr + s * stride_cs + r[:, None].to(tl.int64) * stride_cr
            g += tl.load(base + cols[None, :], mask=rm[:, None], other=0.0)
            u += tl.load(base + INTER + cols[None, :], mask=rm[:, None], other=0.0)
        g = _had(g, hm, HOIST, PRECISE) * tl.load(vb + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        u = _had(u, hm, HOIST, PRECISE) * tl.load(vb + INTER + cols[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
        act = g / (1.0 + tl.exp(-g)) * u
        act = act * tl.load(suh_ptr + slot[:, None] * stride_us + cols[None, :], mask=rm[:, None],
                            other=0.0).to(tl.float32)
        y = _had(act, hm, HOIST, PRECISE)
        tl.store(a_ptr + r[:, None].to(tl.int64) * stride_ar + cols[None, :], y.to(a_ptr.dtype.element_ty),
                 mask=rm[:, None])


@triton.jit
def _exl3_out_kernel(
    c_ptr, slot_ptr, svh_ptr, w_ptr, h_ptr, out_ptr,
    T, SPLITS, stride_cs, stride_cr, stride_vs, stride_wt, stride_ot,
    TOP_K: tl.constexpr, BK: tl.constexpr, BT: tl.constexpr, CB: tl.constexpr, PRECISE: tl.constexpr, HOIST: tl.constexpr,
):
    """out[t, b] = sum_k w[t, k] * (H(y[route t,k]) * svh_down[slot])[b-block], for BT tokens and CB 128-blocks
    (their BT * BK route rows share one tl.dot tile; BK pads top_k)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)[:, None]
    k = tl.arange(0, BK)[None, :]
    m = (k < TOP_K) & (t < T)
    r = tl.reshape(t.to(tl.int64) * TOP_K + k, (BT * BK,))
    rm = tl.reshape(m, (BT * BK,))
    slot = tl.load(slot_ptr + r, mask=rm, other=0).to(tl.int64)
    w = tl.load(w_ptr + t * stride_wt + k, mask=m, other=0.0).to(tl.float32)
    tm = tl.program_id(0) * BT + tl.arange(0, BT)
    if HOIST:
        hm = _load_had(h_ptr)
    else:
        hm = h_ptr
    for b in range(tl.program_id(1) * CB, (tl.program_id(1) + 1) * CB):
        cols = b * 128 + tl.arange(0, 128)
        y = tl.zeros((BT * BK, 128), tl.float32)
        for s in range(SPLITS):
            y += tl.load(c_ptr + s * stride_cs + r[:, None] * stride_cr + cols[None, :], mask=rm[:, None], other=0.0)
        y = _had(y, hm, HOIST, PRECISE) * tl.load(svh_ptr + slot[:, None] * stride_vs + cols[None, :], mask=rm[:, None],
                                              other=0.0).to(tl.float32)
        o = tl.sum(tl.reshape(y, (BT, BK, 128)) * w[:, :, None], axis=1)
        tl.store(out_ptr + tm[:, None].to(tl.int64) * stride_ot + cols[None, :], o.to(out_ptr.dtype.element_ty),
                 mask=(tm < T)[:, None])


@triton.jit
def _exl3_rotate_in_vec_kernel(
    x_ptr, s_ptr, slot_ptr, a_ptr,
    stride_xt, stride_ss, stride_sh, stride_aq,
    NH: tl.constexpr, TOP_K: tl.constexpr,
):
    q = tl.program_id(0)
    r = q // NH
    slot = tl.load(slot_ptr + r).to(tl.int64)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + (r // TOP_K).to(tl.int64) * stride_xt + cols).to(tl.float32)
    s = tl.load(s_ptr + slot * stride_ss + (q % NH) * stride_sh + cols).to(tl.float32)
    tl.store(a_ptr + q.to(tl.int64) * stride_aq + cols, _had128_vec(x * s).to(a_ptr.dtype.element_ty))


@triton.jit
def _exl3_mid_vec_kernel(
    c_ptr, slot_ptr, svh_ptr, suh_ptr, a_ptr,
    INTER, stride_cs, stride_cr, stride_vs, stride_us, stride_ar,
    SPLITS: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_ptr + r).to(tl.int64)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    g = tl.zeros((128,), tl.float32)
    u = tl.zeros((128,), tl.float32)
    for s in tl.static_range(SPLITS):
        base = c_ptr + s * stride_cs + r * stride_cr
        g += tl.load(base + cols)
        u += tl.load(base + INTER + cols)
    vb = svh_ptr + slot * stride_vs
    g = _had128_vec(g) * tl.load(vb + cols).to(tl.float32)
    u = _had128_vec(u) * tl.load(vb + INTER + cols).to(tl.float32)
    act = g / (1.0 + tl.exp(-g)) * u * tl.load(suh_ptr + slot * stride_us + cols).to(tl.float32)
    tl.store(a_ptr + r * stride_ar + cols, _had128_vec(act).to(a_ptr.dtype.element_ty))


# up to this many rows rotate-in and mid run one program per (row, 128-block) instead of 16-row tl.dot tiles
_VEC_ROWS = 256


def exl3_rotate_in(x: torch.Tensor, suh_bank: torch.Tensor, slots: torch.Tensor, top_k: int, *,
                   dtype: torch.dtype = torch.float32, rows: int = 16, num_warps: int = 4,
                   blocks: int | None = None, hoist: bool = True) -> torch.Tensor:
    """``[R, halves, H]`` rotated expert inputs: ``H(x[r // top_k] * suh_bank[slots[r], half])``; ``suh_bank``
    is ``[S, halves, H]``."""
    T, H = x.shape
    R = slots.numel()
    halves = suh_bank.shape[1]
    assert H % 128 == 0 and suh_bank.shape[2] == H and R == T * top_k
    out = torch.empty((R, halves, H), dtype=dtype, device=x.device)
    Q = R * halves
    if Q <= _VEC_ROWS:
        _exl3_rotate_in_vec_kernel[(Q, H // 128)](
            x, suh_bank, slots, out, x.stride(0), suh_bank.stride(0), suh_bank.stride(1), H,
            NH=halves, TOP_K=top_k, num_warps=4,
        )
        return out
    nb = H // 128
    cb = blocks or nb
    assert nb % cb == 0, f"{nb} column blocks do not split into programs of {cb}"
    _exl3_rotate_in_kernel[(triton.cdiv(Q, rows), nb // cb)](
        x, suh_bank, slots, _hadamard_pm1(x.device), out,
        Q, x.stride(0), suh_bank.stride(0), suh_bank.stride(1), H,
        NH=halves, TOP_K=top_k, BR=rows, CB=cb, PRECISE=dtype == torch.float32, HOIST=hoist, num_warps=num_warps,
        num_stages=1,
    )
    return out


def exl3_mid(c: torch.Tensor, slots: torch.Tensor, gate_up_svh: torch.Tensor, down_suh: torch.Tensor, *,
             dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``[R, 1, I]`` down-projection inputs from the gate|up partial sums ``c`` ``[splits, R, 2I]``."""
    splits, R, two_i = c.shape
    inter = two_i // 2
    assert inter % 128 == 0 and gate_up_svh.shape[1] == two_i and down_suh.shape[1] == inter
    out = torch.empty((R, 1, inter), dtype=dtype, device=c.device)
    if R <= _VEC_ROWS:
        _exl3_mid_vec_kernel[(R, inter // 128)](
            c, slots, gate_up_svh, down_suh, out,
            inter, c.stride(0), c.stride(1), gate_up_svh.stride(0), down_suh.stride(0), inter,
            SPLITS=splits, num_warps=4,
        )
        return out
    BR = 64
    nb = inter // 128
    _exl3_mid_kernel[(triton.cdiv(R, BR), nb)](  # one block per program: few row tiles, keep them parallel
        c, slots, gate_up_svh, down_suh, _hadamard_pm1(c.device), out,
        R, inter, splits,
        c.stride(0), c.stride(1), gate_up_svh.stride(0), down_suh.stride(0), inter,
        BR=BR, CB=1, PRECISE=dtype == torch.float32, HOIST=False, num_warps=4, num_stages=1,
    )
    return out


def exl3_out(c: torch.Tensor, slots: torch.Tensor, down_svh: torch.Tensor, topk_weights: torch.Tensor,
             out: torch.Tensor, *, precise: bool = True, tokens: int | None = None, num_warps: int = 4,
             blocks: int | None = None, hoist: bool | None = None, num_stages: int | None = None) -> torch.Tensor:
    """``out [T, H]`` = routed sum of the rotated-out down products ``c`` ``[splits, T*top_k, H]``."""
    splits, R, H = c.shape
    T, top_k = topk_weights.shape
    assert R == T * top_k and H % 128 == 0 and out.shape == (T, H)
    # a token's top_k routes fill one 16-row tl.dot tile: the per-vector form would run them one after another
    BK = triton.next_power_of_2(top_k)
    BT = tokens or max(1, 16 // BK)  # tokens per program: at least 16 rows for the tl.dot
    nb = H // 128
    cb = blocks or (nb if T > 64 else 1)  # few tokens: one program per block keeps the GPU busy
    assert nb % cb == 0, f"{nb} column blocks do not split into programs of {cb}"
    _exl3_out_kernel[(triton.cdiv(T, BT), nb // cb)](
        c, slots, down_svh, topk_weights, _hadamard_pm1(c.device), out,
        T, splits, c.stride(0), c.stride(1), down_svh.stride(0), topk_weights.stride(0), out.stride(0),
        TOP_K=top_k, BK=BK, BT=BT, CB=cb, PRECISE=precise, HOIST=cb > 1 if hoist is None else hoist,
        num_warps=num_warps, **({} if num_stages is None else {"num_stages": num_stages}),
    )
    return out


# ---------------------------------------------------------------------------------------------
# n-gram table rows (exllamav3 exl3_ngram_trellis: one mul1 ring per row, row scale in word 0, per-head bias)
# ---------------------------------------------------------------------------------------------


@triton.jit
def _exl3_ngram_kernel(
    p_ptr, bias_ptr, out_ptr,
    N, HEADS, stride_p, stride_o, stride_b,
    K: tl.constexpr, DIM: tl.constexpr, STREAM_WORDS: tl.constexpr, BLOCK_D: tl.constexpr, BR: tl.constexpr,
    J: tl.constexpr, DOT4: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < N
    i = tl.arange(0, BLOCK_D)
    m = rm[:, None] & (i < DIM)[None, :]
    base = p_ptr + rows[:, None].to(tl.int64) * stride_p
    scale = tl.load(base, mask=rm[:, None], other=0).to(tl.float16, bitcast=True).to(tl.float32)
    state = tl.zeros((BR, BLOCK_D), tl.uint32)
    for j in tl.static_range(J):  # position i's state stacks the symbols of i, i-1, ... from bit 0 up
        bit = ((i + DIM - j) % DIM) * K
        w = bit >> 4
        lo = tl.load(base + 1 + w[None, :], mask=m, other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
        hi = tl.load(base + 1 + ((w + 1) % STREAM_WORDS)[None, :], mask=m, other=0).to(tl.uint16, bitcast=True)
        window = lo | (hi.to(tl.uint32) << 16)
        sym = (window >> (bit & 15).to(tl.uint32)[None, :]) & ((1 << K) - 1)
        state |= sym << (j * K)
    state &= 0xFFFF
    head = (rows % HEADS).to(tl.int64)
    bias = tl.load(bias_ptr + head[:, None] * stride_b + i[None, :], mask=m, other=0.0).to(tl.float32)
    val = _exl3_codebook(state, 2, DOT4) * scale + bias
    tl.store(out_ptr + rows[:, None].to(tl.int64) * stride_o + i[None, :], val.to(out_ptr.dtype.element_ty), mask=m)


def exl3_ngram_dequant(packed: torch.Tensor, K: int, head_bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Decode staged n-gram rows ``packed`` ``[N, words]`` int16 into ``out`` ``[N, dim]``; row r belongs to hash head
    ``r % heads`` (``head_bias`` ``[heads, dim]``), the order the PLE disk store stages them in."""
    N, words = packed.shape
    heads, dim = head_bias.shape
    assert (words - 1) * 16 == dim * K and out.shape == (N, dim), (tuple(packed.shape), K, tuple(head_bias.shape))
    BR = 16
    _exl3_ngram_kernel[(triton.cdiv(N, BR),)](
        packed, head_bias, out, N, heads, packed.stride(0), out.stride(0), head_bias.stride(0),
        K=K, DIM=dim, STREAM_WORDS=words - 1, BLOCK_D=triton.next_power_of_2(dim), BR=BR, J=(15 + K) // K,
        DOT4=_dot4(packed.device), num_warps=4,
    )
    return out


__all__ = [
    "CB_3INST", "CB_MCG", "CB_MUL1",
    "exl3_dequant", "exl3_gemm_down_out", "exl3_gemv", "exl3_moe_gemm", "exl3_mid",
    "exl3_ngram_dequant", "exl3_out", "exl3_rotate_in",
    "group_size",
]
