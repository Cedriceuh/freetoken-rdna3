"""A decode MoE layer's router GEMV (bf16 split-K, kernel/triton/dense_gemv.py) and its shared expert's gate | up
GEMV with the activation (int8, kernel/triton/int8_gemv.py) in one launch (FREETOKEN_ROUTER_SHARED, default 1): both
read the same input and nothing of each other. The first programs run the router's, the others the shared expert's,
each the unchanged program body, all at the shared expert's own warps (its sums depend on them, the router GEMV's do
not: tests/kernels/test_router_shared.py), so the bits do not change. 48 layers of distinct weights in a graph, 7900 XTX:
13.3 -> 9.5 us a layer at one row, 14.1 -> 9.9 at two, 15.2 -> 12.3 at three, 17.8 -> 13.6 at four."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.dense_gemv import _gemv_splitk_bf16_body
from freetoken.kernel.triton.int8_gemv import _ACT_PAIR4, _act_pair_body, _config, _gemv_splitk_int8_body
from freetoken.kernel.triton.moe_router import _ROUTER_KEYS, _router_body

ENABLED = os.environ.get("FREETOKEN_ROUTER_SHARED", "1") == "1"


@triton.jit
def _router_shared_kernel(ra, rw, rpart, RN, RK, r_stride_wn, r_stride_pk, r_k_per, r_stride_am, r_stride_pm, rout,
                          rcnt, r_stride_om, sa, sw, sscale, sout, D, SK, s_stride_wn, s_k_per, s_stride_am,
                          s_stride_om, R_TILES, R_PROGS, R_BLOCK_N: tl.constexpr, R_BLOCK_K: tl.constexpr,
                          ROWS: tl.constexpr, SPLIT: tl.constexpr, SHARE_TILES: tl.constexpr,
                          S_BLOCK_N: tl.constexpr, S_BLOCK_K: tl.constexpr, ONE_PASS: tl.constexpr):
    pid = tl.program_id(0)
    if pid < R_PROGS:
        _gemv_splitk_bf16_body(pid % R_TILES, pid // R_TILES, ra, rw, rpart, RN, RK, r_stride_wn, r_stride_pk,
                               r_k_per, R_BLOCK_N, R_BLOCK_K, ROWS, r_stride_am, r_stride_pm, rout, rcnt,
                               r_stride_om, SPLIT, True, SHARE_TILES)
    else:
        _act_pair_body(pid - R_PROGS, sa, sw, sscale, sout, D, SK, s_stride_wn, s_k_per, s_stride_am, s_stride_om,
                       S_BLOCK_N, S_BLOCK_K, ROWS, ONE_PASS)


def router_and_shared_act(x: torch.Tensor, gate_w: torch.Tensor, gate_cfg: tuple[int, int, int, int],
                          counters: torch.Tensor | None, qweight: torch.Tensor,
                          scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(gemv_bf16(x, gate_w, gate_cfg, counters), gemv_int8_act_mul(x, qweight, scale))`` in one launch, the same
    bits; None where either is not a single launch of those kernels (more than 4 rows, no split-K counters, an
    activation that takes two launches)."""
    from freetoken.kernel.triton.dense_gemv import _SHARED_TILES

    M = x.shape[0]
    if not ENABLED or x.dim() != 2 or not x.is_contiguous() or not 1 <= M <= 4 or counters is None:
        return None
    rbn, rbk, split, _ = gate_cfg
    if split <= 1:
        return None
    N2, SK = qweight.shape
    sbn, sbk, ssplit, swarps = _config(N2, SK)
    if ssplit != 1 or N2 % 2:
        return None
    D = N2 // 2
    s_k_per = triton.cdiv(SK, sbk) * sbk
    one_pass = s_k_per >= SK and sbk >= SK
    pair4 = _ACT_PAIR4 and not one_pass and 2 < M <= 4
    if not one_pass and M > 2 and not pair4:
        return None
    if pair4:
        sbn = 1
    RN, RK = gate_w.shape
    r_k_per = triton.cdiv(triton.cdiv(RK, split), rbk) * rbk
    part = torch.empty((M, split, RN), dtype=torch.float32, device=x.device)
    logits = torch.empty((M, RN), dtype=torch.bfloat16, device=x.device)
    act = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
    r_tiles = triton.cdiv(RN, rbn)
    r_progs = r_tiles * split
    _router_shared_kernel[(r_progs + triton.cdiv(D, sbn),)](
        x, gate_w, part, RN, RK, gate_w.stride(0), part.stride(1), r_k_per, x.stride(0), part.stride(0), logits,
        counters, logits.stride(0), x, qweight, scale, act, D, SK, qweight.stride(0), s_k_per, x.stride(0),
        act.stride(0), r_tiles, r_progs, R_BLOCK_N=rbn, R_BLOCK_K=rbk, ROWS=M, SPLIT=split, SHARE_TILES=_SHARED_TILES,
        S_BLOCK_N=sbn, S_BLOCK_K=sbk, ONE_PASS=one_pass, num_warps=swarps)
    return logits, act


@triton.jit
def _down_topk_kernel(da, dw, dscale, dout, DN, DK, d_stride_wn, d_k_per, d_stride_am, d_stride_om, D_PROGS,
                      scores, tw, ti, s_stride_m, w_stride_m, i_stride_m, ROWS: tl.constexpr, D_BLOCK_N: tl.constexpr,
                      D_BLOCK_K: tl.constexpr, ONE_PASS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                      BLOCK_E: tl.constexpr, BLOCK_TOPK: tl.constexpr, RENORMALIZE: tl.constexpr, KEYS: tl.constexpr):
    pid = tl.program_id(0)
    if pid < D_PROGS:
        _gemv_splitk_int8_body(pid, 0, da, dw, dscale, dout, dout, DN, DK, d_stride_wn, 0, d_k_per, 1, D_BLOCK_N,
                               D_BLOCK_K, ROWS, d_stride_am, d_stride_om, ONE_PASS, 0, 0)
    else:
        _router_body(pid - D_PROGS, scores, scores, tw, ti, scores, ROWS, 1.0, s_stride_m, 1, w_stride_m, 1,
                     i_stride_m, 1, E, TOPK, 1, BLOCK_E, BLOCK_TOPK, False, 1, 1, 1, RENORMALIZE, False, False, KEYS)


def shared_down_and_topk(act: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, logits: torch.Tensor,
                         topk: int, renormalize: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """``(gemv_int8(act, qweight, scale), *fused_topk_softmax(logits, topk, renormalize))`` in one launch: the shared
    expert's down projection and the router's softmax top-k, the same bits where both kernels run one warp a program
    (a one-pass int8 shape, a router of up to 512 experts); None otherwise."""
    M, DK = act.shape
    DN = qweight.shape[0]
    E = logits.shape[1]
    bn, bk, split, warps = _config(DN, DK)
    k_per = triton.cdiv(DK, bk) * bk
    one_pass = split == 1 and k_per >= DK and bk >= DK
    block_e = triton.next_power_of_2(E)
    if (not ENABLED or not 1 <= M <= 4 or not one_pass or warps != 1 or block_e > 512 or 256 // block_e > 1
            or not act.is_contiguous() or not logits.is_contiguous()):
        return None
    out = torch.empty((M, DN), dtype=torch.bfloat16, device=act.device)
    weights = torch.empty((M, topk), dtype=torch.float32, device=act.device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=act.device)
    d_progs = triton.cdiv(DN, bn)
    _down_topk_kernel[(d_progs + M,)](
        act, qweight, scale, out, DN, DK, qweight.stride(0), k_per, act.stride(0), out.stride(0), d_progs, logits,
        weights, ids, logits.stride(0), weights.stride(0), ids.stride(0), ROWS=M, D_BLOCK_N=bn, D_BLOCK_K=bk,
        ONE_PASS=M > 1, E=E, TOPK=topk, BLOCK_E=block_e, BLOCK_TOPK=triton.next_power_of_2(topk),
        RENORMALIZE=renormalize, KEYS=_ROUTER_KEYS, num_warps=1)
    return out, weights, ids


__all__ = ["ENABLED", "router_and_shared_act", "shared_down_and_topk"]
