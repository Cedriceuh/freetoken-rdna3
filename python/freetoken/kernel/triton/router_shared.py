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
from freetoken.kernel.triton.int8_gemv import ROWS_MAX, _ACT_PAIR4, _act_pair_body, _config, _gemv_splitk_int8_body
from freetoken.kernel.triton.moe_router import _ROUTER_KEYS, _router_body

ENABLED = os.environ.get("FREETOKEN_ROUTER_SHARED", "1") == "1"
# FREETOKEN_ROUTER_SHARED_HIP (default 1 on ROCm): the router | shared gate-up launch below as a HIP kernel with its
# sums (kernel/router_shared_exact.py) for the router (16, 128, 4) and gate | up (2, 512) or (1, 4096) tiles at K 2560,
# 1-8 rows
HIP = os.environ.get("FREETOKEN_ROUTER_SHARED_HIP", "1" if torch.version.hip is not None else "0") == "1"
# decode rows both launches take: the batch-invariant GEMVs' rows, at most 8 (the gate | up's two groups of rows)
ROWS_LIMIT = min(ROWS_MAX, 8)


@triton.jit
def _router_shared_kernel(ra, rw, rpart, RN, RK, r_stride_wn, r_stride_pk, r_k_per, r_stride_am, r_stride_pm, rout,
                          rcnt, r_stride_om, sa, sw, sscale, sout, D, SK, s_stride_wn, s_k_per, s_stride_am,
                          s_stride_om, R_TILES, R_PROGS, R_BLOCK_N: tl.constexpr, R_BLOCK_K: tl.constexpr,
                          ROWS: tl.constexpr, SPLIT: tl.constexpr, SHARE_TILES: tl.constexpr,
                          S_BLOCK_N: tl.constexpr, S_BLOCK_K: tl.constexpr, ONE_PASS: tl.constexpr,
                          S1_PROGS=0, ROWS2: tl.constexpr = 0, S2_BLOCK_N: tl.constexpr = 1):
    # ROWS2 > 0: the gate | up's rows in two groups, the first 4 rows (S1_PROGS programs of S_BLOCK_N columns) and the
    # ROWS2 after them (programs of S2_BLOCK_N), as gemv_int8_act_mul would take each group alone
    pid = tl.program_id(0)
    if pid < R_PROGS:
        _gemv_splitk_bf16_body(pid % R_TILES, pid // R_TILES, ra, rw, rpart, RN, RK, r_stride_wn, r_stride_pk,
                               r_k_per, R_BLOCK_N, R_BLOCK_K, ROWS, r_stride_am, r_stride_pm, rout, rcnt,
                               r_stride_om, SPLIT, True, SHARE_TILES)
    elif ROWS2 == 0:
        _act_pair_body(pid - R_PROGS, sa, sw, sscale, sout, D, SK, s_stride_wn, s_k_per, s_stride_am, s_stride_om,
                       S_BLOCK_N, S_BLOCK_K, ROWS, ONE_PASS)
    elif pid < R_PROGS + S1_PROGS:
        _act_pair_body(pid - R_PROGS, sa, sw, sscale, sout, D, SK, s_stride_wn, s_k_per, s_stride_am, s_stride_om,
                       S_BLOCK_N, S_BLOCK_K, 4, ONE_PASS)
    else:
        _act_pair_body(pid - R_PROGS - S1_PROGS, sa + 4 * s_stride_am, sw, sscale, sout + 4 * s_stride_om, D, SK,
                       s_stride_wn, s_k_per, s_stride_am, s_stride_om, S2_BLOCK_N, S_BLOCK_K, ROWS2, ONE_PASS)


def _aligned16(*tensors: torch.Tensor | None) -> bool:
    """The HIP kernel's 16-byte loads: every tensor and its rows 16-byte aligned."""
    return all(t is None or (t.data_ptr() % 16 == 0 and (t.dim() < 2 or t.stride(0) * t.element_size() % 16 == 0))
               for t in tensors)


def router_and_shared_act(x: torch.Tensor, gate_w: torch.Tensor, gate_cfg: tuple[int, int, int, int],
                          counters: torch.Tensor | None, qweight: torch.Tensor, scale: torch.Tensor,
                          shared_gate_w: torch.Tensor | None = None):
    """``(gemv_bf16(x, gate_w, gate_cfg, counters), silu_and_mul(gemv_int8(x, qweight, scale)))`` in one launch, the
    same bits (gemv_int8_act_mul's kernel; 5-8 rows of a K-blocked shape: its programs for the first 4 rows and for
    the others, each group as it would take it alone, rows being independent); None where that does not apply (more
    than 8 rows, no split-K counters, a split-K gate | up). A third item: the shared expert's gate ``[M]`` fp32 when
    ``shared_gate_w`` is given and the HIP kernel ran (moe_shared_gate.py's bits), else None."""
    from freetoken.kernel.triton.dense_gemv import _SHARED_TILES

    M = x.shape[0]
    if not ENABLED or x.dim() != 2 or not x.is_contiguous() or not 1 <= M <= ROWS_LIMIT or counters is None:
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
    rows2, s2bn = 0, 1
    if not one_pass and M > 4:  # two groups: 4 rows, then 1-4 (FREETOKEN_INT8_ACT_PAIR4 for 3-4)
        if not _ACT_PAIR4:
            return None
        rows2 = M - 4
        s2bn = 1 if rows2 > 2 else sbn
    pair4 = _ACT_PAIR4 and not one_pass and 2 < M
    if not one_pass and M > 2 and not pair4:
        return None
    if pair4:
        sbn = 1
    RN, RK = gate_w.shape
    if (HIP and (RN, RK, SK) == (512, 2560, 2560) and tuple(gate_cfg[:3]) == (16, 128, 4)
            and ((sbk, swarps, one_pass) == (512, 1, False) or (sbn, sbk, swarps, one_pass) == (1, 4096, 4, True))
            and gate_w.stride(1) == 1 and qweight.stride(1) == 1 and scale.is_contiguous()
            and _aligned16(x, gate_w, qweight, shared_gate_w)):
        from freetoken.kernel.router_shared_exact import router_and_shared_act_exact
        return router_and_shared_act_exact(x, gate_w, qweight, scale, one_pass, shared_gate_w)
    r_k_per = triton.cdiv(triton.cdiv(RK, split), rbk) * rbk
    part = torch.empty((M, split, RN), dtype=torch.float32, device=x.device)
    logits = torch.empty((M, RN), dtype=torch.bfloat16, device=x.device)
    act = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
    r_tiles = triton.cdiv(RN, rbn)
    r_progs = r_tiles * split
    s1 = triton.cdiv(D, sbn)
    s2 = triton.cdiv(D, s2bn) if rows2 else 0
    _router_shared_kernel[(r_progs + s1 + s2,)](
        x, gate_w, part, RN, RK, gate_w.stride(0), part.stride(1), r_k_per, x.stride(0), part.stride(0), logits,
        counters, logits.stride(0), x, qweight, scale, act, D, SK, qweight.stride(0), s_k_per, x.stride(0),
        act.stride(0), r_tiles, r_progs, R_BLOCK_N=rbn, R_BLOCK_K=rbk, ROWS=M, SPLIT=split, SHARE_TILES=_SHARED_TILES,
        S_BLOCK_N=sbn, S_BLOCK_K=sbk, ONE_PASS=one_pass, S1_PROGS=s1, ROWS2=rows2, S2_BLOCK_N=s2bn, num_warps=swarps)
    return logits, act, None


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
    if (not ENABLED or not 1 <= M <= ROWS_LIMIT or not one_pass or warps != 1 or block_e > 512
            or 256 // block_e > 1 or not act.is_contiguous() or not logits.is_contiguous()):
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


__all__ = ["ENABLED", "ROWS_LIMIT", "router_and_shared_act", "shared_down_and_topk"]
