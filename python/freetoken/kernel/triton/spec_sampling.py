"""A verify step's and a draft's sampling from the LM head's candidates in one kernel (FREETOKEN_SAMPLE_KERNELS), the
bits of engine/spec_sample.py's torch ops (~60 launches a verify step, ~25 a draft).

Every float op is torch's, in torch's order on gfx1100 (checked bit for bit, tests/kernels/test_spec_sampling.py):
the division by the temperature; softmax as ATen's warp softmax (per-lane sums of the elements l, l + 32, ..., then
a butterfly over min(next_pow2(K), 32) lanes, exp(x - max) / sum); cumsum as torch runs it for the call's shape (one
row: rocprim, 4 elements a thread; several: ATen's Sklansky scan over blocks of 2 * its x threads, which the row
count and width set); sum over a row of K <= 64 as ATen's reduce (lanes = the largest power of two <= K, at most 32,
each summing its strided elements in order, then adjacent pairs); exp and log as LLVM's AMDGPU expansion of llvm.exp /
llvm.log for IEEE fp32 (v_exp_f32 / v_log_f32 with the same fma corrections); comparisons, selections and the
first-max argmax.
FMA contraction is off (``enable_fp_fusion=False``): torch's kernels contract nothing here. The uniforms still come
from ``torch.rand``, the same calls in the same order, so the draws are the same.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_K = 64  # torch's sum order above is checked up to here (wider rows: it changes with the row count)


@triton.jit
def _asm1(op: tl.constexpr, x):
    return tl.inline_asm_elementwise(op + " $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _ldexp(x, e):
    return tl.inline_asm_elementwise("v_ldexp_f32 $0, $1, $2", "=v,v,v", [x, e], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _exp(x):
    """torch.exp's fp32 bits (llvm.exp.f32 for gfx1100, IEEE denormals)."""
    ph = x * 1.4426950216293335  # 0x3fb8aa3b
    pl = tl.fma(x, 1.4426950216293335, -ph)
    e = _asm1("v_rndne_f32", ph)
    pl = tl.fma(x, 1.925962855864327e-08, pl)  # 0x32a5705f
    a = (ph - e) + pl
    r = _ldexp(_asm1("v_exp_f32", a), e.to(tl.int32))
    r = tl.where(~(x < -103.2789306640625), r, 0.0)  # 0xc2ce8ed0
    return tl.where(~(x > 88.72283935546875), r, float("inf"))  # 0x42b17218


@triton.jit
def _log(x):
    """torch.log's fp32 bits (llvm.log.f32 for gfx1100, IEEE denormals)."""
    small = x < 1.1754943508222875e-38  # 0x00800000
    y = _asm1("v_log_f32", _ldexp(x, tl.where(small, 32, 0)))
    yh = y * 0.6931471228599548  # 0x3f317217
    yl = tl.fma(y, 0.6931471228599548, -yh)
    yl = tl.fma(y, 5.769998878690785e-08, yl)  # 0x3377d1cf
    r = tl.where(tl.abs(y) < float("inf"), yh + yl, y)
    return r - tl.where(small, 22.180709838867188, 0.0)  # 0x41b17218


@triton.jit
def _pick(x, axis_idx, j):
    """x[:, j] of a 2-D tile (a sum of one value and zeros: exact, -0.0 aside)."""
    return tl.sum(tl.where(axis_idx == j, x, 0.0), axis=1)


@triton.jit
def _lane_sum(x, R: tl.constexpr, IT: tl.constexpr, W: tl.constexpr, LOG_W: tl.constexpr, ADJACENT: tl.constexpr):
    """``x [R, IT * W]``: lane l sums elements l, l + W, ... in order from 0, then the W lanes pairwise: (l, l + W/2)
    first (a butterfly: ATen's warp softmax) or adjacent pairs (ATen's reduce)."""
    v = tl.reshape(x, (R, IT, W))
    it = tl.arange(0, IT)[None, :, None]
    acc = tl.zeros((R, W), tl.float32)
    for i in tl.static_range(IT):
        acc = acc + tl.sum(tl.where(it == i, v, 0.0), axis=1)
    for i in tl.static_range(LOG_W):
        if ADJACENT:
            acc = tl.sum(tl.reshape(acc, (R, W >> (i + 1), 2)), axis=2)
        else:
            acc = tl.sum(tl.reshape(acc, (R, 2, W >> (i + 1))), axis=1)
    return tl.reshape(acc, (R,))


@triton.jit
def _cumsum(x, R: tl.constexpr, NB: tl.constexpr, B: tl.constexpr, LOG_B: tl.constexpr, ONE_ROW: tl.constexpr,
            G: tl.constexpr):
    """torch.cumsum over the last dim of ``x [R, NB * B]`` (padding 0). Several rows: ATen's Sklansky scan over
    blocks of B (its threads along x, from the rows and the row length), each block's first element plus the previous
    blocks' total. One row: rocprim's scan, 4 elements a thread (each thread's running sum, the threads' totals summed
    in order, a thread's first element plus the total before it)."""
    if ONE_ROW:
        v = tl.reshape(x, (R, G, 4))
        gi = tl.arange(0, G)[None, :, None]
        ii = tl.arange(0, 4)[None, None, :]
        out = tl.zeros((R, G, 4), tl.float32)
        pre = tl.zeros((R,), tl.float32)
        for g in tl.static_range(G):
            grp = tl.sum(tl.where(gi == g, v, 0.0), axis=1)  # [R, 4]
            c4 = tl.arange(0, 4)[None, :]
            cur = _pick(grp, c4, 0)
            if g > 0:
                cur = pre + cur
            run = tl.where(c4 == 0, cur[:, None], 0.0)
            tot = _pick(grp, c4, 0)
            for i in tl.static_range(1, 4):
                xi = _pick(grp, c4, i)
                cur = cur + xi
                tot = tot + xi
                run = tl.where(c4 == i, cur[:, None], run)
            out = tl.where(gi == g, run[:, None, :], out)
            pre = tot if g == 0 else pre + tot
        return tl.reshape(out, (R, G * 4))
    v = tl.reshape(x, (R, NB, B))
    blk = tl.arange(0, NB)[None, :, None]
    c2 = tl.arange(0, B)[None, :]
    out = tl.zeros((R, NB, B), tl.float32)
    total = tl.zeros((R,), tl.float32)
    for b in tl.static_range(NB):
        buf = tl.sum(tl.where(blk == b, v, 0.0), axis=1)  # [R, B]
        buf = tl.where(c2 == 0, buf + total[:, None], buf)
        for lv in tl.static_range(LOG_B):
            g = tl.reshape(buf, (R, B >> (lv + 1), 2 << lv))
            cg = tl.arange(0, 2 << lv)[None, None, :]
            last = tl.sum(tl.where(cg == (1 << lv) - 1, g, 0.0), axis=2)  # [R, B / 2s]
            g = tl.where(cg >= (1 << lv), g + last[:, :, None], g)
            buf = tl.reshape(g, (R, B))
        total = _pick(buf, c2, B - 1)
        out = tl.where(blk == b, buf[:, None, :], out)
    return tl.reshape(out, (R, NB * B))


@triton.jit
def _distribution(vals, temp, k, p, col, R: tl.constexpr, K: tl.constexpr, SIT: tl.constexpr, SW: tl.constexpr,
                  LOG_SW: tl.constexpr, LIT: tl.constexpr, L: tl.constexpr, LOG_L: tl.constexpr, NB: tl.constexpr,
                  CB: tl.constexpr, LOG_CB: tl.constexpr, ONE_ROW: tl.constexpr, G: tl.constexpr,
                  HAS_TOPK: tl.constexpr, HAS_TOPP: tl.constexpr):
    """engine/spec_sample.py _distribution's sparse branch on candidate rows ``vals [R, P]`` (columns >= K: -inf)."""
    x = vals / temp[:, None]
    if HAS_TOPK:
        x = tl.where(col >= k[:, None], float("-inf"), x)
    x = tl.where(col < K, x, float("-inf"))
    m = tl.max(x, axis=1)
    e = _exp(x - m[:, None])
    probs = e / _lane_sum(e, R, SIT, SW, LOG_SW, False)[:, None]
    if HAS_TOPP:
        keep = ((_cumsum(probs, R, NB, CB, LOG_CB, ONE_ROW, G) - probs) < p[:, None]) | (col == 0)
        probs = tl.where(keep, probs, 0.0)
        probs = probs / _lane_sum(probs, R, LIT, L, LOG_L, True)[:, None]
    return tl.where(col < K, probs, 0.0)


@triton.jit
def _draw(probs, u, col, P: tl.constexpr):
    """spec_sample._draw: the first index of the max of log(probs) - log(-log(max(u, 1e-20)))."""
    g = _log(probs) - _log(-_log(tl.maximum(u, 1e-20)))
    g = tl.where(col < P, g, float("-inf"))
    best = tl.max(g, axis=1)
    return tl.min(tl.where(g == best[:, None], col, P), axis=1)


@triton.jit(do_not_specialize=["vocab"])
def _draft_kernel(vals_ptr, ids_ptr, temp_ptr, topk_ptr, topp_ptr, u_ptr, tok_ptr, probs_ptr, vocab,
                  K: tl.constexpr, P: tl.constexpr, SIT: tl.constexpr, SW: tl.constexpr, LOG_SW: tl.constexpr,
                  LIT: tl.constexpr, L: tl.constexpr, LOG_L: tl.constexpr, NB: tl.constexpr, CB: tl.constexpr,
                  LOG_CB: tl.constexpr, ONE_ROW: tl.constexpr, G: tl.constexpr, HAS_TOPK: tl.constexpr,
                  HAS_TOPP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, P)[None, :]
    mask = col < K
    vals = tl.load(vals_ptr + row * K + col, mask=mask, other=float("-inf"))
    temp = tl.maximum(tl.load(temp_ptr + row + tl.zeros((1,), tl.int64)), 1e-5)
    k = tl.zeros((1,), tl.int32)
    if HAS_TOPK:
        k = tl.load(topk_ptr + row + tl.zeros((1,), tl.int64))
        k = tl.where(k > 0, k, vocab)
    p = tl.zeros((1,), tl.float32)
    if HAS_TOPP:
        p = tl.load(topp_ptr + row + tl.zeros((1,), tl.int64))
    probs = _distribution(vals, temp, k, p, col, 1, K, SIT, SW, LOG_SW, LIT, L, LOG_L, NB, CB, LOG_CB,
                          ONE_ROW, G, HAS_TOPK, HAS_TOPP)
    u = tl.load(u_ptr + row * K + col, mask=mask, other=0.5)
    j = _draw(probs, u, col, K)
    tok = tl.sum(tl.where(col == j[:, None], tl.load(ids_ptr + row * K + col, mask=mask, other=0), 0), axis=1)
    tl.store(tok_ptr + row + tl.arange(0, 1), tok)
    tl.store(probs_ptr + row * K + col, probs, mask=mask)


def draft_from_candidates(vals: torch.Tensor, ids: torch.Tensor, temperature: torch.Tensor,
                          top_k: torch.Tensor | None, top_p: torch.Tensor | None, vocab: int,
                          gen: torch.Generator | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """spec_sample.draft_distribution(cands=(vals, ids, vocab)) in one kernel after the same torch.rand."""
    B, K = vals.shape
    u = torch.rand((B, K), device=vals.device, generator=gen)
    tok = torch.empty(B, dtype=torch.int64, device=vals.device)
    probs = torch.empty((B, K), dtype=torch.float32, device=vals.device)
    _draft_kernel[(B,)](vals.contiguous(), ids.contiguous(), temperature, top_k if top_k is not None else temperature,
                        top_p if top_p is not None else temperature, u, tok, probs, vocab, K=K,
                        **_geometry(K, B), HAS_TOPK=top_k is not None, HAS_TOPP=top_p is not None, num_warps=1,
                        enable_fp_fusion=False)
    return tok, (ids, probs)


def _geometry(K: int, rows: int) -> dict:
    """The reduction shapes of a K-wide row in a torch call over ``rows`` rows (P = next_pow2(K), at least 4): the
    softmax's lanes and iterations, the sum's lanes (the largest power of two <= K, at most 32), the cumsum's blocks
    (ATen's get_log_num_threads_x_inner_scan: 2 * its x threads, the whole padded row when wider; one row: rocprim)."""
    P = max(4, triton.next_power_of_2(K))
    sw = min(P, 32)
    lanes = min(32, 1 << (K.bit_length() - 1))
    lx, ly = (K - 1).bit_length(), (rows - 1).bit_length()
    cb = min(P, 2 << min(max(4, (9 + lx - ly) // 2), 9))
    return dict(P=P, SIT=P // sw, SW=sw, LOG_SW=sw.bit_length() - 1, LIT=P // lanes, L=lanes,
                LOG_L=lanes.bit_length() - 1, NB=P // cb, CB=cb, LOG_CB=cb.bit_length() - 1, ONE_ROW=rows == 1,
                G=P // 4)


@triton.jit(do_not_specialize=["vocab"])
def _spec_kernel(vals_ptr, ids_ptr, temp_ptr, topk_ptr, topp_ptr, drafts_ptr, qi_ptr, qp_ptr, ua_ptr, ur_ptr, ul_ptr,
                 tok_ptr, nacc_ptr, vocab, M: tl.constexpr, MP: tl.constexpr, K: tl.constexpr, P: tl.constexpr,
                 SIT: tl.constexpr, SW: tl.constexpr, LOG_SW: tl.constexpr, LIT: tl.constexpr, L: tl.constexpr,
                 LOG_L: tl.constexpr, NB: tl.constexpr, CB: tl.constexpr, LOG_CB: tl.constexpr,
                 ONE_ROW: tl.constexpr, G: tl.constexpr, KQ: tl.constexpr,
                 KQP: tl.constexpr, HAS_TOPK: tl.constexpr, HAS_TOPP: tl.constexpr, HAS_Q: tl.constexpr):
    req = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, MP)[:, None]  # the request's rows
    col = tl.arange(0, P)[None, :]
    rmask = r < M
    mask = rmask & (col < K)
    base = (req * M + r) * K + col
    vals = tl.load(vals_ptr + base, mask=mask, other=float("-inf"))
    ids = tl.load(ids_ptr + base, mask=mask, other=-1)
    rr = tl.arange(0, MP)
    temp = tl.maximum(tl.load(temp_ptr + req + rr * 0), 1e-5)  # every row takes its request's params
    k = tl.zeros((MP,), tl.int32)
    if HAS_TOPK:
        k = tl.load(topk_ptr + req + rr * 0)
        k = tl.where(k > 0, k, vocab)
    p = tl.zeros((MP,), tl.float32)
    if HAS_TOPP:
        p = tl.load(topp_ptr + req + rr * 0)
    probs = _distribution(vals, temp, k, p, col, MP, K, SIT, SW, LOG_SW, LIT, L, LOG_L, NB, CB, LOG_CB,
                          ONE_ROW, G, HAS_TOPK, HAS_TOPP)
    probs = tl.where(rmask, probs, 0.0)
    # rows 0..M-2 judge drafts 1..M-1
    dm = rr < M - 1
    d = tl.load(drafts_ptr + req * (M - 1) + rr, mask=dm, other=-1).to(tl.int64)
    is_draft = (ids == d[:, None]) & dm[:, None]
    p_draft = tl.sum(tl.where(is_draft, probs, 0.0), axis=1)
    u = tl.load(ua_ptr + req * (M - 1) + rr, mask=dm, other=1.0)
    if HAS_Q:
        qc = tl.arange(0, KQP)[None, :]
        qm = dm[:, None] & (qc < KQ)
        qbase = (req * (M - 1) + r) * KQ + qc
        q_ids = tl.load(qi_ptr + qbase, mask=qm, other=-2)
        q_probs = tl.load(qp_ptr + qbase, mask=qm, other=0.0)
        q_draft = tl.sum(tl.where(q_ids == d[:, None], q_probs, 0.0), axis=1)
        accepted = (u * q_draft) < p_draft
        # q at each candidate id: [MP, P, KQP] -> [MP, P]
        q_at = tl.sum(tl.where(q_ids[:, None, :] == ids[:, :, None], q_probs[:, None, :], 0.0), axis=2)
        residual = tl.maximum(probs - q_at, 0.0)
        empty = tl.max(residual, axis=1) <= 0.0  # a sum of non-negative values is <= 0 iff all are 0
        residual = tl.where(empty[:, None], probs, residual)
    else:
        accepted = u < p_draft
        residual = tl.where(is_draft, 0.0, probs)
    accepted = accepted & dm
    # n_acc = 1 + the leading accepted drafts
    run = tl.cumsum(tl.where(dm, 1 - accepted.to(tl.int32), 0), axis=0)
    n_acc = 1 + tl.sum(((run == 0) & dm).to(tl.int64), axis=0)
    residual = tl.where(dm[:, None] & (col < K), residual, 0.0)
    residual = residual / tl.maximum(_lane_sum(residual, MP, LIT, L, LOG_L, True), 1e-30)[:, None]
    ur = tl.load(ur_ptr + (req * (M - 1) + r) * K + col, mask=dm[:, None] & (col < K), other=0.5)
    jr = _draw(residual, ur, col, K)  # [MP]
    res_tok = tl.sum(tl.where(col == jr[:, None], ids, 0), axis=1)
    last = r == M - 1
    ul = tl.load(ul_ptr + req * K + col + r * 0, mask=last & (col < K), other=0.5)
    jl = _draw(tl.where(last, probs, 0.0), ul, col, K)
    last_tok = tl.sum(tl.where(last & (col == jl[:, None]), ids, 0), axis=1)
    last_tok = tl.sum(tl.where(rr == M - 1, last_tok, 0), axis=0)
    stop = n_acc - 1
    # tokens: the kept drafts, the drawn token at row stop (the residual's, or the last row's), the rest don't-care
    drawn = tl.where(rr == M - 1, last_tok, res_tok)
    out = tl.where(rr < stop, d, tl.where(rr == stop, drawn, tl.where(rr == M - 1, last_tok, d)))
    tl.store(tok_ptr + req * M + rr, out, mask=rr < M)
    tl.store(nacc_ptr + req + tl.arange(0, 1), n_acc + tl.zeros((1,), tl.int64))


def spec_sample_from_candidates(vals: torch.Tensor, ids: torch.Tensor, drafts: torch.Tensor, temperature: torch.Tensor,
                                top_k: torch.Tensor | None, top_p: torch.Tensor | None, vocab: int,
                                q: tuple[torch.Tensor, torch.Tensor] | None = None,
                                gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """spec_sample.spec_sample(cands=(vals [B, m, K], ids, vocab), q sparse or None) in one kernel after the same
    three torch.rand calls."""
    B, M, K = vals.shape
    dev = vals.device
    u = torch.rand((B, M - 1), device=dev, generator=gen)
    ur = torch.rand((B, M - 1, K), device=dev, generator=gen)
    ul = torch.rand((B, K), device=dev, generator=gen)
    tok = torch.empty((B, M), dtype=torch.int64, device=dev)
    n_acc = torch.empty(B, dtype=torch.int64, device=dev)
    geo = _geometry(K, B * M)
    KQ = q[0].shape[-1] if q is not None else 1
    _spec_kernel[(B,)](
        vals.contiguous(), ids.contiguous(), temperature, top_k if top_k is not None else temperature,
        top_p if top_p is not None else temperature, drafts.contiguous(),
        q[0].contiguous() if q is not None else ids, q[1].contiguous() if q is not None else vals,
        u, ur, ul, tok, n_acc, vocab, M=M, MP=max(2, triton.next_power_of_2(M)), K=K, **geo,
        KQ=KQ, KQP=max(2, triton.next_power_of_2(KQ)), HAS_TOPK=top_k is not None, HAS_TOPP=top_p is not None,
        HAS_Q=q is not None, num_warps=1 if geo["P"] * max(2, triton.next_power_of_2(M)) <= 256 else 4,
        enable_fp_fusion=False,
    )
    return tok, n_acc


def warm(device: torch.device, k: int, width: int, start: int, tp: int, max_reqs: int, rows: tuple[int, ...]) -> None:
    """Compile the candidates' and the draws' kernel variants a decode step can launch with top-k ``k`` (up to
    ``max_reqs`` requests, verify steps of ``rows`` rows each): each new variant first compiles for 0.3-5 s, in the
    middle of a request otherwise. Every random number from its own generator: drawing from the default one would
    shift the serving draws."""
    from freetoken.kernel.triton.vocab_topk import merge_candidates, pack_candidates

    gen = torch.Generator(device=device).manual_seed(0)
    vocab = width * tp
    for kk in (1, k):
        for dtype in (torch.float32, torch.bfloat16):  # the graphs' logits and the draft head's
            packed = pack_candidates(torch.randn(2, width, device=device, generator=gen).to(dtype), kk, start, width)
        for stable in (True, False):
            merge_candidates(torch.cat([packed] * tp), tp, kk, 1 if stable else 2)
    vals, ids = torch.topk(torch.randn(max_reqs * max(rows), 64, device=device, generator=gen), k, dim=-1)
    for b in range(1, max_reqs + 1):
        temps = torch.full((b,), 0.6, device=device)
        top_k = torch.full((b,), k, dtype=torch.int32, device=device)
        top_p = torch.full((b,), 0.95, device=device)
        for p in (top_p, None):
            draft_from_candidates(vals[:b], ids[:b], temps, top_k, p, vocab, gen=gen)
            for m in rows:
                if m < 2:
                    continue
                v, i = vals[: b * m].reshape(b, m, k), ids[: b * m].reshape(b, m, k)
                drafts = i[:, 1:, 0]
                for q in (None, (i[:, 1:].contiguous(), torch.softmax(v[:, 1:], -1).contiguous())):
                    spec_sample_from_candidates(v, i, drafts, temps, top_k, p, vocab, q=q, gen=gen)
    torch.cuda.synchronize(device)


__all__ = ["MAX_K", "draft_from_candidates", "spec_sample_from_candidates", "warm"]
