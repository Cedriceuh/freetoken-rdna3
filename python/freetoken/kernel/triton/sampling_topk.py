"""Top-k-first sampling for small k: one torch.topk over the vocab, then one tiny Triton kernel.

top_k_top_p_sampling_from_probs(softmax(logits / T), k, p) needs a full-vocab softmax, a top-k
threshold search, a renormalization, a top-p threshold search on the renormalized row and an
inverse-CDF draw: about twenty launches, most of them full-vocab passes. With small k the same
distribution is reachable from the k largest logits alone: the renormalized top-k probabilities are
the softmax of the top-k logits divided by T, and top-p then keeps the shortest prefix of the sorted
candidates whose mass reaches p. So: torch.topk (sorted, descending), then a single program per row
does temperature, per-row k mask, softmax, top-p and the draw on at most K_PAD values.

The uniform comes from tl.rand with a host-side counter as seed. Every TP rank samples its own copy
of the logits, so the draws must match across ranks: they do because each rank makes the same calls
in the same order (and the engine broadcasts rank 0's tokens after sampled steps anyway).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SEED_BASE = 0x5EED_0000
_calls = 0


@triton.jit(do_not_specialize=["seed"])
def _topk_tail_kernel(
    vals_ptr, idx_ptr, temp_ptr, topk_ptr, topp_ptr, out_ptr, seed, K,
    HAS_TOPP: tl.constexpr, K_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, K_PAD)
    k = tl.minimum(tl.load(topk_ptr + row), K)
    keep = offs < k
    x = tl.load(vals_ptr + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
    x = tl.where(keep, x / tl.load(temp_ptr + row), float("-inf"))
    e = tl.where(keep, tl.exp(x - tl.max(x, 0)), 0.0)
    p = e / tl.sum(e, 0)
    if HAS_TOPP:
        # keep a candidate while the mass before it is below p (the first one always stays)
        p = tl.where(tl.cumsum(p, 0) - p < tl.load(topp_ptr + row), p, 0.0)
    c = tl.cumsum(p, 0)
    u = tl.rand(seed, row) * tl.max(c, 0)
    choice = tl.min(tl.where((c > u) & (p > 0.0), offs, K_PAD), 0)
    choice = tl.where(choice < K_PAD, choice, 0)  # rounding guard: fall back to the top candidate
    tl.store(out_ptr + row, tl.load(idx_ptr + row * K + choice))


def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor | None,
    k_max: int,
) -> torch.Tensor:
    """Sample one token per row from softmax(logits / T) restricted to top-k then top-p.

    k_max is the host-side max of top_k (no device sync); rows with a smaller k are masked.
    Works on the logits' own dtype: topk on bf16 selects the same set as on their fp32 cast.
    """
    global _calls
    B = logits.size(0)
    vals, idx = torch.topk(logits, k_max, dim=-1)
    out = torch.empty(B, dtype=torch.int64, device=logits.device)
    seed = _SEED_BASE + _calls
    _calls += 1
    _topk_tail_kernel[(B,)](
        vals, idx, temperatures, top_k, top_p if top_p is not None else temperatures, out, seed, k_max,
        HAS_TOPP=top_p is not None, K_PAD=max(16, triton.next_power_of_2(k_max)), num_warps=1,
    )
    return out
