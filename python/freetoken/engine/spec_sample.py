"""Speculative sampling of a verify step.

For draft ``d_{j+1}`` after row ``j``, drawn from the draft distribution ``q_j``, the target distribution ``p_j``
(temperature, then top-k, then top-p, as the sampler draws) accepts it with probability min(1, p_j(d) / q_j(d)). The
first rejected row ``j`` draws from the residual max(0, p_j - q_j), renormalized; when every draft is kept, the row
after the last one draws from ``p_k``. Each emitted token is then distributed exactly as plain sampling would draw it.
Deterministic drafts (the MTP head's argmax) are the point-mass case: accepted with probability p_j(d), the residual
is p_j without d. All on the device (Gumbel-max draws), no host sync.
"""

from __future__ import annotations

import torch

# above this top-k (or with top-k off) the distribution is built over the whole vocabulary
_SPARSE_TOPK_MAX = 256


def _distribution(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor | None,
                  top_p: torch.Tensor | None, top_k_max: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(token ids [N, K], probs [N, K])``: each row's sampling support and its probabilities (zeros past it)."""
    x = logits.float() / temperature.clamp(min=1e-5).unsqueeze(-1)
    vocab = x.shape[-1]
    # host-side decision (no sync): every sampled row's top_k is <= top_k_max (Sampler.prepare); greedy rows,
    # whose top_k is the vocabulary, are overwritten by the caller
    sparse = top_k is not None and top_k_max is not None and 0 < top_k_max <= _SPARSE_TOPK_MAX
    width = top_k_max if sparse else vocab
    vals, ids = torch.topk(x, width, dim=-1) if sparse else torch.sort(x, dim=-1, descending=True)
    cols = torch.arange(width, device=x.device)
    if top_k is not None:
        k = torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab)).unsqueeze(-1)
        if sparse:  # the top-k-first sampler keeps exactly k
            vals = vals.masked_fill(cols >= k, float("-inf"))
        else:  # the full-vocabulary one keeps the ties at the k-th value
            vals = vals.masked_fill(vals < vals.gather(-1, (k.clamp(1, width) - 1)), float("-inf"))
    probs = vals.softmax(-1)
    if top_p is not None:  # the rule of the sampler that draws such a request (sorted descending here)
        p = top_p.unsqueeze(-1)
        if sparse:  # top-k-first: a candidate stays while the mass ahead of it is below top_p (the first always)
            keep = (probs.cumsum(-1) - probs < p) | (cols == 0)
        else:  # full vocabulary: every value >= the first one whose cumulative mass reaches top_p (ties kept);
            # top_p >= 1 cuts nothing (the fp32 cumsum saturates before the tail)
            j = torch.searchsorted(probs.cumsum(-1), p).clamp(max=width - 1)
            keep = (probs >= probs.gather(-1, j)) | (p >= 1.0)
        probs = probs.masked_fill(~keep, 0.0)
        probs = probs / probs.sum(-1, keepdim=True)
    return ids, probs


def _draw(probs: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """Index of a draw from each row of ``probs`` (Gumbel-max; zero-probability entries never win)."""
    u = torch.rand(probs.shape, device=probs.device, generator=gen).clamp_(min=1e-20)
    return (probs.log() - (-u.log()).log()).argmax(-1)


def draft_distribution(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor | None = None,
                       top_p: torch.Tensor | None = None, top_k_max: int | None = None,
                       gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A sampled draft: ``(tokens [B], q [B, V])`` -- one draw from the draft head's ``logits [B, V]`` under each
    request's sampling params, and that distribution over the vocabulary (what :func:`spec_sample` needs)."""
    ids, probs = _distribution(logits, temperature, top_k, top_p, top_k_max)
    tokens = ids.gather(-1, _draw(probs, gen).unsqueeze(-1)).squeeze(-1)
    q = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device).scatter_(-1, ids, probs)
    return tokens, q


def spec_sample(rows_logits: torch.Tensor, drafts: torch.Tensor, temperature: torch.Tensor,
                top_k: torch.Tensor | None = None, top_p: torch.Tensor | None = None,
                top_k_max: int | None = None, gen: torch.Generator | None = None,
                q: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``rows_logits [B, m, V]`` (the target after each of a request's m rows), ``drafts [B, m-1]`` (the tokens at
    rows 1..m-1), per-request sampling params ``[B]``; ``q [B, m-1, V]``: the distributions the drafts were drawn from
    (None: deterministic drafts). Returns ``(tokens [B, m], n_acc [B])``: the first n_acc tokens are the emitted ones
    (the accepted drafts, then the drawn token); the rest are don't-cares."""
    reqs, m, vocab = rows_logits.shape
    per_row = lambda t: None if t is None else t.repeat_interleave(m)  # noqa: E731
    ids, probs = _distribution(rows_logits.reshape(reqs * m, vocab), temperature.repeat_interleave(m),
                               per_row(top_k), per_row(top_p), top_k_max)
    ids, probs = ids.view(reqs, m, -1), probs.view(reqs, m, -1)
    # p_j(d_{j+1}) for the m-1 drafts
    is_draft = ids[:, :-1] == drafts.unsqueeze(-1).to(ids.dtype)
    p_draft = (probs[:, :-1] * is_draft).sum(-1)
    u = torch.rand(p_draft.shape, device=p_draft.device, generator=gen)
    if q is None:
        accepted = (u < p_draft).to(torch.int64)
        residual = probs[:, :-1].masked_fill(is_draft, 0.0)
    else:
        q_draft = q.gather(-1, drafts.unsqueeze(-1).to(torch.int64)).squeeze(-1)
        accepted = (u * q_draft < p_draft).to(torch.int64)  # u < p / q, without the division
        residual = (probs[:, :-1] - q.gather(-1, ids[:, :-1].to(torch.int64))).clamp_(min=0.0)
        # p <= q on the whole support (p == q up to rounding): a rejection is then (almost) impossible; draw from p
        empty = residual.sum(-1, keepdim=True) <= 0
        residual = torch.where(empty, probs[:, :-1], residual)
    n_acc = 1 + accepted.cumprod(-1).sum(-1)
    # the residual draw at every draft row, and the plain draw at the last row
    residual = residual / residual.sum(-1, keepdim=True).clamp(min=1e-30)
    residual_tok = ids[:, :-1].gather(-1, _draw(residual, gen).unsqueeze(-1)).squeeze(-1)  # [B, m-1]
    last_tok = ids[:, -1].gather(-1, _draw(probs[:, -1], gen).unsqueeze(-1)).squeeze(-1)  # [B]
    drawn = torch.cat([residual_tok, last_tok.unsqueeze(-1)], dim=-1)  # the draw if row j is where it stops
    tokens = torch.cat([drafts.to(drawn.dtype), last_tok.unsqueeze(-1)], dim=-1).clone()  # kept drafts...
    stop = n_acc - 1
    tokens.scatter_(1, stop.unsqueeze(-1), drawn.gather(1, stop.unsqueeze(-1)))  # ...then the drawn token
    return tokens, n_acc


__all__ = ["draft_distribution", "spec_sample"]
