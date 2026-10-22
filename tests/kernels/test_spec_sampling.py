"""FREETOKEN_SAMPLE_KERNELS: the fused sampling kernels give the torch ops' bits (engine/spec_sample.py on the LM
head's candidates): the same probabilities, draws, tokens and kept counts for the same generator state (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None,
                                reason="needs a ROCm GPU (the bits checked are ROCm's)")


def _cands(rows, k, spread, gen, bf16=True):
    """Candidate rows as the LM head returns them: [rows, k] values sorted descending (bf16-valued), with ids."""
    x = torch.randn(rows, 248320, generator=gen, device="cuda") * spread + 12
    if bf16:
        x = x.to(torch.bfloat16).float()
    vals, ids = torch.topk(x, k, dim=-1)
    return vals, ids


def _params(n, gen, vocab=248320):
    temps = torch.rand(n, generator=gen, device="cuda") * 1.2 + 0.05
    top_k = torch.randint(1, 30, (n,), generator=gen, device="cuda").to(torch.int32)
    top_k[::4] = vocab  # greedy rows carry the vocabulary
    top_p = torch.rand(n, generator=gen, device="cuda") * 0.7 + 0.3
    top_p[1::3] = 1.0
    return temps, top_k, top_p


@pytest.mark.parametrize("k", [1, 2, 3, 4, 7, 9, 16, 20, 31, 32, 33, 50, 64])
@pytest.mark.parametrize("topk_on,topp_on", [(True, True), (True, False), (False, True), (False, False)])
def test_draft_kernel_matches_torch(k, topk_on, topp_on):
    from freetoken.engine.spec_sample import draft_distribution
    from freetoken.kernel.triton.spec_sampling import draft_from_candidates

    gen = torch.Generator(device="cuda").manual_seed(k * 7 + topk_on * 2 + topp_on)
    for trial in range(4):
        n = 1 + trial * 3
        vals, ids = _cands(n, k, (0.3, 1.0, 3.0, 0.1)[trial], gen)
        temps, top_k, top_p = _params(n, gen)
        top_k = top_k if topk_on else None
        top_p = top_p if topp_on else None
        if not topk_on:  # spec_sample's sparse branch needs top_k: give every row the vocabulary
            top_k_arg = torch.full((n,), 248320, dtype=torch.int32, device="cuda")
        else:
            top_k_arg = top_k
        g1 = torch.Generator(device="cuda").manual_seed(trial)
        g2 = torch.Generator(device="cuda").manual_seed(trial)
        ref_t, (ref_ids, ref_p) = draft_distribution(None, temps, top_k_arg, top_p, k, gen=g1,
                                                     cands=(vals, ids, 248320), fused=False)
        got_t, (got_ids, got_p) = draft_from_candidates(vals, ids, temps, top_k_arg, top_p, 248320, gen=g2)
        assert torch.equal(ref_p, got_p), (k, trial, (ref_p != got_p).sum().item())
        assert torch.equal(ref_t, got_t)


@pytest.mark.parametrize("k", [1, 5, 20, 32, 50, 64])
@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("with_q", [False, True])
def test_spec_kernel_matches_torch(k, m, with_q):
    from freetoken.engine.spec_sample import spec_sample
    from freetoken.kernel.triton.spec_sampling import spec_sample_from_candidates

    gen = torch.Generator(device="cuda").manual_seed(k * 100 + m * 10 + with_q)
    for trial in range(5):
        reqs = 1 + trial
        vals, ids = _cands(reqs * m, k, (0.3, 1.0, 3.0, 0.1, 0.5)[trial], gen)
        vals, ids = vals.view(reqs, m, k), ids.view(reqs, m, k)
        temps, top_k, top_p = _params(reqs, gen)
        if trial == 4:
            top_p = None
        # drafts: often one of the candidates, sometimes not
        pick = torch.randint(0, max(1, min(k, 4)), (reqs, m - 1), generator=gen, device="cuda")
        drafts = ids[:, :-1].gather(-1, pick.unsqueeze(-1)).squeeze(-1)
        drafts[:, ::2] = torch.where(torch.rand(drafts[:, ::2].shape, generator=gen, device="cuda") < 0.3,
                                     torch.randint(0, 248320, drafts[:, ::2].shape, generator=gen, device="cuda"),
                                     drafts[:, ::2])
        q = None
        if with_q:
            qk = max(1, k // 2 + 1)
            q_ids = torch.cat([drafts.unsqueeze(-1), ids[:, 1:, : qk - 1]], -1) if qk > 1 else drafts.unsqueeze(-1)
            q_probs = torch.softmax(torch.randn(q_ids.shape, generator=gen, device="cuda"), -1)
            q = (q_ids.contiguous(), q_probs.contiguous())
        g1 = torch.Generator(device="cuda").manual_seed(50 + trial)
        g2 = torch.Generator(device="cuda").manual_seed(50 + trial)
        ref = spec_sample(None, drafts, temps, top_k, top_p, k, gen=g1, q=q, cands=(vals, ids, 248320), fused=False)
        got = spec_sample_from_candidates(vals, ids, drafts, temps, top_k, top_p, 248320, q=q, gen=g2)
        assert torch.equal(ref[1], got[1]), (trial, ref[1].tolist(), got[1].tolist())
        for b in range(reqs):  # rows past n_acc are don't-cares
            n = int(ref[1][b])
            assert torch.equal(ref[0][b, :n], got[0][b, :n]), (trial, b, ref[0][b].tolist(), got[0][b].tolist())


def test_warm_leaves_the_default_generator_alone():
    """The start-up compile of the kernel variants draws its data from its own generator: the serving draws (the
    default generator's) stay those of a build without it."""
    from freetoken.kernel.triton.spec_sampling import warm

    torch.cuda.manual_seed(1)
    ref = torch.rand(8, device="cuda")
    torch.cuda.manual_seed(1)
    warm(torch.device("cuda"), 20, 8192, 0, 2, 2, (2, 3))
    assert torch.equal(torch.rand(8, device="cuda"), ref)
