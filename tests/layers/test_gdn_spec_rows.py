"""spec_decode's verify rows through the GatedDeltaNet layer (the helpers of ``test_gdn.py``): ``rows`` tokens of a
request in one decode forward give each row the bits of ``rows`` single-token decode steps and leave the same state,
for one request and for a batch; the decode norm runs one row per program."""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Batch
from tests.layers.test_gdn import DEV, HIDDEN, _ctx, _make_layer, _prefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _decode(op, ctx, reqs, hidden: torch.Tensor) -> torch.Tensor:
    """A decode step as the scheduler stages it: each request's state slot in ``linear_table_idx``."""
    batch = Batch(reqs=reqs, phase="decode")
    batch.padded_reqs = reqs
    batch.linear_table_idx = torch.tensor([r.table_idx for r in reqs], dtype=torch.int32, device=DEV)
    with ctx.forward_batch(batch):
        return op.forward(hidden)


@pytest.mark.parametrize("rows", (2, 3, 4))
def test_verify_rows_match_single_row_decode(monkeypatch, rows):
    """A spec_decode verify step (``rows`` tokens of one request in one decode forward) gives each row exactly the
    bits ``rows`` single-token decode steps give, and leaves the same state. The spec buffers are sized for
    FREETOKEN_SPEC_VERIFY_M at import: run with FREETOKEN_SPEC_VERIFY_M=4 to cover 4 rows."""
    import freetoken.spec_decode as spec

    if rows > spec.SPEC_M:
        pytest.skip(f"spec buffers sized for FREETOKEN_SPEC_VERIFY_M={spec.SPEC_M}")
    op, _ = _make_layer(3, "sigmoid")
    ctx = _ctx(monkeypatch, 3)
    torch.manual_seed(5)
    prompt = torch.randn(37, HIDDEN, device=DEV, dtype=torch.bfloat16)
    _, (single, multi), _ = _prefill(op, ctx, [37, 37], seed=0)
    pool = ctx.linear_state_pool
    batch = Batch(reqs=[single, multi], phase="prefill")
    batch.padded_reqs = batch.reqs
    with ctx.forward_batch(batch):
        op.forward(torch.cat([prompt, prompt]))  # the same prompt in both slots
    steps = torch.randn(rows, HIDDEN, device=DEV, dtype=torch.bfloat16)
    want = torch.cat([_decode(op, ctx, [single], steps[j : j + 1]) for j in range(rows)])
    batch = Batch(reqs=[multi], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.linear_table_idx = torch.tensor([multi.table_idx], dtype=torch.int32, device=DEV)
    batch.spec_m = rows
    with ctx.forward_batch(batch):
        got = op.forward(steps)
    assert torch.equal(got, want)
    li = pool.local_index(0)
    assert torch.equal(pool.recurrent_states[li][multi.table_idx], pool.recurrent_states[li][single.table_idx])
    assert torch.equal(pool.conv_states[li][multi.table_idx], pool.conv_states[li][single.table_idx])


def test_decode_norm_runs_one_row_per_program(monkeypatch):
    """The output norm sizes its row blocks by the row count unless told otherwise; on gfx1100 a 2-row block rounded
    1 value in ~3.5k differently (rank 0, 27 heads x 4 verify rows), so decode pins one row per program."""
    import freetoken.kernel.fla as fla

    seen = []
    original = fla.rms_norm_gated

    def spy(**kw):
        seen.append(kw.get("rows_per_block"))
        return original(**kw)

    monkeypatch.setattr(fla, "rms_norm_gated", spy)
    op, _ = _make_layer(3, "sigmoid")
    ctx = _ctx(monkeypatch, 3)
    _, reqs, _ = _prefill(op, ctx, [9], seed=0)
    _decode(op, ctx, reqs, torch.randn(1, HIDDEN, device=DEV, dtype=torch.bfloat16))
    assert seen == [None, 1]  # prefill: sized by rows; decode: one row per program


@pytest.mark.parametrize("rows", (2, 3, 4))
@pytest.mark.parametrize("bs", (2, 4))
def test_batched_verify_rows_match_single_row_decode(monkeypatch, bs, rows):
    """``bs`` requests verifying ``rows`` rows each in one decode forward: each request gets the outputs and the
    final state of ``rows`` single-token steps taken alone."""
    import freetoken.spec_decode as spec

    if rows > spec.SPEC_M:
        pytest.skip(f"spec buffers sized for FREETOKEN_SPEC_VERIFY_M={spec.SPEC_M}")
    op, _ = _make_layer(3, "sigmoid")
    ctx = _ctx(monkeypatch, 3, num_slots=2 * bs + 2)
    torch.manual_seed(7)
    lengths = [23 + 5 * i for i in range(bs)]
    hidden, reqs, _ = _prefill(op, ctx, lengths + lengths, seed=1)
    batch = Batch(reqs=reqs, phase="prefill")  # again, requests i and bs + i on the same prompt
    batch.padded_reqs = reqs
    with ctx.forward_batch(batch):
        op.forward(torch.cat(hidden[:bs] + hidden[:bs], dim=0))
    pool = ctx.linear_state_pool
    singles, multis = reqs[:bs], reqs[bs:]
    steps = torch.randn(bs, rows, HIDDEN, device=DEV, dtype=torch.bfloat16)
    want = [torch.cat([_decode(op, ctx, [r], steps[i, j : j + 1]) for j in range(rows)]) for i, r in enumerate(singles)]
    batch = Batch(reqs=multis, phase="decode")
    batch.padded_reqs = multis
    batch.linear_table_idx = torch.tensor([r.table_idx for r in multis], dtype=torch.int32, device=DEV)
    batch.spec_m = rows
    with ctx.forward_batch(batch):
        got = op.forward(steps.reshape(bs * rows, HIDDEN)).view(bs, rows, -1)
    li = pool.local_index(0)
    for i, (s, mreq) in enumerate(zip(singles, multis)):
        assert torch.equal(got[i], want[i]), f"request {i}"
        assert torch.equal(pool.recurrent_states[li][mreq.table_idx], pool.recurrent_states[li][s.table_idx])
        assert torch.equal(pool.conv_states[li][mreq.table_idx], pool.conv_states[li][s.table_idx])
