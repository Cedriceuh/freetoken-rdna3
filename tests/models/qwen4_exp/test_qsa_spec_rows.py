"""A spec_decode verify step: m rows of one request through the real QSA layer give each row exactly the bits a
single-row decode at that position gives (attention, indexer, the compression ring).

The ring is sized for FREETOKEN_SPEC_VERIFY_M at import: run with FREETOKEN_SPEC_VERIFY_M=4 to cover m = 4."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import freetoken.spec_decode as spec

from .common import Fixture, parsed_config, requires_cuda

QSA_LAYER = 3


@pytest.fixture(autouse=True)
def _row_invariant_projections(monkeypatch):
    """Serving runs these projections through batch-invariant GEMVs (int8, or the bf16 table); the toy shapes here would
    go to F.linear, whose BLAS pick -- hence the bits -- changes with the row count (ROCm 10, (3072, 256)). Decode-sized
    inputs are computed a row at a time, so the test checks the QSA layer itself."""
    from freetoken.layers.quantization.linear import unquantized

    apply = unquantized.TorchLinearKernel.apply

    def per_row(self, layer, x):
        if x.dim() == 2 and 1 < x.shape[0] <= 64:
            return torch.cat([apply(self, layer, x[i : i + 1]) for i in range(x.shape[0])])
        return apply(self, layer, x)

    monkeypatch.setattr(unquantized.TorchLinearKernel, "apply", per_row)


def _batch(fixture: Fixture, req, rows: int) -> SimpleNamespace:
    positions = torch.arange(req.cached_len, req.cached_len + rows, dtype=torch.int32, device=fixture.device)
    out_loc = fixture.page_table[req.table_idx, req.cached_len : req.cached_len + rows].contiguous()
    batch = SimpleNamespace(
        reqs=[req], padded_reqs=[req], phase="decode", size=1, padded_size=1, is_prefill=False, is_decode=True,
        positions=positions, get_attn_positions=lambda: positions, mm_embeds=None, out_loc=out_loc,
        attn_metadata=None, active_table_idx=torch.tensor([req.table_idx], dtype=torch.int32, device=fixture.device),
        spec_m=rows,
    )
    fixture.backend.prepare_metadata(batch)
    return batch


@requires_cuda
@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("start", [300, 301, 302, 303])
def test_verify_rows_match_single_row_decode(m: int, start: int):
    if m > spec.SPEC_M:
        pytest.skip(f"the QSA ring is sized for FREETOKEN_SPEC_VERIFY_M={spec.SPEC_M}")
    config = parsed_config()
    fixture = Fixture(config, num_pages=64)
    attn = fixture.layer(QSA_LAYER)
    steps = 3 * m
    x = torch.randn(start + steps, config.hidden_size, device=fixture.device, dtype=fixture.dtype,
                    generator=torch.Generator(device=fixture.device).manual_seed(start)) * 0.5

    single, rows = fixture.req(0, 0, start), fixture.req(1, 0, start)
    for req in (single, rows):
        attn.forward(x[:start], fixture.batch([req], "prefill"))
    want = []
    for p in range(start, start + steps):
        fixture.step(single)
        want.append(attn.forward(x[p : p + 1], fixture.batch([single], "decode")))
    want = torch.cat(want)

    got = []
    for first in range(start, start + steps, m):  # every row kept: the next step starts after the last one
        fixture.allocate(rows.table_idx, first, first + m)
        rows.cached_len, rows.device_len, rows.extend_len = first, first + m, m
        got.append(attn.forward(x[first : first + m], _batch(fixture, rows, m)))
    got = torch.cat(got)
    for i in range(steps):
        assert torch.equal(got[i], want[i]), f"m={m}: row at position {start + i} differs from single-row decode"


def _batch_reqs(fixture: Fixture, reqs, rows: int) -> SimpleNamespace:
    positions = torch.cat([torch.arange(r.cached_len, r.cached_len + rows, dtype=torch.int32, device=fixture.device)
                           for r in reqs])
    out_loc = torch.cat([fixture.page_table[r.table_idx, r.cached_len : r.cached_len + rows] for r in reqs]).contiguous()
    batch = SimpleNamespace(
        reqs=reqs, padded_reqs=reqs, phase="decode", size=len(reqs), padded_size=len(reqs), is_prefill=False,
        is_decode=True, positions=positions, get_attn_positions=lambda: positions, mm_embeds=None, out_loc=out_loc,
        attn_metadata=None, spec_m=rows,
        active_table_idx=torch.tensor([r.table_idx for r in reqs], dtype=torch.int32, device=fixture.device),
    )
    fixture.backend.prepare_metadata(batch)
    return batch


@requires_cuda
@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("bs", [2, 3, 4])
def test_batched_verify_rows_match_single_row_decode(m: int, bs: int):
    """``bs`` requests verifying ``m`` rows each in one step: every row gets its single-request, single-row bits."""
    if m > spec.SPEC_M:
        pytest.skip(f"the QSA ring is sized for FREETOKEN_SPEC_VERIFY_M={spec.SPEC_M}")
    config = parsed_config()
    fixture = Fixture(config, num_pages=128)
    attn = fixture.layer(QSA_LAYER)
    starts = [300 + 37 * i for i in range(bs)]
    steps = 2 * m
    gen = torch.Generator(device=fixture.device).manual_seed(bs * 10 + m)
    xs = [torch.randn(s + steps, config.hidden_size, device=fixture.device, dtype=fixture.dtype, generator=gen) * 0.5
          for s in starts]
    singles = [fixture.req(i, 0, s) for i, s in enumerate(starts)]
    multis = [fixture.req(bs + i, 0, s) for i, s in enumerate(starts)]
    for req, x, s in zip(singles + multis, xs + xs, starts + starts):
        attn.forward(x[:s], fixture.batch([req], "prefill"))
    want = []
    for req, x, s in zip(singles, xs, starts):
        rows = []
        for p in range(s, s + steps):
            fixture.step(req)
            rows.append(attn.forward(x[p : p + 1], fixture.batch([req], "decode")))
        want.append(torch.cat(rows))
    got = [[] for _ in range(bs)]
    for k in range(0, steps, m):
        for req, s in zip(multis, starts):
            first = s + k
            fixture.allocate(req.table_idx, first, first + m)
            req.cached_len, req.device_len, req.extend_len = first, first + m, m
        x = torch.cat([xs[i][starts[i] + k : starts[i] + k + m] for i in range(bs)])
        out = attn.forward(x, _batch_reqs(fixture, multis, m))
        for i in range(bs):
            got[i].append(out[i * m : (i + 1) * m])
    for i in range(bs):
        g = torch.cat(got[i])
        bad = [j for j in range(steps) if not torch.equal(g[j], want[i][j])]
        assert not bad, f"bs={bs} m={m}: request {i} rows {bad} differ from single-row decode"
