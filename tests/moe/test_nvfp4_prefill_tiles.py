"""The NVFP4 prefill tiles keep the bits: every config, at every block size, keeps the 32-byte K step (BLOCK_SIZE_KB), so
each output sums its K products in the same order whatever the M / N tiles, warps and stages; the short-prefill tiles
cover the sizes they were swept at."""

from __future__ import annotations

import pytest

from freetoken.moe import fused_nvfp4 as fm


@pytest.mark.parametrize("tuned", [False, True])
def test_every_prefill_tile_keeps_the_k_step(monkeypatch, tuned):
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", tuned)
    for M in [*range(1, 4100), 8192, 12288, 16384]:
        for cfg in fm._prefill_configs(M):
            assert cfg["BLOCK_SIZE_KB"] == 32, (M, cfg)


def test_short_prefills_take_the_short_tiles(monkeypatch):
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", True)
    assert fm._prefill_configs(16) == (fm._TUNED_TINY_GATE_UP, fm._TUNED_TINY_DOWN)
    assert fm._prefill_configs(300) == (fm._TUNED_SMALL, fm._TUNED_SMALL)
    assert fm._prefill_configs(1280) == (fm._TUNED_SHORT_GATE_UP, fm._TUNED_SHORT_DOWN)
    assert fm._prefill_configs(2048) == (fm._TUNED_MID, fm._TUNED_MID)
    assert fm._prefill_configs(8192) == (fm._TUNED_BIG_GATE_UP, fm._TUNED_MID)
    # untuned, the shipped tiles stay what they were
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", False)
    assert fm._prefill_configs(16) == (fm._prefill_config(16),) * 2
