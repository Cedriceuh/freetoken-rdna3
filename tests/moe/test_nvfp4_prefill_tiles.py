"""The RDNA3 NVFP4 prefill tiles keep the bits: every tuned config, at every block size, keeps the 32-byte K step
(BLOCK_SIZE_KB), so each output sums its K products in the same order whatever the M / N tiles, warps and stages; the
short-prefill tiles cover the sizes they were swept at, and untuned prefills take upstream's tiles."""

from __future__ import annotations

from freetoken.kernel import nvfp4_moe
from freetoken.moe import fused_nvfp4 as fm

# the shape arguments only matter to the untuned (upstream) tiles
TOP_K, EXPERTS = 10, 512


def test_every_prefill_tile_keeps_the_k_step(monkeypatch):
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", True)
    for M in [*range(1, 4100), 8192, 12288, 16384]:
        for cfg in fm._prefill_configs(M, TOP_K, EXPERTS, 0):
            assert cfg["BLOCK_SIZE_KB"] == 32, (M, cfg)


def test_short_prefills_take_the_short_tiles(monkeypatch):
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", True)
    configs = lambda M: fm._prefill_configs(M, TOP_K, EXPERTS, 0)  # noqa: E731
    assert configs(16) == (fm._TUNED_TINY_GATE_UP, fm._TUNED_TINY_DOWN)
    assert configs(300) == (fm._TUNED_SMALL, fm._TUNED_SMALL)
    assert configs(1280) == (fm._TUNED_SHORT_GATE_UP, fm._TUNED_SHORT_DOWN)
    assert configs(2048) == (fm._TUNED_MID, fm._TUNED_MID)
    assert configs(8192) == (fm._TUNED_BIG_GATE_UP, fm._TUNED_MID)
    # untuned, upstream's tiles (its choice reads the GPU's properties: checked here by the call alone)
    monkeypatch.setattr(fm, "NVFP4_PREFILL_TUNED", False)
    seen = []
    monkeypatch.setattr(nvfp4_moe, "prefill_config", lambda *args: seen.append(args) or {"BLOCK_SIZE_M": 16})
    assert configs(16) == ({"BLOCK_SIZE_M": 16},) * 2 and seen == [(16, TOP_K, EXPERTS, 0)]
