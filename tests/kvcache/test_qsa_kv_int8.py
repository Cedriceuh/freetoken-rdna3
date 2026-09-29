"""QSAKVCache with FREETOKEN_QSA_KV_INT8=1: int8 K/V slabs + fp32 per-(token, kv head) scales.

CPU only (the store and attend kernels are checked on the GPU by rdna3/tests/qsa_kv_int8_check.py).
The planner must price exactly what the pool allocates, or the freed memory never reaches the
MoE expert cache (or the pool over-commits).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.qsa_pool import QSAKVCache

from .test_qsa_pool import FULL_LAYER_IDS, _config, _pool, _spec, _tp  # noqa: F401  (_tp: autouse fixture)


@pytest.fixture
def int8(monkeypatch):
    monkeypatch.setenv("FREETOKEN_QSA_KV_INT8", "1")


def test_int8_pool_allocates_codes_and_scales(int8):
    pool = _pool(num_pages=4)
    assert pool.kv_int8 and pool.dtype is torch.bfloat16  # compute dtype for the index tiers / backend scratch
    assert pool.k_cache(1).dtype is torch.int8 and pool.k_cache(1).shape == (4, 64, 2, 64)
    assert pool.k_scale(1).shape == pool.v_scale(7).shape == (4, 64, 2)
    assert pool.k_scale(1).dtype is torch.float32 and pool.k_scale(1).stride(2) == 1
    assert pool.cmp_k_cache(0).dtype is torch.bfloat16 and pool.pending_ring(0).dtype is torch.bfloat16


def test_int8_rebuild_resizes_the_scales(int8):
    pool = _pool(num_pages=4)
    pool.rebuild(16)
    assert pool.k_cache(3).shape == (16, 64, 2, 64) and pool.k_cache(3).dtype is torch.int8
    assert pool.v_scale(3).shape == (16, 64, 2)


def test_int8_unit_bytes_match_the_cost_model(int8):
    spec = _spec()
    config = _config(spec)
    pool = _pool()
    kv_bytes, _ = pool.unit_bytes()
    assert kv_bytes * 64 == QSAKVCache.kv_cost(config)[0]
    # 4 layers x 2 heads x (K, V): 64 int8 codes + one fp32 scale each, plus the unchanged index slab
    index = 32 * 4 * 2 // 4
    assert kv_bytes == 4 * 2 * 2 * (64 + 4) + index


def test_int8_halves_the_real_model_kv(int8):
    spec = _spec(num_kv_heads=2, head_dim=256, index_head_dim=128, num_index_layers=12, layer_ids=FULL_LAYER_IDS)
    config = _config(spec)
    per_token = QSAKVCache.kv_cost(config)[0] // 64
    bf16 = spec_kv_bytes_per_token(spec, config)
    assert bf16 == 24576 + 768
    assert per_token == 12 * 2 * 2 * (256 + 4) + 768  # 13248: 54 % of the bf16 price


def test_default_pool_is_unchanged(monkeypatch):
    monkeypatch.delenv("FREETOKEN_QSA_KV_INT8", raising=False)
    pool = _pool()
    assert not pool.kv_int8 and pool.k_cache(1).dtype is torch.bfloat16
    with pytest.raises(AssertionError):
        pool.k_scale(1)
