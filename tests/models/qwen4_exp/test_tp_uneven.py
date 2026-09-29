"""Uneven tensor parallelism (FREETOKEN_TP_SPLIT): the ranks' slices must still tile every axis exactly.

CPU only. The splittable axes are the GDN heads, the routed experts' intermediate and the shared
expert's intermediate; everything is checked the way test_tp_shard.py checks the even split, by
reassembling the TP=1 tensor from the ranks' pieces.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from .common import as_rank
from .test_tp_shard import GDN, H, GROUP, _full_pieces, _u8


@pytest.fixture
def split55(monkeypatch):
    monkeypatch.setenv("FREETOKEN_TP_SPLIT", "0.55")


# ----------------------------------------------------------------------------------
# The partition rule
# ----------------------------------------------------------------------------------


def test_partition_is_even_without_the_env(monkeypatch):
    from freetoken.distributed.split import gdn_head_partition, tp_partition

    monkeypatch.delenv("FREETOKEN_TP_SPLIT", raising=False)
    assert [tp_partition(640, 32, rank=r, world_size=2) for r in range(2)] == [(0, 320), (320, 320)]
    assert gdn_head_partition(16, 48, rank=1, world_size=2) == (8, 8, 24, 24)
    with pytest.raises(AssertionError):
        tp_partition(7, 1, rank=0, world_size=2)  # the even split still needs exact division


@pytest.mark.parametrize("raw", ["0.55", "11:9"])
def test_partition_follows_the_shares(monkeypatch, raw):
    from freetoken.distributed.split import gdn_head_partition, intermediate_partition

    monkeypatch.setenv("FREETOKEN_TP_SPLIT", raw)
    assert [intermediate_partition(640, rank=r, world_size=2) for r in range(2)] == [(0, 352), (352, 288)]
    assert [gdn_head_partition(16, 48, rank=r, world_size=2) for r in range(2)] == [(0, 9, 0, 27), (9, 7, 27, 21)]


def test_intermediate_unit_allows_the_balanced_split(monkeypatch):
    from freetoken.distributed.split import gdn_head_partition, intermediate_partition

    monkeypatch.setenv("FREETOKEN_TP_SPLIT", "0.575")
    assert [intermediate_partition(640, rank=r, world_size=2) for r in range(2)] == [(0, 368), (368, 272)]
    assert gdn_head_partition(16, 48, rank=0, world_size=2) == (0, 9, 0, 27)


@pytest.mark.parametrize("total, unit, tp, raw", [(640, 32, 2, "0.6"), (1000, 8, 3, "5:3:2"), (48, 1, 4, "4:3:2:1")])
def test_partition_tiles_the_axis(monkeypatch, total, unit, tp, raw):
    from freetoken.distributed.split import tp_partition

    monkeypatch.setenv("FREETOKEN_TP_SPLIT", raw)
    parts = [tp_partition(total, unit, rank=r, world_size=tp) for r in range(tp)]
    offset = 0
    for lo, size in parts:
        assert lo == offset and size > 0 and size % unit == 0
        offset += size
    assert offset == total


def test_partition_rejects_bad_shares(monkeypatch):
    from freetoken.distributed.split import tp_partition

    for raw in ("1.2", "0", "1:2:3"):
        monkeypatch.setenv("FREETOKEN_TP_SPLIT", raw)
        with pytest.raises(ValueError):
            tp_partition(640, 32, rank=0, world_size=2)
    monkeypatch.setenv("FREETOKEN_TP_SPLIT", "0.99")
    with pytest.raises(ValueError):  # rank 1 would get no 32-row unit of 64 rows
        tp_partition(64, 32, rank=1, world_size=2)


def test_gdn_replicated_heads_ignore_the_split(monkeypatch):
    from freetoken.distributed.split import gdn_head_partition

    monkeypatch.setenv("FREETOKEN_TP_SPLIT", "4:3:2:1")

    # fewer k heads than ranks: every rank shares one head, as div_even(..., allow_replicate=True)
    assert gdn_head_partition(2, 4, rank=3, world_size=4) == (1, 1, 3, 1)


# ----------------------------------------------------------------------------------
# GDN module, state pool and loader
# ----------------------------------------------------------------------------------


def _gdn():
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet

    return Qwen4ExpGatedDeltaNet(**GDN)


@pytest.mark.parametrize("rank, k, v", [(0, 9, 27), (1, 7, 21)])
def test_gdn_declares_the_uneven_heads(split55, rank, k, v):
    with as_rank(rank, 2):
        gdn = _gdn()
    hd = GDN["head_k_dim"]
    assert (gdn.num_k_heads, gdn.num_v_heads) == (k, v)
    assert gdn._in_proj_split == [(2 * k + v) * hd, v * hd, v, v]
    assert gdn.in_proj.local_output_size == sum(gdn._in_proj_split)
    assert gdn.in_proj.full_output_size == (2 * 16 + 48) * hd + 48 * hd + 2 * 48
    assert gdn.conv1d.weight.shape == ((2 * k + v) * hd, 1, 4)
    assert gdn.dt_bias.shape == gdn.A_log.shape == (v,)
    assert gdn.out_proj.local_input_size == v * hd and gdn.out_proj.full_input_size == 48 * hd


@pytest.mark.parametrize("rank", [0, 1])
def test_gdn_uneven_heads_match_the_state_pool(split55, rank):
    from freetoken.kvcache.linear_state_pool import _linear_local_dims
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="gdn", layer_ids=(0,), num_key_heads=GDN["num_k_heads"],
        num_value_heads=GDN["num_v_heads"], key_head_dim=GDN["head_k_dim"],
        value_head_dim=GDN["head_v_dim"], conv_kernel_dim=GDN["conv_kernel_size"],
        output_gate=GDN["output_gate"],
    )
    with as_rank(rank, 2):
        gdn = _gdn()
        _, pool_conv_dim, pool_v_heads = _linear_local_dims(group, 2)
    assert (gdn.conv_dim, gdn.num_v_heads) == (pool_conv_dim, pool_v_heads)


def _group():
    return SimpleNamespace(num_key_heads=16, num_value_heads=48, key_head_dim=2, value_head_dim=2)


def test_gdn_loader_shards_reassemble_uneven(split55):
    from freetoken.models.qwen4_exp.weight import _shard_gdn

    g = _group()
    key_dim, value_dim = 16 * 2, 48 * 2
    conv = torch.arange((2 * key_dim + value_dim) * 3, dtype=torch.float32).view(-1, 3)
    q, k, v = torch.split(conv, [key_dim, key_dim, value_dim])
    shards = [_shard_gdn("in_proj_qkv.weight", conv, g, rank=r, world_size=2) for r in range(2)]
    heads = [(9, 27), (7, 21)]
    parts = [torch.split(s, [kh * 2, kh * 2, vh * 2]) for s, (kh, vh) in zip(shards, heads)]
    for i, full in enumerate((q, k, v)):
        assert torch.equal(torch.cat([p[i] for p in parts]), full)
    out_proj = torch.randn(5, value_dim)
    cols = [_shard_gdn("out_proj.weight", out_proj, g, rank=r, world_size=2) for r in range(2)]
    assert [c.shape[1] for c in cols] == [54, 42] and torch.equal(torch.cat(cols, dim=1), out_proj)
    z = torch.randn(value_dim, 4)
    rows = [_shard_gdn("in_proj_z.weight", z, g, rank=r, world_size=2) for r in range(2)]
    assert torch.equal(torch.cat(rows), z)
    b = torch.randn(48, 4)
    assert [_shard_gdn("in_proj_b.weight", b, g, rank=r, world_size=2).shape[0] for r in range(2)] == [27, 21]


# ----------------------------------------------------------------------------------
# Shared expert: module widths and loader slices
# ----------------------------------------------------------------------------------


def test_shared_expert_loader_and_module_agree(split55):
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert
    from freetoken.models.qwen4_exp.weight import _shard_for_rank

    inter, hidden = 640, 8
    config = SimpleNamespace(num_kv_heads=2, linear_attention_group=_group, shared_expert_intermediate_size=inter,
                             quant=None)
    gate = torch.randn(inter, hidden)
    down = torch.randn(hidden, inter)
    rows, cols = [], []
    for rank, local in ((0, 352), (1, 288)):
        with as_rank(rank, 2):
            rows.append(_shard_for_rank("model.layers.0.mlp.shared_expert.gate_proj.weight", gate, config=config))
            cols.append(_shard_for_rank("model.layers.0.mlp.shared_expert.down_proj.weight", down, config=config))
            with torch.device("meta"):
                mod = _SharedExpert(config, hidden, inter, local_intermediate=local)
        assert rows[-1].shape == (local, hidden) and cols[-1].shape == (hidden, local)
        assert mod.gate_up_proj.output_sizes == (local, local)
        assert mod.gate_up_proj.local_output_size == 2 * local
        assert mod.down_proj.local_input_size == local
    assert torch.equal(torch.cat(rows), gate) and torch.equal(torch.cat(cols, dim=1), down)


def test_shared_expert_default_is_still_even(monkeypatch):
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert

    monkeypatch.delenv("FREETOKEN_TP_SPLIT", raising=False)
    config = SimpleNamespace(quant=None)
    with as_rank(1, 2), torch.device("meta"):
        mod = _SharedExpert(config, 8, 640)
    assert mod.gate_up_proj.output_sizes == (320, 320) and mod.down_proj.local_input_size == 320


# ----------------------------------------------------------------------------------
# Routed NVFP4 experts
# ----------------------------------------------------------------------------------

I_UNEVEN = 640


def _moe_cfg(rank: int, tp: int):
    from freetoken.layers.quantization.moe.base import MoEConfig

    return MoEConfig(num_experts=2, hidden=H, intermediate=I_UNEVEN, top_k=1, tp_rank=rank, tp_size=tp,
                     strategy="offload")


def _pack(pieces, rank: int, tp: int) -> dict[str, torch.Tensor]:
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    kernel, cfg = TritonNvfp4MoEKernel(), _moe_cfg(rank, tp)
    assert kernel.unusable_reason(cfg) is None or "CPU" in kernel.unusable_reason(cfg) or tp == 1
    n = next(iter(pieces.values())).shape[0]
    out = {role: torch.empty((n, *spec.shape), dtype=spec.dtype) for role, spec in kernel.layout(cfg).items()}
    kernel.pack(pieces, cfg, out)
    return out


def test_moe_config_range_is_uneven(split55):
    assert [_moe_cfg(r, 2).local_intermediate_range for r in range(2)] == [(0, 352), (352, 288)]
    assert _moe_cfg(0, 1).local_intermediate_range == (0, I_UNEVEN)


def test_nvfp4_expert_shards_reassemble_uneven(split55):
    pieces = _full_pieces(intermediate=I_UNEVEN)
    ref = _pack(pieces, 0, 1)
    shards = [_pack(pieces, r, 2) for r in range(2)]
    locals_ = [352, 288]
    for role in ("gate_up", "gate_up_scale", "gate_up_global"):
        for half in (0, 1):
            got = torch.cat([_u8(s[role])[:, half * n:(half + 1) * n] for s, n in zip(shards, locals_)], dim=1)
            want = _u8(ref[role])[:, half * I_UNEVEN:(half + 1) * I_UNEVEN]
            assert torch.equal(got, want), (role, half)
    for role in ("down", "down_scale"):
        got = torch.cat([_u8(s[role]) for s in shards], dim=2)
        assert torch.equal(got, _u8(ref[role])), role
    assert shards[0]["down_scale"].shape[-1] == 352 // GROUP and shards[1]["down"].shape[-1] == 288 // 2


def test_nvfp4_kernel_accepts_the_uneven_split(split55):
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    kernel = TritonNvfp4MoEKernel()
    for rank in range(2):
        reason = kernel.unusable_reason(_moe_cfg(rank, 2))
        assert reason is None, reason


# ----------------------------------------------------------------------------------
# Engine plan: every rank plans on its own memory, then all take the smallest plan
# ----------------------------------------------------------------------------------


def test_engine_uneven_plan_takes_the_min_over_ranks(split55, monkeypatch):
    import freetoken.engine.cache_budget as cb
    from freetoken.engine.engine import Engine

    seen = {}

    def fake_resolve(**kw):
        seen.update(kw)
        return 9000, 700, True

    def fake_all_reduce(t, op=None, group=None):
        t.copy_(torch.minimum(t, torch.tensor([8800, 720, 1])))

    monkeypatch.setattr(cb, "resolve_moe_cache_auto", fake_resolve)
    monkeypatch.setattr(cb, "expert_bytes_per_slot", lambda sources: 1 << 20)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    gib = 1 << 30
    fake = SimpleNamespace(
        _pool_cls=SimpleNamespace(kv_cost=lambda config: (1, 0, 64, 0)),
        _baseline_free=20 * gib, _weights_bytes=5 * gib,
        _local_baseline_free=24 * gib, _local_weights_bytes=6 * gib, tp_cpu_group=None,
    )
    config = SimpleNamespace(
        tp_info=SimpleNamespace(size=2, rank=0), memory_ratio=0.8, model_config=SimpleNamespace(
            num_experts=512, num_moe_layers=48, slot_states=()), moe_prefill_overlap=True,
        kv_reserve_tokens=0,
    )
    monkeypatch.setattr("freetoken.engine.engine.state_pool_bytes", lambda config: 0)
    size, pages, overlap = Engine._resolve_auto_moe_cache_size(fake, config, SimpleNamespace(sources=None))
    assert (size, pages, overlap) == (8800, 700, True)
    # the bigger card's 4 GiB surplus goes to the cache whole: ratio x baseline gains exactly the surplus
    assert seen["baseline_free"] == 20 * gib + int(4 * gib / 0.8)
    assert seen["weights_bytes"] == 6 * gib
