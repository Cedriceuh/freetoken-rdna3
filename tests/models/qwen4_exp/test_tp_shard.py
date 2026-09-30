"""qwen4_exp tensor parallelism: every sharded piece must reassemble to the TP=1 tensor exactly.

CPU only. Ported from lukascechovic/FreeToken ``rocm-gfx1201`` (d9fddf7 + 2433af3) onto the
QuantConfig-era modules: the routed NVFP4 experts are sliced in the Triton kernel's ``pack``, the
dense tensors in the loader before fusion, the PLE table on the hash-head axis.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.models.qwen4_exp.ple import GpuResidentTable, NGramEmbedding, PLEMetadata

from .common import VOCAB, as_rank, hash_constants, parsed_config


# ----------------------------------------------------------------------------------
# Routed NVFP4 experts: each rank packs its slice of the intermediate axis
# ----------------------------------------------------------------------------------

H, I, GROUP = 64, 128, 16


def _moe_cfg(rank: int, tp: int, intermediate: int = I):
    from freetoken.layers.quantization.moe.base import MoEConfig

    return MoEConfig(num_experts=2, hidden=H, intermediate=intermediate, top_k=1,
                     tp_rank=rank, tp_size=tp, strategy="offload")


def _full_pieces(n: int = 2, intermediate: int = I, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def codes(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8)

    def scales(*shape):
        return torch.rand(shape, generator=g).to(torch.float8_e4m3fn)

    return {
        "gate": codes(n, intermediate, H // 2), "gate_scale": scales(n, intermediate, H // GROUP),
        "gate_global": torch.rand(n, 1, generator=g),
        "up": codes(n, intermediate, H // 2), "up_scale": scales(n, intermediate, H // GROUP),
        "up_global": torch.rand(n, 1, generator=g),
        "down": codes(n, H, intermediate // 2), "down_scale": scales(n, H, intermediate // GROUP),
        "down_global": torch.rand(n, 1, generator=g),
    }


def _pack(pieces, rank: int, tp: int) -> dict[str, torch.Tensor]:
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    kernel, cfg = TritonNvfp4MoEKernel(), _moe_cfg(rank, tp)
    n = next(iter(pieces.values())).shape[0]
    out = {role: torch.empty((n, *spec.shape), dtype=spec.dtype) for role, spec in kernel.layout(cfg).items()}
    kernel.pack(pieces, cfg, out)
    return out


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype is torch.float8_e4m3fn else t


@pytest.mark.parametrize("tp", [2, 4])
def test_nvfp4_expert_shards_reassemble_every_bank(tp):
    pieces = _full_pieces()
    ref = _pack(pieces, 0, 1)
    shards = [_pack(pieces, r, tp) for r in range(tp)]
    local = I // tp

    def gate_half(t, half):  # gate_up rows are [gate | up] per rank
        return t[:, half * local:(half + 1) * local]

    for role in ("gate_up", "gate_up_scale", "gate_up_global"):
        for half in (0, 1):
            got = torch.cat([gate_half(_u8(s[role]), half) for s in shards], dim=1)
            want = _u8(ref[role])[:, half * I:(half + 1) * I]
            assert torch.equal(got, want), (role, half)
    for role in ("down", "down_scale"):
        got = torch.cat([_u8(s[role]) for s in shards], dim=2)
        assert torch.equal(got, _u8(ref[role])), role
    for s in shards:
        assert torch.equal(s["down_global"], ref["down_global"])  # not on the intermediate axis


def test_nvfp4_expert_shard_is_not_a_flat_chunk_of_the_fused_rows():
    """Guards the per-half cut: a flat row chunk of [gate | up] has the right shape and is wrong."""
    pieces = _full_pieces()
    ref = _pack(pieces, 0, 1)
    rank0 = _pack(pieces, 0, 2)
    assert not torch.equal(rank0["gate_up"], ref["gate_up"][:, :I])


def test_nvfp4_kernel_accepts_tp2_and_rejects_a_split_off_the_block():
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    kernel = TritonNvfp4MoEKernel()
    assert kernel.unusable_reason(_moe_cfg(0, 2)) is None
    assert "NVFP4 block" in kernel.unusable_reason(_moe_cfg(0, 2, intermediate=48))
    assert kernel.layout(_moe_cfg(1, 2))["down"].shape == (H, I // 4)


# ----------------------------------------------------------------------------------
# The PLE n-gram table: sharded on the hash-head axis, one all-gather rebuilds the embedding
# ----------------------------------------------------------------------------------


def _embedding(args, table_rows: torch.Tensor, row_lo: int = 0, row_hi: int | None = None):
    emb = NGramEmbedding(args)
    mult, sizes, offsets = hash_constants(args)
    emb.layer_multipliers = mult
    emb.ngram_heads_vocab_sizes = sizes
    emb.ngram_heads_offsets = offsets
    hi = table_rows.shape[0] if row_hi is None else row_hi
    emb.attach_table(GpuResidentTable(table_rows[row_lo:hi], dtype=torch.float32))
    return emb


def _decode_meta(batch: int, args) -> PLEMetadata:
    torch.manual_seed(11)
    return PLEMetadata(
        input_ids=torch.randint(0, VOCAB, (batch,), dtype=torch.int64),
        cu_seqlens=torch.arange(batch + 1, dtype=torch.int64),
        seq_lens=[1] * batch,
        ngram_context=torch.randint(0, VOCAB, (batch, args.ngram_size - 1), dtype=torch.int64),
        state_slots=torch.arange(batch, dtype=torch.int64),
        fresh_slots=None,
        is_decode=True,
    )


class _FakeComm:
    """Stands in for the process group: ``all_gather`` returns the group's dim-0 concatenation."""

    def __init__(self, gathered: torch.Tensor) -> None:
        self._gathered = gathered

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self._gathered

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the PLE shard must not all-reduce; it all-gathers")


def _ple_setup(seed: int):
    args = parsed_config().qwen4_args
    assert args.num_ngram_heads % 2 == 0
    _, sizes, offsets = hash_constants(args)
    total = int(offsets[-1] + sizes[-1])
    torch.manual_seed(seed)
    table = torch.randn(total, args.ngram_head_dim, dtype=torch.float32)
    return args, offsets, total, table


def _rank_locals(args, offsets, total, table, meta):
    per_rank = args.num_ngram_heads // 2
    locals_ = []
    for rank in range(2):
        with as_rank(rank, 2):
            lo = int(offsets[per_rank * rank])
            hi = total if rank == 1 else int(offsets[per_rank * (rank + 1)])
            emb = _embedding(args, table, lo, hi)
            ids = emb.row_ids(meta)
            assert ids.shape == (meta.input_ids.shape[0], per_rank)
            assert int(ids.min()) >= 0 and int(ids.max()) < hi - lo
            locals_.append(emb.table.lookup(ids))
    return per_rank, locals_


def test_ple_head_shard_reconstructs_the_tp1_embedding_exactly():
    args, offsets, total, table = _ple_setup(3)
    meta = _decode_meta(4, args)
    with as_rank(0, 1):
        reference = _embedding(args, table).forward(meta)
    assert reference.shape == (4, args.ple_embed_dim)

    per_rank, locals_ = _rank_locals(args, offsets, total, table, meta)
    with as_rank(1, 2):
        emb = _embedding(args, table, int(offsets[per_rank]), total)
        emb._comm = _FakeComm(torch.cat(locals_, dim=0))
        got = emb.forward(meta)
    assert torch.equal(got, reference)


def test_ple_head_shard_is_not_reproduced_by_a_naive_dim0_gather():
    args, offsets, total, table = _ple_setup(5)
    meta = _decode_meta(4, args)
    with as_rank(0, 1):
        reference = _embedding(args, table).forward(meta)
    _, locals_ = _rank_locals(args, offsets, total, table, meta)
    assert not torch.equal(torch.cat(locals_, dim=0).reshape(reference.shape), reference)


def test_ple_table_that_serves_all_heads_is_neither_rebased_nor_gathered():
    """The disk store stages every head on every rank: global rows in, the full embedding out."""
    args, _, _, table = _ple_setup(7)
    meta = _decode_meta(4, args)
    with as_rank(0, 1):
        reference = _embedding(args, table).forward(meta)
    full = GpuResidentTable(table, dtype=torch.float32)
    full.serves_all_heads = True
    with as_rank(1, 2):
        emb = _embedding(args, table)
        emb.attach_table(full)
        emb._comm = _FakeComm(None)  # all_gather would hand back None and fail the equality
        assert emb.row_ids(meta).shape == (4, args.num_ngram_heads)
        assert torch.equal(emb.forward(meta), reference)


# ----------------------------------------------------------------------------------
# The QSA attention layer: rank-local head counts + a row-parallel ``o_proj``
# ----------------------------------------------------------------------------------

DEPLOYED = dict(num_q=24, num_kv=2, head_dim=256, hidden=512)


def _attention(config, layer_id: int = 3):
    from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention

    return Qwen4ExpAttention(config, layer_id=layer_id, prefix=f"model.layers.{layer_id}.self_attn")


def test_attention_tp1_geometry_is_unchanged():
    config = parsed_config(**DEPLOYED)
    with as_rank(0, 1):
        attn = _attention(config)
    assert (attn.num_q, attn.num_kv) == (24, 2)
    assert attn._qkv_split == [12288, 512, 512]
    assert attn.qkv_proj.local_output_size == 13312
    assert attn.o_proj.local_input_size == 6144


@pytest.mark.parametrize("rank", [0, 1])
def test_attention_tp2_declares_rank_local_shapes(rank):
    config = parsed_config(**DEPLOYED)
    with as_rank(rank, 2):
        attn = _attention(config)
    assert (attn.num_q, attn.num_kv) == (12, 1)
    assert attn._qkv_split == [6144, 256, 256]
    assert attn.qkv_proj.local_output_size == sum(attn._qkv_split)
    assert attn.o_proj.local_input_size == attn.qo_attn_dim == 3072
    assert attn.qkv_proj.full_output_size == 13312
    assert attn.o_proj.full_input_size == 6144


@pytest.mark.parametrize("tp, expected_calls", [(1, 0), (2, 1)])
def test_attention_o_proj_all_reduces_only_when_sharded(tp, expected_calls, monkeypatch):
    from freetoken.distributed.impl import DistributedCommunicator

    calls = []
    monkeypatch.setattr(DistributedCommunicator, "all_reduce", lambda self, x: (calls.append(x.shape), x)[1])
    config = parsed_config(**DEPLOYED)
    with as_rank(0, tp):
        attn = _attention(config)
    attn.o_proj.weight = torch.zeros_like(attn.o_proj.weight)
    attn.o_proj.forward(torch.zeros(2, attn.qo_attn_dim))
    assert len(calls) == expected_calls


# ----------------------------------------------------------------------------------
# The GDN layer: rank-local head counts + a row-parallel ``out_proj``
# ----------------------------------------------------------------------------------

GDN = dict(
    hidden_size=256, num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128,
    conv_kernel_size=4, rms_norm_eps=1e-6, layer_id=0, output_gate="sigmoid",
)


def _gdn(**overrides):
    from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet

    return Qwen4ExpGatedDeltaNet(**{**GDN, **overrides})


def test_gdn_tp1_geometry_is_unchanged():
    with as_rank(0, 1):
        gdn = _gdn()
    assert (gdn.num_k_heads, gdn.num_v_heads) == (16, 48)
    assert (gdn.key_dim, gdn.value_dim, gdn.conv_dim) == (2048, 6144, 10240)
    assert gdn._in_proj_split == [10240, 6144, 48, 48]
    assert gdn.out_proj.local_input_size == 6144


@pytest.mark.parametrize("rank", [0, 1])
def test_gdn_tp2_declares_rank_local_shapes(rank):
    with as_rank(rank, 2):
        gdn = _gdn()
    assert (gdn.num_k_heads, gdn.num_v_heads) == (8, 24)
    assert (gdn.key_dim, gdn.value_dim, gdn.conv_dim) == (1024, 3072, 5120)
    assert gdn._in_proj_split == [5120, 3072, 24, 24]
    assert sum(gdn._in_proj_split) == gdn.in_proj.local_output_size
    assert gdn.conv1d.weight.shape == (5120, 1, 4)
    assert gdn.dt_bias.shape == gdn.A_log.shape == (24,)
    assert gdn.out_proj.local_input_size == 3072
    assert gdn.out_proj.full_input_size == 6144


@pytest.mark.parametrize("tp", [1, 2])
def test_gdn_conv_dim_matches_the_state_pool_allocation(tp):
    from freetoken.kvcache.linear_state_pool import _linear_local_dims
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    group = LinearGatedDeltaGroupConfig(
        name="gdn", layer_ids=(0,), num_key_heads=GDN["num_k_heads"],
        num_value_heads=GDN["num_v_heads"], key_head_dim=GDN["head_k_dim"],
        value_head_dim=GDN["head_v_dim"], conv_kernel_dim=GDN["conv_kernel_size"],
        output_gate=GDN["output_gate"],
    )
    _, pool_conv_dim, pool_v_heads = _linear_local_dims(group, tp)
    with as_rank(0, tp):
        gdn = _gdn()
    assert gdn.conv_dim == pool_conv_dim
    assert gdn.num_v_heads == pool_v_heads


@pytest.mark.parametrize("tp, expected_calls", [(1, 0), (2, 1)])
def test_gdn_out_proj_all_reduces_only_when_sharded(tp, expected_calls, monkeypatch):
    from freetoken.distributed.impl import DistributedCommunicator

    calls = []
    monkeypatch.setattr(DistributedCommunicator, "all_reduce", lambda self, x: (calls.append(x.shape), x)[1])
    with as_rank(0, tp):
        gdn = _gdn()
    gdn.out_proj.weight = torch.zeros_like(gdn.out_proj.weight)
    gdn.out_proj.forward(torch.zeros(2, gdn.value_dim))
    assert len(calls) == expected_calls


def test_gdn_quantized_in_proj_is_refused_at_tp2():
    quant = SimpleNamespace(scheme_for=lambda prefix: object() if prefix.endswith("in_proj_qkvz") else None)
    with as_rank(0, 2), pytest.raises(NotImplementedError):
        _gdn(quant_config=quant, prefix="model.layers.0.linear_attn")


# ----------------------------------------------------------------------------------
# The dense loader: shard before fusion, composite GDN axes, padded vocab
# ----------------------------------------------------------------------------------


def _group():
    return SimpleNamespace(num_key_heads=4, num_value_heads=8, key_head_dim=3, value_head_dim=3)


def test_gdn_conv_composite_shards_reassemble_by_sub_block():
    from freetoken.models.qwen4_exp.weight import _shard_gdn

    g = _group()
    key_dim, value_dim = 4 * 3, 8 * 3
    full = torch.arange((2 * key_dim + value_dim) * 5, dtype=torch.float32).view(-1, 5)
    shards = [_shard_gdn("in_proj_qkv.weight", full, g, rank=r, world_size=2) for r in range(2)]
    q, k, v = torch.split(full, [key_dim, key_dim, value_dim])
    for r, s in enumerate(shards):
        sq, sk, sv = torch.split(s, [key_dim // 2, key_dim // 2, value_dim // 2])
        assert torch.equal(sq, q.chunk(2)[r]) and torch.equal(sk, k.chunk(2)[r]) and torch.equal(sv, v.chunk(2)[r])
    assert not torch.equal(shards[0], full.chunk(2)[0])  # a flat chunk is a different tensor


def test_gdn_value_axes_and_unclassified_leaf():
    from freetoken.models.qwen4_exp.weight import _shard_gdn

    g = _group()
    out_proj = torch.randn(7, 8 * 3)
    cols = [_shard_gdn("out_proj.weight", out_proj, g, rank=r, world_size=2) for r in range(2)]
    assert torch.equal(torch.cat(cols, dim=1), out_proj)
    a_log = torch.randn(8)
    assert torch.equal(torch.cat([_shard_gdn("A_log", a_log, g, rank=r, world_size=2) for r in range(2)]), a_log)
    norm = torch.randn(3)
    assert _shard_gdn("norm.weight", norm, g, rank=1, world_size=2) is norm
    with pytest.raises(NotImplementedError):
        _shard_gdn("in_proj_new.weight", norm, g, rank=0, world_size=2)


def test_shard_for_rank_splits_attention_pads_vocab_and_refuses_fp8():
    from freetoken.models.qwen4_exp.weight import _shard_for_rank

    config = SimpleNamespace(num_kv_heads=2, linear_attention_group=_group)
    q = torch.randn(8, 4)
    o = torch.randn(4, 8)
    vocab = torch.randn(5, 4)  # does not divide over 2 ranks
    with as_rank(1, 2):
        assert torch.equal(_shard_for_rank("model.layers.0.self_attn.q_proj.weight", q, config=config), q[4:])
        assert torch.equal(_shard_for_rank("model.layers.0.self_attn.o_proj.weight", o, config=config), o[:, 4:])
        local = _shard_for_rank("lm_head.weight", vocab, config=config)
        assert local.shape == (3, 4) and torch.equal(local[:2], vocab[3:]) and not local[2].any()
        norm = torch.randn(4)
        assert _shard_for_rank("model.layers.0.self_attn.q_norm.weight", norm, config=config) is norm
        with pytest.raises(NotImplementedError):
            _shard_for_rank("model.layers.0.self_attn.q_proj.weight", q.to(torch.float8_e4m3fn), config=config)
    with as_rank(0, 1):
        assert _shard_for_rank("lm_head.weight", vocab, config=config) is vocab


# ----------------------------------------------------------------------------------
# The vision tower: replicated, whole on every rank, never all-reduced
# ----------------------------------------------------------------------------------


def _tower():
    from freetoken.models.qwen3_vl.config import VisionConfig
    from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel

    vc = VisionConfig(
        hidden_size=64, depth=2, num_heads=4, intermediate_size=96, patch_size=2, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=32, in_channels=3,
    )
    return Qwen3VLVisionModel(vc)


@pytest.mark.parametrize("rank", [0, 1])
def test_vision_tower_declares_the_tp1_shapes_at_tp2(rank):
    with as_rank(0, 1):
        whole = {k: tuple(v.shape) for k, v in _tower().state_dict().items()}
    with as_rank(rank, 2):
        tower = _tower()
    assert {k: tuple(v.shape) for k, v in tower.state_dict().items()} == whole
    assert tower.blocks.op_list[0].attn.num_heads == 4


def test_vision_tower_never_all_reduces_at_tp2(monkeypatch):
    from freetoken.distributed.impl import DistributedCommunicator

    calls = []
    monkeypatch.setattr(DistributedCommunicator, "all_reduce", lambda self, x: (calls.append(x.shape), x)[1])
    with as_rank(1, 2):
        tower = _tower()
    for op in tower.state_dict().values():
        op.zero_()
    block = tower.blocks.op_list[0]
    block.mlp.forward(torch.zeros(3, 64))
    block.attn.proj.forward(torch.zeros(3, 64))
    tower.merger.forward(torch.zeros(4, 64))
    assert calls == []


def test_shard_for_rank_keeps_vision_tensors_whole():
    from freetoken.models.qwen4_exp.weight import _shard_for_rank

    config = SimpleNamespace(num_kv_heads=2, linear_attention_group=_group)
    qkv, bias = torch.randn(12, 4), torch.randn(4)
    with as_rank(1, 2):
        assert _shard_for_rank("visual.blocks.0.attn.qkv.weight", qkv, config=config) is qkv
        assert _shard_for_rank("visual.merger.linear_fc2.bias", bias, config=config) is bias


@pytest.mark.parametrize("rank, holds_tower", [(0, True), (1, False)])
def test_only_rank0_builds_the_vision_tower(rank, holds_tower):
    """Rank 0 alone encodes at TP>1 (Engine._encode_item broadcasts): the other ranks hold no tower."""
    from freetoken.layers import rotary
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.model import Qwen4ExpForConditionalGeneration

    from .common import hf_config

    hf = hf_config(**DEPLOYED)
    hf.vision_config = SimpleNamespace(
        hidden_size=64, depth=1, num_heads=4, intermediate_size=96, patch_size=2, temporal_patch_size=2,
        spatial_merge_size=2, num_position_embeddings=16, out_hidden_size=512, in_channels=3,
        deepstack_visual_indexes=[],
    )
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        with as_rank(rank, 2):
            config = parse_config(hf)
            with torch.device("meta"):
                model = Qwen4ExpForConditionalGeneration(config)
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()
    assert config.is_multimodal  # every rank still parses the vision section (and its 3-axis rope)
    assert any(k.startswith("visual.") for k in model.state_dict()) is holds_tower
    if not holds_tower:
        model.place_encoder_weights("host")  # nothing to place (the tower's pinned banks need a GPU)
