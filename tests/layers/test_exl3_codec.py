"""EXL3 reference codec: ring unpacking, codebooks, tile layout and Hadamard reconstruction (CPU)."""
import pytest
import torch

from freetoken.layers.quantization import exl3_codec as ex


def _stream_window(symbols: torch.Tensor, K: int, j: int) -> int:
    """The 16 stream bits ending at bit (j+1)*K of one tile's ring, straight from the symbol definition."""
    bits = []
    for v in symbols.tolist():
        bits += [(v >> (K - 1 - b)) & 1 for b in range(K)]
    n = len(bits)
    end = (j + 1) * K
    out = 0
    for p in range(end - 16, end):
        out = (out << 1) | bits[p % n]
    return out


@pytest.mark.parametrize("K", range(1, 9))
def test_unpack_reads_the_ring_window_of_each_weight(K):
    g = torch.Generator().manual_seed(K)
    symbols = torch.randint(0, 1 << K, (2, 3, 256), generator=g)
    trellis = ex.pack_trellis(symbols, K)
    assert trellis.shape == (2, 3, 16 * K) and trellis.dtype == torch.int16
    states = ex.unpack_states(trellis)
    for a, b in ((0, 0), (1, 2)):
        for j in (0, 1, 2, 7, 100, 254, 255):
            assert states[a, b, j].item() == _stream_window(symbols[a, b], K, j)


def test_state_is_a_shift_register_of_the_symbols():
    K = 3
    symbols = torch.randint(0, 8, (256,), generator=torch.Generator().manual_seed(1))
    states = ex.unpack_states(ex.pack_trellis(symbols, K))
    for j in range(1, 256):
        assert states[j].item() == ((states[j - 1].item() << K) | symbols[j].item()) & 0xFFFF


def test_tensor_core_perm_is_the_mma_fragment_order():
    perm = ex.tensor_core_perm()
    assert sorted(perm.tolist()) == list(range(256))
    # thread 0 holds rows 0,1,8,9 of column 0 then of column 8; thread 1 starts at row 2
    assert perm[:8].tolist() == [0, 16, 128, 144, 8, 24, 136, 152]
    assert perm[8:12].tolist() == [32, 48, 160, 176]


def test_mul1_is_affine_in_the_byte_sum():
    states = torch.arange(65536)
    vals = ex.decode_states(states, ex.CB_MUL1)
    x = (states * 0x83DCD12D) & 0xFFFFFFFF
    bytesum = sum((x >> s) & 0xFF for s in (0, 8, 16, 24))
    k_inv = float(torch.tensor(0x1EEE, dtype=torch.int16).view(torch.float16))
    exact = (bytesum.double() - 510) * k_inv
    assert (vals.double() - exact).abs().max().item() < 0.01
    assert abs(vals.double().mean().item()) < 0.02 and abs(vals.double().std().item() - 1.0) < 0.02


@pytest.mark.parametrize("cb", [ex.CB_3INST, ex.CB_MCG])
def test_lop3_codebooks_are_finite_and_centered(cb):
    vals = ex.decode_states(torch.arange(65536), cb).double()
    assert torch.isfinite(vals).all()
    assert abs(vals.mean().item()) < 0.05 * vals.std().item()


def test_codebook_marker_tensors():
    assert ex.codebook_of({"x.trellis": 0, "x.mul1": 0}) == ex.CB_MUL1
    assert ex.codebook_of({"mcg": 0}) == ex.CB_MCG
    assert ex.codebook_of({"x.trellis": 0}) == ex.CB_3INST
    with pytest.raises(ValueError):
        ex.codebook_of({"mcg": 0, "mul1": 0})


def test_half_integer_bitrate_is_rejected():
    with pytest.raises(NotImplementedError):
        ex.trellis_bits(torch.zeros(1, 1, 40, dtype=torch.int16))


def test_decode_places_tiles_row_major_in_k_then_n():
    K = 2
    symbols = torch.randint(0, 4, (2, 3, 256), generator=torch.Generator().manual_seed(2))
    trellis = ex.pack_trellis(symbols, K)
    w = ex.decode_tiles(trellis, ex.CB_MUL1)
    assert w.shape == (32, 48)
    vals = ex.decode_states(ex.unpack_states(trellis), ex.CB_MUL1)
    perm = ex.tensor_core_perm()
    for a, b, j in ((0, 0, 0), (1, 2, 5), (0, 1, 255), (1, 0, 77)):
        p = perm[j].item()
        assert w[16 * a + p // 16, 16 * b + p % 16] == vals[a, b, j]


def test_reconstruct_matches_the_inference_order():
    """W = diag(suh) H Wq H diag(svh): x @ W equals rotate-in, matmul, rotate-out."""
    K, k, n = 3, 256, 384
    g = torch.Generator().manual_seed(3)
    trellis = ex.pack_trellis(torch.randint(0, 8, (k // 16, n // 16, 256), generator=g), K)
    suh = (torch.randn(k, generator=g).sign() * (1 + torch.rand(k, generator=g))).half()
    svh = (torch.rand(n, generator=g) + 0.5).half()
    w = ex.reconstruct(trellis, suh, svh, ex.CB_MUL1, dtype=torch.float64)

    wq = ex.decode_tiles(trellis, ex.CB_MUL1).double()
    had = ex.hadamard_128(dtype=torch.float64)
    x = torch.randn(5, k, generator=g, dtype=torch.float64)
    xh = ((x * suh.double()).view(5, k // 128, 128) @ had).view(5, k)
    y = (((xh @ wq).view(5, n // 128, 128) @ had).view(5, n)) * svh.double()
    torch.testing.assert_close(x @ w, y, rtol=1e-4, atol=1e-4)

    blk_k = torch.block_diag(*[had] * (k // 128))
    blk_n = torch.block_diag(*[had] * (n // 128))
    ref = torch.diag(suh.double()) @ blk_k @ wq @ blk_n @ torch.diag(svh.double())
    torch.testing.assert_close(w, ref, rtol=1e-4, atol=1e-4)


def test_hadamard_is_orthonormal():
    h = ex.hadamard_128(dtype=torch.float64)
    torch.testing.assert_close(h @ h, torch.eye(128, dtype=torch.float64))


def test_packed_signs_unpack_to_plus_minus_one():
    words = torch.tensor([0b101, -32768], dtype=torch.int16)  # bits 0 and 2 of word 0, bit 15 of word 1
    s = ex.unpack_signs(words)
    assert s.shape == (32,)
    neg = (s < 0).nonzero().flatten().tolist()
    assert neg == [0, 2, 31]
    t = {"x.su": words, "x.svh": torch.ones(4).half()}
    assert torch.equal(ex.channel_scales(t, "u"), s)
    assert torch.equal(ex.channel_scales(t, "v"), torch.ones(4).half())


def _pack_ngram_rows(states: torch.Tensor, scales: torch.Tensor, K: int) -> torch.Tensor:
    """exllamav3 ngram_codec.pack_rows: the low K bits of each state, LSB-first, after the scale word."""
    N, dim = states.shape
    bits = ((states.to(torch.int64) & ((1 << K) - 1)).unsqueeze(-1) >> torch.arange(K)) & 1
    words = (bits.reshape(N, dim * K // 16, 16) << torch.arange(16)).sum(-1)
    words = words.to(torch.int32).to(torch.int16)
    return torch.cat([scales.half().view(torch.int16).unsqueeze(1), words], dim=1)


@pytest.mark.parametrize("K", [4, 5, 6])
def test_ngram_rings_round_trip_and_stack_preceding_symbols(K):
    g = torch.Generator().manual_seed(K)
    dim = 160
    symbols = torch.randint(0, 1 << K, (3, dim), generator=g)
    states = torch.zeros_like(symbols)
    for j in range((15 + K) // K):
        states |= torch.roll(symbols, j, dims=1) << (j * K)
    states &= 0xFFFF
    scales = torch.rand(3, generator=g).half()
    got_states, got_scales = ex.ngram_ring_states(_pack_ngram_rows(states, scales, K), K)
    assert torch.equal(got_states, states)
    assert torch.equal(got_scales, scales)
    bias = torch.randn(3, dim, generator=g).half()
    rows = ex.decode_ngram_rows(_pack_ngram_rows(states, scales, K), K, bias)
    torch.testing.assert_close(rows, ex.decode_states(states, ex.CB_MUL1).float() * scales.float()[:, None] + bias.float())


def test_dense_reconstructor_emits_bf16_weight_once_complete():
    from freetoken.models.exl3_weights import Exl3DenseReconstructor

    K, k, n = 2, 128, 256
    g = torch.Generator().manual_seed(4)
    trellis = ex.pack_trellis(torch.randint(0, 1 << K, (k // 16, n // 16, 256), generator=g), K)
    suh, svh = torch.randn(k, generator=g).half(), torch.randn(n, generator=g).half()
    rec = Exl3DenseReconstructor({"m.proj.trellis": "a", "m.proj.mul1": "b", "m.other.trellis": "a"})
    assert rec.feed("m.proj.norm", torch.zeros(1)) is None  # not an EXL3 leaf: passes through
    assert rec.feed("m.proj.mul1", torch.zeros(())) == []
    assert rec.feed("m.proj.trellis", trellis) == []
    assert rec.feed("m.proj.suh", suh) == []
    [(name, w)] = rec.feed("m.proj.svh", svh)
    assert name == "m.proj.weight" and w.dtype == torch.bfloat16 and w.shape == (n, k)
    ref = ex.reconstruct(trellis, suh, svh, ex.CB_MUL1).T
    torch.testing.assert_close(w.float(), ref, rtol=1e-2, atol=1e-2)
    rec.check_done()
    rec.feed("m.other.trellis", trellis)
    with pytest.raises(ValueError):
        rec.check_done()


def test_dialect_reads_expert_bitrate_from_the_checkpoint(tmp_path, monkeypatch):
    import json

    from freetoken.distributed import split

    monkeypatch.setattr(split, "_intermediate_unit", split.INTERMEDIATE_UNIT)  # the dialect widens it to 128

    from safetensors.torch import save_file

    from freetoken.layers.quantization import QuantConfig, QuantKind
    from freetoken.layers.quantization.scheme import exl3_params

    pre = "model.language_model.layers.0.mlp.experts.0"
    tensors = {}
    for proj, (k, n) in (("gate_proj", (256, 128)), ("up_proj", (256, 128)), ("down_proj", (128, 256))):
        tensors[f"{pre}.{proj}.trellis"] = torch.zeros(k // 16, n // 16, 48, dtype=torch.int16)
        tensors[f"{pre}.{proj}.suh"] = torch.ones(k).half()
        tensors[f"{pre}.{proj}.svh"] = torch.ones(n).half()
        tensors[f"{pre}.{proj}.mul1"] = torch.zeros((), dtype=torch.int32)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    hf = {"quantization_config": {"quant_method": "exl3", "bits": 3.05, "codebook": "mul1"}}
    quant = QuantConfig.from_hf(hf, model_path=str(tmp_path))
    assert quant.dialect == "exl3"
    assert quant.scheme_for_name("model.language_model.layers.0.mlp.self_attn.q_proj") is None
    scheme = quant.scheme_for_name("model.language_model.layers.0.mlp.experts")
    assert scheme.kind is QuantKind.EXL3 and exl3_params(scheme) == (3, ex.CB_MUL1)
    assert quant.scheme_for_name("model.language_model.layers.0.mlp.experts.0.down_proj") == scheme
    assert split._intermediate_unit == 128


def test_split_unit_128_keeps_hadamard_blocks_whole(monkeypatch):
    """EXL3 widens the intermediate split unit to 128: 0.55 of 640 lands on 384 / 256, an even split cannot."""
    from freetoken.distributed import split

    monkeypatch.setenv(split.ENV, "0.55")
    split._parse_shares.cache_clear()
    monkeypatch.setattr(split, "_intermediate_unit", split.INTERMEDIATE_UNIT)
    assert [split.intermediate_partition(640, rank=r, world_size=2) for r in (0, 1)] == [(0, 352), (352, 288)]
    split.set_intermediate_unit(128)
    assert [split.intermediate_partition(640, rank=r, world_size=2) for r in (0, 1)] == [(0, 384), (384, 256)]
    split._parse_shares.cache_clear()


def test_exl3_vision_mlp_padding_is_trimmed():
    """exllamav3 zero-pads the vision MLP intermediate to a multiple of 128; the reader cuts it back to the config's."""
    from types import SimpleNamespace

    from freetoken.models.qwen4_exp.weight import _exl3_unpad

    cfg = SimpleNamespace(vision_config=SimpleNamespace(intermediate_size=4304))
    assert _exl3_unpad("visual.blocks.3.mlp.linear_fc1.weight", torch.zeros(4352, 1152), cfg).shape == (4304, 1152)
    assert _exl3_unpad("visual.blocks.3.mlp.linear_fc1.bias", torch.zeros(4352), cfg).shape == (4304,)
    assert _exl3_unpad("visual.blocks.3.mlp.linear_fc2.weight", torch.zeros(1152, 4352), cfg).shape == (1152, 4304)
    assert _exl3_unpad("visual.blocks.3.mlp.linear_fc2.bias", torch.zeros(1152), cfg).shape == (1152,)
    assert _exl3_unpad("layers.0.mlp.shared_expert.down_proj.weight", torch.zeros(2560, 640), cfg).shape == (2560, 640)


@pytest.mark.parametrize("sharded", [True, False])
def test_exl3_ngram_source_reads_sharded_and_single_tables(tmp_path, sharded):
    from safetensors.torch import save_file

    from freetoken.models.qwen4_exp.ple_disk import source_from_exl3_ngram

    pre = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    rows, K, dim = 10, 5, 160
    words = 1 + dim * K // 16
    t = {f"{pre}.head_bias": torch.zeros(16, dim).half(), f"{pre}.head_offsets": torch.arange(16)}
    if sharded:
        t.update({f"{pre}.shard_{i}.trellis": torch.full((rows, words), i, dtype=torch.int16) for i in range(3)})
    else:
        t[f"{pre}.trellis"] = torch.zeros(3 * rows, words, dtype=torch.int16)
    meta = {"format": "exl3_ngram_trellis", "K": str(K), "row_dim": str(dim), "codebook": "mul1"}
    save_file(t, str(tmp_path / "ngram_embedding.safetensors"), metadata=meta)
    src = source_from_exl3_ngram(str(tmp_path / "ngram_embedding.safetensors"))
    assert src.total_rows == 3 * rows and src.row_bytes == 2 * words and src.exl3_bits == K and src.row_dim == dim
    assert len(src.extent_base) == (3 if sharded else 1)


def test_checkpoint_tensor_names_include_files_outside_the_index(tmp_path):
    import json

    from safetensors.torch import save_file

    from freetoken.models.exl3_weights import checkpoint_tensor_names

    save_file({"a.trellis": torch.zeros(1, dtype=torch.int16)}, str(tmp_path / "model.safetensors"))
    save_file({"visual.x.mul1": torch.zeros(1, dtype=torch.int32)}, str(tmp_path / "vision_k6.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a.trellis": "model.safetensors"}}))
    assert checkpoint_tensor_names(str(tmp_path)) == {"a.trellis": "model.safetensors", "visual.x.mul1": "vision_k6.safetensors"}


def test_prefill_workspace_matches_the_measured_peak(monkeypatch):
    import freetoken.moe.fused_exl3 as fused
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.exl3 import TritonExl3MoEKernel

    monkeypatch.setattr(fused, "EXL3_PREFILL_BLOCK", 4096)
    cfg = MoEConfig(num_experts=512, hidden=2560, intermediate=384, top_k=10)
    # torch's peak over one layer's 16384-token prefill on the 7900 XT (4096-token blocks): 620 MiB
    assert TritonExl3MoEKernel().prefill_workspace_bytes(cfg, 16384) == 620 * 2**20
    # a chunk within one block has no chunk-wide output
    assert TritonExl3MoEKernel().prefill_workspace_bytes(cfg, 4096) == 620 * 2**20 - 16384 * 2560 * 2
