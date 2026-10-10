"""YaRN for Qwen3.8-Flash-Next's long context: the engine's cos/sin rows against HF's rotary embedding (CPU)."""

import pytest
import torch

from freetoken.layers.rotary import get_rope, mrope_cos_sin_rows

qwen4 = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

# Qwen's model card: rope_parameters of text_config, max_position_embeddings left at 262144
_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144, "rope_theta": 10000000,
         "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True}
_POS = torch.tensor([0, 1, 7, 4095, 65537, 262143, 262144, 500001, 999999, 1048575])


def _hf_rows(rope_parameters: dict) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = Qwen4ExpTextConfig(head_dim=256, num_attention_heads=24, hidden_size=2560, max_position_embeddings=262144,
                             rope_parameters=dict(rope_parameters))
    cos, sin = qwen4.Qwen4ExpTextRotaryEmbedding(cfg)(torch.zeros(1), _POS[None])
    return cos[0, :, :32], sin[0, :, :32]


@pytest.mark.parametrize("mrope", [True, False], ids=["attention-mrope", "indexer-plain"])
def test_yarn_rows_match_hf(mrope: bool) -> None:
    scaling = tuple((k, v) for k, v in _YARN.items() if not isinstance(v, (list, dict)))
    rope = get_rope(head_dim=256 if mrope else 128, rotary_dim=64, max_position=1048576, base=1e7,
                    rope_scaling=scaling, mrope_section=(11, 11, 10) if mrope else None)
    if mrope:
        rows = mrope_cos_sin_rows(rope._cos_sin_cache, _POS[None].expand(3, -1), rope._section_table)
    else:
        rows = rope._cos_sin_cache[_POS]
    cos, sin = _hf_rows(_YARN)
    torch.testing.assert_close(rows[:, :32], cos, atol=0, rtol=0)
    torch.testing.assert_close(rows[:, 32:], sin, atol=0, rtol=0)
    assert abs(float(rows[0, 0]) - (0.1 * torch.log(torch.tensor(4.0)).item() + 1.0)) < 1e-6  # cos(0) * mscale


def test_default_rows_unchanged_and_exact() -> None:
    plain = {k: v for k, v in _YARN.items() if k not in ("factor", "original_max_position_embeddings")}
    plain["rope_type"] = "default"
    rope = get_rope(head_dim=256, rotary_dim=64, max_position=262144, base=1e7, mrope_section=(11, 11, 10))
    pos = _POS[_POS < 262144]
    rows = mrope_cos_sin_rows(rope._cos_sin_cache, pos[None].expand(3, -1), rope._section_table)
    cos, sin = _hf_rows(plain)
    keep = _POS < 262144
    torch.testing.assert_close(rows[:, :32], cos[keep], atol=0, rtol=0)
    torch.testing.assert_close(rows[:, 32:], sin[keep], atol=0, rtol=0)


def test_extended_plain_table_keeps_the_native_rows() -> None:
    # FREETOKEN_ROPE_MAX_POSITION: a longer plain table, the same bits wherever the native one had a row
    native = get_rope(head_dim=256, rotary_dim=64, max_position=262144, base=1e7, mrope_section=(11, 11, 10))
    longer = get_rope(head_dim=256, rotary_dim=64, max_position=1048576, base=1e7, mrope_section=(11, 11, 10))
    assert longer._cos_sin_cache.shape[0] == 1048576
    assert torch.equal(longer._cos_sin_cache[:262144], native._cos_sin_cache)
