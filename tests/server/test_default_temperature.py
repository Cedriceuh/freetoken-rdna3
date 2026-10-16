"""--default-temperature: the temperature a request that sends none gets, in place of the checkpoint's."""

from __future__ import annotations

import json
from types import SimpleNamespace

from freetoken.server.api_server import default_sampling


def _cfg(tmp_path, **kw):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}))
    base = dict(sampling_defaults="model", use_dummy_weight=False, model_path=str(tmp_path), default_temperature=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_checkpoint_values_without_the_option(tmp_path):
    sampling, source = default_sampling(_cfg(tmp_path))
    assert sampling == {"temperature": 1.0, "top_k": 20, "top_p": 0.95} and source == "model"


def test_option_replaces_only_the_temperature(tmp_path):
    sampling, source = default_sampling(_cfg(tmp_path, default_temperature=0.6))
    assert sampling == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    assert source == "model, temperature from --default-temperature"


def test_option_with_framework_defaults(tmp_path):
    sampling, source = default_sampling(_cfg(tmp_path, sampling_defaults="none", default_temperature=0.6))
    assert sampling == {"temperature": 0.6} and source.startswith("framework")


def test_parse_args_takes_the_option():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    hf = SimpleNamespace(to_dict=lambda: {"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: hf):
        assert parse_args(["--model", "/models/anon"])[0].default_temperature is None
        assert parse_args(["--model", "/models/anon", "--default-temperature", "0.6"])[0].default_temperature == 0.6
