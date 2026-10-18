"""The GDN prefill conv takes its longest request from the host bounds: no device sync in every GDN layer (36 a forward
on Qwen3.8-Flash-Next), the same value as from the device ``cu_seqlens``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kernel.triton import causal_conv1d_triton
from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet


def _op_and_calls(monkeypatch):
    op = object.__new__(Qwen4ExpGatedDeltaNet)
    op.layer_id, op.key_dim, op.value_dim = 0, 4, 6
    op.conv1d = SimpleNamespace(weight=torch.zeros(14, 1, 4))
    pool = SimpleNamespace(local_index=lambda layer_id: 0, conv_states=torch.zeros(1, 2, 14, 3))
    calls = []
    monkeypatch.setattr(causal_conv1d_triton, "causal_conv1d_varlen_rows",
                        lambda *a, max_seq_len, **k: calls.append(max_seq_len))
    return op, pool, calls


def test_longest_request_from_host_bounds_without_item(monkeypatch):
    op, pool, calls = _op_and_calls(monkeypatch)

    def no_sync(self):
        raise AssertionError("a device sync (.item()) in the prefill conv")

    monkeypatch.setattr(torch.Tensor, "item", no_sync)
    cu = torch.tensor([0, 3, 10], dtype=torch.int32)
    op._conv_prefill_rows(torch.zeros(10, 14), pool, cu, None, None, [(0, 3), (3, 10)])
    assert calls == [7, 7, 7]  # q, k and v


@pytest.mark.parametrize("bounds", [[(0, 10)], [(0, 5), (5, 6), (6, 16)]])
def test_host_bounds_give_the_device_value(monkeypatch, bounds):
    op, pool, calls = _op_and_calls(monkeypatch)
    cu = torch.tensor([0] + [b for _, b in bounds], dtype=torch.int32)
    total = bounds[-1][1]
    op._conv_prefill_rows(torch.zeros(total, 14), pool, cu, None, None, bounds)
    op._conv_prefill_rows(torch.zeros(total, 14), pool, cu, None, None)  # no bounds: the device value
    assert calls[:3] == calls[3:] == [max(b - a for a, b in bounds)] * 3
