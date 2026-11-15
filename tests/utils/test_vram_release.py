"""The settle time after a release of cached VRAM (utils/torch_utils.py): off until the loss was seen, or forced."""
import pytest
import torch

from freetoken.utils import torch_utils as tu


@pytest.fixture(autouse=True)
def _unseen(monkeypatch):
    monkeypatch.setattr(tu, "_release_race_seen", False)
    monkeypatch.delenv("FREETOKEN_VRAM_RELEASE_SETTLE", raising=False)


def test_no_wait_until_the_loss_was_seen(monkeypatch):
    assert tu.vram_release_settle() == 0.0
    tu.note_vram_release_race()
    assert tu.vram_release_settle() == 1.0
    monkeypatch.setenv("FREETOKEN_VRAM_RELEASE_SETTLE", "0")  # the variable wins both ways
    assert tu.vram_release_settle() == 0.0
    monkeypatch.setattr(tu, "_release_race_seen", False)
    monkeypatch.setenv("FREETOKEN_VRAM_RELEASE_SETTLE", "0.5")
    assert tu.vram_release_settle() == 0.5


def test_a_read_back_that_differs_makes_later_releases_wait():
    from freetoken.layers.quantization import int8_weight_only as iwo

    host = {"c": torch.tensor([20000003, 20000023])}
    iwo._verify_copies({"c": host["c"].clone()}, host, {"c"}, settle=0)   # clean: nothing noted
    assert tu.vram_release_settle() == 0.0
    iwo._verify_copies({"c": torch.zeros(2, dtype=torch.int64)}, host, {"c"}, settle=0)
    assert tu.vram_release_settle() == 1.0


def test_the_wait_follows_a_release_that_gave_memory_back(monkeypatch):
    waits, held = [], [8 << 30]
    monkeypatch.setenv("FREETOKEN_VRAM_RELEASE_SETTLE", "0.5")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i=None: held[0])
    monkeypatch.setattr(tu.time, "sleep", waits.append)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)          # nothing went back: no wait
    tu.release_cached_vram()
    assert waits == []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: held.__setitem__(0, 1 << 30))
    tu.release_cached_vram()
    assert waits == [0.5]
