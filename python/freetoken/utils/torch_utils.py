from __future__ import annotations

import functools
import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


# A machine was reported (two RX 7900 XTX, kernel 7.2) where writes into VRAM allocated right after a release were lost
# or overwritten for a fraction of a second; a pause of 0.5 s after the release was enough. FREETOKEN_VRAM_RELEASE_SETTLE
# = seconds to wait after every release (0: never). Unset: no wait until this process has seen the loss itself
# (the read-back after the int8 compaction, layers/quantization/int8_weight_only.py), 1 s after that.
_SETTLE_ENV = "FREETOKEN_VRAM_RELEASE_SETTLE"
_SETTLE_SEEN = 1.0
_release_race_seen = False


def note_vram_release_race() -> None:
    """This process lost writes into VRAM re-allocated right after a release: later releases wait."""
    global _release_race_seen
    _release_race_seen = True


def vram_release_settle() -> float:
    raw = os.environ.get(_SETTLE_ENV, "").strip()
    if raw:
        return max(0.0, float(raw))
    return _SETTLE_SEEN if _release_race_seen else 0.0


def release_cached_vram() -> None:
    """``torch.cuda.empty_cache()``, then the settle time when VRAM actually went back to the driver."""
    import torch

    if not torch.cuda.is_available():
        return
    wait = vram_release_settle()
    held = sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())) if wait else 0
    torch.cuda.empty_cache()
    if wait and sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())) < held:
        time.sleep(wait)
