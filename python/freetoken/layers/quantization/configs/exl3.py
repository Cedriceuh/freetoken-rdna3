from __future__ import annotations

import json
import os
import struct
from functools import lru_cache
from typing import Any, ClassVar

from ..exl3_codec import CB_3INST, CB_MCG, CB_MUL1, HAD_DIM
from ..names import is_routed_expert
from ..registry import register_dialect
from ..scheme import QuantKind, QuantScheme, exl3_scheme
from .base import QuantConfig, Stored

_PROJS = ("gate_proj", "up_proj", "down_proj")


@lru_cache(maxsize=None)
def _expert_storage_table(folder: str) -> dict[str, tuple[int, int, int]]:
    """``<experts container>.<proj>`` -> (trellis words per tile, has mul1, has mcg) for expert 0 of every expert layer.

    Only this table stays cached: the full index and the shard headers (~0.2 GiB of Python objects on Qwen3.8) are
    dropped once it is built."""
    from freetoken.models.loader import safetensors_weight_map

    weight_map = safetensors_weight_map(folder)
    wanted: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.endswith(".trellis") and ".experts.0." in key:
            wanted.setdefault(shard, []).append(key)
    table: dict[str, tuple[int, int, int]] = {}
    for shard, keys in wanted.items():
        with open(os.path.join(folder, shard), "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for key in keys:
            module = key[: -len(".trellis")]
            name = module.replace(".experts.0.", ".experts.", 1)
            table[name] = (header[key]["shape"][-1], f"{module}.mul1" in weight_map, f"{module}.mcg" in weight_map)
    return table


@register_dialect
class Exl3Config(QuantConfig):
    """exllamav3 EXL3: every linear is a trellis (``trellis`` / ``suh`` / ``svh`` plus a ``mul1`` or ``mcg`` codebook marker).

    Only the routed experts stay EXL3 (served from the offload cache, ../moe/exl3.py); every other linear is
    reconstructed to bf16 as it loads (models/exl3_weights.py), so the model builds it unquantized. The experts'
    bitrate comes from the checkpoint's tensor shapes, so the dialect needs ``model_path``; one bitrate and codebook
    for every expert layer, because the offload cache holds one bank layout."""

    dialect = "exl3"
    STORAGE: ClassVar[dict[QuantKind, dict[str, str | Stored]]] = {
        QuantKind.EXL3: {"trellis": "trellis", "suh": "suh", "svh": "svh"},
    }

    def __init__(self, q: dict[str, Any], hf_config: Any = None, *, name_map=None, unquantized=()):
        from freetoken.distributed.split import set_intermediate_unit

        super().__init__(name_map, unquantized)
        self.version = q.get("version")
        self._expert: tuple[int, int, str] | None = None  # (bits, codebook, first layer seen)
        # a rank's slice of the expert intermediate must hold whole Hadamard blocks
        set_intermediate_unit(HAD_DIM)

    def scheme_for_name(self, name: str) -> QuantScheme | None:
        # the experts container, or (through the family's packed mapping) one of its first expert's projections
        if not is_routed_expert(name) or self.unquantized(name):
            return None
        name = name[: name.index(".experts") + len(".experts")]
        bits, codebook = self._expert_storage(name)
        if self._expert is None:
            self._expert = (bits, codebook, name)
        elif self._expert[:2] != (bits, codebook):
            raise NotImplementedError(
                f"EXL3 experts of {name} are {bits} bpw / codebook {codebook}, those of {self._expert[2]} "
                f"{self._expert[0]} bpw / codebook {self._expert[1]}: one bank layout serves every layer"
            )
        return exl3_scheme(bits, codebook)

    def _expert_storage(self, name: str) -> tuple[int, int]:
        """(bits, codebook) of the experts under checkpoint module ``name``, from expert 0's tensors."""
        if self.model_path is None:
            raise ValueError("the EXL3 dialect reads expert bitrates from the checkpoint: no model_path was given")
        from freetoken.utils import download_hf_weight

        table = _expert_storage_table(download_hf_weight(self.model_path))
        found = set()
        for proj in _PROJS:
            entry = table.get(f"{name}.{proj}")
            if entry is None:
                raise ValueError(f"EXL3 checkpoint has no {name}.0.{proj}.trellis")
            words, mul1, mcg = entry
            if words % 16:
                raise NotImplementedError(f"{name}.0.{proj}: half-integer EXL3 bitrates ({words / 16} bpw) are not supported")
            # like exllamav3: a linear without a marker tensor is 3inst, whatever the config says
            found.add((words // 16, CB_MUL1 if mul1 else CB_MCG if mcg else CB_3INST))
        if len(found) != 1:
            raise NotImplementedError(f"EXL3 experts of {name} mix bitrates / codebooks across projections: {sorted(found)}")
        return found.pop()
