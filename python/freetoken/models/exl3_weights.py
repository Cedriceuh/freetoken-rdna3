"""EXL3 (exllamav3) checkpoints: the dense linears reconstructed to bf16 as they load, and the routed experts' pieces.

A linear is stored as ``<module>.trellis`` / ``.suh`` / ``.svh`` (``.su`` / ``.sv`` packed signs in old checkpoints)
plus an empty ``.mul1`` or ``.mcg`` codebook marker. Families keep only their routed experts in EXL3 (the offload
cache serves them, layers/quantization/moe/exl3.py); every other linear becomes ``<module>.weight`` ``[out, in]``
here, then follows the family's usual sharding and fusion.
"""

from __future__ import annotations

from typing import Callable, Iterator

import torch

from freetoken.layers.quantization import exl3_codec as ex

EXL3_LEAVES = ("trellis", "suh", "svh", "su", "sv", "mul1", "mcg")
_MARKERS = ("mul1", "mcg")
# columns of one GPU reconstruction pass: the decoded fp32 slab is in_features x this
_RECON_COLS = 4096


def checkpoint_tensor_names(folder: str) -> dict[str, str]:
    """Every tensor of the checkpoint -> its file: the index, plus the safetensors files it does not list (exllamav3
    writes some components beside it, e.g. ``vision_k6.safetensors``)."""
    import glob
    import json
    import os
    import struct

    from freetoken.models.loader import safetensors_weight_map

    names = dict(safetensors_weight_map(folder))
    indexed = set(names.values())
    for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        base = os.path.basename(path)
        if base in indexed:
            continue
        with open(path, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for key in header:
            if key != "__metadata__":
                names.setdefault(key, base)
    return names


def _split(name: str) -> tuple[str, str]:
    module, _, leaf = name.rpartition(".")
    return module, leaf


def reconstruct_weight(
    trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, codebook: int, *, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """``W^T`` ``[out, in]`` of one EXL3 linear in ``dtype``, in column slabs of whole 128-blocks (the right Hadamard
    stays inside a slab): the decode kernel on the GPU, the torch reference on the CPU."""
    A, B, _ = trellis.shape
    k, n = A * 16, B * 16
    if k % ex.HAD_DIM or n % ex.HAD_DIM:
        raise ValueError(f"EXL3 weight {k}x{n} is not a whole number of {ex.HAD_DIM}-blocks")
    out = torch.empty((n, k), dtype=dtype, device=trellis.device)
    step = max(_RECON_COLS // 16 // 8 * 8, 8)
    if not trellis.is_cuda:
        for b0 in range(0, B, step):
            b1 = min(b0 + step, B)
            w = ex.reconstruct(trellis[:, b0:b1], suh, svh[b0 * 16:b1 * 16], codebook)
            out[b0 * 16:b1 * 16] = w.T.to(dtype)
        return out
    from freetoken.kernel.triton.exl3 import exl3_dequant

    had = ex.hadamard_128(trellis.device)
    su = suh.to(device=trellis.device, dtype=torch.float32).view(k, 1)
    sv = svh.to(device=trellis.device, dtype=torch.float32)
    for b0 in range(0, B, step):
        b1 = min(b0 + step, B)
        cols = (b1 - b0) * 16
        w = exl3_dequant(trellis[:, b0:b1].contiguous(), codebook).float()
        w = (had @ w.view(k // ex.HAD_DIM, ex.HAD_DIM, cols)).view(k, cols) * su
        w = (w.view(k, cols // ex.HAD_DIM, ex.HAD_DIM) @ had).view(k, cols) * sv[b0 * 16:b1 * 16]
        out[b0 * 16:b1 * 16] = w.T.to(dtype)
    return out


class Exl3DenseReconstructor:
    """Buffers the tensors of each EXL3 linear and emits ``<module>.weight`` once its trellis and scales are in.

    The codebook comes from the marker names in ``weight_map`` (give it every file: checkpoint_tensor_names), so a
    module completes without waiting for its (empty) marker tensor, which may sit in another shard; like exllamav3, a
    linear without a marker is 3inst."""

    def __init__(self, weight_map: dict[str, str], *, rename: Callable[[str], str | None] = lambda n: n,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype
        self.codebook: dict[str, int] = {}
        for raw in weight_map:
            module, leaf = _split(raw)
            if leaf in _MARKERS:
                name = rename(raw)
                if name is not None:
                    self.codebook[_split(name)[0]] = ex.CB_MUL1 if leaf == "mul1" else ex.CB_MCG
        self.pending: dict[str, dict[str, torch.Tensor]] = {}

    def feed(self, name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]] | None:
        """None when ``name`` is not an EXL3 tensor; otherwise ``[]`` while its module is incomplete, then
        ``[(module.weight, W^T)]``."""
        module, leaf = _split(name)
        if leaf not in EXL3_LEAVES:
            return None
        if leaf in _MARKERS:
            return []
        parts = self.pending.setdefault(module, {})
        parts[leaf] = tensor
        if "trellis" not in parts or not ({"suh", "su"} & parts.keys()) or not ({"svh", "sv"} & parts.keys()):
            return []
        del self.pending[module]
        weight = reconstruct_weight(parts["trellis"], ex.channel_scales(parts, "u"), ex.channel_scales(parts, "v"),
                                    self.codebook.get(module, ex.CB_3INST), dtype=self.dtype)
        return [(f"{module}.weight", weight)]

    def check_done(self) -> None:
        if self.pending:
            raise ValueError(f"incomplete EXL3 linears in the checkpoint: {sorted(self.pending)[:8]}")


def iter_exl3_expert_pieces(
    model_path: str,
    locate: Callable[[str], tuple[int, int, str] | None],
    *,
    expected_experts: int,
    parallel: bool = False,
    workers: int = 8,
    chunk: int = 8 << 20,
    desc: str = "EXL3 experts",
) -> Iterator[tuple[int, int, int, dict[str, torch.Tensor]]]:
    """One piece per routed expert: ``{gate,up,down}_{trellis,suh,svh}`` (packed ``su`` / ``sv`` signs unpacked to fp16).

    ``locate(checkpoint key)`` -> ``(bank_layer, expert, role)`` for the expert tensors to read, role ``gate`` /
    ``up`` / ``down``; markers are skipped (the dialect fixed one codebook for every expert)."""
    import os

    import safetensors

    from freetoken.models.loader import drop_page_cache, safetensors_weight_map
    from freetoken.moe.expert_pieces import per_expert_pieces
    from freetoken.utils import download_hf_weight
    from tqdm import tqdm

    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    wanted: dict[str, tuple[int, int, str]] = {}
    for name in weight_map:
        where = locate(name)
        if where is None:
            continue
        leaf = _split(name)[1]
        if leaf in _MARKERS:
            continue
        if leaf not in ("trellis", "suh", "svh", "su", "sv"):
            raise ValueError(f"{desc}: unexpected expert tensor {name}")
        bank_layer, expert, role = where
        wanted[name] = (bank_layer, expert, f"{role}_{'trellis' if leaf == 'trellis' else 's' + leaf[1] + 'h'}")
    if len(wanted) != expected_experts * 9:
        raise ValueError(f"{desc}: found {len(wanted)} expert tensors, expected {expected_experts * 9}")

    def ingest(name: str, tensor: torch.Tensor) -> torch.Tensor:
        return ex.unpack_signs(tensor) if _split(name)[1] in ("su", "sv") else tensor

    def _serial():
        by_shard: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            if name in wanted:
                by_shard.setdefault(shard, []).append(name)
        for shard in tqdm(sorted(by_shard), desc=f"Loading {desc}"):
            path = os.path.join(folder, shard)
            drop_page_cache(path)
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for name in by_shard[shard]:
                    yield name, ingest(name, f.get_tensor(name))
            drop_page_cache(path)

    def _parallel():
        from freetoken.models.weight import iter_expert_tensors_parallel

        for name, tensor in iter_expert_tensors_parallel(folder, lambda n: n in wanted, workers=workers, chunk=chunk):
            yield name, ingest(name, tensor)

    return per_expert_pieces(_parallel() if parallel else _serial(), wanted.get, tensors_per_expert=9)


__all__ = ["EXL3_LEAVES", "Exl3DenseReconstructor", "checkpoint_tensor_names", "iter_exl3_expert_pieces", "reconstruct_weight"]
