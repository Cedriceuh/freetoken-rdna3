"""Opt-in (FREETOKEN_INT8_DENSE=1): weight-only INT8 for the dense bf16 projections, converted in place after loading.

The routed experts are already NVFP4; everything else a checkpoint leaves bf16 (attention, GDN, shared expert,
hyper-connections, PLE projections, LM head) is read in full on every decode token and dominates batch-1 decode
on RDNA3. Each selected linear keeps an int8 ``[N, K]`` weight and a per-output-row fp32 scale (symmetric,
amax / 127), halving those bytes and the VRAM they hold (the freed memory goes to the MoE / KV pools, which are
sized after the weights).

- decode (one row per decoding request, up to ``FREETOKEN_INT8_ROWS_MAX`` = 8): split-K int8 GEMV
  (``kernel/triton/int8_gemv.py``), each row computed as it would be alone (batch-invariant)
- prefill / larger batches: the weight is dequantized to the activation dtype for the call, then F.linear

Lossy (per-row int8): validated on the author's hardware with proxy benches and a 29-bug agentic bench (same
score with and without), not bit-exact against bf16. Tuned for <= 8 concurrent requests.

Enable with ``FREETOKEN_INT8_DENSE=1``. Routers (``mlp.gate``, ``mlp.shared_expert_gate``) stay bf16: a flipped
expert choice costs more than their bytes. The Qwen VL vision tower (``visual``) stays bf16 too: it only runs on
prefill, where int8 saves no time, and streams its weights from host RAM. ``FREETOKEN_INT8_DENSE_SKIP`` adds
comma-separated substrings of the module path (attribute path from the model root, e.g. ``lm_head,hyper_connection``)
to keep in bf16.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from freetoken.utils import init_logger

logger = init_logger(__name__)

_ALWAYS_SKIP = (".mlp.gate", ".mlp.shared_expert_gate", ".visual")


def enabled() -> bool:
    return os.environ.get("FREETOKEN_INT8_DENSE", "0") == "1"


def _skip_list() -> tuple[str, ...]:
    extra = tuple(filter(None, (p.strip() for p in os.environ.get("FREETOKEN_INT8_DENSE_SKIP", "").split(","))))
    return _ALWAYS_SKIP + extra


class Int8WeightOnlyLinearMethod:
    """Replaces an unquantized linear's method after conversion (duck-typed like LinearMethod.apply/finalize)."""

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        if layer.bias is None and x.is_cuda and x.dim() == 2 and x.dtype is torch.bfloat16 and x.stride(-1) == 1:
            from freetoken.kernel.triton.int8_gemv import ROWS_MAX, gemv_int8

            # one row per decoding request; beyond ROWS_MAX rows (prefill) the dequantized GEMM below is cheaper
            if x.shape[0] == 1 or (1 < x.shape[0] <= ROWS_MAX and x.is_contiguous()):
                return gemv_int8(x, layer.weight, layer.weight_scale)
        # int8 * fp32 computes in fp32 and rounds once into the activation dtype: the bits of
        # (weight.float() * scale).to(dtype), without the full-size fp32 temporary
        w = torch.empty(layer.weight.shape, dtype=x.dtype, device=layer.weight.device)
        torch.mul(layer.weight, layer.weight_scale[:, None], out=w)
        return F.linear(x, w, layer.bias.to(x.dtype) if layer.bias is not None else None)

    def finalize(self, layer: Any) -> None:
        pass


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-row int8: ``w ~= q * scale``."""
    wf = w.float()
    scale = wf.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(wf / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def convert_model(root: Any) -> tuple[int, int]:
    """Convert every eligible bf16 linear under ``root``; returns (layers converted, bytes freed)."""
    from freetoken.layers.base import BaseOP
    from freetoken.layers.quantization.linear.unquantized import UnquantizedLinearMethod

    skip = _skip_list()
    seen: set[int] = set()
    converted = freed = 0

    def eligible(op: Any, path: str) -> bool:
        method = getattr(op, "quant_method", None)
        w = getattr(op, "weight", None)
        # match on the attribute path, not ``op.prefix``: the routers are built without one
        return (isinstance(method, UnquantizedLinearMethod) and isinstance(w, torch.Tensor) and w.dim() == 2
                and w.dtype is torch.bfloat16 and not any(path.endswith(s) or s + "." in path + "." for s in skip)
                and getattr(op, "tied_embedding", None) is None)

    def walk(op: Any, path: str) -> None:
        nonlocal converted, freed
        if id(op) in seen:
            return
        seen.add(id(op))
        if eligible(op, path):
            names.append(path)
            device = op.weight.device
            # quantize on the host: fp32 temporaries on the GPU would grow the caching allocator and the small int8
            # tensors would then pin those segments, so the post-weights free-memory reading (which sizes the MoE and
            # KV pools) would not see the saving at all
            w_host = op.weight.to("cpu")
            freed += w_host.numel() * w_host.element_size()
            op.weight = None
            q, scale = quantize_rows(w_host)
            del w_host
            freed -= q.numel() + scale.numel() * 4
            op.weight = q.to(device)
            op.weight_scale = scale.to(device)
            op.quant_method = Int8WeightOnlyLinearMethod()
            converted += 1
        for key, value in vars(op).items():
            if isinstance(value, BaseOP):
                walk(value, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, BaseOP):
                        walk(item, f"{path}.{key}.{i}")

    names: list[str] = []
    before = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    walk(root, "")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info_rank0(f"INT8 weight-only: device free memory {before / 2**30:.2f} -> "
                          f"{torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
    kinds = sorted({".".join(p.split(".")[-2:]) for p in names})
    logger.info_rank0(f"INT8 weight-only: {converted} dense linears converted, {freed / 2**30:.2f} GiB freed per rank; "
                      f"kinds: {', '.join(kinds)} (kept bf16: {', '.join(skip)})")
    return converted, freed


def compact_device_tensors(root: Any) -> None:
    """Re-allocate every device tensor of the model's state in one pass, largest first.

    Freeing the bf16 weights leaves holes in allocator segments that still hold neighbouring weights; the pools
    sized next are large single allocations that cannot use those holes, and the free-memory reading does not
    count them. A host round trip of the model state (``load_state_dict`` assigns, it does not copy) rebuilds
    the segments densely. Tensors shared between keys stay shared (moved once by identity).
    """
    if not torch.cuda.is_available():
        return
    state = root.state_dict()
    device_keys = {k for k, v in state.items() if v.is_cuda}
    if not device_keys:
        return
    device = state[next(iter(device_keys))].device
    before = torch.cuda.mem_get_info()[0]
    host_by_id: dict[int, torch.Tensor] = {}
    host = {}
    for k, v in state.items():
        if v.is_cuda and id(v) not in host_by_id:  # copy each shared tensor once (setdefault would copy every key)
            host_by_id[id(v)] = v.to("cpu")
        host[k] = host_by_id[id(v)] if v.is_cuda else v
    del state
    root.load_state_dict(dict(host))
    torch.cuda.empty_cache()
    dev_by_id: dict[int, torch.Tensor] = {}
    order = sorted(host, key=lambda k: -(host[k].numel() * host[k].element_size()))
    placed = {}
    for k in order:
        v = host[k]
        if k in device_keys and id(v) not in dev_by_id:  # one device allocation per shared tensor, no throwaway copy
            dev_by_id[id(v)] = v.to(device)
        placed[k] = dev_by_id[id(v)] if k in device_keys else v
    root.load_state_dict(placed)
    torch.cuda.empty_cache()
    logger.info_rank0(f"INT8 weight-only: compacted {len(device_keys)} device tensors, free memory "
                      f"{before / 2**30:.2f} -> {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")


__all__ = ["compact_device_tensors", "convert_model", "enabled", "quantize_rows", "Int8WeightOnlyLinearMethod"]
