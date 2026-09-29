from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from freetoken.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from freetoken.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    greedy_mask: torch.Tensor | None = None
    top_k_max: int | None = None  # host-side max(top_k) over the SAMPLED rows, for the small-k path (no device sync)


# Largest top_k served by top-k-first sampling (kernel/triton/sampling_topk.py); above it, with top_k
# off, or when flashinfer is installed, the full-vocab kernels run. 0 disables it. Default: on for ROCm
# (256): ~0.07 ms per step against 0.2-1 ms for the full-vocab path there (a sort); off elsewhere unless set.
_FAST_TOPK_ENV = os.environ.get("FREETOKEN_FAST_TOPK_MAX")


def _fast_topk_max() -> int:
    if _FAST_TOPK_ENV is not None:
        return int(_FAST_TOPK_ENV)
    from freetoken.kernel.backend import is_rocm

    return 256 if is_rocm() else 0


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        import flashinfer.sampling as sampling
    else:
        # exact top-k / top-p; on ROCm from a sorted threshold (see _SORTED_THRESHOLD in the module)
        import freetoken.kernel.triton.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        is_greedy = [p.is_greedy for p in params]
        if all(is_greedy):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        # Greedy outputs are selected explicitly in sample(); use neutral sampling
        # parameters for those rows instead of approximating argmax at low temperature.
        ts = [1.0 if g else max(p.temperature, MIN_T) for p, g in zip(params, is_greedy)]
        top_ks = [
            p.top_k if not g and p.top_k >= 1 else self.vocab_size
            for p, g in zip(params, is_greedy)
        ]
        top_ps = [
            1.0 if g else min(max(p.top_p, MIN_P), 1.0)
            for p, g in zip(params, is_greedy)
        ]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        greedy_mask = (
            make_device_tensor(is_greedy, torch.bool, self.device) if any(is_greedy) else None
        )
        # greedy rows carry top_k = vocab_size but are overwritten by argmax: leave them out of the max, or a
        # greedy request next to a small-k one would push the whole batch off the small-k path
        sampled_ks = [k for k, g in zip(top_ks, is_greedy) if not g]
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p, greedy_mask=greedy_mask,
                                 top_k_max=max(sampled_ks) if top_k is not None else None)

    @staticmethod
    def _use_topk_first(args: BatchSamplingArgs) -> bool:
        if args.top_k is None or args.top_k_max is None or args.top_k_max > _fast_topk_max():
            return False
        from freetoken.kernel.backend import is_flashinfer_installed

        return not is_flashinfer_installed()

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            if self._use_topk_first(args):
                from freetoken.kernel.triton.sampling_topk import top_k_top_p_sampling_from_logits

                tokens = top_k_top_p_sampling_from_logits(
                    logits, args.temperatures, args.top_k, args.top_p, args.top_k_max
                )
            else:
                tokens = sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
            if args.greedy_mask is not None:
                # Mixed batches still run probability sampling for all rows, but
                # greedy rows must follow argmax's deterministic tie-breaking.
                greedy_tokens = torch.argmax(logits, dim=-1).to(tokens.dtype)
                tokens = torch.where(args.greedy_mask, greedy_tokens, tokens)
            return tokens
