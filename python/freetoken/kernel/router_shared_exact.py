"""FREETOKEN_ROUTER_SHARED_HIP: kernel/triton/router_shared.py's router | shared gate-up launch at 1-8 rows as a HIP
kernel with the Triton kernel's sums (csrc/jit/router_shared_exact.cuh): the same bits, its loads issued ahead."""
from __future__ import annotations

from functools import lru_cache

import torch

from .utils import load_jit


@lru_cache(maxsize=None)
def _module():
    return load_jit("router_shared_exact", "v9", cuda_files=["router_shared_exact.cuh"],
                    cuda_wrappers=[("run", "&RouterSharedExact::run")])


def router_and_shared_act_exact(x: torch.Tensor, gate_w: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor,
                                one_pass: bool, shared_gate_w: torch.Tensor | None = None):
    """(logits [M, 512], act [M, D], gate [M] fp32 or None) for a 512 x 2560 bf16 router and a 2D x 2560 int8
    gate | up, 1-8 rows; one_pass: the gate | up's (1, 4096, 4 warps) tile, else its (2, 512, 1 warp) one;
    shared_gate_w [2560] bf16: also the shared expert's gate as moe_shared_gate.py computes it."""
    M = x.shape[0]
    D = qweight.shape[0] // 2
    logits = torch.empty((M, gate_w.shape[0]), dtype=torch.bfloat16, device=x.device)
    act = torch.empty((M, D), dtype=torch.bfloat16, device=x.device)
    gate = torch.empty((M,), dtype=torch.float32, device=x.device) if shared_gate_w is not None else scale
    _module().run(x, gate_w, logits, qweight, scale, act, int(one_pass),
                  shared_gate_w if shared_gate_w is not None else x[0], gate, int(shared_gate_w is not None))
    return logits, act, (gate if shared_gate_w is not None else None)

