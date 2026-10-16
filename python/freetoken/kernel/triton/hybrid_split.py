"""The two GPU ends of a hybrid MoE decode layer (layers/moe.py ``_decode_hybrid``), one kernel each, carrying the
handshake with the CPU coordinator (moe/cpu_executor.py ``decode_publish`` / ``decode_merge``).

Publish: from the routes the hybrid ensure kernel split (``cpu_ids``: an expert id for a route the CPU computes, -1 for
the GPU's), write the CPU's inputs straight into its pinned, device-mapped IO rows (the activation rows, the ids, the
router weights), then raise the pinned ``ready`` word to the step's task (a constant per captured graph) with a
system-scope release once every wave's stores are out, ``done`` reset to 0 before it; a step without a CPU route sets
``done`` at once and rings nothing. Merge: spin on ``done`` until the coordinator sets it, then the GPU partial plus the
CPU's, read in place from host memory. They replace the copies around the doorbell's stream memops (each a graph node,
~3.2 us here) and the host-to-device copy plus add. The spin reads with a system-scope atomic: a volatile load is served
from a GPU cache and never sees the CPU's store.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def fence_lanes(device) -> int:
    """The publish kernel's threads (8 warps): each one issues a release before ready goes up."""
    return 8 * getattr(torch.cuda.get_device_properties(device), "warp_size", 32)


@triton.jit(do_not_specialize=["task"])
def _publish_kernel(x_ptr, ids_ptr, w_ptr, hx_ptr, hids_ptr, hw_ptr, ready_ptr, done_ptr, has_cpu_ptr, fence_ptr, task, M,
                    H, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
                    LANES: tl.constexpr):
    # one program for all the rows: ready goes up after all of them
    rows = tl.arange(0, BLOCK_M)
    mr = rows < M
    k = tl.arange(0, BLOCK_K)
    mk = mr[:, None] & (k < K)[None, :]
    idx = rows[:, None] * K + k[None, :]
    ids = tl.load(ids_ptr + idx, mask=mk, other=-1)
    lane = tl.arange(0, LANES)
    # the coordinator consumes ready by resetting it: done and ready are written by one lane, never repeated
    first = lane == 0
    if tl.max(tl.max(ids, axis=1), axis=0) < 0:
        tl.store(has_cpu_ptr, 0)
        tl.atomic_xchg(done_ptr, 1, sem="release", scope="sys")
    else:
        for start in range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            m = mr[:, None] & (offs < H)[None, :]
            xi = rows[:, None] * H + offs[None, :]
            tl.store(hx_ptr + xi, tl.load(x_ptr + xi, mask=m), mask=m)
        tl.store(hids_ptr + idx, ids, mask=mk)
        tl.store(hw_ptr + idx, tl.load(w_ptr + idx, mask=mk, other=0.0), mask=mk)
        tl.store(has_cpu_ptr, 1)
        # every wave waits for its own host stores (a release per lane on device scratch), then the first lane, which
        # reset done before, raises ready: the coordinator, once it sees the task, finds done at 0
        tl.atomic_xchg(done_ptr + lane * 0, tl.zeros((LANES,), dtype=tl.int64), mask=first, sem="release", scope="sys")
        tl.atomic_add(fence_ptr + lane, 0, sem="release", scope="sys")
        tl.debug_barrier()
        tl.atomic_xchg(ready_ptr + lane * 0, tl.zeros((LANES,), dtype=tl.int64) + task, mask=first, sem="release",
                       scope="sys")


def hybrid_publish(hidden: torch.Tensor, cpu_ids: torch.Tensor, weights: torch.Tensor, host_x: torch.Tensor,
                   host_ids: torch.Tensor, host_w: torch.Tensor, ready: torch.Tensor, done: torch.Tensor,
                   has_cpu: torch.Tensor, fence: torch.Tensor, task: int) -> None:
    """``host_*``: device views of the executor's pinned IO rows; ``ready`` / ``done``: device views of its pinned flag
    words (one int64 each); ``has_cpu``: device int32, whether this step rang; ``fence``: device int32 scratch of at
    least :func:`fence_lanes` elements; ``task``: the executor's task word for (layer, rows)."""
    M, H = hidden.shape
    K = cpu_ids.shape[1]
    block_m = triton.next_power_of_2(M)
    _publish_kernel[(1,)](hidden, cpu_ids, weights, host_x, host_ids, host_w, ready, done, has_cpu, fence, task, M, H, K,
                          BLOCK_M=block_m, BLOCK_H=max(256, 8192 // block_m), BLOCK_K=triton.next_power_of_2(K),
                          LANES=fence_lanes(hidden.device), num_warps=8)


@triton.jit
def _merge_wait_kernel(g_ptr, h_ptr, out_ptr, done_ptr, has_cpu_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(g_ptr + offs, mask=m).to(tl.float32)
    if tl.load(has_cpu_ptr) != 0:
        v = tl.atomic_add(done_ptr, 0, sem="acquire", scope="sys")
        while v == 0:
            v = tl.atomic_add(done_ptr, 0, sem="acquire", scope="sys")
        s += tl.load(h_ptr + offs, mask=m).to(tl.float32)
    tl.store(out_ptr + offs, s.to(out_ptr.dtype.element_ty), mask=m)


def hybrid_merge_wait(gpu_routed: torch.Tensor, host_y: torch.Tensor, done: torch.Tensor,
                      has_cpu: torch.Tensor) -> torch.Tensor:
    """gpu_routed + the CPU partial of the step :func:`hybrid_publish` rang (``host_y``: a device view of its pinned
    output rows), once ``done`` is set; gpu_routed alone (in a new tensor) when the step had no CPU route."""
    out = torch.empty_like(gpu_routed)
    n = gpu_routed.numel()
    _merge_wait_kernel[(triton.cdiv(n, 2048),)](gpu_routed, host_y, out, done, has_cpu, n, BLOCK=2048, num_warps=4)
    return out


__all__ = ["fence_lanes", "hybrid_merge_wait", "hybrid_publish"]
