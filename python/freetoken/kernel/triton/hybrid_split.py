"""The two GPU ends of a hybrid MoE decode layer (layers/moe.py ``_decode_hybrid``), one kernel each.

Split: from the routing after the capped fetch (``slots``: a cache slot, or -1 for a route the CPU computes), write
the CPU's inputs straight into its pinned, device-mapped host buffers (the activation rows, the CPU routes' raw expert
ids or -1, the router weights) and the GPU's own (the weights with the CPU routes zeroed, the slots clamped to 0
where the kernel reads every route). Merge: the GPU partial plus the CPU's, read in place from host memory. They
replace the masks, two ``where``, a clamp, three device-to-host copies and a host-to-device copy plus add.

The ``_signal`` / ``_wait`` variants also carry the handshake with the CPU coordinator, in place of three stream memops
(each costs a graph node, ~3.2 us here): the split kernel bumps the slot's step number (device memory), then raises the
host ``ready`` flag to it with a system-scope release once every wave's stores are out; the merge kernel spins on the
host ``done`` flag until the coordinator hands that number back. The spin reads with a system-scope atomic: a volatile
load is served from a GPU cache and never sees the CPU's store.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _split_kernel(x_ptr, raw_ptr, slot_ptr, w_ptr, hx_ptr, hids_ptr, hw_ptr, gw_ptr, gslot_ptr, H, K: tl.constexpr,
                  CLAMP: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    for start in range(0, H, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        m = offs < H
        tl.store(hx_ptr + row * H + offs, tl.load(x_ptr + row * H + offs, mask=m), mask=m)
    k = tl.arange(0, BLOCK_K)
    mk = k < K
    slot = tl.load(slot_ptr + row * K + k, mask=mk, other=0)
    raw = tl.load(raw_ptr + row * K + k, mask=mk, other=0)
    w = tl.load(w_ptr + row * K + k, mask=mk, other=0.0).to(tl.float32)
    on_gpu = slot >= 0
    tl.store(hids_ptr + row * K + k, tl.where(on_gpu, -1, raw).to(tl.int32), mask=mk)
    tl.store(hw_ptr + row * K + k, w, mask=mk)
    tl.store(gw_ptr + row * K + k, tl.where(on_gpu, w, 0.0), mask=mk)
    if CLAMP:
        tl.store(gslot_ptr + row * K + k, tl.maximum(slot, 0), mask=mk)


def hybrid_split(hidden: torch.Tensor, raw_ids: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                 host_x: torch.Tensor, host_ids: torch.Tensor, host_w: torch.Tensor, clamp: bool):
    """-> (GPU weights, GPU slots). ``host_*`` are device views of the CPU executor's pinned buffers."""
    M, H = hidden.shape
    K = slots.shape[1]
    gpu_w = torch.empty(M, K, dtype=torch.float32, device=hidden.device)
    gpu_slots = torch.empty_like(slots) if clamp else slots
    _split_kernel[(M,)](hidden, raw_ids, slots, weights, host_x, host_ids, host_w, gpu_w, gpu_slots, H, K,
                        CLAMP=clamp, BLOCK_H=1024, BLOCK_K=triton.next_power_of_2(K), num_warps=4)
    return gpu_w, gpu_slots


@triton.jit
def _split_signal_kernel(x_ptr, raw_ptr, slot_ptr, w_ptr, hx_ptr, hids_ptr, hw_ptr, gw_ptr, gslot_ptr, seq_ptr,
                         ready_ptr, fence_ptr, M, H, K: tl.constexpr, CLAMP: tl.constexpr, BLOCK_M: tl.constexpr,
                         BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr, LANES: tl.constexpr):
    # one program for all the rows: the ready flag goes up after all of them
    rows = tl.arange(0, BLOCK_M)
    mr = rows < M
    for start in range(0, H, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        m = mr[:, None] & (offs < H)[None, :]
        idx = rows[:, None] * H + offs[None, :]
        tl.store(hx_ptr + idx, tl.load(x_ptr + idx, mask=m), mask=m)
    k = tl.arange(0, BLOCK_K)
    mk = mr[:, None] & (k < K)[None, :]
    idx = rows[:, None] * K + k[None, :]
    slot = tl.load(slot_ptr + idx, mask=mk, other=0)
    raw = tl.load(raw_ptr + idx, mask=mk, other=0)
    w = tl.load(w_ptr + idx, mask=mk, other=0.0).to(tl.float32)
    on_gpu = slot >= 0
    tl.store(hids_ptr + idx, tl.where(on_gpu, -1, raw).to(tl.int32), mask=mk)
    tl.store(hw_ptr + idx, w, mask=mk)
    tl.store(gw_ptr + idx, tl.where(on_gpu, w, 0.0), mask=mk)
    if CLAMP:
        tl.store(gslot_ptr + idx, tl.maximum(slot, 0), mask=mk)
    step = tl.load(seq_ptr) + 1
    tl.store(seq_ptr, step)
    # every wave waits for its own host stores (a release per lane on device scratch), then one lane raises ready
    tl.atomic_add(fence_ptr + tl.arange(0, LANES), 0, sem="release", scope="sys")
    tl.debug_barrier()
    tl.atomic_xchg(ready_ptr, step, sem="release", scope="sys")


def fence_lanes(device) -> int:
    """The split kernel's threads (8 warps): each one issues a release before ready goes up."""
    return 8 * getattr(torch.cuda.get_device_properties(device), "warp_size", 32)


@triton.jit
def _merge_kernel(g_ptr, h_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(g_ptr + offs, mask=m).to(tl.float32) + tl.load(h_ptr + offs, mask=m).to(tl.float32)
    tl.store(out_ptr + offs, s.to(out_ptr.dtype.element_ty), mask=m)


def hybrid_split_signal(hidden: torch.Tensor, raw_ids: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor,
                        host_x: torch.Tensor, host_ids: torch.Tensor, host_w: torch.Tensor, clamp: bool,
                        seq: torch.Tensor, ready: torch.Tensor, fence: torch.Tensor):
    """:func:`hybrid_split`, then the step number in ``seq`` (device, one int64) raised into ``ready`` (a device view of
    the slot's host flag). ``fence``: device int32 scratch of at least :func:`fence_lanes` elements."""
    M, H = hidden.shape
    K = slots.shape[1]
    gpu_w = torch.empty(M, K, dtype=torch.float32, device=hidden.device)
    gpu_slots = torch.empty_like(slots) if clamp else slots
    block_m = triton.next_power_of_2(M)
    _split_signal_kernel[(1,)](hidden, raw_ids, slots, weights, host_x, host_ids, host_w, gpu_w, gpu_slots, seq, ready,
                               fence, M, H, K, CLAMP=clamp, BLOCK_M=block_m, BLOCK_H=max(256, 8192 // block_m),
                               BLOCK_K=triton.next_power_of_2(K), LANES=fence_lanes(hidden.device), num_warps=8)
    return gpu_w, gpu_slots


@triton.jit
def _merge_wait_kernel(g_ptr, h_ptr, out_ptr, seq_ptr, done_ptr, n, BLOCK: tl.constexpr):
    step = tl.load(seq_ptr)
    v = tl.atomic_add(done_ptr, 0, sem="acquire", scope="sys")
    while v < step:
        v = tl.atomic_add(done_ptr, 0, sem="acquire", scope="sys")
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    s = tl.load(g_ptr + offs, mask=m).to(tl.float32) + tl.load(h_ptr + offs, mask=m).to(tl.float32)
    tl.store(out_ptr + offs, s.to(out_ptr.dtype.element_ty), mask=m)


def hybrid_merge_wait(gpu_routed: torch.Tensor, host_y: torch.Tensor, seq: torch.Tensor,
                      done: torch.Tensor) -> torch.Tensor:
    """:func:`hybrid_merge` once ``done`` (a device view of the slot's host flag) holds the step number in ``seq``."""
    out = torch.empty_like(gpu_routed)
    n = gpu_routed.numel()
    _merge_wait_kernel[(triton.cdiv(n, 2048),)](gpu_routed, host_y, out, seq, done, n, BLOCK=2048, num_warps=4)
    return out


def hybrid_merge(gpu_routed: torch.Tensor, host_y: torch.Tensor) -> torch.Tensor:
    """gpu_routed + the CPU partial, read from the device view of its pinned buffer (once the stream waited for it)."""
    out = torch.empty_like(gpu_routed)
    n = gpu_routed.numel()
    _merge_kernel[(triton.cdiv(n, 2048),)](gpu_routed, host_y, out, n, BLOCK=2048, num_warps=4)
    return out


__all__ = ["fence_lanes", "hybrid_merge", "hybrid_merge_wait", "hybrid_split", "hybrid_split_signal"]
