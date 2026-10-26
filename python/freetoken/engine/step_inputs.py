"""FREETOKEN_STEP_INPUTS: a graph-replayed decode step's per-step inputs staged by the host in one pinned buffer, then
one copy to the GPU and one kernel (kernel/triton/step_inputs.py) that writes the captured graph's input buffers,
instead of ~25 small copies, casts and gathers a step (the scheduler's positions, row mapping and write tuple, the
graph's ``copy_from``, the QSA backend's restage). The same values land in the same buffers: the step's bits do not
change.

The batch's tensors point at the graph's buffers (``GraphCaptureBuffer.set_batch``) and at the copied buffer from the
scheduler on; their values arrive when the scheduler's forward launches the copy and the kernel, before anything
reads them on the GPU (the PLE store reads the tokens back right after).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)
ENABLED = os.environ.get("FREETOKEN_STEP_INPUTS", "1") == "1"


class DecodeStepInputs:
    """Two pinned host buffers and two device buffers, alternating by step: a host buffer is rewritten once the copy
    out of it two steps earlier has run, a device buffer once the steps that read it are queued before."""

    def __init__(self, engine, mrope: bool) -> None:
        from freetoken.spec_decode import SPEC_M

        runner = engine.graph_runner
        self.engine, self.mrope = engine, mrope
        self.B = max(runner.graph_bs_list)
        R = self.B * SPEC_M
        o = 0
        self.off_pos, o = o, o + R
        self.off_mrope, o = o, o + 3 * R
        self.off_lin, o = o, o + self.B
        self.off_kv, o = o, o + self.B
        self.off_last, o = o, o + self.B
        o += o % 2  # the int64 words start 8-byte aligned
        self.off_map, o = o // 2, o + 2 * self.B
        self.off_wrow, o = o // 2, o + 2 * self.B
        self.off_wpos, o = o // 2, o + 2 * self.B
        self.words = o
        self.host = [torch.zeros(o, dtype=torch.int32, pin_memory=True) for _ in range(2)]
        self.dev = [torch.zeros(o, dtype=torch.int32, device=engine.device) for _ in range(2)]
        self.copied: list[torch.cuda.Event | None] = [None, None]
        self.k = 0
        backend = engine.attn_backend
        self.qsa = backend if hasattr(backend, "stage_static") else None
        logger.info_rank0(f"decode step inputs: one copy and one kernel a step ({o * 4} bytes staged)"
                          if self.qsa is not None else "decode step inputs: not for this attention backend")

    def applies(self, batch) -> bool:
        """A graph decode step of m rows per padded request (``device_len`` already counting them)."""
        runner = self.engine.graph_runner
        return (batch.is_decode and self.qsa is not None and batch.padded_size <= self.B
                and runner.can_use_cuda_graph(batch) and (batch.padded_size, batch.spec_m) in runner.graph_map
                and all(r.extend_len == batch.spec_m for r in batch.padded_reqs))

    def stage_rows(self, batch, slots: list[int] | None) -> None:
        """Rows of ``batch`` (each request's ``device_len`` counting the step's rows): positions, mrope positions, the
        page-table row, GatedDeltaNet slot (``slots``, else the table row) and K/V length of each padded request; the
        batch's tensors and attention metadata then point at the graph's and this step's buffers."""
        from freetoken.attention.qsa_sparse import QSASparseMetadata

        k = self.k = self.k ^ 1
        if self.copied[k] is not None:
            self.copied[k].synchronize()
        h = self.host[k].numpy()
        h64 = h.view(np.int64)
        reqs = batch.padded_reqs
        B, m = batch.padded_size, batch.spec_m
        R = B * m
        lens_q = [m] * B
        for i, r in enumerate(reqs):
            h[self.off_pos + i * m: self.off_pos + (i + 1) * m] = np.arange(r.cached_len, r.device_len, dtype=np.int32)
            if self.mrope:
                full = r.mrope_positions_full
                if full is not None and r.device_len <= full.shape[1]:
                    cols = full[:, r.cached_len: r.device_len].cpu().numpy()
                else:
                    cols = np.broadcast_to(np.arange(r.cached_len + r.mrope_delta, r.device_len + r.mrope_delta,
                                                     dtype=np.int32), (3, m))
                for t in range(3):
                    h[self.off_mrope + t * R + i * m: self.off_mrope + t * R + (i + 1) * m] = cols[t]
            h64[self.off_map + i] = r.table_idx
            h[self.off_lin + i] = r.table_idx if slots is None else slots[i]
            h[self.off_kv + i] = r.device_len
        h[self.off_last: self.off_last + B] = np.arange(1, B + 1, dtype=np.int32) * m - 1
        dev = self.dev[k]
        self.engine.graph_runner.buffer.set_batch(batch)
        batch.active_table_idx = dev.view(torch.int64)[self.off_map: self.off_map + B]
        qo_indptr = torch.tensor([0] + lens_q, dtype=torch.int32).cumsum_(0).to(torch.int32)
        md = QSASparseMetadata(is_decode=True, last_indices=dev[self.off_last: self.off_last + B],
                               qo_indptr_cpu=qo_indptr, kv_len_cpu=torch.tensor([r.device_len for r in reqs],
                                                                                dtype=torch.int32))
        batch.attn_metadata = md
        self.qsa.stage_static(md, B, m)
        batch.staged_inputs = k

    def stage_writes(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """The scheduler's write tuple (each request's table row, and its ``device_len`` or -1), from this step's
        buffer."""
        k = batch.staged_inputs
        h64 = self.host[k].numpy().view(np.int64)
        n = len(batch.reqs)
        h64[self.off_wrow: self.off_wrow + n] = [r.table_idx for r in batch.reqs]
        h64[self.off_wpos: self.off_wpos + n] = [(r.device_len if r.can_decode else -1) for r in batch.reqs]
        d64 = self.dev[k].view(torch.int64)
        return d64[self.off_wrow: self.off_wrow + n], d64[self.off_wpos: self.off_wpos + n]

    def launch(self, batch, token_pool: torch.Tensor) -> None:
        """The copy of this step's buffer and the kernel that writes the graph's inputs, on the current stream."""
        from freetoken.kernel.triton.step_inputs import decode_inputs

        k = batch.staged_inputs
        self.dev[k].copy_(self.host[k], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.copied[k] = event
        buf = self.engine.graph_runner.buffer
        B, m = batch.padded_size, batch.spec_m
        decode_inputs(self.dev[k], token_pool, self.engine.page_table, buf.input_ids, buf.out_loc, buf.positions,
                      buf.mrope_positions, buf.table_idx, self.qsa.static_targets(),
                      (self.off_pos, self.off_mrope, self.off_lin, self.off_kv, self.off_map), B, m)


__all__ = ["ENABLED", "DecodeStepInputs"]
