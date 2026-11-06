from __future__ import annotations

import ctypes
import functools
import gc
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.spec_decode import SHALLOW_M, SPEC_BS_DEEP, SPEC_BS_MAX, SPEC_M, SPEC_MS, rows_of
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # flat [3 * rows] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # GDN query indptr = arange(bs+1) * rows per request; one constant per captured row count, filled once.
    fla_cu_seqlens: dict

    @classmethod
    def init(cls, bs: int, device: torch.device, mrope: bool = False, row_counts: tuple[int, ...] = (1,)) -> GraphCaptureBuffer:
        rows = bs * max(row_counts)  # token rows: up to max(row_counts) per request
        return GraphCaptureBuffer(
            input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            out_loc=torch.zeros(rows, dtype=torch.int32, device=device),
            positions=torch.zeros(rows, dtype=torch.int32, device=device),
            # flat: each row count views it as a contiguous [3, rows] (the rope kernels' layout), so no attention layer
            # copies a strided [3, rows] slice of a [3, max_rows] buffer
            mrope_positions=(
                torch.zeros(3 * rows, dtype=torch.int32, device=device) if mrope else None
            ),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens={m: torch.arange(bs + 1, dtype=torch.int32, device=device) * m for m in row_counts},
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        bs, m = batch.padded_size, rows_of(batch)
        _slice, _rows = slice(bs), slice(bs * m)
        batch.input_ids = self.input_ids[_rows]
        batch.out_loc = self.out_loc[_rows]
        batch.positions = self.positions[_rows]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[: 3 * bs * m].view(3, bs * m)
        batch.linear_table_idx = self.table_idx[_slice]
        # GDN metadata reads the persistent cu_seqlens (constant arange * rows) and the persistent table_idx slot map,
        # so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[m][: bs + 1], cache_indices=self.table_idx[_slice]
        )
        pool = get_global_ctx().linear_state_pool
        if batch.phase == "verify" and pool is not None:
            batch.fla_metadata.verify_scratch = pool.verify_scratch

    def copy_from(self, batch: Batch) -> None:
        rows = batch.padded_size * rows_of(batch)
        _slice, _rows = slice(batch.padded_size), slice(rows)
        self.input_ids[_rows] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_rows] = batch.out_loc
        self.positions[_rows] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[: 3 * rows].view(3, rows).copy_(batch.mrope_positions)
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx
        elif batch.fla_metadata is not None:  # a batch the scheduler staged no slots for: its GDN metadata's
            self.table_idx[_slice] = batch.fla_metadata.cache_indices


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    explicit = cuda_graph_max_bs is not None
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []
    if explicit and cuda_graph_max_bs <= 8:
        # a few concurrent requests (e.g. --cuda-graph-max-bs 4): capture every size, so 3 requests do not run
        # padded to 4 (the padding row routes to extra experts: more cache misses, slower than 4 real requests)
        return list(range(1, cuda_graph_max_bs + 1))

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


@functools.cache
def _cu_graph_upload():
    if sys.platform != "linux" or torch.version.hip is not None:
        return None
    fn = ctypes.CDLL("libcuda.so.1").cuGraphUpload
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _upload_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> None:
    """Upload the graph at capture time.

    Otherwise the first replay pushes every node behind the graph's own stream WAIT, and a large graph then blocks cudaGraphLaunch until the WAIT is released. PLE wait-sync releases it only after the launch returns, so that deadlocks.
    """
    upload = _cu_graph_upload()
    if upload is None:
        return
    rc = upload(graph.raw_cuda_graph_exec(), stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"cuGraphUpload failed with CUresult {rc}")
    stream.synchronize()


def capture_shapes(ms: tuple[int, ...], bs_list: list[int], bs_max: int, bs_deep: int) -> list[tuple[int, int]]:
    """The (rows per request, batch size) graphs to capture. Steps of more than ``bs_max`` requests run ``ms[0]`` rows
    (rows_for), so the other row counts only need the batch sizes up to ``bs_max``'s padded one; past ``bs_deep``
    requests at most SHALLOW_M rows each, so the deeper row counts stop at its padded batch size."""
    verify_bs = next((bs for bs in bs_list if bs >= bs_max), max(bs_list))
    deep_bs = next((bs for bs in bs_list if bs >= bs_deep), max(bs_list))
    return [(m, bs) for m in ms for bs in bs_list if m == ms[0] or bs <= (verify_bs if m <= SHALLOW_M else deep_bs)]


def sizing_shapes(shapes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The (rows per request, batch size) shapes to run eagerly before any capture. The spec_decode buffers are sized
    by the eager forwards: the head's streams once, by the first one's rows; the per-request rollback buffers grow with
    the requests, freeing the old ones under any graph captured before. So the most rows, then the verify shape with
    the most requests."""
    verify = [s for s in shapes if s[0] > 1]
    if not verify:
        return []
    most_rows = max(shapes, key=lambda s: (s[0] * s[1], s[1]))
    most_reqs = max(verify, key=lambda s: (s[1], s[0] * s[1]))
    return list(dict.fromkeys([most_rows, most_reqs]))


class GraphRunner:
    """CUDA graphs of ``forward`` over padded batches of ``rows_per_req`` rows per request, one graph per batch size; ``replay`` returns the real requests' rows of each output."""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        forward: Callable[[], Any],
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        *,
        rows_per_req: int = 1,
        phase: Literal["decode", "verify"] = "decode",
        ctx_override: Dict[str, Any] | None = None,
        batch_hook: Callable[[Batch], None] | None = None,
        pool: Any = None,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        # the padded rows replay the dummy request, so it must carry rows_per_req rows
        assert rows_per_req == 1 or dummy_req.extend_len == rows_per_req, (rows_per_req, dummy_req.extend_len)
        self.forward = forward
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        self.rows_per_req = rows_per_req
        self.phase = phase
        self.ctx_override = ctx_override or {}
        self.batch_hook = batch_hook
        self.graph_pool = pool
        self.out_dtype = out_dtype
        self._out: List[torch.Tensor] | None = None
        self._out_is_tuple = False
        self._out_rows: Dict[tuple[int, int], tuple[int, ...]] = {}
        # rows per request the graphs are captured for: spec_decode's row counts on the decode family (one graph set
        # each; (1,) without it), rows_per_req on a verify family. The GDN metadata lives in the capture buffer.
        self.row_counts = SPEC_MS if phase == "decode" and rows_per_req == 1 else (rows_per_req,)
        self._capture_graphs(max_seq_len)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _store_output(self, key: tuple[int, int], out: Any) -> None:
        self._out_is_tuple = isinstance(out, tuple)
        outs = out if self._out_is_tuple else (out,)
        if self._out is None:
            # the largest batch is captured first, so its outputs size the buffers
            assert self.out_dtype is None or len(outs) == 1, "out_dtype applies to a single output"
            self._out = [torch.empty_like(o, dtype=self.out_dtype or o.dtype) for o in outs]
        for buf, o in zip(self._out, outs, strict=True):
            buf[: o.shape[0]] = o
        self._out_rows[key] = tuple(o.shape[0] for o in outs)

    def _capture_graphs(self, max_seq_len: int):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[tuple[int, int], torch.cuda.CUDAGraph] = {}  # (batch size, rows per request)
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")
        if self.row_counts == SPEC_MS and SPEC_M > 1 and SPEC_BS_MAX > self.max_graph_bs:
            # the verify steps' rollback buffers are sized by the captures: a larger eager verify step would move
            # them away from the addresses the graphs write
            raise ValueError(f"FREETOKEN_SPEC_BS_MAX={SPEC_BS_MAX} exceeds the largest CUDA graph batch size "
                             f"({self.max_graph_bs}): lower it or raise --cuda-graph-max-bs")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list, rows_per_req=self.rows_per_req)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, self.device, mrope=self.mrope, row_counts=self.row_counts)
        self._reset_moe_offload_cache()

        # spec_decode: one graph per (batch size, rows per request); the capture batch carries m rows per request.
        # The most rows first (then the most rows per request).
        shapes = capture_shapes(self.row_counts, self.graph_bs_list, SPEC_BS_MAX, SPEC_BS_DEEP)
        pbar = tqdm(
            sorted(shapes, key=lambda s: (s[0] * s[1], s[0]), reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="graph",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        if self.phase == "decode":
            for m, bs in sizing_shapes(shapes):
                self._eager_shape(m, bs)
        pool = self.graph_pool
        for m, bs in pbar:
            # the dummy request carries rows_per_req rows; a spec_decode row count widens it for this capture
            self.dummy_req.device_len += m - self.rows_per_req
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase=self.phase, rows_per_req=self.rows_per_req)
            batch.padded_reqs = batch.reqs
            batch.spec_m = m if self.phase == "decode" else 1
            if self.batch_hook is not None:
                self.batch_hook(batch)
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch, **self.ctx_override):
                self._store_output((bs, m), self.forward())
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._store_output((bs, m), self.forward())
                _upload_graph(graph, self.stream)
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[(bs, m)] = graph
            self.dummy_req.device_len -= m - self.rows_per_req
        self.graph_pool = pool

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _eager_shape(self, m: int, bs: int) -> None:
        """One eager decode forward of ``bs`` requests x ``m`` rows on the dummy request (sizes the spec buffers)."""
        self.dummy_req.device_len += m - 1
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
        batch.padded_reqs = batch.reqs
        batch.spec_m = m
        self.attn_backend.prepare_for_capture(batch)
        self.buffer.set_batch(batch)
        dummy_slot = (self.dummy_req.linear_slot_idx
                      if self.dummy_req.linear_slot_idx is not None
                      else self.dummy_req.table_idx)
        self.buffer.table_idx[:bs].fill_(dummy_slot)
        with get_global_ctx().forward_batch(batch, **self.ctx_override):
            self.forward()
        self._reset_moe_offload_cache()
        self.dummy_req.device_len -= m - 1

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        m = rows_of(batch)
        return (
            batch.phase == self.phase
            and batch.rows_per_req == self.rows_per_req
            and batch.size <= self.max_graph_bs
            and any(gm == m and bs >= batch.size for bs, gm in self.graph_map)
        )

    def replay(self, batch: Batch) -> Any:
        assert self.can_use_cuda_graph(batch) and self._out is not None
        padded, m = batch.padded_size, rows_of(batch)
        g = self.graph_map[(padded, m)]
        if getattr(batch, "staged_inputs", None) is None:  # else the step-inputs kernel wrote them (step_inputs.py)
            self.buffer.copy_from(batch)
            self.attn_backend.prepare_for_replay(batch)
        g.replay()
        # a spec_decode verify step views its logits per padded request ([padded_size, m, V]); any other step keeps
        # its real requests' rows
        whole = self.phase == "decode" and m > 1
        outs = tuple(
            buf[: rows if whole else rows // padded * batch.size]
            for buf, rows in zip(self._out, self._out_rows[(padded, m)], strict=True)
        )
        return outs if self._out_is_tuple else outs[0]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors (input/out_loc/positions/...) and the output buffers (_out).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self._out = None
        gc.collect()
