"""Disk-backed PLE table (--ple-backend disk): the C++ store hashes n-gram windows and batch-reads rows from the checkpoint's fp8 shard tensors into pinned staging; the captured ``lookup`` is a fixed-shape H2D copy + dequant.

Hash windows are pure functions of ``req.input_ids`` + ``device_len`` (prefix hits, restores and COW forks need no bookkeeping); the decode input token lives device-side under overlap scheduling and is read back here.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import safetensors
import torch

from freetoken.mm import MM_PAD_SHIFT_VALUE, restore_placeholder

from freetoken.core import Batch

# FREETOKEN_PLE_FILL_TIMING=N (bench runs): every N decode steps, log the host's mean wait for the step's token
# readback and the mean fill time after it (wait-sync: the captured graph waits on the fill's flag at the PLE layer;
# launch-gating: the GPU idles through both, the forward launches after the fill)
_FILL_TIMING = int(os.environ.get("FREETOKEN_PLE_FILL_TIMING", "0"))
_FILL_ACC: dict = {}


def _fill_tally(m: int, wait: float, fill: float) -> None:
    w, f, n = _FILL_ACC.get(m, (0.0, 0.0, 0))
    _FILL_ACC[m] = (w + wait, f + fill, n + 1)
    if sum(v[2] for v in _FILL_ACC.values()) >= _FILL_TIMING:
        from freetoken.utils import init_logger

        for rows, (w, f, n) in sorted(_FILL_ACC.items()):
            init_logger(__name__).info_rank0(f"[ple-fill] {rows} rows, {n} steps: readback wait {w / n * 1e3:.2f} ms, "
                                             f"fill {f / n * 1e3:.2f} ms")
        _FILL_ACC.clear()
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.spec_decode import rows_of
from freetoken.utils import init_logger

from .weight import (
    _PLE_SCALE_SUFFIX,
    _PLE_SHARD_RE,
    _PLE_ST_DTYPE,
    _ple_table_files,
    _safetensors_header,
)

_IO_URING_ENV = "FREETOKEN_PLE_IO_URING"
_SYNC_ENV = "FREETOKEN_PLE_SYNC"  # auto | wait | gate

logger = init_logger(__name__)


def _context(ids: torch.Tensor, position: int, eos: int) -> list[int]:
    """The two token ids before ``position``; eos pads past the start."""
    return [int(ids[position - 2]) if position >= 2 else eos,
            int(ids[position - 1]) if position >= 1 else eos]


@dataclass(frozen=True)
class PleRowSource:
    """On-disk row layout: equal extents, row i of an extent at ``base + i * row_stride`` (a repacked flat file is one extent with its own stride)."""

    paths: list[str]
    extent_file: list[int]
    extent_base: list[int]
    rows_per_extent: int
    row_bytes: int
    row_stride: int
    scale: float
    # EXL3 n-gram rings (exllamav3 exl3_ngram_trellis): K bits per element, row_dim elements per row, a bias per hash head;
    # exl3_bits == 0 is the fp8 table, one byte per element
    exl3_bits: int = 0
    row_dim: int = 0
    head_bias: torch.Tensor | None = None

    @property
    def total_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent


def source_from_safetensors(folder: str) -> PleRowSource:
    """Map the checkpoint's ``ngram_embedding.shard_<i>`` tensors in place: one extent per shard, no copy."""
    rows = cols = 0
    scale: torch.Tensor | None = None
    paths: list[str] = []
    path_idx: dict[str, int] = {}
    shards: dict[int, tuple[int, int]] = {}
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE shard {key} has dtype {meta['dtype']}, expected {_PLE_ST_DTYPE}")
            if rows and tuple(meta["shape"]) != (rows, cols):
                raise ValueError(f"PLE shard {key} is {meta['shape']}, expected {[rows, cols]}")
            rows, cols = meta["shape"]
            if path not in path_idx:
                path_idx[path] = len(paths)
                paths.append(path)
            idx = int(match.group("shard"))
            if idx in shards:
                raise ValueError(f"duplicate PLE shard {idx} in {path}")
            shards[idx] = (path_idx[path], base + meta["data_offsets"][0])
    if sorted(shards) != list(range(len(shards))) or not shards:
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(shards)[:8]}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")
    order = [shards[i] for i in range(len(shards))]
    return PleRowSource(paths, [f for f, _ in order], [b for _, b in order], rows, cols, cols, float(scale))


_EXL3_NGRAM_FILE = "ngram_embedding.safetensors"
# split in shards (``shard_<i>.trellis``) or one ``trellis`` tensor, depending on the exllamav3 release that wrote it
_EXL3_SHARD_RE = re.compile(r"\.ple\.ple_embedding\.ngram_embedding(?:\.shard_(?P<shard>\d+))?\.trellis$")
_EXL3_HEAD_BIAS_SUFFIX = ".ple.ple_embedding.ngram_embedding.head_bias"


def source_from_exl3_ngram(path: str) -> PleRowSource:
    """exllamav3's ``ngram_embedding.safetensors``: ``shard_<i>.trellis`` (or a single ``trellis``) int16
    ``[rows, 1 + dim*K/16]`` rings mapped in place, plus the per-head bias the decode adds."""
    header, base = _safetensors_header(path)
    meta = header.get("__metadata__") or {}
    if meta.get("format") != "exl3_ngram_trellis" or meta.get("codebook", "mul1") != "mul1":
        raise ValueError(f"{path}: not an exllamav3 mul1 n-gram table (metadata {meta})")
    K, dim = int(meta["K"]), int(meta["row_dim"])
    rows = words = 0
    shards: dict[int, int] = {}
    head_bias = None
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if key.endswith(_EXL3_HEAD_BIAS_SUFFIX):
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                head_bias = f.get_tensor(key).to(torch.float16)
            continue
        match = _EXL3_SHARD_RE.search(key)
        if match is None:
            continue
        if info["dtype"] != "I16" or (rows and tuple(info["shape"]) != (rows, words)):
            raise ValueError(f"EXL3 n-gram shard {key} is {info['dtype']} {info['shape']}, expected I16 {[rows, words]}")
        rows, words = info["shape"]
        idx = int(match.group("shard") or 0)
        if idx in shards:
            raise ValueError(f"duplicate EXL3 n-gram shard {idx} in {path} ({key})")
        shards[idx] = base + info["data_offsets"][0]
    if not shards or sorted(shards) != list(range(len(shards))):
        raise ValueError(f"EXL3 n-gram shard indices are not contiguous 0..N-1: {sorted(shards)[:8]}")
    if head_bias is None or head_bias.shape[1] != dim or (words - 1) * 16 != dim * K:
        raise ValueError(f"EXL3 n-gram table: {words} words per row do not hold {dim} x {K} bits, or no head_bias")
    order = [shards[i] for i in range(len(shards))]
    return PleRowSource([path], [0] * len(order), order, rows, 2 * words, 2 * words, 1.0,
                        exl3_bits=K, row_dim=dim, head_bias=head_bias)


def resolve_row_source(folder: str) -> PleRowSource:
    """Pick the row source for a checkpoint; the seam where a repacked format would plug in."""
    exl3 = os.path.join(folder, _EXL3_NGRAM_FILE)
    if os.path.exists(exl3):
        return source_from_exl3_ngram(exl3)
    return source_from_safetensors(folder)


class DiskRowTable:
    """``PLETableBackend`` whose rows are read from disk per fill (--ple-backend disk)."""

    # every rank stages all hash heads from the host store (hashed from the full constants), so
    # NGramEmbedding neither rebases nor all-gathers at TP>1
    serves_all_heads = True

    def __init__(
        self,
        source: PleRowSource,
        hash_constants: dict,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from freetoken.kernel.row_store import PleStore

        self.num_rows = source.total_rows
        self.head_dim = source.row_dim or source.row_bytes  # fp8: one byte per element
        self._row_bytes = source.row_bytes
        self._exl3_bits = source.exl3_bits
        self.dtype = dtype
        self.heads = int(hash_constants["num_ngram_heads"])
        self.scale = source.scale
        self.eos_token_id = int(hash_constants["eos_token_id"])
        self.image_token_id = hash_constants.get("image_token_id")
        sizes = [int(x) for x in hash_constants["per_head_vocab_sizes"]]
        offsets = [int(x) for x in hash_constants["per_head_offsets"]]
        need = max(o + s for o, s in zip(offsets, sizes))
        if need > source.total_rows:
            raise ValueError(
                f"PLE row source holds {source.total_rows} rows but the hash addresses {need}; incomplete checkpoint?"
            )
        self._store = PleStore(
            paths=list(source.paths),
            extent_file=list(source.extent_file),
            extent_base=list(source.extent_base),
            rows_per_extent=source.rows_per_extent,
            row_bytes=source.row_bytes,
            row_stride=source.row_stride,
            multipliers=[int(x) for x in hash_constants["layer_multipliers"]],
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            eos_token_id=self.eos_token_id,
            use_io_uring=os.getenv(_IO_URING_ENV, "1") != "0",
        )
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._token_bytes = self.heads * self._row_bytes
        self._head_bias = None if source.head_bias is None else source.head_bias.to(self._device)
        if self._head_bias is not None and self._head_bias.shape[0] != self.heads:
            raise ValueError(f"EXL3 n-gram table has {self._head_bias.shape[0]} head biases, the model hashes {self.heads} heads")
        # allocated up front: pinned alloc inside stream capture is illegal; one replay consumes it at a time
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * self._token_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()  # padded decode lanes read whatever sits here
        # outlives any one graph: a cache rebuild recaptures against the same pointer
        self._graph_dev = torch.empty(
            max_graph_rows * self._token_bytes, dtype=torch.uint8, device=self._device
        )
        eager_bytes = max_extend_tokens * self._token_bytes
        self._eager_pinned = alloc_pinned_tensor(eager_bytes, dtype=torch.uint8)
        self._eager_pinned.zero_()  # the warmup prefill stages nothing and reads whatever sits here
        self._eager_dev = torch.empty(eager_bytes, dtype=torch.uint8, device=self._device)
        # probe picks flag-sync (graph WAITs at the consume, host fills then signals) or launch-gating
        from freetoken.kernel.row_store import probe_wait_sync

        self._wait_sync = probe_wait_sync(os.getenv(_SYNC_ENV, "auto"), self._device)
        # one flag for all graphs: the readback event orders a fill after the previous graph, so signals never overlap
        self._flag = alloc_pinned_tensor(1, dtype=torch.int64)
        self._flag.zero_()
        self._token_readback = alloc_pinned_tensor(max_graph_rows, dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        sync = "wait-sync" if self._wait_sync else "launch-gating"
        logger.info_rank0(f"PLE disk backend: {self._store.io_backend()}, {sync}")

    # ---------------- host side (engine thread, before the forward launches) ----------------

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        """Stage per-request token runs (two context ids, then the new tokens) in batch order."""
        pinned = self._graph_pinned if graph else self._eager_pinned
        offset = 0
        for run in runs:
            self._store.stage(run.data_ptr(), run.numel() - 2, pinned.data_ptr() + offset * self._token_bytes)
            offset += run.numel() - 2
        self._store.flush(self._flag.data_ptr() if graph and self._wait_sync else 0)

    def _ple_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.image_token_id is None:
            return input_ids
        return restore_placeholder(input_ids, self.image_token_id)

    def _ple_context(self, ids: torch.Tensor, position: int) -> list[int]:
        if self.image_token_id is None:
            return _context(ids, position, self.eos_token_id)
        return [self.image_token_id if t >= MM_PAD_SHIFT_VALUE else t for t in _context(ids, position, self.eos_token_id)]

    def host_fill_batch(self, batch: Batch, use_graph: bool):
        """Stage this batch's rows; returns the post-dispatch fill callable under flag-sync, else None."""
        eos = self.eos_token_id
        if batch.is_decode:
            reqs = list(batch.reqs)
            m = rows_of(batch)  # rows per request (spec_decode): the request's token, then its drafts
            if use_graph and self._wait_sync:
                bs = batch.padded_size
                self._token_readback[: bs * m].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(torch.cuda.current_stream(self._device))

                def _complete() -> None:
                    try:
                        t0 = time.perf_counter() if _FILL_TIMING else 0.0
                        self._readback_event.synchronize()
                        t1 = time.perf_counter() if _FILL_TIMING else 0.0
                        tokens = self._token_readback[: bs * m].to(torch.int64).tolist()
                        runs = [torch.tensor([*self._ple_context(r.input_ids, r.device_len - 1), *tokens[i * m:(i + 1) * m]],
                                             dtype=torch.int64)
                                for i, r in enumerate(reqs)]
                        self.fill(runs, graph=True)
                        if _FILL_TIMING:
                            _fill_tally(m, t1 - t0, time.perf_counter() - t1)
                    except BaseException:
                        from freetoken.kernel.row_store import signal

                        # unblock the stream before surfacing; the step's output is discarded
                        signal(self._flag)
                        raise

                return _complete
            # launch-gating: this D2H is the step's readback and orders the fill after sampling
            t0 = time.perf_counter() if _FILL_TIMING else 0.0
            tokens = batch.input_ids.to("cpu").to(torch.int64).tolist()
            t1 = time.perf_counter() if _FILL_TIMING else 0.0
            runs = [torch.tensor([*self._ple_context(r.input_ids, r.device_len - 1), *tokens[i * m:(i + 1) * m]],
                                 dtype=torch.int64)
                    for i, r in enumerate(reqs)]
            self.fill(runs, graph=use_graph)
            if _FILL_TIMING:
                _fill_tally(m, t1 - t0, time.perf_counter() - t1)
            return None
        runs = [
            torch.cat((
                torch.tensor(self._ple_context(req.input_ids, req.cached_len), dtype=torch.int64),
                self._ple_ids(req.input_ids[req.cached_len : req.device_len]).to(torch.int64),
            ))
            for req in batch.padded_reqs
        ]
        self.fill(runs, graph=False)
        return None

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one dispatch: stage on enter, run the deferred fill+signal on exit."""
        deferred = self.host_fill_batch(batch, use_graph)
        yield
        # no try/finally: a failed launch leaves no WAIT pending, so the fill must not run
        if deferred is not None:
            deferred()

    # ---------------- device side (PLETableBackend protocol) ----------------

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = row_ids.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self._wait_sync:
            from freetoken.kernel.row_store import wait_reset

            wait_reset(torch.cuda.current_stream(self._device), self._flag)
        pinned, dev = (
            (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned, self._eager_dev)
        )
        nbytes = rows * self._token_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        if self._exl3_bits:
            from freetoken.kernel.triton.exl3 import exl3_ngram_dequant

            packed = dev[:nbytes].view(torch.int16).view(rows * self.heads, self._row_bytes // 2)
            values = exl3_ngram_dequant(packed, self._exl3_bits, self._head_bias,
                                        torch.empty((rows * self.heads, self.head_dim), dtype=self.dtype, device=self._device))
        else:
            values = dev[:nbytes].view(torch.float8_e4m3fn).to(self.dtype)
            if self.scale != 1.0:
                values = values * self.scale
        values = values.view(*row_ids.shape[:-1], -1)
        if out is None:
            return values
        out.copy_(values)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None
