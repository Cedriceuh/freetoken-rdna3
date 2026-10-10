"""Env-gated decode profiling for bench runs (all off by default; read once, when the profiler is created).

FT_STATS_EVERY=N   every N graph-replayed decode steps, log the host step period vs the GPU time inside the forward
                   (CUDA events) and, with FT_MOE_STATS=1, the MoE offload cache decode miss counters.
FT_PROF_STEPS=N    torch.profiler window over N decode steps starting at decode step FT_PROF_START; per-rank kernel
FT_PROF_START=S    and op tables plus a chrome trace are written to FT_PROF_DIR (default ./ftprof, created if missing).
FT_PROF_STACK=1    with FT_PROF_STEPS: also record Python stacks and input shapes (ops_by_stack / ops_by_shape tables;
                   the chrome trace then ties each kernel to the line that launched it)
FT_PROF_EAGER=1    also count eager decode steps (run with --cuda-graph-max-bs 0): ROCm's torch.profiler does not see
                   the kernels inside a replayed graph, so a per-kernel breakdown needs eager decode (kernel device
                   times are the same; only the launch gaps between them differ).
"""

from __future__ import annotations

import os
import time

import torch

from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger

logger = init_logger(__name__)


class DecodeProfiler:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.every = int(os.environ.get("FT_STATS_EVERY", "0"))
        self.prof_start = int(os.environ.get("FT_PROF_START", "0"))
        self.prof_steps = int(os.environ.get("FT_PROF_STEPS", "0"))
        self.out = os.environ.get("FT_PROF_DIR", "ftprof")
        self.active = bool(self.every or self.prof_steps)
        self.eager = os.environ.get("FT_PROF_EAGER") == "1"
        self.n = 0
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.periods: list[float] = []
        self.t_prev: float | None = None
        self.cur = None
        self.prof = None

    def start(self, use_graph: bool) -> None:
        if not (self.active and use_graph):
            return
        now = time.perf_counter()
        if self.every and self.t_prev is not None and now - self.t_prev < 1.0:  # consecutive decode steps only
            self.periods.append(now - self.t_prev)
        self.t_prev = now
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(self.engine.stream)
        self.cur = (e0, e1)
        if self.prof_steps and self.n == self.prof_start and self.prof is None:
            self.prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                               with_stack=os.environ.get("FT_PROF_STACK") == "1",
                                               record_shapes=os.environ.get("FT_PROF_STACK") == "1")
            self.prof.__enter__()
            logger.warning(f"[ftprof] rank {get_tp_info().rank}: torch.profiler started at decode step {self.n}")

    def end(self, use_graph: bool) -> None:
        if not (self.active and use_graph) or self.cur is None:
            return
        e0, e1 = self.cur
        self.cur = None
        e1.record(self.engine.stream)
        if self.every:  # only the stats window reads them back (and clears them)
            self.events.append((e0, e1))
        self.n += 1
        rank = get_tp_info().rank
        if self.prof is not None and self.prof != "done" and self.n == self.prof_start + self.prof_steps:
            torch.cuda.synchronize(self.engine.device)
            self.prof.__exit__(None, None, None)
            os.makedirs(self.out, exist_ok=True)
            ka = self.prof.key_averages()
            with open(f"{self.out}/rank{rank}_kernels_by_device_time.txt", "w") as f:
                f.write(ka.table(sort_by="self_device_time_total", row_limit=80, max_name_column_width=110))
            with open(f"{self.out}/rank{rank}_ops_by_cpu_time.txt", "w") as f:
                f.write(ka.table(sort_by="self_cpu_time_total", row_limit=50, max_name_column_width=110))
            if os.environ.get("FT_PROF_STACK") == "1":
                with open(f"{self.out}/rank{rank}_ops_by_stack.txt", "w") as f:
                    f.write(self.prof.key_averages(group_by_stack_n=8).table(
                        sort_by="self_device_time_total", row_limit=120, max_name_column_width=60,
                        max_src_column_width=200))
                with open(f"{self.out}/rank{rank}_ops_by_shape.txt", "w") as f:
                    f.write(self.prof.key_averages(group_by_input_shape=True).table(
                        sort_by="device_time_total", row_limit=150, max_name_column_width=60,
                        max_shapes_column_width=80))
            self.prof.export_chrome_trace(f"{self.out}/rank{rank}_trace.json")
            logger.warning(f"[ftprof] rank {rank}: profile of {self.prof_steps} decode steps written to {self.out}")
            self.prof = "done"
        if self.every and self.n % self.every == 0:
            torch.cuda.synchronize(self.engine.device)
            gpu_ms = sum(a.elapsed_time(b) for a, b in self.events) / len(self.events)
            per = self.periods[-len(self.events):]
            host_ms = 1000 * sum(per) / len(per) if per else float("nan")
            msg = (f"[ftprof] rank {rank} decode steps {self.n - len(self.events)}-{self.n}: step period {host_ms:.2f} ms, "
                   f"GPU time inside forward {gpu_ms:.2f} ms ({100 * gpu_ms / host_ms:.0f}% of the period)")
            cache = self.engine.moe_offload_cache
            if cache is not None and getattr(cache, "collect_stats", False):
                ms = cache.decode_miss_stats()
                msg += " | moe " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in ms.items())
                cache.reset_stats()
            logger.warning(msg)
            self.events.clear()
            del self.periods[:-1]  # keep the last period only (the next window reads its own)
