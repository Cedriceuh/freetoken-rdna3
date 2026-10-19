"""A shorter host path for Triton launches (FREETOKEN_TRITON_FAST_LAUNCH).

``JITFunction.run`` spends most of a launch's host time before the C launcher: the binder specializes every argument
(on ROCm the AMD backend also asks each tensor's storage size in Python, for its buffer ops), then the cache key, the
knobs and the launch metadata. A short prefill launches ~1,500 kernels, and the host paces it, at ~50 us a launch in
the server. Here a launch computes a cheaper key that determines everything the binder looks at (each tensor's dtype,
16-byte alignment and storage size under 2 GiB, each int's 1 / multiple of 16 / width, the constexpr values, the
options) and, on a hit, calls the same compiled kernel's launcher with the same arguments as ``run`` would: the same
binary, so the same bits. A miss takes Triton's own path, which also compiles, and its kernel is remembered under the
key; an argument type the key does not cover, a pre-run hook or a launch hook always take Triton's path. The launcher
call mirrors Triton 3.8.0's ``HIPLauncher``; another Triton version keeps Triton's path everywhere.

``FREETOKEN_TRITON_FAST_LAUNCH=check`` also runs Triton's binder on every hit and raises if it picks another kernel.
"""

from __future__ import annotations

import inspect
import os

import torch

_MODE = os.environ.get("FREETOKEN_TRITON_FAST_LAUNCH", "1")
_TRITON = "3.8.0"  # the version whose launcher calls the fast path repeats
_installed = False


class _Unsupported(Exception):
    pass


def _arg_key(v):
    """What the binder's specialization of ``v`` depends on (a superset; more keys only means more cache entries)."""
    t = type(v)
    if t is int:
        return (0, v == 1, not v & 15, -0x80000000 <= v <= 0x7FFFFFFF, v >= 0x8000000000000000)
    if isinstance(v, torch.Tensor):
        return (v.dtype, not v.data_ptr() & 15, v.untyped_storage().size() <= 0x7FFFFFFF)
    if t is float:
        return 2
    if t is bool:
        return 3
    if v is None:
        return 4
    if t is tuple:
        return (5,) + tuple(_arg_key(x) for x in v)
    raise _Unsupported


class _Static:
    """Per-JITFunction facts the fast path needs: parameter names, constexpr flags, defaults."""

    __slots__ = ("names", "constexpr", "defaults", "nparams", "cache")

    def __init__(self, fn):
        self.names = [p.name for p in fn.params]
        self.constexpr = [p.is_constexpr for p in fn.params]
        sig = fn.signature.parameters
        self.defaults = [sig[n].default for n in self.names]
        self.nparams = len(self.names)
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.values()):
            raise _Unsupported
        self.cache = {}


def _direct(kernel):
    """The HIP launcher's C entry and its fixed arguments, when the kernel needs no scratch memory: what
    ``HIPLauncher.__call__`` passes it then, without its per-launch Python."""
    launcher = kernel.run
    if (type(launcher).__name__ != "HIPLauncher" or getattr(launcher, "global_scratch_size", 1)
            or getattr(launcher, "profile_scratch_size", 1)):
        return None
    return (launcher.launch, launcher.launch_cooperative_grid, kernel.function, kernel.packed_metadata,
            launcher.warp_size, launcher.arg_annotations, launcher.kernel_signature)


def install() -> bool:
    """Patch ``JITFunction.run`` (once per process). Returns whether the fast path is on."""
    global _installed
    if _MODE not in ("1", "check") or _installed:
        return _installed
    import triton
    from triton import knobs
    from triton.runtime import driver
    from triton.runtime.jit import JITFunction, compute_cache_key

    # Triton 3.8's hooks are chains, empty unless a profiler adds one (then each launch takes Triton's path)
    enter, leave = knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook
    if triton.__version__ != _TRITON or not hasattr(enter, "calls") or not hasattr(leave, "calls"):
        return False
    orig_run = JITFunction.run
    check = _MODE == "check"
    get_device = driver.active.get_current_device
    get_stream = driver.active.get_current_stream
    empty = inspect.Parameter.empty

    def run(self, *args, grid, warmup, **kwargs):
        if warmup or self.pre_run_hooks or enter.calls or leave.calls:
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        st = self.__dict__.get("_ft_fast_static")
        if st is None:
            try:
                st = _Static(self)
            except _Unsupported:
                st = False
            self._ft_fast_static = st
        if st is False:
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        n = len(args)
        if n > st.nparams:
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        vals = list(args)
        if n < st.nparams:
            opts = dict(kwargs)
            for i in range(n, st.nparams):
                name = st.names[i]
                if name in opts:
                    vals.append(opts.pop(name))
                else:
                    d = st.defaults[i]
                    if d is empty:
                        return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
                    vals.append(d)
        else:
            opts = kwargs
        device = get_device()
        try:
            # a constexpr keyed with its type (True == 1 == 1.0 as dict keys, not for the compiler); a float by its repr
            # (-0.0 == 0.0, and NaN never equals itself)
            key = (device, tuple(((float, repr(v)) if type(v) is float else (type(v), v)) if c else _arg_key(v)
                                 for v, c in zip(vals, st.constexpr)),
                   tuple((k, type(v), v) for k, v in opts.items()) if opts else ())
            kernel = st.cache.get(key)
        except (_Unsupported, TypeError):  # TypeError: an unhashable constexpr or option
            return orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        if kernel is None:
            kernel = orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
            if kernel is not None:
                st.cache[key] = (kernel, _direct(kernel))
            return kernel
        kernel, direct = kernel
        if check:
            kw = dict(kwargs)
            kw["debug"] = kw.get("debug", self.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            kernel_cache, kernel_key_cache, _, _, binder = self.device_caches[device]
            _, spec, options = binder(*args, **kw)
            if kernel_cache.get(compute_cache_key(kernel_key_cache, spec, options)) is not kernel:
                raise RuntimeError(f"FREETOKEN_TRITON_FAST_LAUNCH=check: {self.fn.__qualname__} picked another kernel")
        if callable(grid):
            grid = grid(dict(zip(st.names, vals)))
        g = len(grid)
        if direct is not None:
            launch, coop, function, packed, warp_size, annotations, signature = direct
            launch(coop, grid[0], grid[1] if g > 1 else 1, grid[2] if g > 2 else 1, get_stream(device), function,
                   None, None, packed, None, None, None, warp_size, annotations, signature, tuple(vals))
        else:
            kernel.run(grid[0], grid[1] if g > 1 else 1, grid[2] if g > 2 else 1, get_stream(device), kernel.function,
                       kernel.packed_metadata, None, None, None, *vals)
        return kernel

    JITFunction.run = run
    _installed = True
    return True
