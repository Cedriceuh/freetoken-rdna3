"""FREETOKEN_TRITON_FAST_LAUNCH: the fast path's key must determine Triton's own specialization (two arguments with
the same key specialize alike), and a launch through it must run the kernel Triton's path would run."""

import itertools

import pytest
import torch

triton = pytest.importorskip("triton")


def _hip_specialize(v):
    from triton._C.libtriton import native_specialize_impl
    from triton.backends.amd.compiler import HIPBackend

    return native_specialize_impl(HIPBackend, v, False, True, True)


def test_equal_keys_specialize_alike(monkeypatch):
    from triton import knobs

    from freetoken.kernel.triton.fast_launch import _arg_key

    monkeypatch.setattr(knobs.amd, "use_buffer_ops", True)
    ints = [0, 1, 2, 15, 16, 17, 32, -1, -16, 2**31 - 16, 2**31 - 1, 2**31, 2**31 + 16, -2**31, -2**31 - 1,
            2**32, 2**63 - 1, 2**63, 2**64 - 16]
    tensors = [torch.zeros(256, dtype=dt)[o:] for dt in (torch.uint8, torch.float16, torch.int32, torch.float32)
               for o in (0, 1, 2, 3, 4, 8, 16)]  # pointers 16-byte aligned or not
    values = ints + tensors + [True, False, 0.5, 16.0, None]
    for a, b in itertools.combinations(values, 2):
        if _arg_key(a) == _arg_key(b):
            assert _hip_specialize(a) == _hip_specialize(b), (a, b)


def test_tensor_over_2gib_storage_keys_apart():
    from freetoken.kernel.triton.fast_launch import _arg_key

    small = torch.empty(64, dtype=torch.uint8)
    big = torch.empty(2**31 + 64, dtype=torch.uint8)  # untouched pages: not committed
    assert _arg_key(small) != _arg_key(big[:64])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="a ROCm GPU is required")
def test_fast_launch_runs_the_same_kernel(monkeypatch):
    import triton.language as tl

    import freetoken.kernel.triton.fast_launch as fl

    @triton.jit
    def _axpy(x, y, out, n, a, BLOCK: tl.constexpr, SCALE: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        tl.store(out + o, tl.load(x + o, mask=m) * a + tl.load(y + o, mask=m) * SCALE, mask=m)

    def launch(x, y, n, a, scale, grid_fn=False):
        out = torch.zeros_like(x)
        grid = (lambda META: (triton.cdiv(META["n"], META["BLOCK"]),)) if grid_fn else (triton.cdiv(n, 256),)
        _axpy[grid](x, y, out, n, a, BLOCK=256, SCALE=scale)
        return out

    buf = torch.randn(4100, device="cuda")
    cases = [(buf[:4096], buf[1:4097], 4096, 0.5, 1), (buf[4:4100], buf[:4096], 4000, 2.0, 2),
             (buf[:4096], buf[:4096], 1, 1.0, 1), (buf[3:4099], buf[2:4098], 4096, -1.5, 3)]
    expected = [launch(*c) for c in cases]

    monkeypatch.setattr(fl, "_MODE", "check")
    monkeypatch.setattr(fl, "_installed", False)
    orig = triton.runtime.jit.JITFunction.run
    try:
        assert fl.install()
        for _ in range(3):  # first round fills the fast path's cache, the next ones hit it
            for c, e in zip(cases, expected):
                assert torch.equal(launch(*c), e)
                assert torch.equal(launch(*c, grid_fn=True), e)
        # one kernel per distinct specialization: the misaligned pointers and n == 1 key apart
        assert len(_axpy._ft_fast_static.cache) == len(_axpy.device_caches[torch.cuda.current_device()][0])
    finally:
        triton.runtime.jit.JITFunction.run = orig
