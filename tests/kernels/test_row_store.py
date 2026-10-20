"""The hash-agnostic ``RowStore`` (``kernel/csrc/row_store``): byte-exact staged rows across extents and files,
dedup of repeated ids, a destination stride, and bounds checks before any I/O. The PLE hash wrapper on top of
it is covered by ``tests/models/qwen4_exp/test_ple_disk.py``."""

from __future__ import annotations

import pytest
import torch

_row_store = pytest.importorskip("freetoken.kernel._row_store")

ROW_BYTES = 264  # deliberately unaligned rows exercise reads across page boundaries


def _table(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, ROW_BYTES), dtype=torch.uint8, generator=gen)


def _stage(store, ids: torch.Tensor, dst_stride: int = 0) -> torch.Tensor:
    stride = dst_stride or store.row_bytes
    staging = torch.zeros(ids.numel() * stride, dtype=torch.uint8)
    store.stage_rows(ids.data_ptr(), ids.numel(), staging.data_ptr(), dst_stride)
    store.flush(0)
    return staging.view(ids.numel(), stride)


@pytest.mark.parametrize("use_io_uring", [True, False])
def test_rows_come_back_bitwise_across_extents_and_files(tmp_path, use_io_uring):
    # two files; file 0 carries extents 0 and 2 behind an unrelated header, file 1 carries extent 1
    per_extent = 1000
    table = _table(3 * per_extent, seed=1)
    header = b"safetensors-like header, not a row" + bytes(4096 - 34)
    f0 = tmp_path / "shard0.bin"
    f1 = tmp_path / "shard1.bin"
    f0.write_bytes(header + table[:per_extent].numpy().tobytes() + table[2 * per_extent :].numpy().tobytes())
    f1.write_bytes(table[per_extent : 2 * per_extent].numpy().tobytes())
    store = _row_store.RowStore(
        paths=[str(f0), str(f1)],
        extent_file=[0, 1, 0],
        extent_base=[len(header), 0, len(header) + per_extent * ROW_BYTES],
        rows_per_extent=per_extent,
        row_bytes=ROW_BYTES,
        row_stride=ROW_BYTES,
        use_io_uring=use_io_uring,
    )
    assert store.total_rows == 3 * per_extent and store.row_bytes == ROW_BYTES

    ids = torch.tensor([0, 999, 1000, 1999, 2000, 2999, 1234, 1234, 42, 0], dtype=torch.int64)
    got = _stage(store, ids)
    assert torch.equal(got, table[ids])

    # a wider destination stride leaves the gap untouched
    got = _stage(store, ids, dst_stride=ROW_BYTES + 40)
    assert torch.equal(got[:, :ROW_BYTES], table[ids]) and int(got[:, ROW_BYTES:].sum()) == 0

    # a fill larger than the reader's in-flight window pipelines through
    many = torch.randint(0, 3 * per_extent, (5000,), dtype=torch.int64, generator=torch.Generator().manual_seed(2))
    assert torch.equal(_stage(store, many), table[many])

    # flushing nothing still signals the flag
    flag = torch.zeros(1, dtype=torch.int64)
    store.flush(flag.data_ptr())
    assert int(flag[0]) == 1


@pytest.mark.parametrize("rings", ["1", "4"])
@pytest.mark.parametrize("use_io_uring", [True, False])
def test_rows_across_pages_come_back_bitwise_on_split_fills(tmp_path, monkeypatch, rings, use_io_uring):
    # with direct I/O a row across a page boundary reads its two pages apart (one read across two 4 KiB units slows
    # the whole batch); a fill of >= 256 rows splits over FREETOKEN_ROW_STORE_RINGS io_uring rings
    monkeypatch.setenv("FREETOKEN_ROW_STORE_RINGS", rings)
    crosses = lambda i: (i * ROW_BYTES) // 4096 != ((i + 1) * ROW_BYTES - 1) // 4096
    rows = next(n for n in range(8000, 9000) if crosses(n - 1))  # the last row's second page is the file's partial tail
    table = _table(rows, seed=4)
    path = tmp_path / "t.bin"
    path.write_bytes(table.numpy().tobytes())
    store = _row_store.RowStore(paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=rows,
                                row_bytes=ROW_BYTES, row_stride=ROW_BYTES, use_io_uring=use_io_uring)
    if "O_DIRECT" not in store.io_backend():
        pytest.skip(f"no direct I/O under {tmp_path}: rows are read whole")
    assert (" x4" in store.io_backend()) == (rings == "4" and store.io_backend().startswith("io_uring"))

    # only rows across pages: every one needs two reads in flight at once
    crossing = torch.tensor([i for i in range(rows) if crosses(i)], dtype=torch.int64)
    assert crossing.numel() >= 256 and int(crossing[-1]) == rows - 1
    assert torch.equal(_stage(store, crossing), table[crossing])
    # mixed, with duplicates, larger than every ring's window
    gen = torch.Generator().manual_seed(5)
    many = torch.randint(0, rows, (6000,), dtype=torch.int64, generator=gen)
    many[:500] = crossing.repeat(2)[:500]
    assert torch.equal(_stage(store, many), table[many])
    # below the split threshold: one ring
    assert torch.equal(_stage(store, many[:100]), table[many[:100]])


@pytest.mark.parametrize("use_io_uring", [True, False])
def test_ram_cache_serves_rows_read_before_bitwise(tmp_path, use_io_uring):
    rows = 3000
    table = _table(rows, seed=6)
    path = tmp_path / "t.bin"
    path.write_bytes(table.numpy().tobytes())
    make = lambda cache_rows: _row_store.RowStore(
        paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=rows, row_bytes=ROW_BYTES,
        row_stride=ROW_BYTES, use_io_uring=use_io_uring, cache_rows=cache_rows)
    store = make(1024)
    assert store.cache_rows == 1024 and "RAM cache of 1024 rows" in store.io_backend()
    gen = torch.Generator().manual_seed(8)
    first = torch.randperm(rows, generator=gen)[:500]
    assert torch.equal(_stage(store, first), table[first]) and store.cache_hits == 0
    # the same rows again come from the cache, mixed with new ones and duplicates
    again = torch.cat([first[:300], torch.randperm(rows, generator=gen)[:400], first[:50]])
    hits_before = store.cache_hits
    assert torch.equal(_stage(store, again), table[again])
    assert store.cache_hits - hits_before >= 350  # first[:300] and its 50 repeats, at least
    # a cache much smaller than the rows read evicts, and stays bitwise
    small = make(16)
    many = torch.randint(0, rows, (4000,), dtype=torch.int64, generator=gen)
    for _ in range(3):
        assert torch.equal(_stage(small, many), table[many])
    assert small.cache_rows == 16 and make(0).cache_rows == 0


def test_geometry_and_bounds_are_checked_up_front(tmp_path):
    table = _table(64, seed=3)
    path = tmp_path / "t.bin"
    path.write_bytes(table.numpy().tobytes())
    make = lambda **kw: _row_store.RowStore(
        paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=64,
        row_bytes=ROW_BYTES, row_stride=ROW_BYTES, use_io_uring=False, **kw,
    )
    store = make()
    ids = torch.tensor([3, 64], dtype=torch.int64)
    staging = torch.zeros(2 * ROW_BYTES, dtype=torch.uint8)
    with pytest.raises(IndexError):
        store.stage_rows(ids.data_ptr(), 2, staging.data_ptr(), 0)
    with pytest.raises(RuntimeError, match="overlap"):
        store.stage_rows(ids[:1].data_ptr(), 1, staging.data_ptr(), ROW_BYTES - 1)
    with pytest.raises(RuntimeError, match="extent needs"):
        _row_store.RowStore(paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=65,
                            row_bytes=ROW_BYTES, row_stride=ROW_BYTES, use_io_uring=False)
    with pytest.raises(RuntimeError, match="exceeds a page"):
        _row_store.RowStore(paths=[str(path)], extent_file=[0], extent_base=[0], rows_per_extent=1,
                            row_bytes=4097, row_stride=4097, use_io_uring=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_wait_sync_probe_only_accepts_a_captured_wait_that_holds():
    """probe_wait_sync says True only where a WAIT captured in a graph blocks every replay until the host signals
    (ROCm 10 on gfx1100; where the second replay runs through a captured wait, the probe says False)."""
    import threading
    import time

    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.kernel.row_store import probe_wait_sync, signal, wait_reset

    dev = torch.device("cuda", torch.cuda.current_device())
    if not probe_wait_sync("auto", dev):
        pytest.skip("stream memops unusable in capture on this stack: launch-gating (the safe fallback)")
    flag = alloc_pinned_tensor(1, dtype=torch.int64)
    flag.zero_()
    x = torch.zeros(1, device=dev)
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side), torch.cuda.graph(g, stream=side):
        wait_reset(side, flag)
        x.add_(1.0)
    for i in range(3):
        threading.Timer(0.05, lambda: signal(flag)).start()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize(dev)
        assert time.perf_counter() - t0 >= 0.04, f"replay {i} did not wait for the host signal"
        assert int(flag[0]) == 0
    assert int(x.item()) == 3
