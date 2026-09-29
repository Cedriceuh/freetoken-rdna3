"""GPU check of the host KV tier copies (kvcache/host_kv_pool.py), one GPU, no server.

Pool tensors at the real Qwen3.8-Flash-Next per-rank shapes (TP=2, XTX share): 12 QSA layers of
1 KV head x 256 bf16, the compressed index slab (12 layers, ratio 4, 128 dims), 36 GDN layers of
conv (5632 x 3 bf16) + recurrent (26 heads x 128 x 128 fp32) state.

1. pages: random device pages -> host -> OTHER device pages, bit-exact, both directions timed;
2. snapshots: GDN slot -> host -> other slot, bit-exact, timed;
3. ordering: a save issued right after a kernel that writes the pages sees the new values, and a
   load is visible to a kernel issued right after it on the caller's stream.

    python3 rdna3/tests/host_kv_check.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from freetoken.kvcache.host_kv_pool import HostKVPool  # noqa: E402

PS, RATIO = 64, 4
P, S = 1025, 8


class Pool:
    pass


def main() -> int:
    dev = torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(0)
    kv = Pool()
    kv._kv_buffer = torch.randn(2, 12, P, PS, 1, 256, device=dev, dtype=torch.bfloat16, generator=g)
    kv._kv_scale = None
    kv.index_ratio = RATIO
    kv.cmp_scratch_base = P * PS // RATIO
    kv._cmp_k_buffer = torch.randn(12, kv.cmp_scratch_base + 5, 128, device=dev, dtype=torch.bfloat16, generator=g)
    kv._rope_positions = None
    lin = Pool()
    lin.conv_states = torch.randn(36, S, 5632, 3, device=dev, dtype=torch.bfloat16, generator=g)
    lin.recurrent_states = torch.randn(36, S, 26, 128, 128, device=dev, dtype=torch.float32, generator=g)
    lin.slot_states = {}

    host = HostKVPool(kv, lin, PS, num_pages=600, num_snaps=4, device=dev)
    ok = True

    # 1. pages round trip: 512 scattered source pages -> host (non-contiguous host pages too) -> other pages
    n = 512
    perm = torch.randperm(P - 1, generator=torch.Generator().manual_seed(1)).tolist()
    src, dst = perm[:n], perm[n : 2 * n]
    hp = host.alloc_pages(n)
    hp = hp[: n // 2][::-1] + hp[n // 2 :]  # a reversed half: every run of length 1
    ref_kv = kv._kv_buffer[:, :, src].clone()
    cmp_v = kv._cmp_k_buffer[:, : kv.cmp_scratch_base].unflatten(1, (P, PS // RATIO))
    ref_cmp = cmp_v[:, src].clone()
    src_t = torch.tensor(src, device=dev)
    dst_t = torch.tensor(dst, device=dev)
    # warm-up round (first-use allocator / registration costs), same shapes, then the timed one
    host.save_pages(src_t, hp)
    host.load_pages(hp, dst_t)
    kv._kv_buffer[:, :, src] = ref_kv
    cmp_v[:, src] = ref_cmp
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    host.save_pages(src_t, hp)
    t_issue = time.perf_counter() - t0
    torch.cuda.synchronize()
    t_save = time.perf_counter() - t0
    kv._kv_buffer[:, :, dst] = 0
    cmp_v[:, dst] = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    host.load_pages(hp, dst_t)
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    same = torch.equal(kv._kv_buffer[:, :, dst], ref_kv) and torch.equal(cmp_v[:, dst], ref_cmp)
    untouched = torch.equal(kv._kv_buffer[:, :, src], ref_kv)
    mb = n * host.page_bytes / 1e6
    print(f"pages: {n} x {host.page_bytes} B = {mb:.0f} MB  save {t_save * 1e3:.1f} ms"
          f" ({mb / 1e3 / t_save:.1f} GB/s, issue {t_issue * 1e3:.1f} ms)  load {t_load * 1e3:.1f} ms"
          f" ({mb / 1e3 / t_load:.1f} GB/s)  bit-exact={same} src-intact={untouched}")
    ok &= same and untouched

    # 2. snapshot round trip
    ref_c, ref_r = lin.conv_states[:, 2].clone(), lin.recurrent_states[:, 2].clone()
    hs = host.alloc_snap()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    host.save_snap(2, hs)
    torch.cuda.synchronize()
    t_ss = time.perf_counter() - t0
    lin.conv_states[:, 5] = 0
    lin.recurrent_states[:, 5] = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    host.load_snap(hs, 5)
    torch.cuda.synchronize()
    t_sl = time.perf_counter() - t0
    same = torch.equal(lin.conv_states[:, 5], ref_c) and torch.equal(lin.recurrent_states[:, 5], ref_r)
    smb = host.snap_bytes / 1e6
    print(f"snapshot: {smb:.1f} MB  save {t_ss * 1e3:.1f} ms  load {t_sl * 1e3:.1f} ms  bit-exact={same}")
    ok &= same

    # 3. stream ordering against the caller's stream (no explicit synchronize in between)
    s = torch.cuda.Stream(device=dev)
    with torch.cuda.stream(s):
        page = torch.tensor([7], device=dev)
        big = torch.randn(4096, 4096, device=dev)
        for _ in range(20):          # keep the stream busy so a missing wait would show
            big = big @ big * 1e-3
        kv._kv_buffer[:, :, 7] = 3.0
        h1 = host.alloc_pages(1)
        host.save_pages(page, h1)     # must see the 3.0 written just before
        kv._kv_buffer[:, :, 9] = -1.0
        host.load_pages(h1, torch.tensor([9], device=dev))
        probe = kv._kv_buffer[:, :, 9].float().mean()   # must see the loaded 3.0
    torch.cuda.synchronize()
    ordered = float(probe) == 3.0
    print(f"ordering: save-after-write and read-after-load on the caller's stream: {ordered}")
    ok &= ordered
    print("ALL OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
