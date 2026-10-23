"""FREETOKEN_MOE_MISS_READS on EXL3: the decode GEMVs read the slots the expert cache is filling this step from the
layer's pinned host bank and write them into the slots on the way. The outputs are those of copying the rows first
and reading the slots (the same words, the same sums), and the slots end up holding the rows the copy would have
written; at the per-rank shapes, one or several routes a program, CPU routes (-1) among them, 0 to all slots
missing."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SHAPES = [(2560, 768, 2), (384, 2560, 1), (2560, 512, 2), (256, 2560, 1), (2560, 1280, 2), (640, 2560, 1)]


def _plan(slots, n_miss, E, g, dev):
    """evict_slots / src_indices / num_indices as the cache's lru_ensure leaves them: n_miss distinct slots of the
    step (each filled once), their experts' host rows."""
    used = slots[slots >= 0].unique()
    pick = used[torch.randperm(used.numel(), device=dev, generator=g)[:n_miss]]
    ev = torch.full((slots.numel(),), -7, dtype=torch.int32, device=dev)
    src = torch.zeros((slots.numel(),), dtype=torch.int32, device=dev)
    ev[:pick.numel()] = pick.int()
    src[:pick.numel()] = torch.randperm(E, device=dev, generator=g)[:pick.numel()].int()
    num = torch.tensor([pick.numel()], dtype=torch.int64, device=dev)
    return ev, src, num, pick.numel()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_gemv_reading_misses_from_host_matches_copy_then_gemv(shape, K, tokens):
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
    from freetoken.kernel.triton.exl3 import exl3_gemv
    from freetoken.kernel.triton.route_groups import route_groups
    from freetoken.layers.quantization import exl3_codec as ex

    dev, top_k, S, E = "cuda", 10, 24, 40
    kin, n, halves = shape
    g = torch.Generator(device=dev).manual_seed(kin + n + 7 * K + tokens)
    row = (kin // 16, n // 16, 16 * K)
    bank = torch.randint(-32768, 32767, (S, *row), device=dev, dtype=torch.int16, generator=g)
    host = alloc_pinned_tensor(E, *row, dtype=torch.int16)
    host.copy_(torch.randint(-32768, 32767, (E, *row), device=dev, dtype=torch.int16, generator=g).cpu())
    ptrs = torch.tensor([0, device_ptr(host)], dtype=torch.int64, device=dev)  # this bank at index 1
    for cpu_routes in (False, True):
        slots = torch.cat([torch.randperm(S, device=dev, generator=g)[:top_k] for _ in range(tokens)]).int()
        if cpu_routes:
            slots[torch.rand(slots.shape, device=dev, generator=g) < 0.15] = -1
        a = torch.randn(slots.numel(), halves, kin, device=dev, generator=g) * 2
        n_used = slots[slots >= 0].unique().numel()
        for n_miss in sorted({0, 1, n_used // 2, n_used}):
            ev, src, num, k = _plan(slots, n_miss, E, g, dev)
            ref = bank.clone()
            ref[ev[:k].long()] = host[src[:k].cpu().long()].to(dev)
            want = exl3_gemv(a, ref, slots, ex.CB_MUL1, skip_inactive=True)
            for group in (1, 2, 4) if tokens > 1 else (1,):
                got_bank = bank.clone()
                grp = route_groups(slots, group) if group > 1 else None
                got = exl3_gemv(a, got_bank, slots, ex.CB_MUL1, skip_inactive=True, group=group, groups=grp,
                                misses=(ev, src, num, ptrs), bank=1)
                assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (n_miss, group, cpu_routes)
                assert torch.equal(got_bank, ref), (n_miss, group, cpu_routes)  # the slots filled, the others as were


@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_decode_moe_reading_misses_keeps_the_bits_and_fills_the_slots(tokens):
    import freetoken.moe.fused_exl3 as F
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
    from freetoken.layers.quantization import exl3_codec as ex

    dev, top_k, S, E, H, inter, K = "cuda", 10, 24, 40, 2560, 384, 3
    g = torch.Generator(device=dev).manual_seed(100 + tokens)
    gu_row, dn_row = (H // 16, 2 * inter // 16, 16 * K), (inter // 16, H // 16, 16 * K)
    gate_up = torch.randint(-32768, 32767, (S, *gu_row), device=dev, dtype=torch.int16, generator=g)
    down = torch.randint(-32768, 32767, (S, *dn_row), device=dev, dtype=torch.int16, generator=g)
    h_gu = alloc_pinned_tensor(E, *gu_row, dtype=torch.int16)
    h_gu.copy_(torch.randint(-32768, 32767, (E, *gu_row), device=dev, dtype=torch.int16, generator=g).cpu())
    h_dn = alloc_pinned_tensor(E, *dn_row, dtype=torch.int16)
    h_dn.copy_(torch.randint(-32768, 32767, (E, *dn_row), device=dev, dtype=torch.int16, generator=g).cpu())
    ptrs = torch.tensor([device_ptr(h_gu), 0, 0, device_ptr(h_dn)], dtype=torch.int64, device=dev)
    gate_up_suh = torch.randn(S, 2, H, device=dev, generator=g).sign().to(torch.float16)
    gate_up_svh = (torch.randn(S, 2 * inter, device=dev, generator=g) * 0.01).to(torch.float16)
    down_suh = torch.randn(S, inter, device=dev, generator=g).sign().to(torch.float16)
    down_svh = (torch.randn(S, H, device=dev, generator=g) * 0.01).to(torch.float16)
    x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16, generator=g)
    ids = torch.stack([torch.randperm(S, device=dev, generator=g)[:top_k] for _ in range(tokens)]).int()
    w = torch.rand(tokens, top_k, device=dev, generator=g)
    n_used = ids.unique().numel()
    ev, src, num, k = _plan(ids.view(-1), (n_used + 1) // 2, E, g, dev)
    ref_gu, ref_dn = gate_up.clone(), down.clone()
    ref_gu[ev[:k].long()] = h_gu[src[:k].cpu().long()].to(dev)
    ref_dn[ev[:k].long()] = h_dn[src[:k].cpu().long()].to(dev)
    want = F.fused_experts_exl3(x, ref_gu, gate_up_suh, gate_up_svh, ref_dn, down_suh, down_svh, w, ids.clone(),
                                codebook=ex.CB_MUL1, is_prefill=False)
    gu, dn = gate_up.clone(), down.clone()
    banks = {"gate_up": (0, gu[0].numel() * 2), "down": (3, dn[0].numel() * 2)}
    got = F.fused_experts_exl3(x, gu, gate_up_suh, gate_up_svh, dn, down_suh, down_svh, w, ids.clone(),
                               codebook=ex.CB_MUL1, is_prefill=False, misses=((ev, src, num, ptrs), banks))
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert torch.equal(gu, ref_gu) and torch.equal(dn, ref_dn)
