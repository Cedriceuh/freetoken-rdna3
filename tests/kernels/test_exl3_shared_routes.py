"""FREETOKEN_EXL3_SHARED_ROUTES: the decode GEMV with the routes of one slot in one program (each trellis tile decoded
once for up to 2 or 4 of them) gives the bits of one program a route, at the per-rank shapes' tuned configs and the
default one, with slots read by more routes than a program takes and CPU routes (-1) among them."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# (KIN, N, halves): gate|up and down at intermediate 384 / 256 per rank (tuned configs), the whole 640 (one card, the
# default config) and a small shape on the default
SHAPES = [(2560, 768, 2), (384, 2560, 1), (2560, 512, 2), (256, 2560, 1), (2560, 1280, 2), (640, 2560, 1), (512, 256, 2)]


def _slots(tokens, top_k, S, g, dev, cpu_routes):
    # each token's top-k are distinct slots; tokens share slots (a small bank: some slots in more than 4 routes)
    rows = [torch.randperm(S, device=dev, generator=g)[:top_k] for _ in range(tokens)]
    slots = torch.cat(rows).int()
    if cpu_routes:
        slots[torch.rand(slots.shape, device=dev, generator=g) < 0.15] = -1
    return slots


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("tokens", [2, 3, 4, 8])
def test_shared_routes_match_one_program_a_route(shape, K, tokens):
    from freetoken.kernel.triton.exl3 import exl3_gemv
    from freetoken.layers.quantization import exl3_codec as ex

    dev, top_k, S = "cuda", 10, 24
    kin, n, halves = shape
    g = torch.Generator(device=dev).manual_seed(kin + n + K * 7 + tokens)
    bank = torch.randint(-32768, 32767, (S, kin // 16, n // 16, 16 * K), device=dev, dtype=torch.int16, generator=g)
    for cpu_routes in (False, True):
        slots = _slots(tokens, top_k, S, g, dev, cpu_routes)
        a = torch.randn(slots.numel(), halves, kin, device=dev, generator=g) * 2
        for skip in (False, True) if cpu_routes else (False,):
            sl = slots if skip else slots.clamp(min=0)
            want = exl3_gemv(a, bank, sl, ex.CB_MUL1, skip_inactive=skip)
            for group in (2, 4):
                got = exl3_gemv(a, bank, sl, ex.CB_MUL1, skip_inactive=skip, group=group)
                assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (group, skip)


@pytest.mark.parametrize("tokens", [2, 4, 8])
def test_decode_moe_with_shared_routes_keeps_the_bits(monkeypatch, tokens):
    import freetoken.moe.fused_exl3 as F
    from freetoken.layers.quantization import exl3_codec as ex

    dev, top_k, S, H, inter, K = "cuda", 10, 24, 2560, 384, 3
    g = torch.Generator(device=dev).manual_seed(tokens)
    gate_up = torch.randint(-32768, 32767, (S, H // 16, 2 * inter // 16, 16 * K), device=dev, dtype=torch.int16,
                            generator=g)
    down = torch.randint(-32768, 32767, (S, inter // 16, H // 16, 16 * K), device=dev, dtype=torch.int16, generator=g)
    gate_up_suh = torch.randn(S, 2, H, device=dev, generator=g).sign().to(torch.float16)
    gate_up_svh = (torch.randn(S, 2 * inter, device=dev, generator=g) * 0.01).to(torch.float16)
    down_suh = torch.randn(S, inter, device=dev, generator=g).sign().to(torch.float16)
    down_svh = (torch.randn(S, H, device=dev, generator=g) * 0.01).to(torch.float16)
    x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16, generator=g)
    ids = _slots(tokens, top_k, S, g, dev, False).view(tokens, top_k)
    w = torch.rand(tokens, top_k, device=dev, generator=g)
    args = (x, gate_up, gate_up_suh, gate_up_svh, down, down_suh, down_svh, w, ids)
    monkeypatch.setattr(F, "SHARED_ROUTES", 1)
    want = F.fused_experts_exl3(*args, codebook=ex.CB_MUL1, is_prefill=False)
    for group in (2, 4):
        monkeypatch.setattr(F, "SHARED_ROUTES", group)
        got = F.fused_experts_exl3(*args, codebook=ex.CB_MUL1, is_prefill=False)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), group


def _groups_reference(slots, group):
    out, seen = [], {}
    for r, s in enumerate(slots):
        seen.setdefault(s, []).append(r)
    for r, s in enumerate(slots):
        mine = seen[s]
        k = mine.index(r)
        if k % group:
            out.append([0, 0, 0, 0])
            continue
        run = mine[k:k + group]
        out.append([len(run)] + run[1:] + [0] * (4 - len(run)))
    return out


@pytest.mark.parametrize("tokens", [2, 3, 4, 8])
@pytest.mark.parametrize("group", [2, 4])
def test_rotate_in_writes_the_route_groups_and_keeps_its_bits(tokens, group):
    from freetoken.kernel.triton.exl3 import exl3_rotate_in
    from freetoken.kernel.triton.route_groups import route_groups

    dev, H, top_k, S = "cuda", 2560, 10, 24
    g = torch.Generator(device=dev).manual_seed(tokens * 3 + group)
    x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16, generator=g) * 3
    suh = (torch.randn(S, 2, H, device=dev, generator=g) * 2).to(torch.float16)
    slots = _slots(tokens, top_k, S, g, dev, True)
    want = exl3_rotate_in(x, suh, slots, top_k)
    grp = torch.full((slots.numel(), 4), -7, dtype=torch.int32, device=dev)
    got = exl3_rotate_in(x, suh, slots, top_k, groups=grp, group=group)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    ref = _groups_reference(slots.tolist(), group)
    assert grp[:, 0].tolist() == [row[0] for row in ref]
    lead = grp[:, 0] > 0
    for r in lead.nonzero().flatten().tolist():  # the member indices a leader reads
        n = ref[r][0]
        assert grp[r, 1:n].tolist() == ref[r][1:n], r
    alone = route_groups(slots, group)
    assert torch.equal(alone[:, 0], grp[:, 0])
    for r in lead.nonzero().flatten().tolist():
        n = ref[r][0]
        assert torch.equal(alone[r, 1:n], grp[r, 1:n]), r
