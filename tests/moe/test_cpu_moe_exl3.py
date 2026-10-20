"""CPU MoE executor, EXL3 experts (csrc/cpu_moe/kernels/exl3.cpp): one decode step on synthetic mul1 trellis banks, packed as
layers/quantization/moe/exl3.py packs them, against the reference codec (exl3_codec.reconstruct); then GPU checks of
the EXL3 decode kernels these banks feed (skipped without a GPU)."""

from __future__ import annotations

import platform
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as Fn

from freetoken.layers.quantization.exl3_codec import CB_MUL1, pack_trellis, reconstruct

pytestmark = pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="the EXL3 CPU path is AVX2")


def _linear(k, n, bits, gen):
    symbols = torch.randint(0, 1 << bits, (k // 16, n // 16, 256), generator=gen)
    trellis = pack_trellis(symbols, bits)
    sign = torch.where(torch.rand(k, generator=gen) < 0.5, -1, 1).half()
    suh = (torch.rand(k, generator=gen) * 0.5 + 0.75).half() * sign
    svh = (torch.rand(n, generator=gen) * 0.02 + 0.01).half()
    return trellis, suh, svh


def _experts(E, H, I, bits, seed):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(E):
        out.append({p: _linear(H if p != "down" else I, I if p != "down" else H, bits, gen)
                    for p in ("gate", "up", "down")})
    return out


def _cache(experts, lo, i):
    t0, t1, cols = lo // 16, (lo + i) // 16, slice(lo, lo + i)
    bank = {
        "gate_up": torch.stack([torch.cat([d["gate"][0][:, t0:t1], d["up"][0][:, t0:t1]], 1) for d in experts]),
        "gate_up_suh": torch.stack([torch.stack([d["gate"][1], d["up"][1]]) for d in experts]),
        "gate_up_svh": torch.stack([torch.cat([d["gate"][2][cols], d["up"][2][cols]]) for d in experts]),
        "down": torch.stack([d["down"][0][t0:t1] for d in experts]),
        "down_suh": torch.stack([d["down"][1][cols] for d in experts]),
        "down_svh": torch.stack([d["down"][2] for d in experts]),
    }
    return SimpleNamespace(bank_sources={k: [v.contiguous()] for k, v in bank.items()}, num_experts=len(experts),
                           num_layers=1, quant_format="exl3")


def _run(ex, x, ids, w):
    """One task on the executor's IO rows: x, ids, w in, y out."""
    M = x.shape[0]
    ex._io["x"][:M].copy_(x)
    ex._io["ids"][:M].copy_(ids)
    ex._io["w"][:M].copy_(w)
    ex._ext.run(0, M)
    return ex._io["y"][:M].clone()


def _reference(experts, x, ids, w, lo, i):
    cols = slice(lo, lo + i)
    y = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64)
    for m in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[m, k])
            if e < 0:
                continue
            Wg, Wu, Wd = (reconstruct(*experts[e][p], CB_MUL1).double() for p in ("gate", "up", "down"))
            xd = x[m].double()
            y[m] += float(w[m, k]) * ((Fn.silu(xd @ Wg[:, cols]) * (xd @ Wu[:, cols])) @ Wd[cols])
    return y


@pytest.mark.parametrize("bits", [2, 3, 4])
@pytest.mark.parametrize("M", [1, 3])
@pytest.mark.parametrize("lo,i", [(0, 256), (128, 128)])
def test_cpu_exl3_matches_reference_codec(bits, M, lo, i):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    E, H, I, top_k = 6, 256, 256, 3
    experts = _experts(E, H, I, bits, seed=bits * 10 + M)
    ex = CpuMoeExecutor(_cache(experts, lo, i), top_k=top_k, activation="silu", apply_router_weight_on_input=False,
                        num_threads=4, max_tokens=4, device=torch.device("cpu"))
    gen = torch.Generator().manual_seed(M)
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16()
    ids = torch.stack([torch.randperm(E, generator=gen)[:top_k] for _ in range(M)]).int()
    ids[0, 0] = -1  # a route the GPU computes: skipped here
    w = torch.softmax(torch.randn(M, top_k, generator=gen), 1).float()
    y = _run(ex, x, ids, w)
    ref = _reference(experts, x.float(), ids, w, lo, i)
    err = ((y.double() - ref).norm() / ref.norm()).item()
    assert err < 5e-3, err  # bf16 input/output and no per-weight fp16 rounding


def test_cpu_moe_task_larger_than_max_tokens_is_refused():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    experts = _experts(2, 128, 128, 3, seed=0)
    ex = CpuMoeExecutor(_cache(experts, 0, 128), top_k=1, activation="silu", apply_router_weight_on_input=False,
                        num_threads=2, max_tokens=2, device=torch.device("cpu"))
    with pytest.raises(Exception, match="outside the executor's layers or IO"):
        ex._ext.run(0, 3)


def test_cpu_exl3_rows_sharing_an_expert_match_one_row_at_a_time():
    """Rows routed to the same expert decode its trellis once (exl3_gemv_cols_mt): the sums are the ones each row gets
    alone, bit for bit."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    E, H, I, top_k, M = 3, 256, 256, 2, 4
    experts = _experts(E, H, I, 3, seed=7)
    ex = CpuMoeExecutor(_cache(experts, 0, I), top_k=top_k, activation="silu", apply_router_weight_on_input=False,
                        num_threads=4, max_tokens=M, device=torch.device("cpu"))
    gen = torch.Generator().manual_seed(3)
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16()
    ids = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 1]], dtype=torch.int32)  # every expert on two or three rows
    w = torch.softmax(torch.randn(M, top_k, generator=gen), 1).float()
    y = _run(ex, x, ids, w)
    for m in range(M):
        ym = _run(ex, x[m:m + 1], ids[m:m + 1], w[m:m + 1])
        assert torch.equal(ym[0], y[m]), m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_exl3_decode_skips_cpu_routes():
    """Hybrid decode on the GPU: a route the CPU computes reaches it as slot -1 with weight 0; the GEMVs write zeros
    for it without decoding (FREETOKEN_HYBRID_SKIP_CPU_ROUTES), bit-identical to running it on a valid slot."""
    from freetoken.moe import fused_exl3

    if not fused_exl3.SKIP_CPU_ROUTES:
        pytest.skip("FREETOKEN_HYBRID_SKIP_CPU_ROUTES=0")
    dev = torch.device("cuda")
    E, H, I, top_k, M = 4, 256, 256, 3, 2
    bank = {k: v[0].to(dev) for k, v in _cache(_experts(E, H, I, 3, seed=5), 0, I).bank_sources.items()}
    gen = torch.Generator().manual_seed(9)
    x = (torch.randn(M, H, generator=gen) * 0.5).bfloat16().to(dev)
    ids = torch.tensor([[0, 2, 3], [1, 0, 2]], dtype=torch.int32, device=dev)
    cpu = torch.tensor([[False, True, False], [True, False, True]], device=dev)
    w = torch.where(cpu, 0.0, torch.rand(M, top_k, generator=gen).to(dev))

    def run(slots):
        return fused_exl3.fused_experts_exl3(x, bank["gate_up"], bank["gate_up_suh"], bank["gate_up_svh"], bank["down"],
                                             bank["down_suh"], bank["down_svh"], w, slots, codebook=CB_MUL1,
                                             is_prefill=False)

    assert torch.equal(run(ids), run(torch.where(cpu, -1, ids)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_exl3_out_one_split_matches_two():
    """exl3_out with one down partial sum (rank 1's 256 slice on xtx-xt; SPLITS left unspecialized, the faster
    variant) gives the bits of the same sums split in two exact halves."""
    from freetoken.kernel.triton.exl3 import exl3_out

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4)
    S, H, top_k = 64, 2560, 10
    svh = torch.rand(S, H, device=dev, generator=gen).half()
    for T in (1, 2, 4):
        c = torch.randn(1, T * top_k, H, device=dev, generator=gen)
        slots = torch.randint(0, S, (T * top_k,), device=dev, dtype=torch.int32, generator=gen)
        w = torch.rand(T, top_k, device=dev, generator=gen)
        one = exl3_out(c, slots, svh, w, torch.empty(T, H, device=dev, dtype=torch.bfloat16))
        two = exl3_out(torch.cat([c * 0.5, c * 0.5]), slots, svh, w,
                       torch.empty(T, H, device=dev, dtype=torch.bfloat16))
        assert torch.equal(one, two), T


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("R", [2, 8, 40])
def test_gpu_exl3_gemv_four_n_tiles_match_one(R):
    """Rank 1's gate|up GEMV config (kernel/triton/exl3.py ``_GEMV_TUNED[(160, 16)]``: 4 N tiles a program) gives the
    bits of 1 N tile a program, the same K tile, split and warps."""
    from freetoken.kernel.triton.exl3 import exl3_gemv

    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(R)
    S, KIN, N, bits = 6, 2560, 512, 3  # gate|up of a 256-wide slice: two halves of 256 columns
    bank = torch.stack([pack_trellis(torch.randint(0, 1 << bits, (KIN // 16, N // 16, 256), generator=gen), bits)
                        for _ in range(S)]).to(dev)
    a = torch.randn(R, 2, KIN, generator=gen).to(dev)
    slots = torch.randint(0, S, (R,), generator=gen, dtype=torch.int32).to(dev)
    four = exl3_gemv(a, bank, slots, CB_MUL1, kt=1, nt=4, split_k=8, num_warps=2)
    one = exl3_gemv(a, bank, slots, CB_MUL1, kt=1, nt=1, split_k=8, num_warps=2)
    assert torch.equal(four, one)
