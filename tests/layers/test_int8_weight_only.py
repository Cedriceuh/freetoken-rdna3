"""Weight-only INT8 conversion: routers stay bf16, converted linears stay close to bf16 (CPU)."""
import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers import BaseOP, LinearReplicated, OPList


class _Mlp(BaseOP):
    def __init__(self):
        self.gate = LinearReplicated(64, 8, has_bias=False)                  # router: no prefix, stays bf16
        self.shared_expert_gate = LinearReplicated(64, 1, has_bias=False)    # stays bf16
        self.up = LinearReplicated(64, 96, has_bias=False, prefix="model.layers.0.mlp.up")


class _Layer(BaseOP):
    def __init__(self):
        self.mlp = _Mlp()  # the real path: model.layers.<i>.mlp.gate


class _Root(BaseOP):
    def __init__(self):
        self.layers = OPList([_Layer()])
        self.lm_head = LinearReplicated(64, 32, has_bias=False, prefix="lm_head")


def _root():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(0)
    root = _Root()
    for op in (root.layers.op_list[0].mlp.gate, root.layers.op_list[0].mlp.shared_expert_gate, root.layers.op_list[0].mlp.up, root.lm_head):
        op.weight = (torch.randn(op.weight.shape) * 0.05).to(torch.bfloat16)
    return root


def test_routers_stay_bf16_and_the_rest_converts(monkeypatch):
    from freetoken.layers.quantization import int8_weight_only as iwo

    root = _root()
    x = torch.randn(4, 64).to(torch.bfloat16)
    ref_up = root.layers.op_list[0].mlp.up.forward(x).float()
    n, freed = iwo.convert_model(root)
    assert n == 2 and freed > 0
    assert root.layers.op_list[0].mlp.gate.weight.dtype is torch.bfloat16
    assert root.layers.op_list[0].mlp.shared_expert_gate.weight.dtype is torch.bfloat16
    assert root.layers.op_list[0].mlp.up.weight.dtype is torch.int8 and root.lm_head.weight.dtype is torch.int8
    got = root.layers.op_list[0].mlp.up.forward(x).float()  # M=4: dequant + F.linear
    assert ((got - ref_up).norm() / ref_up.norm()).item() < 0.02


def test_skip_env_keeps_a_path_in_bf16(monkeypatch):
    from freetoken.layers.quantization import int8_weight_only as iwo

    monkeypatch.setenv("FREETOKEN_INT8_DENSE_SKIP", "lm_head")
    root = _root()
    n, _ = iwo.convert_model(root)
    assert n == 1 and root.lm_head.weight.dtype is torch.bfloat16


def test_compaction_is_a_no_op_on_cpu_and_keeps_values():
    from freetoken.layers.quantization import int8_weight_only as iwo

    root = _root()
    iwo.convert_model(root)
    before = {k: v.clone() for k, v in root.state_dict().items()}
    iwo.compact_device_tensors(root)  # no CUDA here: returns early, nothing moves
    after = root.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)


class _Tower(BaseOP):
    def __init__(self):
        self.proj = LinearReplicated(64, 64, has_bias=True, prefix="visual.blocks.0.attn.proj")


def test_vision_tower_stays_bf16():
    from freetoken.layers.quantization import int8_weight_only as iwo

    root = _root()
    root.visual = _Tower()
    root.visual.proj.weight = (torch.randn(64, 64) * 0.05).to(torch.bfloat16)
    n, _ = iwo.convert_model(root)
    assert n == 2 and root.visual.proj.weight.dtype is torch.bfloat16
    assert root.lm_head.weight.dtype is torch.int8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape", [(336, 10240), (704, 2560), (320, 10240), (4000, 2560)])
def test_row_groups_give_each_row_its_single_row_bits(shape):
    """K-blocked shapes run several rows through _gemv_int8_row_group (each weight tile read once for up to 4 rows):
    every row must come out exactly as a batch of one, for 2 to 8 rows (groups of 4, then the rest)."""
    from freetoken.kernel.triton.int8_gemv import gemv_int8

    torch.manual_seed(sum(shape))
    n, k = shape
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    s = torch.rand(n, device="cuda") * 0.01
    x = torch.randn(8, k, device="cuda").bfloat16()
    alone = torch.stack([gemv_int8(x[i : i + 1], w, s)[0] for i in range(8)])
    for m in (2, 3, 4, 5, 8):
        assert torch.equal(gemv_int8(x[:m], w, s), alone[:m]), (shape, m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shape", [(336, 10240), (320, 10240)])
@pytest.mark.parametrize("hc", [2, 4])
def test_hc_silu_epilogue_matches_the_separate_kernel(shape, hc):
    """gemv_int8(..., silu_hc=hc, silu_cols=320) (the hyper-connection down projection storing its lora columns through
    hc_silu) gives the bits of the GEMV then hc_silu on those columns, the inject columns untouched, 1 to 8 rows; the up
    projection reading that strided slice gives the bits of reading a contiguous copy."""
    from freetoken.kernel.triton.hc import hc_silu
    from freetoken.kernel.triton.int8_gemv import gemv_int8

    torch.manual_seed(hc + shape[0])
    n, k = shape
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    s = torch.rand(n, device="cuda") * 0.01
    w_up = torch.randint(-127, 128, (10240, 320), dtype=torch.int8, device="cuda")
    s_up = torch.rand(10240, device="cuda") * 0.01
    x = (torch.randn(8, k, device="cuda") * 0.5).bfloat16()
    for m in range(1, 9):
        ref = gemv_int8(x[:m], w, s)
        ref[:, :320] = hc_silu(ref[:, :320], hc)
        got = gemv_int8(x[:m], w, s, silu_hc=hc, silu_cols=320)
        assert torch.equal(got, ref), m
        assert torch.equal(gemv_int8(got[:, :320], w_up, s_up), gemv_int8(got[:, :320].contiguous(), w_up, s_up)), m


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("hc", [2, 4])
def test_hc_mix_kernel_matches_gemv_then_gate_mix(hc):
    """gemv_int8_hc_mix (the up projection and hc_gate_mix in one kernel) gives the bits of gemv_int8 then
    hc_gate_mix, 1 to 8 rows, reading a strided input like the down projection's low-rank slice."""
    from freetoken.kernel.triton.hc import hc_gate_mix
    from freetoken.kernel.triton.int8_gemv import gemv_int8, gemv_int8_hc_mix

    hd = 2560
    for seed in range(6):  # a mismatch can be one value in thousands: several draws
        torch.manual_seed(10 * seed + hc)
        w = torch.randint(-127, 128, (hc * hd, 320), dtype=torch.int8, device="cuda")
        s = torch.rand(hc * hd, device="cuda") * 0.01 * (1 + seed % 3)
        down = (torch.randn(8, 336, device="cuda") * 0.5).bfloat16()
        rn = (torch.randn(8, hc * hd, device="cuda") * (0.5 + seed % 4)).bfloat16()
        for m in range(1, 9):
            x = down[:m, :320]
            got = gemv_int8_hc_mix(x, w, s, rn[:m], hc)
            if got is None:
                pytest.skip("the up projection's config is not the one-pass (1, 512, 1, 1) one")
            assert torch.equal(got, hc_gate_mix(rn[:m], gemv_int8(x, w, s), hc)), (seed, m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("d", [384, 256, 352, 288, 320])
def test_act_mul_kernel_matches_gemv_then_silu_and_mul(d):
    """gemv_int8_act_mul (the shared expert's gate | up projection storing silu(gate) * up) gives the bits of gemv_int8
    then the Triton silu_and_mul, 1 to 8 rows (None where the separate kernels run: K-blocked shapes past 4 rows), at
    each rank's slice width, over several random draws."""
    from freetoken.kernel.triton.activation import silu_and_mul
    from freetoken.kernel.triton.int8_gemv import gemv_int8, gemv_int8_act_mul

    for seed in range(6):  # a mismatch can be one value in thousands: several draws
        torch.manual_seed(10 * seed + d)
        w = torch.randint(-127, 128, (2 * d, 2560), dtype=torch.int8, device="cuda")
        s = torch.rand(2 * d, device="cuda") * 0.005 * (1 + seed % 4)
        x = (torch.randn(8, 2560, device="cuda") * (0.3 + seed % 3)).bfloat16()
        for m in range(1, 9):
            got = gemv_int8_act_mul(x[:m], w, s)
            if got is not None:
                assert torch.equal(got, silu_and_mul(gemv_int8(x[:m], w, s))), (seed, m)
