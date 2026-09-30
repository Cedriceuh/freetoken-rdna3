"""Experimental weight-only INT8 conversion: routers stay bf16, converted linears stay close to bf16 (CPU)."""
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
