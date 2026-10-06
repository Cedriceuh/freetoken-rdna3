"""Qwen4ExpMoE takes the fused MoE epilogue (FREETOKEN_FUSED_MOE_EPILOGUE) only where it is exact: Triton NVFP4 / EXL3
experts (they allocate their output) on the GPU decode path, without an all-reduce inside the experts; the bf16 kernel
writes into its input, the hybrid / CPU layers add partial sums and FREETOKEN_FUSE_MOE_ALLREDUCE=0 at TP>1 reduces each
expert's output, so they keep the three-kernel epilogue."""

from types import SimpleNamespace

import pytest

from freetoken.layers.quantization.moe.exl3 import TritonExl3MoEKernel
from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel
from freetoken.layers.quantization.moe.unquantized import FusedMoEKernel
from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE


def _moe(kernel_cls, decode_target="gpu", cpu_layers=(), with_cache=True, tp=2, block_all_reduce=True):
    cache = SimpleNamespace(decode_target=decode_target, is_cpu_layer=lambda layer_id: layer_id in cpu_layers)
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(kernel=object.__new__(kernel_cls)),
        offload_cache=cache if with_cache else None,
        layer_id=3,
    )
    return SimpleNamespace(experts=experts, _tp_size=tp, _comm=object() if tp > 1 and block_all_reduce else None)


@pytest.mark.parametrize("kernel_cls", [TritonNvfp4MoEKernel, TritonExl3MoEKernel])
def test_quantized_triton_experts_on_the_gpu_path_take_the_fused_epilogue(kernel_cls):
    assert Qwen4ExpMoE._epilogue_fusable(_moe(kernel_cls))
    # a cpu decode target leaves the layers it does not list on the GPU path
    assert Qwen4ExpMoE._epilogue_fusable(_moe(kernel_cls, decode_target="cpu", cpu_layers=(7,)))
    assert Qwen4ExpMoE._epilogue_fusable(_moe(kernel_cls, tp=1))


@pytest.mark.parametrize(
    "moe",
    [
        _moe(FusedMoEKernel),  # bf16 experts overwrite hidden_states
        _moe(TritonNvfp4MoEKernel, decode_target="hybrid"),  # GPU partial + CPU partial
        _moe(TritonNvfp4MoEKernel, decode_target="cpu", cpu_layers=(3,)),  # this layer decodes on the CPU
        _moe(TritonExl3MoEKernel, with_cache=False),
        _moe(TritonNvfp4MoEKernel, block_all_reduce=False),  # FREETOKEN_FUSE_MOE_ALLREDUCE=0: the experts all-reduce
    ],
)
def test_other_expert_paths_keep_the_three_kernel_epilogue(moe):
    assert not Qwen4ExpMoE._epilogue_fusable(moe)
