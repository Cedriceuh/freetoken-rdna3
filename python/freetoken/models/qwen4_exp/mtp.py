"""Qwen3.8-Flash-Next MTP draft head (``mtp.*`` in the checkpoint), for speculative decoding.

One extra decoder layer that reads the main model's hyper-connection streams at row i and the embedding of
token i+1, and guesses token i+2 (SGLang ``qwen4_exp_mtp.py``, ``_fuse_residual_linear_shared``)::

    e  = fc_embedding(norm_e(embed(token_{i+1})))                         [T, hidden]
    h  = fc_hidden(norm_h(R_i) viewed [T, hc, hidden])                     per stream
    R' = layer(h + e)                                                      full attention + MoE, hyper-connections
    logits = lm_head(mtp_mixer.mix(R')[0])                                 the main model's head

``norm_h`` normalizes each of the hc streams on its own statistic: measured offline (on
1442 greedy tokens) the draft is accepted 0.875 of the time that way, 0.843 with one statistic over all hc*hidden.
The checkpoint stores the MTP's 512 routed experts stacked in bf16; :func:`quantize_nvfp4` turns them into the
main experts' modelopt NVFP4 layout (same acceptance offline), so they ride the existing NVFP4 banks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList

from .hc import GatedResidual, GroupedPlusOneRMSNorm

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

# E2M1 magnitudes indexed by the low 3 bits of a code; bit 3 is the sign (kernel/triton/nvfp4_dequant.py)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_MAX = 448.0
_E2M1_MAX = 6.0


def quantize_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """modelopt NVFP4 of one ``[out, in]`` weight: packed e2m1 codes ``[out, in/2]`` uint8 (element 2b in the low
    nibble), e4m3 scales per 16 inputs ``[out, in/16]``, and the per-tensor fp32 scale, with
    ``w ~= e2m1 * scale * global``. The global scale puts the largest block scale on e4m3's maximum."""
    out_dim, in_dim = w.shape
    assert in_dim % 16 == 0, in_dim
    x = w.float()
    amax = x.abs().max().clamp(min=1e-12)
    g = amax / (_E2M1_MAX * _E4M3_MAX)
    blocks = x.view(out_dim, in_dim // 16, 16)
    block_amax = blocks.abs().amax(-1, keepdim=True)
    scale = (block_amax / _E2M1_MAX / g).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
    denom = scale.float() * g
    q = torch.where(denom > 0, blocks / denom.clamp(min=1e-30), torch.zeros_like(blocks))
    grid = torch.tensor(_E2M1, device=w.device)
    mag = (q.abs().unsqueeze(-1) - grid).abs().argmin(-1)  # nearest magnitude (ties -> the smaller)
    codes = (mag | ((q < 0) & (mag > 0)).to(mag.dtype) << 3).to(torch.uint8).view(out_dim, in_dim)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scale.view(out_dim, in_dim // 16).contiguous(), g.reshape(1)


class Qwen4ExpMTP(BaseOP):
    """The MTP draft head: input fusion, one decoder layer (full attention + MoE, layer id ``num_layers``) and its
    own top mixer. Shares the main model's embedding and lm_head. State-dict keys are the checkpoint's ``mtp.*``
    names (``mtp.layers.0...`` for the decoder layer); its routed experts live in the offload cache's last bank."""

    def __init__(self, config: ModelConfig, *, prefix: str = "mtp") -> None:
        from .model import Qwen4ExpDecoderLayer

        args = config.qwen4_args
        self.hc_count, self.hidden_size = args.hc_count, config.hidden_size
        width = self.hc_count * self.hidden_size
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(self.hidden_size, config.rms_norm_eps, 1)
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.fc_embedding = LinearReplicated(self.hidden_size, self.hidden_size, has_bias=False)
        self.fc_hidden = LinearReplicated(self.hidden_size, self.hidden_size, has_bias=False)
        self.layers = OPList([Qwen4ExpDecoderLayer(config, config.num_layers, prefix=f"{prefix}.layers.0")])
        # The quant config excludes mtp.* (bf16 in the checkpoint), but these experts are quantized at load into the
        # offload cache's NVFP4 banks: build the layer under a main-expert name so it takes the banks' method.
        from freetoken.layers.moe import make_moe_layer

        mlp = self.layers.op_list[0].mlp
        experts = make_moe_layer(config, layer_id=config.num_layers, renormalize=config.norm_topk_prob,
                                 quant_config=config.quant, prefix="model.layers.0.mlp.experts")
        experts.defer_all_reduce = mlp.experts.defer_all_reduce
        mlp.experts = experts
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")

    def forward(self, streams: torch.Tensor | list, next_embeds: torch.Tensor, batch: Batch,
                keep: torch.Tensor | None = None, kv_only: bool = False) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``streams [T, hc*hidden]``: the main model's streams at rows i (before its top mixer), or a one-element list
        holding them that the head empties, so they are freed once normed (a 16k-token prefill chunk's are 320 MiB,
        and the head's mix holds three more of that size); ``next_embeds [T, hidden]``: the embedding of token i+1.
        Returns the head input (guess for token i+2) and the MTP layer's own streams (the input of a chained guess),
        for the rows ``keep`` (all T when None): the attention runs on every row, whose KV later drafts read; the MoE
        after it only where a guess is read (rows are independent past the attention). ``kv_only``: stop after the
        attention (its KV written), return None."""
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(next_embeds))
        if isinstance(streams, list):
            streams = streams.pop()
        h = self.pre_fc_norm_hidden.forward(streams).view(-1, self.hidden_size)  # 2-D: the int8 GEMV takes few rows
        del streams
        # the embedding added in place: the same bf16 sums as ``a + b``, without a second [T, hc*hidden] tensor
        x = self.fc_hidden.forward(h).view(-1, self.hc_count, self.hidden_size)
        del h
        x = x.add_(e.unsqueeze(-2)).flatten(-2)
        del e
        layer = self.layers.op_list[0]
        block_input, inject = layer.attn_hyper_connection.mix(x)
        x = layer.attn_hyper_connection.combine(x, layer.self_attn.forward(block_input, batch), inject)
        if kv_only:
            return None
        if keep is not None:
            x = x.index_select(0, keep)
        block_input, inject = layer.mlp_hyper_connection.mix(x)
        x = layer.mlp_hyper_connection.combine(x, layer.mlp.forward(block_input), inject)
        return self.hyper_connection_mixer.mix(x)[0], x


__all__ = ["Qwen4ExpMTP", "quantize_nvfp4"]
