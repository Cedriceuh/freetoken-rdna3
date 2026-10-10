from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.kernel.causal_conv1d import causal_conv1d_decode, causal_conv1d_varlen
from freetoken.distributed import get_tp_info
from freetoken.distributed.split import gdn_head_partition
from freetoken.layers import BaseOP, GatedRMSNorm, LinearColParallelMerged, LinearRowParallel
from freetoken.layers.quantization import QuantConfig
from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla, gdn_prefill_chunk_fla
from freetoken.spec_decode import SPEC_M, RollbackTarget, register, rows_of


# FREETOKEN_FEWER_COPIES (default 1 on ROCm, 0 elsewhere, see models/qwen4_exp/attention.py): the gated norm reads z in
# place
_FEWER_COPIES = os.environ.get("FREETOKEN_FEWER_COPIES", "1" if torch.version.hip is not None else "0") == "1"
# FREETOKEN_GDN_PREFILL_BLOCK (default 4096, 0 = off): a prefill's delta rule runs this many tokens of a request at a
# time, the recurrent state chained through its pool slot: the same bits (the kernels work per 64-token chunk, the
# state in fp32), a quarter of the temporaries at a 16k-token chunk (~0.8 GiB less on a 7900 XT)
_GDN_PREFILL_BLOCK = int(os.environ.get("FREETOKEN_GDN_PREFILL_BLOCK", "4096"))
assert _GDN_PREFILL_BLOCK % 64 == 0, "FREETOKEN_GDN_PREFILL_BLOCK must be a multiple of the 64-token chunk"
_SPEC_BUFS: list = []  # [(recurrent [layers, reqs, m-1, ...], conv [layers, m-1, reqs, ...])] once allocated
_SPEC_ROWS: list = []


def _spec_buffers(pool, reqs: int, device):
    """spec_decode buffers shared by every GDN layer (one slice each): the recurrent kernel's per-step states and the
    conv state after each row but the last; one rollback target per pool tensor covers all the layers."""
    if not _SPEC_BUFS or _SPEC_BUFS[0][0].shape[1] < reqs:
        assert not torch.cuda.is_current_stream_capturing(), "spec buffers must exist before the capture"
        m = SPEC_M
        rec, conv = pool.recurrent_states, pool.conv_states  # [layers, slots, ...]
        inter = rec.new_empty((rec.shape[0], reqs, m - 1, *rec.shape[2:]))
        snaps = conv.new_empty((conv.shape[0], m - 1, reqs, *conv.shape[2:]))
        _SPEC_BUFS[:] = [(inter, snaps)]
        _SPEC_ROWS[:] = [torch.arange(reqs, dtype=torch.int32, device=device)]
        pad = pool.padding_slot
        register("gdn.recurrent", RollbackTarget(rec, lambda step, b: inter[:, b, step], dim=1, padding_slot=pad))
        register("gdn.conv", RollbackTarget(conv, lambda step, b: snaps[:, step, b], dim=1, padding_slot=pad))
    return _SPEC_BUFS[0]


class _DepthwiseConv1d(BaseOP):
    """Holds the depthwise conv weight ``[conv_dim, 1, K]`` (key ``conv1d.weight``)."""

    def __init__(self, conv_dim: int, kernel: int):
        self.weight = torch.empty(conv_dim, 1, kernel)


class Qwen4ExpGatedDeltaNet(BaseOP):
    """GatedDeltaNet op using the vendored flash-linear-attention triton kernels
    (``freetoken.kernel.fla``) for the recurrence and a per-request
    recurrent + conv state held in ``ctx.linear_state_pool`` (keyed by ``Req.table_idx``).

    Parameter names match HF (``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a``/
    ``conv1d``/``A_log``/``dt_bias``/``norm``/``out_proj``). Handles prefill (incl. chunked
    continuation) and single-token decode; state is fresh when ``req.cached_len == 0``.

    ``output_gate`` is the gate activation name from ``LinearGatedDeltaGroupConfig``
    ("sigmoid" for Qwen3.8-Flash-Next).
    """

    def __init__(
        self, hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
        conv_kernel_size, rms_norm_eps, layer_id, output_gate: str = "sigmoid",
        *, quant_config: QuantConfig | None = None, prefix: str = "",
    ):
        self.layer_id = layer_id
        # The fla chunk/decode kernels read+write the recurrent state and the per-chunk h as
        # [V, K] while the LinearStatePool declares it [K, V]; these coincide (and the
        # hybrid-radix snapshot scatter h[h_row]->slot is a plain copy) only when the two head
        # dims are equal. Qwen3.5/3.6/3.8 satisfy this (128/128); guard any future config.
        assert head_k_dim == head_v_dim, (
            f"GatedDeltaNet requires head_k_dim == head_v_dim, got {head_k_dim} != {head_v_dim}"
        )
        # Head counts and every width derived from them are RANK-LOCAL, divided the way
        # LinearStatePool._linear_local_dims divides them (gdn_head_partition: even, or the uneven
        # FREETOKEN_TP_SPLIT), so the conv/recurrent state slots and this module's tensors agree;
        # full widths here would slice the next rank's heads out of the GEMM.
        tp_size = get_tp_info().size
        _, self.num_k_heads, _, self.num_v_heads = gdn_head_partition(num_k_heads, num_v_heads)
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = self.num_k_heads * head_k_dim
        self.value_dim = self.num_v_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel_size
        full_value_dim = num_v_heads * head_v_dim
        full_conv_dim = 2 * num_k_heads * head_k_dim + full_value_dim
        # The fused projections get this rank's segments explicitly. Evenly split, they must be the
        # quotient LinearColParallelMerged would have computed (it is not when only one head count
        # replicates); uneven, the ranks' segments add up to the full widths by construction.
        even = self.num_k_heads * tp_size == num_k_heads or tp_size > num_k_heads
        assert not even or self.conv_dim * tp_size == full_conv_dim, (
            f"local conv_dim {self.conv_dim} does not tile {full_conv_dim} at tp_size={tp_size}: "
            f"k heads {num_k_heads} -> {self.num_k_heads}, v heads {num_v_heads} -> {self.num_v_heads}"
        )
        # quantized checkpoints quantize qkv|z but not b|a, so the fusion splits into a qkvz GEMM and a ba GEMM with their own schemes (matches sglang / vLLM)
        self._split_in_proj = (
            quant_config is not None and quant_config.scheme_for(f"{prefix}.in_proj_qkvz") is not None
        )

        # b|a are one column per v head, so they shard with the v heads like everything else.
        self._in_proj_split = [self.conv_dim, self.value_dim, self.num_v_heads, self.num_v_heads]
        if self._split_in_proj:
            if tp_size > 1:
                # the quantized in_proj scales do not follow a head split; refuse rather than build a wrong layer
                raise NotImplementedError(f"GDN quantized in_proj is not implemented at tp_size={tp_size}")
            self.in_proj_qkvz = LinearColParallelMerged(
                hidden_size, [self.conv_dim, self.value_dim], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_qkvz",
            )
            self.in_proj_ba = LinearColParallelMerged(
                hidden_size, [num_v_heads, num_v_heads], has_bias=False,
                quant_config=quant_config, prefix=f"{prefix}.in_proj_ba",
            )
        else:
            # Fused input projection (one GEMM instead of four): qkv | z | b | a, declared with the
            # FULL widths -- the layer shards them itself.
            self.in_proj = LinearColParallelMerged(
                hidden_size, [full_conv_dim, full_value_dim, num_v_heads, num_v_heads], has_bias=False,
                local_output_sizes=self._in_proj_split, quant_config=quant_config, prefix=f"{prefix}.in_proj",
            )
            assert sum(self._in_proj_split) == self.in_proj.local_output_size, (
                f"declared split {self._in_proj_split} does not tile the local GEMM width "
                f"{self.in_proj.local_output_size} at tp_size={tp_size}"
            )
        self.conv1d = _DepthwiseConv1d(self.conv_dim, conv_kernel_size)
        # Recurrence-gating params kept in fp32 (exp/softplus is precision-sensitive,
        # and the fla kernel reads them as fp32) -- matches HF/sglang, and avoids a
        # per-call .float() upcast in the decode wrapper. The weight loader exempts
        # *.A_log / *.dt_bias from the model-dtype downcast. One entry per v head -> rank-local.
        self.dt_bias = torch.empty(self.num_v_heads, dtype=torch.float32)
        self.A_log = torch.empty(self.num_v_heads, dtype=torch.float32)
        self.norm = GatedRMSNorm(head_v_dim, eps=rms_norm_eps, activation=output_gate)
        # Column-sharded v heads leave each rank a partial sum: row-parallel over the FULL value_dim,
        # all-reduced at TP>1 (a plain linear at TP=1).
        self.out_proj = LinearRowParallel(
            full_value_dim, hidden_size, has_bias=False, local_input_size=self.value_dim,
            quant_config=quant_config, prefix=f"{prefix}.out_proj",
        )

    def _gate_params(self, a: torch.Tensor, b: torch.Tensor):
        beta = b.sigmoid()
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)
        return g, beta

    def _conv_weight(self) -> torch.Tensor:
        return self.conv1d.weight.squeeze(1)  # [conv_dim, kernel] for the fused kernel

    def _conv_prefill(self, conv_in, pool, cu_seqlens, cache_indices, has_initial_state) -> torch.Tensor:
        """Varlen causal conv (fused sgl_kernel) with silu; reads/updates each request's
        conv state in place by ``cache_indices`` slot. ``conv_in`` [total, conv_dim].
        ``cu_seqlens`` / ``cache_indices`` / ``has_initial_state`` come from FLAMetadata."""
        li = pool.local_index(self.layer_id)
        x = conv_in.transpose(0, 1).contiguous()  # [conv_dim, total]
        out = causal_conv1d_varlen(x, self._conv_weight(), pool.conv_states[li],
                                   cu_seqlens, cache_indices, has_initial_state)
        return out.transpose(0, 1)  # [total, conv_dim]

    def _conv_decode(self, conv_in: torch.Tensor, table_idx: torch.Tensor, pool) -> torch.Tensor:
        """Single-token causal conv update (fused sgl_kernel) by ``table_idx`` slot;
        updates conv state in place, no host loop -> CUDA-graph capturable.
        ``conv_in`` [B, conv_dim] -> silu(conv) [B, conv_dim]."""
        li = pool.local_index(self.layer_id)
        return causal_conv1d_decode(conv_in, pool.conv_states[li], self._conv_weight(), table_idx)

    def _spec_conv_decode(self, conv_in: torch.Tensor, slots: torch.Tensor, pool, li: int, m: int):
        """``m`` rows per request: one conv launch over the rows, the conv state after every row but the last saved,
        and the recurrent kernel's per-step buffer. Returns ``(mixed [B*m, conv_dim], intermediate [reqs, SPEC_M-1,
        HV, V, K])``. Buffers are sized for SPEC_M rows and allocated at the first (largest) decode forward with
        several rows, which runs eagerly before any capture."""
        reqs = conv_in.shape[0] // m
        conv = pool.conv_states[li]
        inter_all, conv_all = _spec_buffers(pool, reqs, conv_in.device)
        self._spec_inter, self._spec_conv = inter_all[li], conv_all[li]
        self._spec_rows = _SPEC_ROWS[0]
        from freetoken.kernel.triton.causal_conv1d_triton import causal_conv1d_decode_rows

        # one launch for the m rows, the state after each row but the last written into the rollback snapshots
        mixed = causal_conv1d_decode_rows(conv_in.view(reqs, m, -1), conv, self._conv_weight(), slots,
                                          snapshots=self._spec_conv[:, :reqs])
        return mixed.view(reqs * m, -1), self._spec_inter[:reqs]

    def _write_track_snapshot(self, pool, li: int, conv_in: torch.Tensor,
                              h: torch.Tensor | None, fla, h_rows: torch.Tensor | None = None) -> None:
        """Snapshot this layer's recurrent + conv state at the chunk-aligned track boundary
        into a donatable pool slot, on the forward stream (hybrid-radix extra_buffer path).
        SSM: ``recurrent_states[li, dst] = h[0, h_row]`` -- a DIRECT copy (h is [V,K], the
        state pool is [K,V]; they coincide because GDN requires head_k_dim == head_v_dim).
        Conv: the last (kernel-1) raw conv-input timesteps ending at the boundary."""
        rec = pool.recurrent_states[li]
        if h_rows is None:  # the blocked prefill hands over the tracked rows only
            h_rows = h[0, fla.track_h_row]
        rec.index_copy_(0, fla.track_dst, h_rows.to(rec.dtype))
        cv = pool.conv_states[li]
        # conv_in [total, conv_dim]; gather the (kernel-1) window per tracked req.
        conv_win = conv_in[fla.track_conv_src].transpose(-1, -2).contiguous()  # [nt, conv_dim, K-1]
        cv.index_copy_(0, fla.track_dst, conv_win.to(cv.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        pool = ctx.linear_state_pool
        total = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
        # built once and shared by all GDN layers. The scheduler/graph set it; build it
        # lazily here (cached on the batch) for direct-op callers (tests).
        fla = batch.fla_metadata
        if fla is None:
            from freetoken.attention.linear import build_fla_metadata

            fla = build_fla_metadata(batch, hidden_states.device)
            batch.fla_metadata = fla

        if self._split_in_proj:
            qkvz = self.in_proj_qkvz.forward(hidden_states)
            conv_in, z = torch.split(qkvz, [self.conv_dim, self.value_dim], dim=-1)
            ba = self.in_proj_ba.forward(hidden_states)
            b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        else:
            proj = self.in_proj.forward(hidden_states)
            conv_in, z, b, a = torch.split(proj, self._in_proj_split, dim=-1)
        z = z.reshape(total, self.num_v_heads, self.head_v_dim)
        li = pool.local_index(self.layer_id)

        if batch.is_decode:
            # Fused fla decode kernel: gating + in-kernel l2norm + recurrent update +
            # per-request state read/write-by-index, all in one kernel (no gather/scatter,
            # no clone, no external l2norm). q/k stay at num_k_heads (kernel handles GQA).
            inter = None
            m = rows_of(batch)
            if m > 1:  # m rows per request (spec_decode): one conv launch, states kept for the rollback
                mixed, inter = self._spec_conv_decode(conv_in, fla.cache_indices, pool, li, m)
            else:
                mixed = self._conv_decode(conv_in, fla.cache_indices, pool)  # [B, conv_dim]
            B = mixed.shape[0]
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, B, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, B, self.num_v_heads, self.head_v_dim).to(dtype)
            core_out = gdn_decode_fla(
                q, k, v, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
                state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                intermediate_states=inter, intermediate_indices=None if inter is None else self._spec_rows[: inter.shape[0]],
            )
        else:
            mixed = self._conv_prefill(
                conv_in, pool, fla.cu_seqlens, fla.cache_indices, fla.has_initial_state)
            # fla chunk handles GQA in-kernel: q/k stay at num_k_heads, v at num_v_heads.
            qf, kf, vf = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = qf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            k = kf.reshape(1, total, self.num_k_heads, self.head_k_dim).to(dtype)
            v = vf.reshape(1, total, self.num_v_heads, self.head_v_dim).to(dtype)
            g, beta = self._gate_params(a, b)
            g = g.reshape(1, total, self.num_v_heads)
            beta = beta.float().reshape(1, total, self.num_v_heads)
            # The chunk kernel reads + writes back initial_state[cache_indices] in place;
            # fresh sequences (cached_len==0) must start from a zeroed slot.
            if fla.fresh_state_indices is not None:
                pool.recurrent_states[li].index_fill_(0, fla.fresh_state_indices, 0.0)
            track = fla.track_dst is not None
            if _GDN_PREFILL_BLOCK > 0 and fla.seq_bounds is not None and total > _GDN_PREFILL_BLOCK:
                core_out, h_rows = _gdn_prefill_blocked(q, k, v, g, beta, pool.recurrent_states[li], fla,
                                                        self.head_k_dim ** -0.5, track)
                if track:
                    self._write_track_snapshot(pool, li, conv_in, None, fla, h_rows=h_rows)
            else:
                result = gdn_prefill_chunk_fla(
                    q, k, v, g, beta,
                    state_source=pool.recurrent_states[li], indices=fla.cache_indices,
                    cu_seqlens=fla.cu_seqlens, scale=self.head_k_dim ** -0.5,
                    return_h=track,
                )
                if track:
                    core_out, h = result
                    self._write_track_snapshot(pool, li, conv_in, h, fla)
                else:
                    core_out = result

        core_out = core_out.reshape(-1, self.head_v_dim)
        if not _FEWER_COPIES:
            z = z.reshape(-1, self.head_v_dim)  # a copy when z is a strided slice of several rows
        # decode: one row per norm program, so a verify step's rows (and a batch's) get their single-row bits
        out = self.norm.forward(core_out, z, rows_per_block=1 if batch.is_decode else None).reshape(total, -1)
        return self.out_proj.forward(out)


def _gdn_blocks(fla, device: torch.device) -> list:
    """(start, end, request, cu_seqlens) per block of each request's prefill rows; built once per forward from the host
    bounds (one pinned non-blocking copy) and shared by every GDN layer."""
    blocks = getattr(fla, "_gdn_blocks", None)
    if blocks is None:
        spans = [(a, min(a + _GDN_PREFILL_BLOCK, end), i)
                 for i, (start, end) in enumerate(fla.seq_bounds) for a in range(start, end, _GDN_PREFILL_BLOCK)]
        cu = torch.tensor([[0, b - a] for a, b, _ in spans], dtype=torch.int64,
                          pin_memory=torch.cuda.is_available()).to(device, non_blocking=True)
        blocks = fla._gdn_blocks = [(a, b, i, cu[j]) for j, (a, b, i) in enumerate(spans)]
    return blocks


def _gdn_prefill_blocked(q, k, v, g, beta, state: torch.Tensor, fla, scale: float, track: bool):
    """``gdn_prefill_chunk_fla`` over the whole prefill, a block of one request at a time (the state chained through its
    slot); returns (o, the h rows of the track entries or None)."""
    blocks = _gdn_blocks(fla, q.device)
    want: dict[int, list] = {}
    if track:
        for e, (i, c) in enumerate(fla.track_seq_chunk):
            start = fla.seq_bounds[i][0]
            for j, (a, b, ii, _) in enumerate(blocks):
                if ii == i and a - start <= c * 64 < b - start:
                    want.setdefault(j, []).append((e, c - (a - start) // 64))
                    break
    o = rows = None
    for j, (a, b, i, cu) in enumerate(blocks):
        res = gdn_prefill_chunk_fla(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], state_source=state,
                                    indices=fla.cache_indices[i:i + 1], cu_seqlens=cu, scale=scale, return_h=j in want)
        if j in want:
            res, h = res
            if rows is None:
                rows = h.new_empty((len(fla.track_seq_chunk), *h.shape[2:]))
            for e, local in want[j]:
                rows[e] = h[0, local]
        if o is None:
            o = res.new_empty((q.shape[1], *res.shape[1:]))
        o[a:b] = res
    return o, rows


__all__ = ["Qwen4ExpGatedDeltaNet"]
