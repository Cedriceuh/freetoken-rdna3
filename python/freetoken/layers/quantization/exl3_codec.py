"""EXL3 (exllamav3 trellis format) reference codec, in plain torch.

A linear layer is stored as ``trellis`` int16 ``[in/16, out/16, 16*K]`` (one 256-weight tile per
16x16 block, K bits per weight), ``suh`` / ``svh`` fp16 input / output channel scales (``su`` /
``sv`` packed sign bits in old checkpoints) and an empty ``mcg`` or ``mul1`` tensor that names the
codebook. The weight, ``[in, out]`` as in ``y = x @ W``, is

    W = diag(suh) @ H @ Wq @ H @ diag(svh)

with H the normalized 128-point Sylvester Hadamard applied per 128-block and Wq the decoded tiles.
A tile is a tail-biting ring of 256*K bits, read MSB-first from little-endian uint32 words; weight
j is the 16-bit window that ends at bit (j+1)*K, mapped through the codebook, and it lands at the
row-major tile position ``tensor_core_perm()[j]``.

This mirrors exllamav3's CUDA kernels (``exl3_dq.cuh``, ``codebook.cuh``, ``pack.cu``) and is the
oracle for the RDNA3 kernels: slow, exact, CPU or GPU.
"""

from __future__ import annotations

from functools import lru_cache

import torch

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
HAD_DIM = 128

_MUL_3INST, _ADD_3INST = 89226354, 64248484
_MUL_MCG = 0xCBAC1FED
_MUL_MUL1 = 0x83DCD12D
_MASK32 = 0xFFFFFFFF
# mul1: fp16(0x6400 + bytesum) = 1024 + bytesum, then one fp16 fma with these two constants
_MUL1_K_INV = 0x1EEE
_MUL1_K_BIAS = 0xC931


def codebook_of(tensors: dict) -> int:
    """Codebook named by the marker tensors of one EXL3 linear (``<key>.mcg`` / ``<key>.mul1``)."""
    has_mcg = any(k == "mcg" or k.endswith(".mcg") for k in tensors)
    has_mul1 = any(k == "mul1" or k.endswith(".mul1") for k in tensors)
    if has_mcg and has_mul1:
        raise ValueError("EXL3 linear names both the mcg and the mul1 codebook")
    return CB_MCG if has_mcg else CB_MUL1 if has_mul1 else CB_3INST


def trellis_bits(trellis: torch.Tensor) -> int:
    """Bits per weight K of a ``[.., 16*K]`` trellis; half-integer K (mul1 ``frac`` tiles) is not supported."""
    words = trellis.shape[-1]
    if words % 16:
        raise NotImplementedError(f"EXL3 half-integer bitrate ({words / 16} bpw) is not supported")
    K = words // 16
    if not 1 <= K <= 8:
        raise ValueError(f"EXL3 trellis of {words} words per tile is not 1..8 bits per weight")
    return K


@lru_cache(maxsize=None)
def _tensor_core_perm_cpu() -> torch.Tensor:
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = t // 4
        for half, c in enumerate((c0, c0 + 8)):
            for i, r in enumerate(rows):
                perm[t * 8 + half * 4 + i] = r * 16 + c
    return torch.tensor(perm, dtype=torch.long)


def tensor_core_perm(device: torch.device | str | None = None) -> torch.Tensor:
    """Row-major tile position of each of the 256 encoded weights (NVIDIA mma fragment order)."""
    perm = _tensor_core_perm_cpu()
    return perm if device is None else perm.to(device)


def _uint32_words(trellis: torch.Tensor) -> torch.Tensor:
    """Tile words as int64 holding uint32 values: int16 pairs read little-endian."""
    t = trellis.contiguous()
    if t.dtype != torch.int16:
        raise TypeError(f"EXL3 trellis must be int16, got {t.dtype}")
    return t.view(torch.int32).to(torch.int64) & _MASK32


def unpack_states(trellis: torch.Tensor) -> torch.Tensor:
    """The 256 16-bit trellis states of every tile, in encoded order: ``[.., 16*K] -> [.., 256]`` int64."""
    K = trellis_bits(trellis)
    g = _uint32_words(trellis)
    n_words = 8 * K
    j = torch.arange(256, dtype=torch.int64, device=trellis.device)
    b0 = j * K + K - 16 + 256 * K  # first bit of the window, kept non-negative across the ring
    b1 = b0 + 16
    i0 = b0 // 32
    i1 = (b1 - 1) // 32  # == i0 when the window sits in one word
    shift = (i1 + 1) * 32 - b1
    a = g[..., i0 % n_words]
    b = g[..., i1 % n_words]
    return (((a << 32) | b) >> shift) & 0xFFFF


def _halves(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The two fp16 values packed in a uint32 (low, high), as float64."""
    lo = (x & 0xFFFF).to(torch.int32).to(torch.int16)
    hi = ((x >> 16) & 0xFFFF).to(torch.int32).to(torch.int16)
    return lo.view(torch.float16).double(), hi.view(torch.float16).double()


def _fp16_const(bits: int) -> float:
    return float(torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(torch.float16))


def decode_states(states: torch.Tensor, codebook: int) -> torch.Tensor:
    """Codebook value of each 16-bit state, fp16, rounded as the CUDA kernels round it.

    Sums and the mul1 fma are computed exactly in float64 and rounded once to fp16, which is what
    ``__hadd`` / ``__hfma`` do."""
    s = states.to(torch.int64)
    if codebook == CB_MUL1:
        x = (s * _MUL_MUL1) & _MASK32
        bytesum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)
        h = (1024 + bytesum).double()
        return (h * _fp16_const(_MUL1_K_INV) + _fp16_const(_MUL1_K_BIAS)).half()
    if codebook == CB_MCG:
        x = (s * _MUL_MCG) & _MASK32
    elif codebook == CB_3INST:
        x = (s * _MUL_3INST + _ADD_3INST) & _MASK32
    else:
        raise ValueError(f"unknown EXL3 codebook {codebook}")
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60  # lop3 0x6a: (a & b) ^ c
    lo, hi = _halves(x)
    return (lo + hi).half()


def decode_tiles(trellis: torch.Tensor, codebook: int) -> torch.Tensor:
    """Decoded weights Wq ``[in, out]`` fp16 from a ``[in/16, out/16, 16*K]`` trellis (no Hadamard, no scales)."""
    if trellis.dim() != 3:
        raise ValueError(f"EXL3 trellis must be 3-D, got shape {tuple(trellis.shape)}")
    A, B, _ = trellis.shape
    vals = decode_states(unpack_states(trellis), codebook)  # [A, B, 256], encoded order
    inv = torch.argsort(tensor_core_perm(trellis.device))
    tiles = vals[..., inv].view(A, B, 16, 16)  # row-major tile: [a, b, r, c] -> W[16a + r, 16b + c]
    return tiles.permute(0, 2, 1, 3).reshape(A * 16, B * 16)


@lru_cache(maxsize=None)
def _hadamard_cpu(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / (n**0.5)


def hadamard_128(device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Normalized 128-point Sylvester Hadamard (symmetric, its own inverse)."""
    return _hadamard_cpu(HAD_DIM).to(device=device, dtype=dtype)


def unpack_signs(bitfield: torch.Tensor) -> torch.Tensor:
    """Old-checkpoint ``su`` / ``sv``: int16 words of sign bits (bit b of word w -> element 16w + b, set = -1)."""
    bits = bitfield.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    masks = 1 << torch.arange(16, dtype=torch.int32, device=bits.device)
    neg = ((bits.unsqueeze(-1) & masks) != 0).flatten()
    return (1.0 - 2.0 * neg.to(torch.float32)).half()


def channel_scales(tensors: dict, side: str) -> torch.Tensor:
    """``suh`` (side "u") or ``svh`` (side "v") of one linear, unpacking ``su`` / ``sv`` when that is what is stored."""
    for key, value in tensors.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf == f"s{side}h":
            return value
    for key, value in tensors.items():
        leaf = key.rsplit(".", 1)[-1]
        if leaf == f"s{side}":
            return unpack_signs(value)
    raise KeyError(f"EXL3 linear has neither s{side}h nor s{side}")


def reconstruct(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    codebook: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Full weight ``W [in, out]`` in the original basis (``y = x @ W``)."""
    wq = decode_tiles(trellis, codebook).to(torch.float32)
    k, n = wq.shape
    if k % HAD_DIM or n % HAD_DIM:
        raise ValueError(f"EXL3 weight {k}x{n} is not a whole number of {HAD_DIM}-blocks")
    if suh.numel() != k or svh.numel() != n:
        raise ValueError(f"EXL3 scales {suh.numel()}/{svh.numel()} do not match weight {k}x{n}")
    had = hadamard_128(wq.device)
    w = (had @ wq.view(k // HAD_DIM, HAD_DIM, n)).view(k, n)
    w = w * suh.to(torch.float32).view(k, 1)
    w = (w.view(k, n // HAD_DIM, HAD_DIM) @ had).view(k, n)
    w = w * svh.to(torch.float32).view(1, n)
    return w.to(dtype)


def reconstruct_linear(tensors: dict, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``reconstruct`` from the checkpoint tensors of one linear, keyed by leaf name or full key."""
    trellis = next(v for k, v in tensors.items() if k.rsplit(".", 1)[-1] == "trellis")
    return reconstruct(
        trellis, channel_scales(tensors, "u"), channel_scales(tensors, "v"), codebook_of(tensors), dtype=dtype
    )


def pack_trellis(symbols: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse of the ring layout: K-bit symbols ``[.., 256]`` -> trellis words ``[.., 16*K]`` int16.

    Port of exllamav3's ``pack_trellis_kernel`` (MSB-first 16-bit words, halves swapped per uint32);
    used to build synthetic checkpoints and to test ``unpack_states``."""
    if symbols.shape[-1] != 256:
        raise ValueError("EXL3 tiles hold 256 symbols")
    lead = symbols.shape[:-1]
    v = symbols.to(torch.int64) & ((1 << K) - 1)
    bit_idx = torch.arange(K - 1, -1, -1, dtype=torch.int64, device=v.device)
    stream = ((v.unsqueeze(-1) >> bit_idx) & 1).reshape(*lead, 16 * K, 16)
    weights = 1 << torch.arange(15, -1, -1, dtype=torch.int64, device=v.device)
    words = (stream * weights).sum(-1)  # [.., 16K] uint16 values, stream order
    words = words.view(*lead, 8 * K, 2).flip(-1).reshape(*lead, 16 * K)
    return words.to(torch.int32).to(torch.int16)


def ngram_ring_states(packed: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """exllamav3 ``exl3_ngram_trellis`` rows (one tail-biting ring per row, mul1): ``[N, 1 + dim*K/16]`` int16 ->
    (``[N, dim]`` 16-bit states, ``[N]`` fp16 row scales).

    Word 0 holds the scale's bits; the other words are the ring's little-endian bitstream, where bits
    ``[i*K, (i+1)*K)`` are the low K bits of position i's state and its higher bits are the symbols of the preceding
    positions (mod dim)."""
    words = packed.shape[1] - 1
    dim = words * 16 // K
    scales = packed[:, 0].contiguous().view(torch.float16)
    stream = packed[:, 1:].contiguous().to(torch.int64) & 0xFFFF
    bit = torch.arange(dim, device=packed.device) * K
    window = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % words] << 16)
    symbols = (window >> (bit & 15)) & ((1 << K) - 1)
    states = torch.zeros_like(symbols)
    for j in range((15 + K) // K):
        states |= torch.roll(symbols, j, dims=1) << (j * K)
    return states & 0xFFFF, scales


def decode_ngram_rows(packed: torch.Tensor, K: int, bias: torch.Tensor) -> torch.Tensor:
    """n-gram rows ``mul1(state) * scale + bias`` (fp32); ``bias`` is each row's head bias, ``[N, dim]``."""
    states, scales = ngram_ring_states(packed, K)
    return decode_states(states, CB_MUL1).float() * scales.float().unsqueeze(1) + bias.float()


__all__ = [
    "CB_3INST",
    "CB_MCG",
    "CB_MUL1",
    "HAD_DIM",
    "channel_scales",
    "codebook_of",
    "decode_states",
    "decode_ngram_rows",
    "decode_tiles",
    "hadamard_128",
    "ngram_ring_states",
    "pack_trellis",
    "reconstruct",
    "reconstruct_linear",
    "tensor_core_perm",
    "trellis_bits",
    "unpack_signs",
    "unpack_states",
]
