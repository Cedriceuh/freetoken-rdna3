"""The EXL3 kernels' trellis decoding against the reference (exl3_codec), in the forms FREETOKEN_EXL3_FAST_MUL1 and
FREETOKEN_EXL3_ALIGNBIT select: the mul1 codebook gives every one of the 65536 states its value (the exact fma
rounded once to fp16), with the RDNA3 dot4 byte sum and with the plain integer one, and the tile dequant (the states'
windows, every codebook, 2 to 8 bits) the reference's tiles. Each form in its own process: the flags are read at
import."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_CHILD = textwrap.dedent(
    """
    import torch
    import triton
    import triton.language as tl

    from freetoken.kernel.triton import exl3 as X
    from freetoken.layers.quantization import exl3_codec as ex


    @triton.jit
    def _codebook(out, DOT4: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + i, X._exl3_codebook(i.to(tl.uint32), 2, DOT4))


    ref = ex.decode_states(torch.arange(65536), ex.CB_MUL1).float().view(torch.int32)
    for dot4 in sorted({False, X._dot4(torch.device("cuda"))}):
        out = torch.empty(65536, device="cuda")
        _codebook[(64,)](out, DOT4=dot4, BLOCK=1024)
        bad = (out.cpu().view(torch.int32) != ref).nonzero().flatten()
        assert bad.numel() == 0, f"dot4={dot4}: {bad.numel()} states differ, first {bad[:8].tolist()}"
    g = torch.Generator().manual_seed(0)
    for K in range(2, 9):
        trellis = torch.randint(-32768, 32768, (4, 8, 16 * K), dtype=torch.int16, generator=g)
        for cb in (ex.CB_3INST, ex.CB_MCG, ex.CB_MUL1):
            got = X.exl3_dequant(trellis.cuda(), cb).cpu()
            assert torch.equal(got.view(torch.int16), ex.decode_tiles(trellis, cb).view(torch.int16)), (K, cb)
    print("ok", bool(X._FAST_MUL1), bool(X._ALIGNBIT))
    """
)


@pytest.mark.parametrize("form", ["1", "0"])
def test_trellis_decoding_gives_the_reference_values(tmp_path, form):
    script = tmp_path / "decode.py"
    script.write_text(_CHILD)
    env = dict(os.environ, FREETOKEN_EXL3_FAST_MUL1=form, FREETOKEN_EXL3_ALIGNBIT=form)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    on = str(form == "1")
    assert r.stdout.split() == ["ok", on, str(form == "1" and torch.version.hip is not None)], r.stdout
