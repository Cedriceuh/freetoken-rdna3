"""FREETOKEN_QSA_KV_HOST: host-mapped device tensors and the prefill page staging give the bits of VRAM (GPU)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_host_mapped_tensor_reads_and_writes_in_place() -> None:
    from freetoken.kernel.host_mapped import host_mapped_empty

    h = host_mapped_empty((4096, 256), torch.bfloat16, torch.device("cuda", torch.cuda.current_device()))
    assert h.is_cuda
    ref = torch.randn(4096, 256, device="cuda").bfloat16()
    h.copy_(ref)
    idx = torch.randint(0, 4096, (777,), device="cuda")
    assert torch.equal(h[idx], ref[idx])  # a GPU gather across PCIe
    h[idx] = -ref[idx]  # a GPU scatter into host memory
    ref[idx] = -ref[idx]
    assert torch.equal(h.cpu(), ref.cpu())


def test_stage_pages_copies_exactly_the_pages_asked() -> None:
    from freetoken.kernel.host_mapped import host_mapped_empty
    from freetoken.kernel.triton.qsa.stage import stage_pages

    pages, shape = 300, (64, 1, 256)
    h = host_mapped_empty((pages, *shape), torch.bfloat16, torch.device("cuda", torch.cuda.current_device()))
    h.copy_(torch.randn(pages, *shape, device="cuda").bfloat16())
    dst = torch.zeros(pages, *shape, device="cuda", dtype=torch.bfloat16)
    want = torch.tensor([5, 0, 299, 17, 17, 120], device="cuda", dtype=torch.int32)  # repeats are harmless
    stage_pages(h, dst, want)
    hit = torch.zeros(pages, dtype=torch.bool, device="cuda")
    hit[want.long()] = True
    assert torch.equal(dst[hit], h[hit])
    assert not dst[~hit].any()
