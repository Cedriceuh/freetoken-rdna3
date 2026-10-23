import pytest
import torch

from freetoken.kernel import indexing, store_cache
from freetoken.kernel.index import num_splits_for


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA or ROCm GPU is required"
)


def test_indexing_jit_matches_torch_on_cold_and_warm_loads():
    weights = torch.arange(8 * 64, dtype=torch.float32, device="cuda").reshape(8, 64)

    for values in ([7, 2, 0], [1, 6, 3]):
        indices = torch.tensor(values, dtype=torch.int32, device="cuda")
        actual = indexing(weights, indices)
        torch.testing.assert_close(actual, weights[indices.long()])


@pytest.mark.parametrize(("width", "expected_splits"), [(256, 2), (512, 4)])
def test_indexing_jit_copies_rows_split_across_multiple_warps(width, expected_splits):
    weights = torch.arange(8 * width, dtype=torch.float32, device="cuda").reshape(8, width)
    assert num_splits_for(weights.shape[1] * weights.element_size()) == expected_splits

    for values in ([7, 2, 0], [1, 6, 3]):
        indices = torch.tensor(values, dtype=torch.int64, device="cuda")
        actual = indexing(weights, indices)
        torch.testing.assert_close(actual, weights[indices])


def test_masked_indexing_zeros_indices_outside_vocab_range():
    width = 256
    weights = torch.arange(6 * width, dtype=torch.float32, device="cuda").reshape(6, width)
    indices = torch.tensor([9, 10, 15, 16], dtype=torch.int32, device="cuda")

    actual = indexing(weights, indices, vocab_range=(10, 6))
    expected = torch.stack(
        (torch.zeros_like(weights[0]), weights[0], weights[5], torch.zeros_like(weights[0]))
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_jit_matches_torch_on_cold_and_warm_loads(index_dtype):
    k_cache = torch.zeros((8, 64), dtype=torch.float32, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    indices = torch.tensor([5, 0, 3], dtype=index_dtype, device="cuda")

    for offset in (0.0, 1000.0):
        k = torch.arange(3 * 64, dtype=torch.float32, device="cuda").reshape(3, 64)
        k = k + offset
        v = k + 500.0
        store_cache(k_cache, v_cache, indices, k, v)
        torch.testing.assert_close(k_cache[indices], k)
        torch.testing.assert_close(v_cache[indices], v)

    untouched = torch.tensor([1, 2, 4, 6, 7], device="cuda")
    torch.testing.assert_close(k_cache[untouched], torch.zeros((5, 64), device="cuda"))
    torch.testing.assert_close(v_cache[untouched], torch.zeros((5, 64), device="cuda"))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_takes_k_and_v_with_their_own_row_strides(index_dtype):
    """The attention's v is a column slice of the qkv projection's rows (FREETOKEN_FEWER_COPIES stores it from there):
    the store reads k and v each with its own row stride and writes the bytes a contiguous copy would."""
    rows, width = 5, 512
    qkv = torch.randn((rows, 4 * width + 2 * width), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")  # the rope's output: contiguous
    v = qkv[:, 4 * width + width :]
    assert v.stride(0) != k.stride(0)
    indices = torch.tensor([7, 2, 9, 0, 4], dtype=index_dtype, device="cuda")
    k_cache = torch.zeros((12, width), dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    store_cache(k_cache, v_cache, indices, k, v)
    assert torch.equal(k_cache[indices], k)
    assert torch.equal(v_cache[indices], v.contiguous())
    untouched = torch.tensor([1, 3, 5, 6, 8, 10, 11], device="cuda")
    assert not k_cache[untouched].any() and not v_cache[untouched].any()
