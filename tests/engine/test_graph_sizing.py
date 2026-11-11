"""The eager forwards before the CUDA graph captures size the spec_decode buffers for every captured shape."""

from freetoken.engine.graph import capture_shapes, sizing_shapes, verify_requests


def _shapes(ms, bs_list, verify_bs, deep_bs):
    return capture_shapes(ms, bs_list, verify_bs, deep_bs)


def test_sizing_covers_the_most_rows_the_most_verify_requests_and_the_most_states():
    # 4 rows only up to 2 requests, 2 rows up to 4: (4, 2) and (2, 4) have 8 rows each; the requests' buffers need 4,
    # the rollback states the 6 of (4, 2) (two requests x three rows kept; (2, 4) keeps 4)
    shapes = _shapes((1, 2, 3, 4), [1, 2, 3, 4], verify_bs=4, deep_bs=2)
    assert sizing_shapes(shapes) == [(2, 4), (4, 2)]
    # 4 rows up to 3 requests, 2 up to 4: the most rows (12) and the most requests (4) are two shapes
    shapes = _shapes((1, 2, 3, 4), [1, 2, 3, 4], verify_bs=4, deep_bs=3)
    assert sizing_shapes(shapes) == [(4, 3), (2, 4)]
    # xtx-xt's setting (verify up to 2 requests): one shape has both
    shapes = _shapes((1, 2, 3, 4), [1, 2, 3, 4], verify_bs=2, deep_bs=2)
    assert sizing_shapes(shapes) == [(4, 2)]
    assert sizing_shapes([(1, 1), (1, 2), (1, 4)]) == []  # no verify rows: nothing to size


def test_deep_rows_stop_at_their_batch_size():
    shapes = capture_shapes((1, 2, 3, 4), [1, 2, 3, 4], 4, 2)
    assert {bs for m, bs in shapes if m > 2} == {1, 2} and {bs for m, bs in shapes if m <= 2} == {1, 2, 3, 4}
    assert capture_shapes((1, 2, 3, 4), [1, 2, 4], 2, 2) == [(m, bs) for m in (1, 2, 3, 4) for bs in (1, 2, 4)
                                                             if m == 1 or bs <= 2]


def test_a_server_of_fewer_requests_verifies_up_to_its_own_count():
    # a profile's FREETOKEN_SPEC_BS_MAX=4 under --max-running-requests 3 --cuda-graph-max-bs 3: three requests at most
    assert verify_requests(4, 3) == 3 and verify_requests(4, 8) == 4 and verify_requests(2, None) == 2
    # and its graphs are those of FREETOKEN_SPEC_BS_MAX=3: two rows up to three requests, the deeper ones up to two
    assert capture_shapes((1, 2, 3, 4), [1, 2, 3], 4, 2) == capture_shapes((1, 2, 3, 4), [1, 2, 3], 3, 2)
