"""spec_decode verify steps allocate their draft rows ahead of device_len: the next step must not allocate those
pages again, and a finishing request returns them (CPU, real CacheManager)."""
from __future__ import annotations

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager

PAGE = 4


def _req(cm, ids, out_len):
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=0, output_len=out_len,
              uid=0, sampling_params=SamplingParams(), cache_handle=cm.match_req(_pend(ids)).cuda_handle)
    cm.lock(req.cache_handle)
    return req


def _pend(ids):
    from types import SimpleNamespace

    return SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids))


def _spec_step(cm, req, m):
    """One verify step the way the scheduler prepares it: m rows allocated, then one token kept."""
    req.device_len += m - 1
    cm.allocate_paged([req])
    req.device_len -= m - 1
    req.complete_one()


def test_ahead_pages_are_allocated_once_and_freed_at_finish():
    for cache_type in ("naive", "radix"):
        page_table = torch.zeros(2, 64, dtype=torch.int32)
        cm = CacheManager(16, PAGE, page_table, cache_type)
        req = _req(cm, [1, 2, 3, 4, 5, 6], out_len=20)
        cm.allocate_paged([req])  # the prompt: positions 0..5, two pages
        req.complete_one()        # prefill done: cached 6, device 7
        for _ in range(9):        # positions 6..14 decoded, each step also allocating the next row
            _spec_step(cm, req, m=2)
        rows = page_table[0, : req.cached_len + 1]
        pages = (rows // PAGE).tolist()
        assert pages == sorted(pages) and len(set(pages)) == -(-(req.cached_len + 1) // PAGE)
        # no page handed out twice: every allocated page left the free list exactly once
        assert len(cm.free_slots) == 16 - len(set(pages))
        cm.cache_req(req, finished=True)
        cm.check_integrity()


def test_finish_on_a_page_boundary_returns_the_page_of_the_rejected_draft():
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(16, PAGE, page_table, "radix")
    req = _req(cm, [1, 2, 3, 4, 5, 6, 7], out_len=10)  # 7 prompt tokens: the draft row at position 8 opens page 2
    cm.allocate_paged([req])
    req.complete_one()
    _spec_step(cm, req, m=2)   # decodes position 7, its draft row lands on position 8 (page 2)
    assert req.cached_len == 8 and req.alloc_len == 9
    cm.cache_req(req, finished=True)
    cm.check_integrity()
