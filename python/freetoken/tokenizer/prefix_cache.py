"""Token reuse across the prompts of one conversation.

An agent resends its whole conversation every turn, and encoding it costs time in proportion to its length (one core,
~0.4 s at 174k tokens and ~0.5 s at 224k on a Threadripper 3970X). This worker also detokenizes the running requests
between two encodes, so that time delays every stream as well. Each turn usually extends the previous prompt, so only
its new tail needs encoding.

A fast (HF ``tokenizers``) tokenizer splits its input on the added tokens before it normalizes, pre-tokenizes and
encodes the text between them, and each such piece is encoded on its own. The tokens up to and including an added
token that strips no whitespace therefore depend only on the text up to it, and the tokens after it only on the text
after it. :class:`PromptPrefixCache` keeps the last prompts it encoded and, for a new prompt, finds the longest common
prefix with one of them, takes that prompt's tokens up to the last cut token (``<|im_start|>``-like) inside the shared
part and encodes only the rest: the same tokens as encoding the whole prompt.

FREETOKEN_TOKENIZE_PREFIX_CACHE=n: keep the last n prompts (default 8; 0 = off).
FREETOKEN_TOKENIZE_PREFIX_CHECK=1: also encode every prompt whole, log a mismatch and use the whole encoding (tests).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import List

import torch
from freetoken.utils import init_logger

logger = init_logger(__name__)

_ENTRIES = int(os.environ.get("FREETOKEN_TOKENIZE_PREFIX_CACHE", "8"))
_CHECK = os.environ.get("FREETOKEN_TOKENIZE_PREFIX_CHECK", "0") == "1"
# prompts shorter than this are encoded whole and not kept (the encode costs well under a millisecond)
_MIN_CHARS = 16384
_CUT_CANDIDATES = ("<|im_start|>", "<|start_header_id|>", "<|turn_start|>", "<start_of_turn>")
_PROBES = ("a", " a", "  a", "\n", "\n\nx", "user\nhello world", "assistant\n<think>\n\n</think>\n\n", "123 456",
           "́e", "̸x", "é", "日本語", "  \t\n x", "<|im_end|>\n", "'s", "", " ")


def _common_prefix(a: str, b: str) -> int:
    """Length of the longest common prefix of ``a`` and ``b`` (block compares, then a binary search in the block)."""
    n = min(len(a), len(b))
    lo, step = 0, 1 << 16
    while lo < n:
        hi = min(n, lo + step)
        if a[lo:hi] == b[lo:hi]:
            lo = hi
            continue
        while hi - lo > 1:  # a[:lo] == b[:lo], a[:hi] != b[:hi]
            mid = (lo + hi) // 2
            if a[lo:mid] == b[lo:mid]:
                lo = mid
            else:
                hi = mid
        return lo
    return n


class _Entry:
    __slots__ = ("text", "ids", "char_ends", "tok_ends")

    def __init__(self, text: str, ids: torch.Tensor, char_ends: List[int], tok_ends: List[int]) -> None:
        self.text, self.ids, self.char_ends, self.tok_ends = text, ids, char_ends, tok_ends


class PromptPrefixCache:
    """Encodes templated prompts (no special tokens added) reusing the tokens of a recent prompt's prefix; see the
    module docstring. ``usable`` is False when the tokenizer gives no such guarantee (then every call encodes whole)."""

    def __init__(self, tokenizer, entries: int = _ENTRIES) -> None:
        self.tokenizer = tokenizer
        self.entries = entries
        self._cache: OrderedDict[int, _Entry] = OrderedDict()
        self._next = 0
        self.cut, self.cut_id = None, None
        self.usable = entries > 0 and self._pick_cut()
        self._lock = threading.Lock()  # the frontend's count_tokens runs tokenize in worker threads
        self.reused = 0  # tokens the last encode took from a cached prompt

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _pick_cut(self) -> bool:
        tok = self.tokenizer
        if not getattr(tok, "is_fast", False) or getattr(tok, "split_special_tokens", False):
            return False
        added = getattr(tok, "added_tokens_decoder", None) or {}
        for cand in _CUT_CANDIDATES:
            hit = [(i, t) for i, t in added.items() if getattr(t, "content", None) == cand]
            if len(hit) != 1:
                continue
            i, t = hit[0]
            if t.lstrip or t.rstrip or t.single_word or t.normalized:
                continue
            # no other added token may contain the cut text (it would win the match), and the split must hold on the
            # probes: encode(a + cut + b) == encode(a + cut) + encode(b)
            if any(cand in getattr(o, "content", "") and j != i for j, o in added.items()):
                continue
            if all(self._encode(a + cand + b) == self._encode(a + cand) + self._encode(b)
                   for a in _PROBES for b in _PROBES):
                self.cut, self.cut_id = cand, i
                return True
        return False

    def _entry(self, text: str, ids: torch.Tensor) -> _Entry | None:
        char_ends, pos = [], text.find(self.cut)
        while pos >= 0:
            char_ends.append(pos + len(self.cut))
            pos = text.find(self.cut, pos + 1)
        tok_ends = ((ids == self.cut_id).nonzero().view(-1) + 1).tolist()
        if len(tok_ends) != len(char_ends):  # the cut text did not always become the cut token: keep nothing
            return None
        return _Entry(text, ids, char_ends, tok_ends)

    def encode(self, text: str) -> torch.Tensor:
        """``tokenizer.encode(text, add_special_tokens=False)`` as an int32 vector."""
        if not self.usable or len(text) < _MIN_CHARS:
            return torch.tensor(self._encode(text), dtype=torch.int32)
        with self._lock:
            return self._encode_cached(text)

    def _encode_cached(self, text: str) -> torch.Tensor:
        best, best_key, best_len = None, None, 0
        for key, e in self._cache.items():
            first = e.char_ends[0] if e.char_ends else 0
            if first and first <= len(text) and text[:first] == e.text[:first]:
                n = _common_prefix(text, e.text)
                if n > best_len:
                    best, best_key, best_len = e, key, n
        ids, superseded = None, False
        self.reused = 0
        if best is not None:
            # the last cut token whose end lies inside the shared part
            lo, hi = 0, len(best.char_ends)
            while lo < hi:
                mid = (lo + hi) // 2
                if best.char_ends[mid] <= best_len:
                    lo = mid + 1
                else:
                    hi = mid
            if lo:
                c, t = best.char_ends[lo - 1], best.tok_ends[lo - 1]
                ids = torch.cat([best.ids[:t], torch.tensor(self._encode(text[c:]), dtype=torch.int32)])
                # the new prompt holds the old one up to its last cut: whatever shares a prefix with the old one
                # shares as much with the new one, so the old entry can go
                superseded = lo == len(best.char_ends)
                self.reused = t
        if ids is None or _CHECK:
            whole = torch.tensor(self._encode(text), dtype=torch.int32)
            if ids is not None and not torch.equal(ids, whole):
                logger.warning("tokenize prefix cache: %d tokens differ from the whole encoding (%d); using the whole",
                               ids.numel(), whole.numel())
            ids = whole
        e = self._entry(text, ids)
        if e is not None:
            if superseded:
                del self._cache[best_key]
            self._cache[self._next] = e
            self._next += 1
            while len(self._cache) > self.entries:
                self._cache.popitem(last=False)
        elif best_key is not None:
            self._cache.move_to_end(best_key)
        return ids


__all__ = ["PromptPrefixCache"]
