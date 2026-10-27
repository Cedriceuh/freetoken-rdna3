"""FREETOKEN_DRAFT_VOCAB: the MTP head's decode drafts scored over part of the vocabulary, always with the tokenizer's
added tokens (end of turn, think and tool-call tokens): ``latin`` (the two-card profiles) the tokens whose text is
Latin script, whitespace, punctuation or the symbols code and markdown use, plus the partial UTF-8 characters
(143,589 of 248,320 for Qwen3.8-Flash-Next, read from ``tokenizer.json``); a count n the first n ids (byte-level BPE
gives the first ids to the most frequent merges, but French words have ids above 160,000); or the path of a JSON list
of token ids.

The verify step keeps the whole LM head, so greedy answers are the target's whatever the drafts are; a sampled draft is
drawn from the reduced head's distribution, which exact speculative sampling takes as it comes. The ids are split over
the ranks in the proportions of their LM head shards, in id order; a rank holds its part's int8 rows (gathered once at
start over the CPU group).
"""

from __future__ import annotations

import json
import os

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# latin, a count n (the first n ids and the added ones) or the path of a JSON list of token ids
DRAFT_VOCAB = os.environ.get("FREETOKEN_DRAFT_VOCAB", "0")


def added_token_ids(model_path: str) -> list[int]:
    """The tokenizer's added tokens (``tokenizer.json``), which the reduced head always scores."""
    try:
        with open(os.path.join(model_path, "tokenizer.json"), encoding="utf-8") as f:
            return sorted(int(t["id"]) for t in json.load(f).get("added_tokens", []))
    except (OSError, ValueError, KeyError):
        return []


# Latin script, whitespace, punctuation and the symbols code and markdown use (FREETOKEN_DRAFT_VOCAB=latin)
_LATIN_RANGES = ((0x0000, 0x024F), (0x1E00, 0x1EFF), (0x2000, 0x206F), (0x20A0, 0x20CF), (0x2100, 0x214F),
                 (0x2190, 0x23FF), (0x2500, 0x25FF), (0x2600, 0x27BF), (0xFE0F, 0xFE0F), (0x1F300, 0x1FAFF))


def _byte_decoder() -> dict[str, int]:
    """Byte-level BPE's printable stand-ins -> the bytes they encode (GPT-2's bytes_to_unicode)."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("\xa1"), ord("\xac") + 1), *range(ord("\xae"), ord("\xff") + 1)]
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def latin_token_ids(model_path: str) -> list[int]:
    """The vocabulary's tokens whose text is Latin script (``_LATIN_RANGES``), and those of a partial UTF-8 character
    (a byte of any script). From ``tokenizer.json``'s byte-level BPE vocabulary."""
    with open(os.path.join(model_path, "tokenizer.json"), encoding="utf-8") as f:
        vocab = json.load(f)["model"]["vocab"]
    dec = _byte_decoder()
    out = []
    for piece, i in vocab.items():
        try:
            text = bytes(dec[ch] for ch in piece).decode("utf-8")
        except (KeyError, UnicodeDecodeError):  # not byte-level, or part of a character
            out.append(int(i))
            continue
        if all(any(a <= ord(ch) <= b for a, b in _LATIN_RANGES) for ch in text):
            out.append(int(i))
    return sorted(out)


def _even_shards(vocab: int, tp: int) -> list[tuple[int, int]]:
    per = -(-vocab // tp)
    return [(r * per, max(0, min(per, vocab - r * per))) for r in range(tp)]


class DraftVocabHead:
    """This rank's part of the reduced head: int8 rows, their fp32 scales and their global ids (ascending)."""

    def __init__(self, lm_head, weight: torch.Tensor, weight_scale: torch.Tensor, ids: torch.Tensor):
        self.lm_head = lm_head
        self.weight = weight
        self.weight_scale = weight_scale
        self.bias = None
        self.ids = ids  # int32, on the device
        self.num_embeddings = lm_head.num_embeddings
        self.tp_size = lm_head.tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.layers.quantization.int8_weight_only import Int8WeightOnlyLinearMethod

        return Int8WeightOnlyLinearMethod().apply(self, x)

    def candidates(self, logits: torch.Tensor, k: int, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """As ``ParallelLMHead.candidates`` over the reduced vocabulary: ``(vals, global ids, argmax)``, every rank."""
        from freetoken.kernel.triton.vocab_topk import merge_candidates, pack_candidates

        packed = pack_candidates(logits, k, 0, self.ids.numel(), ids_map=self.ids)
        gathered = self.lm_head._comm.all_gather(packed) if self.tp_size > 1 else packed
        return merge_candidates(gathered, self.tp_size, k, rows)


def build_draft_head(lm_head, model_path: str, cpu_group, n: int | str = DRAFT_VOCAB) -> DraftVocabHead | None:
    """The reduced head of ``lm_head`` (int8 rows, FREETOKEN_INT8_DENSE), or None (off, or another LM head)."""
    from freetoken.distributed import get_tp_info

    weight, scale = getattr(lm_head, "weight", None), getattr(lm_head, "weight_scale", None)
    listed = None
    if n == "latin":
        listed = latin_token_ids(model_path)
    elif isinstance(n, str) and not n.strip().lstrip("-").isdigit():
        with open(n, encoding="utf-8") as f:
            listed = [int(i) for i in json.load(f)]
    elif int(n) <= 0:
        return None
    if not isinstance(weight, torch.Tensor) or weight.dtype is not torch.int8 or scale is None:
        logger.warning_rank0("FREETOKEN_DRAFT_VOCAB needs the int8 LM head (FREETOKEN_INT8_DENSE): drafts keep the "
                             "whole head")
        return None
    vocab, tp, rank = lm_head.num_embeddings, lm_head.tp_size, get_tp_info().rank
    base = listed if listed is not None else range(min(int(n), vocab))
    ids = sorted({i for i in base if 0 <= i < vocab} | {i for i in added_token_ids(model_path) if i < vocab})
    shards = lm_head.vocab_shards or _even_shards(vocab, tp)
    # each rank's part of ids, in id order, in the proportions of its LM head shard
    total = sum(rows for _, rows in shards)
    cum, bounds = 0, [0]
    for _, rows in shards:
        cum += rows
        bounds.append(round(len(ids) * cum / total))
    mine = ids[bounds[rank]: bounds[rank + 1]]
    start, width = lm_head.vocab_range
    own = [i for i in ids if start <= i < start + width]  # the rows of ids this rank's shard holds
    idx = torch.tensor([i - start for i in own], dtype=torch.int64, device=weight.device)
    own_w, own_s = weight.index_select(0, idx).cpu(), scale.index_select(0, idx).float().cpu()
    if tp == 1:
        rows_w, rows_s = own_w, own_s
    else:
        # every rank's own rows on every rank (padded to the largest count), then this rank keeps its part
        counts = [sum(1 for i in ids if s0 <= i < s0 + w0) for s0, w0 in shards]
        pad = max(counts)
        k_dim = weight.shape[1]
        send_w = torch.zeros((pad, k_dim), dtype=torch.int8)
        send_w[: own_w.shape[0]] = own_w
        send_s = torch.zeros(pad, dtype=torch.float32)
        send_s[: own_s.shape[0]] = own_s
        got_w = [torch.empty_like(send_w) for _ in range(tp)]
        got_s = [torch.empty_like(send_s) for _ in range(tp)]
        torch.distributed.all_gather(got_w, send_w, group=cpu_group)
        torch.distributed.all_gather(got_s, send_s, group=cpu_group)
        all_w = torch.cat([g[:c] for g, c in zip(got_w, counts)])  # the shards are contiguous id ranges, in order
        all_s = torch.cat([g[:c] for g, c in zip(got_s, counts)])
        rows_w, rows_s = all_w[bounds[rank]: bounds[rank + 1]], all_s[bounds[rank]: bounds[rank + 1]]
    head = DraftVocabHead(lm_head, rows_w.to(weight.device).contiguous(),
                          rows_s.to(weight.device, scale.dtype).contiguous(),
                          torch.tensor(mine, dtype=torch.int32, device=weight.device))
    logger.info_rank0(f"FREETOKEN_DRAFT_VOCAB={n}: the drafts score {len(ids)} of {vocab} tokens "
                      f"({', '.join(str(b - a) for a, b in zip(bounds, bounds[1:]))} rows a rank)")
    return head
