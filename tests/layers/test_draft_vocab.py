"""FREETOKEN_DRAFT_VOCAB's token sets, read from ``tokenizer.json`` (CPU)."""

from __future__ import annotations

import json


def _tokenizer(tmp_path, pieces: dict[str, int], added: list[int]):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": pieces},
                                                          "added_tokens": [{"id": i} for i in added]}))
    return str(tmp_path)


def test_latin_keeps_latin_text_and_partial_characters(tmp_path):
    """Byte-level pieces: ASCII, accented Latin ("é" is "Ã©" in byte-level form) and a lone byte of a longer character
    are kept; CJK and Cyrillic pieces are not."""
    from freetoken.layers.draft_vocab import latin_token_ids

    pieces = {"Ġthe": 0, "Ã©": 1, "æ": 2, "ä¸Ń": 3, "ÐŁ": 4, "Ġrefactor": 5, "âĢĶ": 6}
    # 2: one byte of a three-byte character (kept: it can be any script's); 3: U+4E2D; 4: U+041F; 6: U+2014 (an em dash)
    assert latin_token_ids(_tokenizer(tmp_path, pieces, [])) == [0, 1, 2, 5, 6]


def test_the_added_tokens_are_read_from_the_tokenizer(tmp_path):
    from freetoken.layers.draft_vocab import added_token_ids

    assert added_token_ids(_tokenizer(tmp_path, {}, [248046, 248044])) == [248044, 248046]
    assert added_token_ids(str(tmp_path / "missing")) == []
