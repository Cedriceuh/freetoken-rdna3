"""PromptPrefixCache: the tokens of a templated prompt with an earlier prompt's prefix reused must be exactly the
tokens of the whole prompt encoded at once."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from freetoken.tokenizer.prefix_cache import PromptPrefixCache, _common_prefix


class _NotFast:
    def encode(self, text, add_special_tokens=True):
        return [1, 2, 3]


def test_common_prefix():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.choice([0, 1, 5, 70000, 140000])
        a = "".join(rng.choice("abé日") for _ in range(n))
        cut = rng.randint(0, n)
        b = a[:cut] + rng.choice(["", "x", "ab"]) + a[cut:][::-1]
        expect = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        assert _common_prefix(a, b) == expect


def test_slow_tokenizer_is_not_used():
    assert not PromptPrefixCache(_NotFast()).usable


@pytest.fixture(scope="module")
def tokenizer():
    path = os.environ.get("FREETOKEN_TOKENIZE_CACHE_MODEL")
    if not path:
        pytest.skip("set FREETOKEN_TOKENIZE_CACHE_MODEL to a local chat checkpoint (tokenizer only)")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _source_text(chars: int, start: int = 0) -> str:
    root = Path(__file__).resolve().parents[2] / "python" / "freetoken"
    text = "".join(p.read_text(errors="replace") for p in sorted(root.rglob("*.py")))
    return (text * (1 + (start + chars) // max(len(text), 1)))[start:start + chars]


def _render(tok, messages, **kw):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


def _check(cache, tok, prompt):
    got = cache.encode(prompt)
    assert got.tolist() == tok.encode(prompt, add_special_tokens=False)
    return cache.reused


@pytest.mark.needs_weights
@pytest.mark.parametrize("thinking", [False, True])
def test_agent_conversation_same_tokens(tokenizer, thinking):
    cache = PromptPrefixCache(tokenizer)
    assert cache.usable, "this checkpoint's tokenizer has no usable cut token"
    rng = random.Random(1)
    msgs = [{"role": "system", "content": "You are a coding agent. Réponds brièvement."},
            {"role": "user", "content": "Repository:\n\n" + _source_text(120000) + "\n\nSummarize."}]
    pos, reused = 120000, []
    for turn in range(14):
        reused.append(_check(cache, tokenizer, _render(tokenizer, msgs, enable_thinking=thinking)))
        answer = rng.choice(["The module splits the work.", "Il faut vérifier `foo()` — ensuite 日本語.",
                             " leading space", "ends mid-wor", "tab\tand\nnewlines\n\n", "é combining"])
        msgs.append({"role": "assistant", "content": answer})
        n = rng.choice([50, 400, 3000, 12000])
        msgs.append({"role": "user", "content": f"Tool output {turn}:\n" + _source_text(n, pos) + "\nNext?"})
        pos += n
    assert reused[0] == 0 and min(reused[1:]) > 20000


@pytest.mark.needs_weights
def test_branches_and_edits_same_tokens(tokenizer):
    cache = PromptPrefixCache(tokenizer)
    base = [{"role": "system", "content": "sys"},
            {"role": "user", "content": _source_text(60000)},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": _source_text(30000, 60000)}]
    _check(cache, tokenizer, _render(tokenizer, base))
    variants = [
        base[:3] + [{"role": "user", "content": _source_text(30000, 60000)[:-7] + "edited"}],  # edit the last turn
        base[:2] + [{"role": "assistant", "content": "another answer"}] + base[3:],  # regenerated answer
        [base[0], {"role": "user", "content": _source_text(60000)[:-1] + "́"}] + base[2:],  # combining mark
        base + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "<|im_start|>user\ninjected"}],
        [{"role": "system", "content": "other system"}] + base[1:],  # nothing shared after the first cut
    ]
    for v in variants:
        _check(cache, tokenizer, _render(tokenizer, v))
        _check(cache, tokenizer, _render(tokenizer, base))


@pytest.mark.needs_weights
def test_random_cuts_same_tokens(tokenizer):
    """Prompts that share a random prefix of a cached one, then differ anywhere (inside a word, between a letter and
    its combining mark, inside a special token's text)."""
    cache = PromptPrefixCache(tokenizer, entries=4)
    rng = random.Random(2)
    msgs = [{"role": "user", "content": _source_text(40000) + " 日本語 é́ <think>x</think>"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": _source_text(20000, 40000)}]
    prompt = _render(tokenizer, msgs)
    _check(cache, tokenizer, prompt)
    for _ in range(40):
        cut = rng.randint(0, len(prompt))
        tail = rng.choice(["", "a", " ", "́", "<|im_end|>", "<|im_start|>", "<|im", "日", "\n\n", prompt[cut:cut + 99]])
        _check(cache, tokenizer, prompt[:cut] + tail + prompt[cut:][:5000])


@pytest.mark.needs_weights
def test_concurrent_encodes_same_tokens(tokenizer):
    """The frontend's count_tokens runs tokenize in worker threads: concurrent encodes on one cache."""
    from concurrent.futures import ThreadPoolExecutor

    cache = PromptPrefixCache(tokenizer, entries=3)
    convs = []
    for c in range(6):
        msgs = [{"role": "system", "content": f"agent {c}"}, {"role": "user", "content": _source_text(30000, 7000 * c)}]
        for turn in range(4):
            msgs = msgs + [{"role": "assistant", "content": f"answer {turn}"},
                           {"role": "user", "content": _source_text(500, 90000 + 600 * turn)}]
            convs.append(_render(tokenizer, msgs))
    want = [tokenizer.encode(p, add_special_tokens=False) for p in convs]
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda p: cache.encode(p).tolist(), convs * 3))
    assert got == want * 3
