#!/usr/bin/env python3
"""Decode and prompt reading as a conversation grows to the profile's maximum context (docs/rdna3/benchmarks.md).

    python3 rdna3/bench/depth_sweep.py [--url http://127.0.0.1:1919] [--model qwen3.8-flash-next] [--max 248000]
                                       [--passes 2] [--label xtx-xt]

Standard library only. One conversation per pass: a cold ~10.5k-token read, then blocks the server has never seen (like
large tool results) until the depth reaches --max (ten steps to 248k, seven to 124k). At each step:
  TG          decode tokens/s over a 256-token answer about the new block, after its first token (greedy)
  PP          speed of reading that block: its new tokens / time to the first token
  agent turn  time to the first token of a following ~1k-token turn on the now-cached conversation
Prints a Markdown table with the median of the passes. Run it alone on the server: two passes take ~6 min. --max must
stay ~2k below the server's context (the step's agent turn and answers come on top): 248000 for 250k, 124000 for 131k.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request

WORDS = ("system kernel memory cache token expert layer vector matrix branch commit review parser socket thread queue "
         "buffer config module import return value index table record field stream signal driver device packet frame "
         "schema client server request answer update commit rollback metric window batch graph node edge weight").split()
TARGETS = (10_500, 16_000, 25_000, 40_000, 60_000, 87_000, 123_000, 165_000, 207_000, 250_000)


def text(n_words: int, rng: random.Random) -> str:
    lines, line = [], []
    for _ in range(n_words):
        line.append(rng.choice(WORDS))
        if len(line) == 12:
            lines.append(" ".join(line)); line = []
    return "\n".join(lines + [" ".join(line)])


def chat(url: str, model: str, messages: list, max_tokens: int) -> tuple[float, float, int, int, str]:
    """(time to first token, decode tok/s after it, prompt tokens, completion tokens, text) of one greedy request."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0, stamps, pieces, usage = time.time(), [], [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("error"):  # e.g. the prompt is longer than the server's context
                raise RuntimeError(f"server error: {ev['error'].get('message', ev['error'])}")
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    stamps.append(time.time()); pieces.append(piece)
    if not stamps:
        raise RuntimeError("the server streamed no text")
    n = int(usage.get("completion_tokens", len(stamps)))
    tg = (n - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 and stamps[-1] > stamps[0] else float("nan")
    return (stamps[0] - t0 if stamps else float("nan")), tg, int(usage.get("prompt_tokens", 0)), n, "".join(pieces)


def one_pass(url: str, model: str, targets: list[int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    messages, depth, per_word, rows = [], 0, 1.3, []
    for target in targets:
        words = max(200, int((target - depth) / per_word))
        block = text(words, rng)
        messages.append({"role": "user", "content": f"Here is a document (part {len(rows) + 1}):\n\n{block}\n\n"
                                                    "Describe this part in detail, line by line."})
        ttft, tg, prompt, n, answer = chat(url, model, messages, 256)
        new = prompt - depth
        per_word = max(1.0, new / words) if not rows else per_word
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": text(750, rng) + "\n\nIn one sentence, what are these words about?"})
        turn, _, prompt2, n2, answer2 = chat(url, model, messages, 32)
        messages.append({"role": "assistant", "content": answer2})
        depth = prompt2 + n2
        rows.append({"depth": prompt, "tg": tg, "pp": new / ttft, "new": new, "turn": turn})
        print(f"  depth {prompt / 1000:6.1f}k  TG {tg:5.1f} tok/s  PP {new / ttft:6.0f} tok/s ({new} new)  "
              f"agent turn {turn:.2f} s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:1919")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--max", type=int, default=248_000, help="deepest step, in tokens (124000 for one card)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    targets = [t for t in TARGETS if t < a.max * 0.9] + [a.max]  # the last step reaches --max
    passes = []
    for p in range(a.passes):
        print(f"pass {p + 1}/{a.passes}", flush=True)
        passes.append(one_pass(a.url, a.model, targets, seed=p))
    print(f"\n{a.label + ': ' if a.label else ''}median of {a.passes} pass(es)\n")
    print("| Depth | TG (tok/s) | PP of the new block (tok/s) | Agent turn (s) |\n|---:|---:|---:|---:|")
    for i in range(len(targets)):
        col = lambda k: statistics.median(p[i][k] for p in passes)
        print(f"| {col('depth') / 1000:.1f}k | {col('tg'):.1f} | {col('pp'):.0f} | {col('turn'):.2f} |")


if __name__ == "__main__":
    main()
