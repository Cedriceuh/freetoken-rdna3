#!/usr/bin/env python3
"""Quick end-to-end measurement of a running server, to compare setups or report numbers for an untested profile.

    python3 rdna3/bench/quick_bench.py [--url http://127.0.0.1:1919] [--model qwen3.8-flash-next] [--label my-setup]

Standard library only. Four measurements, each on text the server has never seen (so no cache hit inflates them):
  decode      tokens/s of a 512-token answer, after its first token (the model's default sampling, thinking off)
  cold read   time to first token of a ~8.3k-token prompt with nothing cached (as in docs/rdna3/benchmarks.md)
  agent turn  time to first token when ~1.5k new tokens are appended to that (now cached) conversation
  tool call   whether a tool call comes back parsed
Prints a Markdown table to paste in an issue. Takes about a minute; run it when nothing else uses the server.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = ("system kernel memory cache token expert layer vector matrix branch commit review parser socket thread queue "
         "buffer config module import return value index table record field stream signal driver device packet frame "
         "schema client server request answer update commit rollback metric window batch graph node edge weight").split()


def text(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def post(url: str, body: dict, stream: bool) -> tuple[float | None, float | None, int, dict]:
    """(time to first token, time of last token, completion tokens, message or usage) for one request."""
    body = {**body, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    first = last = None
    usage, message = {}, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not stream:
            doc = json.loads(r.read())
            return time.time() - t0, None, int(doc.get("usage", {}).get("completion_tokens", 0)), doc["choices"][0]["message"]
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time()
                    first = first or now
                    last = now
    return (first - t0 if first else None), (last - t0 if last else None), int(usage.get("completion_tokens", 0)), usage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:1919")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    try:
        urllib.request.urlopen(f"{a.url}/health", timeout=10).read()
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"no server at {a.url} ({e}); start one first (rdna3/serve.sh) or pass --url")
    rng = random.Random(time.time_ns())
    nothink = {"chat_template_kwargs": {"enable_thinking": False}}
    base = {"model": a.model, **nothink}
    tag = f"[run {rng.getrandbits(32):08x}]"  # makes every prompt new to the prefix cache

    # warm-up: the first use of a new image compiles and autotunes kernels, for decode and for prefill
    post(a.url, {**base, "messages": [{"role": "user", "content": f"{tag} Say hello."}], "max_tokens": 8}, True)
    post(a.url, {**base, "max_tokens": 8, "messages": [
        {"role": "user", "content": f"{tag} warm-up\n{text(1500, rng)}\nOne word."}]}, True)

    ttft, tlast, n, _ = post(a.url, {**base, "max_tokens": 512, "messages": [
        {"role": "user", "content": f"{tag} Write a long, detailed technical story about a database migration."}]}, True)
    decode = (n - 1) / (tlast - ttft) if n > 1 and tlast and tlast > ttft else 0.0

    history = [{"role": "user", "content": f"{tag} Here is a log to keep in mind:\n{text(8300, rng)}\nSummarize it in one line."}]
    cold, _, _, usage = post(a.url, {**base, "max_tokens": 16, "messages": history}, True)
    cold_tokens = int(usage.get("prompt_tokens", 0))
    history += [{"role": "assistant", "content": "A log of system events."},
                {"role": "user", "content": f"Tool result:\n{text(1100, rng)}\nWhat changed? One line."}]
    turn, _, _, _ = post(a.url, {**base, "max_tokens": 16, "messages": history}, True)

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                                               "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                              "required": ["city"]}}}]
    # greedy, so the check is about the parser, not about whether a sampled answer chose the tool
    _, _, _, msg = post(a.url, {**base, "max_tokens": 128, "temperature": 0, "tools": tools, "messages": [
        {"role": "user", "content": "What is the weather in Tokyo? Use the tool."}]}, False)
    tool_ok = bool(msg.get("tool_calls"))
    tool_note = "" if tool_ok else f" (answer: {(msg.get('content') or '')[:80]!r})"

    stats = json.loads(urllib.request.urlopen(f"{a.url}/v1/stats", timeout=10).read())
    gpus = ", ".join(g.get("name", "?") for g in stats.get("gpus", [])) or "?"  # rank 0's device only
    print(f"\n| {a.label or 'setup'} | |\n|---|---|")
    print(f"| model / context | {stats.get('model', {}).get('id', a.model)} / {stats.get('model', {}).get('ctx', '?')} |")
    print(f"| GPU of rank 0 (as the server reports it) | {gpus} |")
    print(f"| decode, 512 tokens | {decode:.1f} tok/s |")
    print(f"| cold read, {cold_tokens} tokens | {cold:.2f} s ({cold_tokens / cold:.0f} tok/s) |" if cold else "| cold read | failed |")
    print(f"| agent turn, ~1.5k new tokens | {turn:.2f} s |" if turn else "| agent turn | failed |")
    print(f"| tool call parsed | {'yes' if tool_ok else 'NO' + tool_note} |")


if __name__ == "__main__":
    main()
