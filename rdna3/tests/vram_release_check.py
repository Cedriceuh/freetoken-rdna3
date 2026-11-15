"""GPU check: do copies into VRAM allocated right after a release hold? One GPU, no server, torch only.

The int8 compaction (layers/quantization/int8_weight_only.py) moves the model's tensors to RAM, gives their VRAM back
to the driver and allocates it again. On a machine reported with two RX 7900 XTX and a 7.2 kernel most of those copies
were lost unless the engine waited ~0.5 s after the release (FREETOKEN_VRAM_RELEASE_SETTLE). This does the same with
random tensors: fill the card, release, allocate again, copy from RAM, read back, read back again a second later.

    python3 rdna3/tests/vram_release_check.py [--gib 12] [--wait 0] [--rounds 3] [--device 0]

Prints the tensors that differ from their host copy per round; exit 1 if any did. `--wait 0.5` waits after the release:
a machine that fails without it and passes with it has the reported behaviour.
"""
import argparse
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=12.0, help="VRAM to fill, GiB")
    ap.add_argument("--wait", type=float, default=0.0, help="seconds between the release and the new allocations")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device(f"cuda:{args.device}")
    g = torch.Generator().manual_seed(0)
    # the model's mix: a few hundred large int8 / bf16 weights, as many small fp32 / int64 ones
    big = int(args.gib * 2**30 / 600)
    host = []
    for i in range(600):
        host.append(torch.randint(-128, 128, (big,), dtype=torch.int8, generator=g))
        n = (3, 16, 18, 640, 6180)[i % 5]
        host.append(torch.randn(n, generator=g) if i % 2 else torch.randint(1, 1 << 40, (n,), generator=g))
    placed = [h.to(dev) for h in host]  # the first placement, as after the weights are loaded
    torch.cuda.synchronize(dev)
    failed = 0
    for r in range(args.rounds):
        placed = None
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        time.sleep(args.wait)
        order = sorted(host, key=lambda t: -t.numel() * t.element_size())  # largest first, as the compaction
        placed = [h.to(dev) for h in order]
        bad = sum(not torch.equal(p.to("cpu"), h) for p, h in zip(placed, order))
        time.sleep(1.0)
        late = sum(not torch.equal(p.to("cpu"), h) for p, h in zip(placed, order))
        print(f"round {r}: {bad} of {len(order)} tensors differ from their host copy, {late} a second later"
              f" (wait {args.wait} s, {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB on {torch.cuda.get_device_name(dev)})")
        failed += bad + late
    print("FAIL: copies into VRAM allocated right after a release were lost" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
