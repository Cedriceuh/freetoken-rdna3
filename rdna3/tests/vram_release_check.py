"""GPU check: do copies into VRAM allocated right after a release hold? One GPU, no server.

The int8 compaction (layers/quantization/int8_weight_only.py) moves the model's tensors to RAM, gives their VRAM back
to the driver and allocates it again. On a machine reported with two RX 7900 XTX and a 7.2 kernel most of those copies
were lost unless the engine waited ~0.5 s after the release (FREETOKEN_VRAM_RELEASE_SETTLE).

    python3 rdna3/tests/vram_release_check.py [--gib 12] [--wait 0] [--rounds 3] [--device 0]
    python3 rdna3/tests/vram_release_check.py --engine [--pinned-gib 2] [--mlock] [...]

Default: torch only, random tensors: fill the card, release, allocate again, copy from RAM, read back, read back again
a second later. It passed on the reported machine, alone and with both cards visible.

`--engine`: the engine's own code on a model of plain linears: the weights go to the card, `convert_model` turns them
to int8 one by one (each read to RAM, quantized there, written back, its bf16 VRAM freed), `compact_device_tensors`
makes the round trip, and its read-back says how many tensors differ; then the model is compared with what it held
before the round trip. `--pinned-gib` keeps that much pinned host memory alive meanwhile, `--mlock` locks the
process's pages first (the engine itself locks them after the load, not before).

Prints the tensors that differ per round; exit 1 if any did. `--wait 0.5` waits after each release: a machine that
fails without it and passes with it has the reported behaviour.
"""
import argparse
import sys
import time

import torch


def engine_round(dev: torch.device, gib: float) -> tuple[int, int, int]:
    """One load -> int8 conversion -> compaction with the engine's functions; (tensors, differing at the engine's
    read-back, differing from the model before the round trip once everything has settled)."""
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers import BaseOP, LinearReplicated, OPList
    from freetoken.layers.quantization import int8_weight_only as iwo

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    side = 4096

    class Mixer(BaseOP):
        def __init__(self, i: int) -> None:
            self.proj = LinearReplicated(side, side, has_bias=False, prefix=f"model.layers.{i}.mixer.proj")

    class Block(BaseOP):
        def __init__(self, i: int) -> None:
            self.mixer = Mixer(i)
            self.dt_bias = torch.empty(18)
            self.sizes = torch.empty(16, dtype=torch.int64)

    class Model(BaseOP):
        def __init__(self, n: int) -> None:
            self.layers = OPList([Block(i) for i in range(n)])

    n = max(1, int(gib * 2**30 / (side * side * 2)))
    model = Model(n)
    g = torch.Generator().manual_seed(0)
    base = torch.randn(side, side, generator=g) * 0.05
    for i, block in enumerate(model.layers.op_list):  # the load: every tensor made in RAM, then written to the card
        block.mixer.proj.weight = base.roll(i, 1).to(torch.bfloat16).to(dev)
        block.dt_bias = torch.randn(18, generator=g).to(dev)
        block.sizes = torch.randint(1, 1 << 40, (16,), generator=g).to(dev)
    del base
    converted, _ = iwo.convert_model(model)
    assert converted == n, (converted, n)
    torch.cuda.synchronize(dev)
    time.sleep(2.0)  # the reference: what the model holds once nothing moves any more, read twice
    before = {k: v.to("cpu") for k, v in model.state_dict().items()}
    assert all(torch.equal(v.to("cpu"), before[k]) for k, v in model.state_dict().items()), "the reference is not stable"
    first: list[int] = []
    verify = iwo._verify_copies

    def counting(placed, host, keys, *a, **kw):
        first.append(sum(not torch.equal(placed[k].to("cpu"), host[k]) for k in host if k in keys))
        return verify(placed, host, keys, *a, **kw)

    iwo._verify_copies = counting
    try:
        iwo.compact_device_tensors(model)
    finally:
        iwo._verify_copies = verify
    time.sleep(1.0)
    after = model.state_dict()
    final = sum(not torch.equal(v.to("cpu"), before[k]) for k, v in after.items())
    return len(after), first[0] if first else -1, final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=12.0, help="VRAM to fill, GiB")
    ap.add_argument("--wait", type=float, default=0.0, help="seconds between the release and the new allocations")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--engine", action="store_true", help="the engine's int8 conversion and compaction on a model of linears")
    ap.add_argument("--pinned-gib", type=float, default=0.0, help="--engine: pinned host memory kept alive, GiB")
    ap.add_argument("--mlock", action="store_true", help="--engine: mlockall (current, future, on fault) first")
    args = ap.parse_args()
    if args.engine:
        return engine_main(args)
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


def engine_main(args) -> int:
    import ctypes
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
    # the engine's releases read this: unset = no wait until its read-back has seen a difference
    if args.wait:
        os.environ["FREETOKEN_VRAM_RELEASE_SETTLE"] = str(args.wait)
    else:
        os.environ.pop("FREETOKEN_VRAM_RELEASE_SETTLE", None)
    if args.mlock and ctypes.CDLL("libc.so.6", use_errno=True).mlockall(1 | 2 | 4) != 0:
        print(f"mlockall failed: {os.strerror(ctypes.get_errno())}")
    dev = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(dev)
    pinned = torch.empty(int(args.pinned_gib * 2**30), dtype=torch.uint8).pin_memory() if args.pinned_gib else None
    failed = 0
    for r in range(args.rounds):
        from freetoken.utils import torch_utils

        torch_utils._release_race_seen = False  # each round as a fresh process would see it
        total, first, final = engine_round(dev, args.gib)
        print(f"round {r}: {first} of {total} tensors differ at the engine's read-back after the compaction, {final} "
              f"from the model before it once settled (wait {args.wait} s, pinned {args.pinned_gib} GiB, "
              f"mlock {args.mlock}, {torch.cuda.get_device_name(dev)})")
        failed += max(first, 0) + final
    del pinned
    print("FAIL: the engine's compaction lost copies" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
