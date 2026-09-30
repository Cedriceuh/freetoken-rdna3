"""Compare the loader's rank-local tensors with the buffers the model declares, per TP rank (CPU, meta model).

Runs inside the image with the tree mounted at /src and the checkpoint at /models/m (the paths below); ``--vision``
checks the model built with its vision tower (rdna3/serve.sh --vision)."""
import sys
import torch
sys.path.insert(0, "/src")
from tests.models.qwen4_exp.common import as_rank, install_quant_config, meta_state_dict
from freetoken.models.qwen4_exp.weight import iter_weights

M = "/models/m"
VISION = "--vision" in sys.argv[1:]
install_quant_config(M)
for rank in (0, 1):
    with as_rank(rank, 2):
        sd = meta_state_dict(M, vision=VISION)
        bad, seen = [], set()
        for name, t in iter_weights(M, torch.device("cpu"), include_moe_experts=False, include_non_moe=True,
                                   include_vision=VISION and any(k.startswith("visual.") for k in sd)):
            seen.add(name)
            p = sd.get(name)
            if p is None:
                bad.append(("EXTRA", name, tuple(t.shape)))
            elif tuple(p.shape) != tuple(t.shape) or p.dtype != t.dtype:
                bad.append(("SHAPE", name, tuple(p.shape), str(p.dtype), tuple(t.shape), str(t.dtype)))
        missing = [k for k in sd if k not in seen and ".experts." not in k]
        print(f"rank {rank}: {len(bad)} mismatches, {len(missing)} model buffers not loaded", flush=True)
        for b in bad[:25]:
            print("  ", b, flush=True)
        for k in missing[:15]:
            print("   MISSING", k, tuple(sd[k].shape), flush=True)
