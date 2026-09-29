"""Standalone test of freetoken's triton sampling kernels on one GPU (hang detector: run under `timeout`)."""
import sys, time, torch
import importlib, os
s = importlib.import_module(os.environ.get("SAMPLING_MOD", "freetoken.kernel.triton.sampling"))
V, B = 248320, 1
torch.manual_seed(0)
logits = torch.randn(B, V, device="cuda") * 3
t = torch.ones(B, device="cuda")
dev = torch.cuda.get_device_properties(0)
# softmax plan (G CTAs per row, chunk) and the exact top-k / top-p kernels' plan (one CTA per row on ROCm)
print("device", dev.name, "CUs", dev.multi_processor_count, "softmax plan", s._plan(B, V, logits.device),
      "exact plan", s._fused_plan(B, V, logits.device, s._ONE_CTA_PER_ROW), flush=True)
probs = s.softmax(logits, t); torch.cuda.synchronize(); print("softmax ok", flush=True)
which = sys.argv[1]
t0 = time.time()
if which == "plain":
    out = s.sampling_from_probs(probs)
elif which == "topp":
    out = s.top_p_sampling_from_probs(probs, 0.95)
elif which == "topk":
    out = s.top_k_sampling_from_probs(probs, 20)
else:
    out = s.top_k_top_p_sampling_from_probs(probs, torch.full((B,), 20, device="cuda", dtype=torch.int32),
                                            torch.full((B,), 0.95, device="cuda"))
torch.cuda.synchronize()
print(which, "ok", out.tolist(), f"{(time.time() - t0) * 1e3:.1f} ms", flush=True)
