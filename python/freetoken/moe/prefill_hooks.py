"""Hooks a caller sets around a long prefill's routed experts (one element: mutable, read at call time)."""

# FREETOKEN_PREFILL_AR_OVERLAP: called as BLOCK_DONE[0](start, end, out) once the routed output of tokens [start, end)
# is queued, when the experts run a chunk in token blocks (NVFP4 / EXL3 prefill), so the caller can finish and
# all-reduce those rows while the next block computes
BLOCK_DONE: list = [None]

# set when a prefill started an all-reduce on RCCL's stream that the compute stream then waited for on the GPU
# (FREETOKEN_PREFILL_AR_OVERLAP, FREETOKEN_GDN_AR_OVERLAP): the engine synchronizes the device before the next decode
# step, because the decode graphs replayed after such a cross-stream wait ran a third slower at four requests until the
# host synchronized (journey.md section 33)
ASYNC_AR: list = [False]
