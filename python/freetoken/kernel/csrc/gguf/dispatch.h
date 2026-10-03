// Minimal AT_DISPATCH helper for the vendored GGUF kernels (borrowed from
// sgl-kernel csrc/quantization/gguf, which are ports of llama.cpp). The donor
// pulls these macros from its large include/utils.h; we only need the float
// dispatch, so vendor just that to keep the JIT compile self-contained. No ATen here: under HIP its headers include
// rocThrust, which the ROCm pip SDK does not ship; the torch side lives in gguf_bind.cpp, a host-only file.
#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Warp-shuffle wrappers the donor pulls from sgl-kernel's utils.h.
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
// HIP's *_sync shuffles want a 64-bit mask (a static_assert rejects the 32-bit one); every call here passes the full
// mask, so the plain shuffle over the same 32 lanes is the same operation
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor((var), (lane_mask), WARP_SIZE)
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) __shfl_xor((var), (lane_mask), (width))
#endif
#else
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync((mask), (var), (lane_mask), (width))
#endif
#endif

// Element type of the activations / outputs, mapped from the tensor's dtype by gguf_bind.cpp (which rejects others)
enum GgufDtype : int { kGgufFloat = 0, kGgufHalf = 1, kGgufBFloat16 = 2 };

// The scalar_t AT_DISPATCH_FLOATING_TYPES would give (float, c10::Half, c10::BFloat16), from a GgufDtype
#define GGUF_DISPATCH_FLOAT_TYPES(DTYPE, NAME, ...)   \
  switch (DTYPE) {                                    \
    case kGgufFloat: {                                \
      using scalar_t = float;                         \
      __VA_ARGS__();                                  \
      break;                                          \
    }                                                 \
    case kGgufHalf: {                                 \
      using scalar_t = c10::Half;                     \
      __VA_ARGS__();                                  \
      break;                                          \
    }                                                 \
    case kGgufBFloat16: {                             \
      using scalar_t = c10::BFloat16;                 \
      __VA_ARGS__();                                  \
      break;                                          \
    }                                                 \
  }
