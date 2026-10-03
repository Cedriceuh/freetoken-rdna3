// Adatped from
// https://github.com/vllm-project/vllm/blob/755ed7b05be4743237d3339c4ff8c22bcaae04f4/csrc/quantization/gguf/gguf_kernel.cu
// Device side only: the torch wrappers and the Python bindings are in gguf_bind.cpp (see dispatch.h for why).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

// dont use clang-format here, it breaks the include order
// clang-format off
#include "dispatch.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"
// clang-format off

// Q8 gemv
template <typename scalar_t>
static __global__ void
quantize_q8_1(const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx, const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), amax, mask, 32));
    sum += SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx, const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

void ggml_dequantize_launch(int dtype, void* W, void* DW, int64_t type, int64_t m, int64_t n, cudaStream_t stream) {
  GGUF_DISPATCH_FLOAT_TYPES(dtype, "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W, (scalar_t*)DW, m * n, stream);
  });
}

void ggml_mul_mat_vec_a8_launch(int dtype, void* X, void* quant_X, void* W, void* Y, int64_t type, int64_t row, int col,
                                int vecs, cudaStream_t stream) {
  GGUF_DISPATCH_FLOAT_TYPES(dtype, "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X, (void*)quant_X, col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W, (void*)quant_X, (scalar_t*)Y, col, row, vecs, stream);
        break;
    }
  });
}

void ggml_mul_mat_a8_launch(int dtype, void* X, void* quant_X, void* W, void* Y, int64_t type, int64_t row, int col,
                            int padded, int batch, cudaStream_t stream) {
  GGUF_DISPATCH_FLOAT_TYPES(dtype, "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X, (void*)quant_X, col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
    }
  });
}

void ggml_moe_a8_launch(int dtype, void* X, void* quant_X, void* W, void* Y, void* sorted_token_ids, void* expert_ids,
                        void* num_tokens_post_padded, int64_t W_stride0, int64_t sorted_token_ids_size0, int64_t type,
                        int64_t row, int64_t top_k, int64_t tokens, int col, int padded, cudaStream_t stream) {
  GGUF_DISPATCH_FLOAT_TYPES(dtype, "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X, (void*)quant_X, col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X,
            (void*)W,
            (scalar_t*)Y,
            (int*)sorted_token_ids,
            (int*)expert_ids,
            (int*)num_tokens_post_padded,
            W_stride0,
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids_size0,
            stream);
        break;
    }
  });
}

void ggml_moe_a8_vec_launch(int dtype, void* X, void* quant_X, void* W, void* Y, void* topk_ids,
                            int64_t quant_X_stride0, int64_t top_k, int64_t type, int64_t row, int64_t tokens, int col,
                            cudaStream_t stream) {
  GGUF_DISPATCH_FLOAT_TYPES(dtype, "ggml_moe_a8_vec", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X, (void*)quant_X, col, tokens, stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W,
            (void*)quant_X,
            (scalar_t*)Y,
            (int*)topk_ids,
            top_k,
            tokens,
            col,
            row,
            quant_X_stride0,
            stream);
        break;
    }
  });
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}
