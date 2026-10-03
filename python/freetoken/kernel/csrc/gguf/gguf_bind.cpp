// Torch side of the vendored GGUF kernels (gguf_kernel.cu): host-only, so torch's HIP headers -- which include
// rocThrust, absent from the ROCm pip SDK -- never reach the device compile.
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <optional>

// Launchers in gguf_kernel.cu (dtype: GgufDtype in dispatch.h)
void ggml_dequantize_launch(int dtype, void* W, void* DW, int64_t type, int64_t m, int64_t n, cudaStream_t stream);
void ggml_mul_mat_vec_a8_launch(int dtype, void* X, void* quant_X, void* W, void* Y, int64_t type, int64_t row, int col,
                                int vecs, cudaStream_t stream);
void ggml_mul_mat_a8_launch(int dtype, void* X, void* quant_X, void* W, void* Y, int64_t type, int64_t row, int col,
                            int padded, int batch, cudaStream_t stream);
void ggml_moe_a8_launch(int dtype, void* X, void* quant_X, void* W, void* Y, void* sorted_token_ids, void* expert_ids,
                        void* num_tokens_post_padded, int64_t W_stride0, int64_t sorted_token_ids_size0, int64_t type,
                        int64_t row, int64_t top_k, int64_t tokens, int col, int padded, cudaStream_t stream);
void ggml_moe_a8_vec_launch(int dtype, void* X, void* quant_X, void* W, void* Y, void* topk_ids,
                            int64_t quant_X_stride0,
                            int64_t top_k, int64_t type, int64_t row, int64_t tokens, int col, cudaStream_t stream);
int64_t ggml_moe_get_block_size(int64_t type);

static int gguf_dtype(at::ScalarType t, const char* name) {
  switch (t) {
    case at::ScalarType::Float:
      return 0;
    case at::ScalarType::Half:
      return 1;
    case at::ScalarType::BFloat16:
      return 2;
    default:
      TORCH_CHECK(false, name, " not implemented for '", t, "'");
  }
  return -1;
}

torch::Tensor ggml_dequantize(
    torch::Tensor W,  // quant weight
    int64_t type,
    int64_t m,
    int64_t n,
    std::optional<at::ScalarType> const& dtype) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto dtype_ = dtype.value_or(torch::kFloat16);
  auto options = torch::TensorOptions().dtype(dtype_).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  ggml_dequantize_launch(gguf_dtype(DW.scalar_type(), "ggml_dequantize"), W.data_ptr(), DW.data_ptr(), type, m, n,
                         stream);

  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({vecs, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({vecs, padded / 32 * 9}, options);
  ggml_mul_mat_vec_a8_launch(gguf_dtype(X.scalar_type(), "ggml_mul_mat_vec_a8"), X.data_ptr(), quant_X.data_ptr(),
                             W.data_ptr(), Y.data_ptr(), type, row, col, vecs, stream);
  return Y;
}

torch::Tensor ggml_mul_mat_a8(
    torch::Tensor W,  // quant weight
    torch::Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  ggml_mul_mat_a8_launch(gguf_dtype(X.scalar_type(), "ggml_mul_mat_a8"), X.data_ptr(), quant_X.data_ptr(), W.data_ptr(),
                         Y.data_ptr(), type, row, col, padded, batch, stream);
  return Y;
}

torch::Tensor ggml_moe_a8(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::empty({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  ggml_moe_a8_launch(gguf_dtype(X.scalar_type(), "ggml_moe_a8"), X.data_ptr(), quant_X.data_ptr(), W.data_ptr(),
                     Y.data_ptr(), sorted_token_ids.data_ptr(), expert_ids.data_ptr(),
                     num_tokens_post_padded.data_ptr(), W.stride(0), sorted_token_ids.sizes()[0], type, row, top_k,
                     tokens, col, padded, stream);
  return Y;
}

torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X,  // input
    torch::Tensor W,  // expert weights
    torch::Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options = torch::TensorOptions().dtype(X.dtype()).device(W.device());
  at::Tensor Y = torch::zeros({tokens * top_k, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({tokens, padded / 32 * 9}, options);
  ggml_moe_a8_vec_launch(gguf_dtype(X.scalar_type(), "ggml_moe_a8_vec"), X.data_ptr(), quant_X.data_ptr(), W.data_ptr(),
                         Y.data_ptr(), topk_ids.data_ptr(), quant_X.stride(0), top_k, type, row, tokens, col, stream);
  return Y;
}

// ---- FreeToken pybind bindings (donor registers these via TORCH_LIBRARY; we
// expose them through torch.utils.cpp_extension.load's pybind module instead) ----
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_dequantize", &ggml_dequantize, "");
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8, "");
  m.def("ggml_mul_mat_a8", &ggml_mul_mat_a8, "");
  m.def("ggml_moe_a8", &ggml_moe_a8, "");
  m.def("ggml_moe_a8_vec", &ggml_moe_a8_vec, "");
  m.def("ggml_moe_get_block_size", &ggml_moe_get_block_size, "");
}
