#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// Host wrapper over cudaMemcpyBatchAsync (CUDA >= 13.0, the 8-argument signature;
// 12.8/12.9 carried an extra failIdx parameter): enqueue N independent
// pointer-to-pointer copies with ONE runtime call, on an explicit (non-legacy)
// stream. Callers hand pre-resolved raw addresses; copies within a batch are
// unordered, so entries must be pairwise independent.
//
// ROCm: hipMemcpyBatchAsync (HIP >= 7.1) carries the failIdx parameter (the CUDA 12.8
// shape); run_loop is the same contract as one hipMemcpyAsync per entry, for runtimes
// whose batch call is missing or slower.
struct BatchMemcpy {
    static void run(
        tvm::ffi::TensorView dst_ptrs,
        tvm::ffi::TensorView src_ptrs,
        tvm::ffi::TensorView sizes,
        int64_t stream_handle
    ) {
#if FREETOKEN_USE_ROCM
        const auto n = verify(dst_ptrs, src_ptrs, sizes, stream_handle);
        if (n == 0) {
            return;
        }
#if defined(HIP_VERSION) && HIP_VERSION >= 70100000
        auto attr = ::hipMemcpyAttributes{};
        attr.srcAccessOrder = ::hipMemcpySrcAccessOrderStream;
        std::size_t attr_idx = 0;
        std::size_t fail_idx = 0;
        const auto err = ::hipMemcpyBatchAsync(
            reinterpret_cast<void**>(dst_ptrs.data_ptr()),
            reinterpret_cast<void**>(src_ptrs.data_ptr()),
            reinterpret_cast<std::size_t*>(sizes.data_ptr()),
            n,
            &attr,
            &attr_idx,
            1,
            &fail_idx,
            reinterpret_cast<::hipStream_t>(stream_handle)
        );
        if (err != ::hipSuccess) {
            // clear the sticky last error, or the caller's fallback (the loop mode) would fail on it next
            (void)::hipGetLastError();
            host::CUDA_CHECK(err);
        }
#else
        // older HIP: no hipMemcpyBatchAsync; the module still builds so the loop mode stays available
        ::host::panic(std::source_location::current(), "hipMemcpyBatchAsync needs HIP >= 7.1 at build time");
#endif
#elif CUDART_VERSION >= 13000
        using namespace host;
        auto N = SymbolicSize{"batch length"};
        auto ptr_dtype = SymbolicDType{};
        TensorMatcher({N})
            .with_dtype<int64_t>(ptr_dtype)
            .with_device<kDLCPU>()
            .verify(dst_ptrs)
            .verify(src_ptrs)
            .verify(sizes);
        const auto n = static_cast<std::size_t>(N.unwrap());
        if (n == 0) {
            return;
        }
        RuntimeCheck(stream_handle != 0, "cudaMemcpyBatchAsync rejects the legacy NULL stream");
        auto attr = ::cudaMemcpyAttributes{};
        attr.srcAccessOrder = ::cudaMemcpySrcAccessOrderStream;
        std::size_t attr_idx = 0;
        CUDA_CHECK(::cudaMemcpyBatchAsync(
            reinterpret_cast<void* const*>(dst_ptrs.data_ptr()),
            reinterpret_cast<const void* const*>(src_ptrs.data_ptr()),
            reinterpret_cast<const std::size_t*>(sizes.data_ptr()),
            n,
            &attr,
            &attr_idx,
            1,
            reinterpret_cast<::cudaStream_t>(stream_handle)
        ));
#else
        ::host::panic(
            std::source_location::current(),
            "this cudaMemcpyBatchAsync binding requires CUDA >= 13.0 at build time"
        );
#endif
    }

    static void run_loop(
        tvm::ffi::TensorView dst_ptrs,
        tvm::ffi::TensorView src_ptrs,
        tvm::ffi::TensorView sizes,
        int64_t stream_handle
    ) {
        const auto n = verify(dst_ptrs, src_ptrs, sizes, stream_handle);
        const auto* dst = static_cast<const int64_t*>(dst_ptrs.data_ptr());
        const auto* src = static_cast<const int64_t*>(src_ptrs.data_ptr());
        const auto* nbytes = static_cast<const int64_t*>(sizes.data_ptr());
        for (std::size_t i = 0; i < n; ++i) {
#if FREETOKEN_USE_ROCM
            host::CUDA_CHECK(::hipMemcpyAsync(
                reinterpret_cast<void*>(dst[i]),
                reinterpret_cast<const void*>(src[i]),
                static_cast<std::size_t>(nbytes[i]),
                ::hipMemcpyDefault,
                reinterpret_cast<::hipStream_t>(stream_handle)
            ));
#else
            host::CUDA_CHECK(::cudaMemcpyAsync(
                reinterpret_cast<void*>(dst[i]),
                reinterpret_cast<const void*>(src[i]),
                static_cast<std::size_t>(nbytes[i]),
                ::cudaMemcpyDefault,
                reinterpret_cast<::cudaStream_t>(stream_handle)
            ));
#endif
        }
    }

private:
    static auto verify(
        tvm::ffi::TensorView dst_ptrs,
        tvm::ffi::TensorView src_ptrs,
        tvm::ffi::TensorView sizes,
        int64_t stream_handle
    ) -> std::size_t {
        using namespace host;
        auto N = SymbolicSize{"batch length"};
        auto ptr_dtype = SymbolicDType{};
        TensorMatcher({N})
            .with_dtype<int64_t>(ptr_dtype)
            .with_device<kDLCPU>()
            .verify(dst_ptrs)
            .verify(src_ptrs)
            .verify(sizes);
        RuntimeCheck(stream_handle != 0, "batch memcpy rejects the legacy NULL stream");
        return static_cast<std::size_t>(N.unwrap());
    }
};
