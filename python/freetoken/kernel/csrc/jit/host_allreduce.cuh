#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

// Two-rank all-reduce through host memory, for the small (decode-size) messages of tensor
// parallelism on GPUs without peer-to-peer access. Both ranks map the same host pages
// (shared memory registered fine-grained / uncached, see kernel/host_allreduce.py); one
// block per rank:
//   1. copy the local tensor into this rank's slot,  2. publish a sequence number,
//   3. wait for the peer's sequence number,          4. add the peer's slot into the tensor.
// The sequence number lives in device memory and is bumped by the kernel itself, so the
// launch has fixed arguments and can be captured in a CUDA graph. Slots are double
// buffered on the sequence parity: a rank rewrites a slot two calls later, which it can
// only reach after the peer has published the call in between, i.e. after the peer has
// read that slot. The sum is computed in fp32 in the same order on both ranks (a + b with
// the two ranks' values, commutative), so both ranks get the same bits.
//
// Layout of the shared region: [0, 64) rank 0 flag, [64, 128) rank 1 flag, then from
// kDataOffset four slots of `slot_bytes`: slot (rank * 2 + parity).
// A peer that never arrives (desynchronized ranks) traps after `timeout_ticks` of the
// 100 MHz wall clock instead of hanging the server.

namespace {

constexpr int kThreads = 512;
constexpr int64_t kDataOffset = 4096;

__device__ __forceinline__ uint64_t load_system(const uint64_t* p) {
#if FREETOKEN_USE_ROCM
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    return *reinterpret_cast<const volatile uint64_t*>(p);
#endif
}

__device__ __forceinline__ int64_t wall_ticks() {
#if FREETOKEN_USE_ROCM
    return static_cast<int64_t>(wall_clock64());
#else
    int64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t / 10;  // ns -> 100 MHz ticks
#endif
}

// element types: float, or bf16 carried as its raw 16 bits (round-to-nearest-even on the way
// back, the same rounding the collective libraries use)
struct bf16_bits { uint16_t v; };

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(bf16_bits v) { return __uint_as_float(static_cast<uint32_t>(v.v) << 16); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ bf16_bits from_f<bf16_bits>(float v) {
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return bf16_bits{static_cast<uint16_t>((u >> 16) | 0x40u)};  // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return bf16_bits{static_cast<uint16_t>(u >> 16)};
}

template <typename T>
__global__ void host_allreduce_kernel(
    T* __restrict__ x, int64_t n, char* base, int rank, int64_t slot_bytes,
    unsigned long long* counter, int64_t timeout_ticks) {
    __shared__ unsigned long long seq_s;
    if (threadIdx.x == 0) seq_s = *counter + 1;
    __syncthreads();
    const unsigned long long seq = seq_s;
    const int parity = static_cast<int>(seq & 1ull);
    T* mine = reinterpret_cast<T*>(base + kDataOffset + (rank * 2 + parity) * slot_bytes);
    const T* peer = reinterpret_cast<const T*>(base + kDataOffset + ((1 - rank) * 2 + parity) * slot_bytes);
    unsigned long long* my_flag = reinterpret_cast<unsigned long long*>(base + rank * 64);
    const unsigned long long* peer_flag = reinterpret_cast<const unsigned long long*>(base + (1 - rank) * 64);

    // 1. local tensor -> my slot: 16-byte stores when this rank's tensor allows it (the slot
    // layout is element-indexed either way, so the ranks may pick different widths)
    const bool vec = (reinterpret_cast<uintptr_t>(x) % 16 == 0) && ((n * sizeof(T)) % 16 == 0);
    if (vec) {
        const int64_t nv = n * sizeof(T) / 16;
        const uint4* src = reinterpret_cast<const uint4*>(x);
        uint4* dst = reinterpret_cast<uint4*>(mine);
        for (int64_t i = threadIdx.x; i < nv; i += kThreads) dst[i] = src[i];
    } else {
        for (int64_t i = threadIdx.x; i < n; i += kThreads) mine[i] = x[i];
    }
    __threadfence_system();
    __syncthreads();

    // 2-3. publish, then wait for the peer
    if (threadIdx.x == 0) {
#if FREETOKEN_USE_ROCM
        __hip_atomic_store(my_flag, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *reinterpret_cast<volatile unsigned long long*>(my_flag) = seq;
#endif
        const int64_t t0 = wall_ticks();
        while (load_system(reinterpret_cast<const uint64_t*>(peer_flag)) < seq) {
#if FREETOKEN_USE_ROCM
            __builtin_amdgcn_s_sleep(1);
#endif
            if (wall_ticks() - t0 > timeout_ticks) __builtin_trap();
        }
        *counter = seq;
    }
    __syncthreads();
#if FREETOKEN_USE_ROCM
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // system scope: drop any cached peer bytes
#else
    __threadfence_system();
#endif

    // 4. x += peer, summed in fp32 (identical bits on both ranks)
    if (vec) {
        const int64_t nv = n * sizeof(T) / 16;
        constexpr int kPer = 16 / sizeof(T);
        uint4* xv = reinterpret_cast<uint4*>(x);
        const uint64_t* pv = reinterpret_cast<const uint64_t*>(peer);
        for (int64_t i = threadIdx.x; i < nv; i += kThreads) {
            uint4 a = xv[i];
            uint64_t b2[2] = {load_system(pv + 2 * i), load_system(pv + 2 * i + 1)};
            const T* at = reinterpret_cast<const T*>(&a);
            const T* bt = reinterpret_cast<const T*>(b2);
            alignas(16) T out[kPer];  // read back through a uint4 below
#pragma unroll
            for (int k = 0; k < kPer; ++k) out[k] = from_f<T>(to_f(at[k]) + to_f(bt[k]));
            xv[i] = *reinterpret_cast<const uint4*>(out);
        }
    } else {
        for (int64_t i = threadIdx.x; i < n; i += kThreads) {
            T b;
            __builtin_memcpy(&b, const_cast<const T*>(reinterpret_cast<const volatile T*>(peer + i)), sizeof(T));
            x[i] = from_f<T>(to_f(x[i]) + to_f(b));
        }
    }
}

}  // namespace

struct HostAllReduce {
    static void run(
        tvm::ffi::TensorView x,
        tvm::ffi::TensorView counter,
        int64_t base_ptr,
        int64_t rank,
        int64_t slot_bytes,
        int64_t timeout_ticks
    ) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto N = SymbolicSize{"numel"};
        auto dtype = SymbolicDType{};
        auto cdtype = SymbolicDType{};
        TensorMatcher({N}).with_dtype(dtype).with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({1}).with_dtype<int64_t>(cdtype).with_device<kDLCUDA, kDLROCM>(device).verify(counter);
        const int64_t n = N.unwrap();
        const auto dt = dtype.unwrap();
        const bool is_f32 = dt.code == kDLFloat && dt.bits == 32;
        const bool is_bf16 = dt.code == kDLBfloat && dt.bits == 16;
        RuntimeCheck(dt.lanes == 1 && (is_f32 || is_bf16), "host all-reduce: float32 or bfloat16 only");
        RuntimeCheck(n * (dt.bits / 8) <= slot_bytes, "host all-reduce: message larger than the slot");
        RuntimeCheck(rank == 0 || rank == 1, "host all-reduce: two ranks only");
        auto* base = reinterpret_cast<char*>(base_ptr);
        auto* ctr = static_cast<unsigned long long*>(counter.data_ptr());
        auto launch = LaunchKernel(1, kThreads, device.unwrap());
        if (is_f32) {
            launch(host_allreduce_kernel<float>, static_cast<float*>(x.data_ptr()), n, base,
                   static_cast<int>(rank), slot_bytes, ctr, timeout_ticks);
        } else {
            launch(host_allreduce_kernel<bf16_bits>, static_cast<bf16_bits*>(x.data_ptr()), n, base,
                   static_cast<int>(rank), slot_bytes, ctr, timeout_ticks);
        }
    }
};
