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
// 100 MHz wall clock instead of hanging the server. With a `wait_log` (debug, see
// FREETOKEN_HOST_ALLREDUCE_WAITLOG), each call stores how long it waited for the peer, in ticks, at
// wait_log[seq % kWaitLogLen].

namespace {

constexpr int kThreads = 512;
constexpr int64_t kDataOffset = 4096;
constexpr unsigned long long kWaitLogLen = 1ull << 16;

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

// kAhead: peer chunks (16 bytes) a thread loads before it sums any. At 4 a message of up to 32 KB (six decode rows of
// 2560 bf16) crosses PCIe in one round of reads instead of one per kThreads * 16 bytes; the same sums either way
#if FREETOKEN_USE_ROCM
using native_u64x2 = uint64_t __attribute__((ext_vector_type(2)));
#endif

// kWide: the peer's 16 bytes in one non-temporal load (the region is registered uncached, read after the system-scope
// acquire below) instead of two 8-byte system-scope atomic loads
template <typename T, int kAhead, bool kWide>
__device__ __forceinline__ void allreduce_block(
    T* __restrict__ x, int64_t n, char* base, int rank, int64_t slot_bytes,
    unsigned long long* counter, int64_t timeout_ticks, unsigned long long* wait_log) {
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
        if (wait_log != nullptr) wait_log[seq % kWaitLogLen] = static_cast<unsigned long long>(wall_ticks() - t0);
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
        for (int64_t i0 = threadIdx.x; i0 < nv; i0 += kAhead * kThreads) {
            uint64_t b2[kAhead][2];
#pragma unroll
            for (int j = 0; j < kAhead; ++j) {  // every load in flight before the first sum
                const int64_t i = i0 + j * kThreads;
                if (i < nv) {
#if FREETOKEN_USE_ROCM
                    if constexpr (kWide) {
                        const native_u64x2 v = __builtin_nontemporal_load(reinterpret_cast<const native_u64x2*>(pv + 2 * i));
                        b2[j][0] = v.x;
                        b2[j][1] = v.y;
                        continue;
                    }
#endif
                    b2[j][0] = load_system(pv + 2 * i);
                    b2[j][1] = load_system(pv + 2 * i + 1);
                }
            }
#pragma unroll
            for (int j = 0; j < kAhead; ++j) {
                const int64_t i = i0 + j * kThreads;
                if (i < nv) {
                    uint4 a = xv[i];
                    const T* at = reinterpret_cast<const T*>(&a);
                    const T* bt = reinterpret_cast<const T*>(b2[j]);
                    alignas(16) T out[kPer];  // read back through a uint4 below
#pragma unroll
                    for (int k = 0; k < kPer; ++k) out[k] = from_f<T>(to_f(at[k]) + to_f(bt[k]));
                    xv[i] = *reinterpret_cast<const uint4*>(out);
                }
            }
        }
    } else {
        for (int64_t i = threadIdx.x; i < n; i += kThreads) {
            T b;
            __builtin_memcpy(&b, const_cast<const T*>(reinterpret_cast<const volatile T*>(peer + i)), sizeof(T));
            x[i] = from_f<T>(to_f(x[i]) + to_f(b));
        }
    }
}


template <typename T, int kAhead, bool kWide = false>
__global__ void host_allreduce_kernel(
    T* __restrict__ x, int64_t n, char* base, int rank, int64_t slot_bytes,
    unsigned long long* counter, int64_t timeout_ticks, unsigned long long* wait_log) {
    allreduce_block<T, kAhead, kWide>(x, n, base, rank, slot_bytes, counter, timeout_ticks, wait_log);
}

// --- FREETOKEN_HOST_AR_HC_COMBINE: the all-reduce of a block's output, then kernel/triton/hc.py's
// _hc_combine_rmsnorm_kernel on it, in one launch. Block 0 all-reduces (allreduce_block) and raises a device flag;
// blocks 1 .. rows * 4 wait for it, then each does one (row, stream) program of the Triton kernel with its 128 threads
// and its arithmetic: the inject gate 2 sigmoid(s / 4), the combine fma(inj, y, res) rounded to bf16 and stored over
// the residual, the sum of squares of a thread's 32 values (registers 8 rr + j of element 1024 rr + 8 t + j) as a
// pairwise tree, the lanes at distances 8, 4, 2, 1 then 16, the warps as (w0 + w2) + (w1 + w3), a correctly rounded
// division by the stream's width, v_rsq, then y = rrms x and fma(y, w, y) rounded to bf16. The last consumer resets
// the flag for the next launch (stream order).
__device__ __forceinline__ float hc_bf16_to_f32(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

// round to nearest even, a NaN as 0x7FFF: Triton's fp32 -> bf16 conversion on gfx11
__device__ __forceinline__ uint16_t hc_f32_to_bf16(float f) {
    if (f != f) return 0x7FFFu;
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

template <int M>
__device__ __forceinline__ float hc_dpp_xor(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + M, 0xF, 0xF, false));
}

// a 32-lane sum as Triton's reduction adds it (distances 8, 4, 2, 1 within a row of 16, then 16)
__device__ __forceinline__ float hc_lanes32(float t) {
    t += hc_dpp_xor<8>(t);
    t += hc_dpp_xor<4>(t);
    t += hc_dpp_xor<2>(t);
    t += hc_dpp_xor<1>(t);
    return t + __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(t), 0x76543210u, 0xFEDCBA98u, false, false));
}

__device__ __forceinline__ float hc_tree16(const float* v) {
    const float a0 = v[0] + v[1], a1 = v[2] + v[3], a2 = v[4] + v[5], a3 = v[6] + v[7];
    const float a4 = v[8] + v[9], a5 = v[10] + v[11], a6 = v[12] + v[13], a7 = v[14] + v[15];
    return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}

template <int kAhead, bool kWide>
__global__ void __launch_bounds__(kThreads) host_allreduce_hc_combine_kernel(
    bf16_bits* __restrict__ x, int64_t n, char* base, int rank, int64_t slot_bytes, unsigned long long* counter,
    int64_t timeout_ticks, unsigned long long* wait_log, uint16_t* __restrict__ res, const uint16_t* __restrict__ inj,
    const uint16_t* __restrict__ nw, uint16_t* __restrict__ rn, int hd, int64_t stride_res, int64_t stride_inj,
    int64_t stride_rn, int w_shared, float eps, int* flags) {
    __shared__ float red[4];
    if (blockIdx.x == 0) {
        allreduce_block<bf16_bits, kAhead, kWide>(x, n, base, rank, slot_bytes, counter, timeout_ticks, wait_log);
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();  // the sums, device-wide, before the flag
            __hip_atomic_store(flags, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        return;
    }
    const int p = blockIdx.x - 1, r = p / 4, st = p % 4;
    const int t = threadIdx.x;
    const bool act = t < 128;
    const int warp = t / 32, lane = t % 32;
    uint16_t* rp = res + r * stride_res + st * hd;
    const uint16_t* yp = reinterpret_cast<const uint16_t*>(x) + static_cast<int64_t>(r) * hd;
    const uint16_t* wp = nw + (w_shared ? 0 : st * hd);
    // what does not depend on the all-reduce (the residual, the norm weight, the inject gate) while block 0 waits
    uint4 rv[4], wv[4];
    float injv = 0.f;
    if (act) {
        const float s = hc_bf16_to_f32(inj[r * stride_inj + st]) * 0.25f;
        const float e = __builtin_amdgcn_exp2f(s * -1.44269502162933349609375f);
        injv = 2.0f * (1.0f / (1.0f + e));
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int e0 = rr * 1024 + t * 8;
            if (e0 < hd) {  // hd is a multiple of 8
                rv[rr] = *reinterpret_cast<const uint4*>(rp + e0);
                wv[rr] = *reinterpret_cast<const uint4*>(wp + e0);
            }
        }
    }
    if (threadIdx.x == 0) {
        while (__hip_atomic_load(flags, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0) __builtin_amdgcn_s_sleep(1);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    uint32_t xw[4][4];  // the combined values, packed bf16 pairs (element 1024 rr + 8 t + j)
    if (act) {
        uint4 yv[4];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int e0 = rr * 1024 + t * 8;
            if (e0 < hd) yv[rr] = *reinterpret_cast<const uint4*>(yp + e0);
        }
        float sq[32];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int e0 = rr * 1024 + t * 8;
            const uint32_t rw[4] = {rv[rr].x, rv[rr].y, rv[rr].z, rv[rr].w};
            const uint32_t yw[4] = {yv[rr].x, yv[rr].y, yv[rr].z, yv[rr].w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t pair = 0;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float xf = 0.f;
                    uint16_t xb = 0;
                    if (e0 < hd) {
                        const float rf = hc_bf16_to_f32(static_cast<uint16_t>(h ? (rw[i] >> 16) : (rw[i] & 0xFFFFu)));
                        const float yf = hc_bf16_to_f32(static_cast<uint16_t>(h ? (yw[i] >> 16) : (yw[i] & 0xFFFFu)));
                        xb = hc_f32_to_bf16(__builtin_fmaf(injv, yf, rf));
                        xf = hc_bf16_to_f32(xb);
                    }
                    sq[rr * 8 + 2 * i + h] = xf * xf;
                    pair |= static_cast<uint32_t>(xb) << (16 * h);
                }
                xw[rr][i] = pair;
            }
        }
        const float part = hc_lanes32(hc_tree16(sq) + hc_tree16(sq + 16));
        if (lane == 0) red[warp] = part;
    }
    __syncthreads();
    if (act) {
        const float total = (red[0] + red[2]) + (red[1] + red[3]);
        const float rrms = __builtin_amdgcn_rsqf(total / static_cast<float>(hd) + eps);
        uint16_t* op = rn + r * stride_rn + st * hd;
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            const int e0 = rr * 1024 + t * 8;
            if (e0 >= hd) continue;
            const uint32_t ww[4] = {wv[rr].x, wv[rr].y, wv[rr].z, wv[rr].w};
            uint32_t yo[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t pair = 0;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float wf = hc_bf16_to_f32(static_cast<uint16_t>(h ? (ww[i] >> 16) : (ww[i] & 0xFFFFu)));
                    const float xf = hc_bf16_to_f32(static_cast<uint16_t>(h ? (xw[rr][i] >> 16) : (xw[rr][i] & 0xFFFFu)));
                    const float y1 = rrms * xf;
                    pair |= static_cast<uint32_t>(hc_f32_to_bf16(__builtin_fmaf(y1, wf, y1))) << (16 * h);
                }
                yo[i] = pair;
            }
            *reinterpret_cast<uint4*>(rp + e0) = make_uint4(xw[rr][0], xw[rr][1], xw[rr][2], xw[rr][3]);
            *reinterpret_cast<uint4*>(op + e0) = make_uint4(yo[0], yo[1], yo[2], yo[3]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {  // the last consumer resets the flag and the count for the next launch
        if (__hip_atomic_fetch_add(flags + 1, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == static_cast<int>(gridDim.x) - 2) {
            __hip_atomic_store(flags + 1, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(flags, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// FREETOKEN_HOST_ALLGATHER: the two-rank all-gather of small messages through the same slots, flags and sequence
// counter (calls interleave with the all-reduces in the same order on both ranks): this rank's words into its slot,
// publish, wait for the peer, then out = [rank 0's words, rank 1's words]. Copies only
__global__ void host_allgather_kernel(const uint32_t* __restrict__ x, uint32_t* __restrict__ out, int64_t nw, char* base,
                                      int rank, int64_t slot_bytes, unsigned long long* counter, int64_t timeout_ticks,
                                      unsigned long long* wait_log) {
    __shared__ unsigned long long seq_s;
    if (threadIdx.x == 0) seq_s = *counter + 1;
    __syncthreads();
    const unsigned long long seq = seq_s;
    const int parity = static_cast<int>(seq & 1ull);
    uint32_t* mine = reinterpret_cast<uint32_t*>(base + kDataOffset + (rank * 2 + parity) * slot_bytes);
    const uint32_t* peer = reinterpret_cast<const uint32_t*>(base + kDataOffset + ((1 - rank) * 2 + parity) * slot_bytes);
    unsigned long long* my_flag = reinterpret_cast<unsigned long long*>(base + rank * 64);
    const unsigned long long* peer_flag = reinterpret_cast<const unsigned long long*>(base + (1 - rank) * 64);
    uint32_t* own_out = out + rank * nw;
    uint32_t* peer_out = out + (1 - rank) * nw;
    for (int64_t i = threadIdx.x; i < nw; i += kThreads) {
        const uint32_t v = x[i];
        mine[i] = v;
        own_out[i] = v;
    }
    __threadfence_system();
    __syncthreads();
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
        if (wait_log != nullptr) wait_log[seq % kWaitLogLen] = static_cast<unsigned long long>(wall_ticks() - t0);
        *counter = seq;
    }
    __syncthreads();
#if FREETOKEN_USE_ROCM
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");  // system scope: drop any cached peer bytes
#else
    __threadfence_system();
#endif
    for (int64_t i = threadIdx.x; i < nw; i += kThreads) {
        uint32_t v;
        __builtin_memcpy(&v, const_cast<const uint32_t*>(reinterpret_cast<const volatile uint32_t*>(peer + i)), 4);
        peer_out[i] = v;
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
        int64_t timeout_ticks,
        int64_t wait_log_ptr,
        int64_t ahead
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
        auto* wait_log = reinterpret_cast<unsigned long long*>(wait_log_ptr);
        // ahead: 0 = one peer chunk at a time; 1 = four ahead, 8-byte atomic loads; 2 = four ahead, 16-byte loads
        RuntimeCheck(ahead >= 0 && ahead <= 2, "host all-reduce: ahead must be 0, 1 or 2");
        auto launch = LaunchKernel(1, kThreads, device.unwrap());
        if (is_f32) {
            launch(ahead == 2 ? host_allreduce_kernel<float, 4, true>
                   : ahead == 1 ? host_allreduce_kernel<float, 4> : host_allreduce_kernel<float, 1>,
                   static_cast<float*>(x.data_ptr()), n, base, static_cast<int>(rank), slot_bytes, ctr, timeout_ticks,
                   wait_log);
        } else {
            launch(ahead == 2 ? host_allreduce_kernel<bf16_bits, 4, true>
                   : ahead == 1 ? host_allreduce_kernel<bf16_bits, 4> : host_allreduce_kernel<bf16_bits, 1>,
                   static_cast<bf16_bits*>(x.data_ptr()), n, base, static_cast<int>(rank), slot_bytes, ctr,
                   timeout_ticks, wait_log);
        }
    }
};

struct HostAllGather {
    static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView out, tvm::ffi::TensorView counter, int64_t base_ptr,
                    int64_t rank, int64_t slot_bytes, int64_t timeout_ticks, int64_t wait_log_ptr) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto N = SymbolicSize{"words"};
        auto N2 = SymbolicSize{"gathered words"};
        auto cdtype = SymbolicDType{};
        TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({N2}).with_dtype<int32_t>().with_device<kDLCUDA, kDLROCM>(device).verify(out);
        TensorMatcher({1}).with_dtype<int64_t>(cdtype).with_device<kDLCUDA, kDLROCM>(device).verify(counter);
        const int64_t n = N.unwrap();
        RuntimeCheck(N2.unwrap() == 2 * n, "host all-gather: the output holds both ranks' words");
        RuntimeCheck(n * 4 <= slot_bytes, "host all-gather: message larger than the slot");
        RuntimeCheck(rank == 0 || rank == 1, "host all-gather: two ranks only");
        auto launch = LaunchKernel(1, kThreads, device.unwrap());
        launch(host_allgather_kernel, static_cast<const uint32_t*>(x.data_ptr()), static_cast<uint32_t*>(out.data_ptr()),
               n, reinterpret_cast<char*>(base_ptr), static_cast<int>(rank), slot_bytes,
               static_cast<unsigned long long*>(counter.data_ptr()), timeout_ticks,
               reinterpret_cast<unsigned long long*>(wait_log_ptr));
    }
};

struct HostAllReduceHcCombine {
    static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView counter, int64_t base_ptr, int64_t rank,
                    int64_t slot_bytes, int64_t timeout_ticks, int64_t wait_log_ptr, int64_t ahead,
                    tvm::ffi::TensorView res, tvm::ffi::TensorView inj, tvm::ffi::TensorView nw,
                    tvm::ffi::TensorView rn, double eps, tvm::ffi::TensorView flags) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto M = SymbolicSize{"rows"};
        auto H = SymbolicSize{"stream width"};
        auto K = SymbolicSize{"streams width"};
        auto C = SymbolicSize{"inject cols"};
        auto W = SymbolicSize{"norm weight"};
        auto rs = SymbolicSize{"res row stride"};
        auto is = SymbolicSize{"inj row stride"};
        auto ns = SymbolicSize{"rn row stride"};
        auto bd = SymbolicDType{};
        auto cdtype = SymbolicDType{};
        TensorMatcher({M, H}).with_strides({H, 1}).with_dtype(bd).with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({M, K}).with_strides({rs, 1}).with_dtype(bd).with_device<kDLCUDA, kDLROCM>(device).verify(res);
        TensorMatcher({M, C}).with_strides({is, 1}).with_dtype(bd).with_device<kDLCUDA, kDLROCM>(device).verify(inj);
        TensorMatcher({W}).with_dtype(bd).with_device<kDLCUDA, kDLROCM>(device).verify(nw);
        TensorMatcher({M, K}).with_strides({ns, 1}).with_dtype(bd).with_device<kDLCUDA, kDLROCM>(device).verify(rn);
        TensorMatcher({1}).with_dtype<int64_t>(cdtype).with_device<kDLCUDA, kDLROCM>(device).verify(counter);
        TensorMatcher({2}).with_dtype<int32_t>().with_device<kDLCUDA, kDLROCM>(device).verify(flags);
        const auto dt = bd.unwrap();
        RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16 && dt.lanes == 1, "hc combine all-reduce: bfloat16");
        const int64_t m = M.unwrap(), hd = H.unwrap(), wn = W.unwrap();
        RuntimeCheck(m >= 1 && m <= 16, "hc combine all-reduce: 1-16 rows");
        RuntimeCheck(K.unwrap() == 4 * hd && hd % 8 == 0 && hd > 3072 - 1024 && hd <= 4096, "hc combine all-reduce: 4 streams");
        RuntimeCheck(C.unwrap() >= 4 && (wn == hd || wn == 4 * hd), "hc combine all-reduce: inject cols, norm weight");
        RuntimeCheck(m * hd * 2 <= slot_bytes && (rank == 0 || rank == 1), "hc combine all-reduce: slot, ranks");
        RuntimeCheck(ahead >= 0 && ahead <= 2, "hc combine all-reduce: ahead must be 0, 1 or 2");
        for (auto* p : {x.data_ptr(), res.data_ptr(), nw.data_ptr(), rn.data_ptr()}) {
            RuntimeCheck(reinterpret_cast<uintptr_t>(p) % 16 == 0, "hc combine all-reduce: 16-byte aligned rows");
        }
        RuntimeCheck(rs.unwrap() % 8 == 0 && ns.unwrap() % 8 == 0, "hc combine all-reduce: 16-byte row strides");
        auto launch = LaunchKernel(static_cast<int>(1 + 4 * m), kThreads, device.unwrap());
        auto* xa = static_cast<bf16_bits*>(x.data_ptr());
        auto* ctr = static_cast<unsigned long long*>(counter.data_ptr());
        auto* b = reinterpret_cast<char*>(base_ptr);
        auto* wl = reinterpret_cast<unsigned long long*>(wait_log_ptr);
        auto* ra = static_cast<uint16_t*>(res.data_ptr());
        auto* ia = static_cast<const uint16_t*>(inj.data_ptr());
        auto* na = static_cast<const uint16_t*>(nw.data_ptr());
        auto* qa = static_cast<uint16_t*>(rn.data_ptr());
        auto* fa = static_cast<int*>(flags.data_ptr());
        const int n = static_cast<int>(m * hd);
        launch(ahead == 2 ? host_allreduce_hc_combine_kernel<4, true>
               : ahead == 1 ? host_allreduce_hc_combine_kernel<4, false> : host_allreduce_hc_combine_kernel<1, false>,
               xa, static_cast<int64_t>(n), b, static_cast<int>(rank), slot_bytes, ctr, timeout_ticks, wl, ra, ia, na, qa,
               static_cast<int>(hd), rs.unwrap(), is.unwrap(), ns.unwrap(), static_cast<int>(wn == hd),
               static_cast<float>(eps), fa);
    }
};
