#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// The int8 weight-only GEMV of kernel/triton/int8_gemv.py for its one-pass tile (BLOCK_N = 2, BLOCK_K = 4096, 4 warps)
// at 1-8 activation rows, with the Triton kernel's sums (FREETOKEN_INT8_EXACT_GEMV). An int8 weight times a bf16
// activation is exact in fp32, so only the order of the additions sets the bits; it is the Triton kernel's: each
// thread holds 16 columns of each of the two 2048-column halves of the tile and sums its 32 products as a pairwise
// tree (adjacent pairs), the lanes then add across lane distances 8, 4, 2, 1 and 16, the warps as
// (w0 + w2) + (w1 + w3).
// Products past K are +0 in Triton's sums: a half of zeros adds nothing, so it is skipped (the lanes that sum only
// zeros there), the rows' cross-warp sums share one barrier instead of one each, and rows 5-8 read the weight tile
// again from the caches.
namespace {

constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kWarps8 = 8;

__device__ __forceinline__ float bf16_to_f32(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

// round to nearest even, a NaN as 0x7FFF: Triton's fp32 -> bf16 conversion on gfx11
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
    if (f != f) return 0x7FFFu;
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float int8_at(uint32_t word, int byte) {
    return static_cast<float>(static_cast<int8_t>((word >> (8 * byte)) & 0xFF));
}

// the 8 warps' partials of one sum: w with w + 4, then + 2, then + 1
template <int W>
__device__ __forceinline__ float warps8_sum(const float (&red)[8][W], int i) {
    const float a = (red[0][i] + red[4][i]) + (red[2][i] + red[6][i]);
    const float b = (red[1][i] + red[5][i]) + (red[3][i] + red[7][i]);
    return a + b;
}

// tl.sigmoid as Triton lowers it: 1 / (1 + 2^(x * -log2(e))) with the hardware exp2 and a correctly rounded division
// (Triton scales the exp2 argument below -126, where 1 + e is 1 either way)
__device__ __forceinline__ float sigmoid_triton(float x) {
    const float e = __builtin_amdgcn_exp2f(x * -1.44269502162933349609375f);
    return 1.0f / (1.0f + e);
}

// the pairwise tree of 16 values, adjacent pairs first
__device__ __forceinline__ float tree16(const float (&v)[16]) {
    float a0 = v[0] + v[1], a1 = v[2] + v[3], a2 = v[4] + v[5], a3 = v[6] + v[7];
    float a4 = v[8] + v[9], a5 = v[10] + v[11], a6 = v[12] + v[13], a7 = v[14] + v[15];
    float b0 = a0 + a1, b1 = a2 + a3, b2 = a4 + a5, b3 = a6 + a7;
    return (b0 + b1) + (b2 + b3);
}

// rows r0 .. r0 + RB - 1 (RB <= 4) of the one-pass tile: each thread's trees and the lanes, into red[warp][2 r + n]
template <int RB>
__device__ __forceinline__ void one_pass_rows(const uint16_t* __restrict__ a, const int8_t* __restrict__ w, int n0,
                                              int N, int K, int64_t stride_wn, int64_t stride_am, int r0,
                                              float (&red)[kWarps][16], int warp, int lane) {
    float half_sum[2][RB][2];
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int k0 = rep * 2048 + warp * 512 + lane * 16;
        const bool live = k0 < K;  // K is a multiple of 16: a thread's 16 columns are all in or all out
#pragma unroll
        for (int r = 0; r < RB; ++r) half_sum[rep][r][0] = half_sum[rep][r][1] = 0.f;
        if (!live) continue;
        float av[RB][16];
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            const uint4* ap = reinterpret_cast<const uint4*>(a + (r0 + r) * stride_am + k0);
            uint4 lo = ap[0], hi = ap[1];
            const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                av[r][2 * i] = bf16_to_f32(static_cast<uint16_t>(words[i] & 0xFFFFu));
                av[r][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(words[i] >> 16));
            }
        }
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            const int n = n0 + nn;
            if (n >= N) continue;
            const uint4 wq = *reinterpret_cast<const uint4*>(w + n * stride_wn + k0);
            const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
            float wf[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) wf[i] = int8_at(ww[i / 4], i % 4);
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                float p[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) p[i] = wf[i] * av[r][i];
                half_sum[rep][r][nn] = tree16(p);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RB; ++r) {
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            // the thread's 32 products: the two halves' trees, then their sum (a dead half is +0: x + 0 = x)
            float t = half_sum[0][r][nn] + half_sum[1][r][nn];
            t += __shfl_xor(t, 8);
            t += __shfl_xor(t, 4);
            t += __shfl_xor(t, 2);
            t += __shfl_xor(t, 1);
            t += __shfl_xor(t, 16);
            if (lane == 0) red[warp][2 * (r0 + r) + nn] = t;
        }
    }
}

template <int ROWS>
__global__ void __launch_bounds__(kThreads) int8_gemv_exact_kernel(
    const uint16_t* __restrict__ a, const int8_t* __restrict__ w, const float* __restrict__ scale,
    uint16_t* __restrict__ out, int N, int K, int64_t stride_wn, int64_t stride_am, int64_t stride_om) {
    __shared__ float red[kWarps][ROWS * 2];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = blockIdx.x * 2;
    float acc[ROWS][2];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r][0] = acc[r][1] = 0.f;
    float half_sum[2][ROWS][2];
#pragma unroll
    for (int rep = 0; rep < 2; ++rep) {
        const int k0 = rep * 2048 + warp * 512 + lane * 16;
        const bool live = k0 < K;  // K is a multiple of 16: a thread's 16 columns are all in or all out
#pragma unroll
        for (int r = 0; r < ROWS; ++r) half_sum[rep][r][0] = half_sum[rep][r][1] = 0.f;
        if (!live) continue;
        float av[ROWS][16];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint4* ap = reinterpret_cast<const uint4*>(a + r * stride_am + k0);
            uint4 lo = ap[0], hi = ap[1];
            const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                av[r][2 * i] = bf16_to_f32(static_cast<uint16_t>(words[i] & 0xFFFFu));
                av[r][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(words[i] >> 16));
            }
        }
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            const int n = n0 + nn;
            if (n >= N) continue;
            const uint4 wq = *reinterpret_cast<const uint4*>(w + n * stride_wn + k0);
            const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
            float wf[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) wf[i] = int8_at(ww[i / 4], i % 4);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                float p[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) p[i] = wf[i] * av[r][i];
                half_sum[rep][r][nn] = tree16(p);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int nn = 0; nn < 2; ++nn) {
            // the thread's 32 products: the two halves' trees, then their sum (a dead half is +0: x + 0 = x)
            float t = half_sum[0][r][nn] + half_sum[1][r][nn];
            t += __shfl_xor(t, 8);
            t += __shfl_xor(t, 4);
            t += __shfl_xor(t, 2);
            t += __shfl_xor(t, 1);
            t += __shfl_xor(t, 16);
            acc[r][nn] = t;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            red[warp][2 * r] = acc[r][0];
            red[warp][2 * r + 1] = acc[r][1];
        }
    }
    __syncthreads();
    if (threadIdx.x < ROWS * 2) {
        const int r = threadIdx.x / 2, nn = threadIdx.x % 2, n = n0 + nn;
        if (n < N) {
            const int i = 2 * r + nn;
            const float s = (red[0][i] + red[2][i]) + (red[1][i] + red[3][i]);
            out[r * stride_om + n] = f32_to_bf16_rne(s * scale[n]);
        }
    }
}

// 5-8 rows (two to four requests decoding): four rows a pass, the weight tile read again from the caches; the
// same sums for each row
template <int ROWS>
__global__ void __launch_bounds__(kThreads) int8_gemv_exact_rows8_kernel(
    const uint16_t* __restrict__ a, const int8_t* __restrict__ w, const float* __restrict__ scale,
    uint16_t* __restrict__ out, int N, int K, int64_t stride_wn, int64_t stride_am, int64_t stride_om) {
    __shared__ float red[kWarps][16];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = blockIdx.x * 2;
    // up to 4 rows at a time (the weight tile read again from the caches for rows 5-8): each row as alone
    one_pass_rows<(ROWS < 4 ? ROWS : 4)>(a, w, n0, N, K, stride_wn, stride_am, 0, red, warp, lane);
    if constexpr (ROWS > 4) one_pass_rows<ROWS - 4>(a, w, n0, N, K, stride_wn, stride_am, 4, red, warp, lane);
    __syncthreads();
    if (threadIdx.x < ROWS * 2) {
        const int r = threadIdx.x / 2, nn = threadIdx.x % 2, n = n0 + nn;
        if (n < N) {
            const int i = 2 * r + nn;
            const float s = (red[0][i] + red[2][i]) + (red[1][i] + red[3][i]);
            out[r * stride_om + n] = f32_to_bf16_rne(s * scale[n]);
        }
    }
}

// The K-blocked tile (BLOCK_N = 1, BLOCK_K = 2048, 8 warps; the hyper-connections' down projection, 336 x 10240) at
// 1-8 rows, the row-group kernel's sums: each thread keeps 8 columns of the 2048-wide block and adds each K block's
// products to them in K order (acc = 0 + p, then acc + p: exact products, so the fused form adds the same), then the
// 8 as a pairwise tree, the lanes at distances 8, 4, 2, 1 and 16, the warps pairing w with w + 4, then + 2, then + 1.
// HC > 0: the columns below SILU_COLS stored through hc_silu (silu(bf16 value / HC), rounded to bf16).
__device__ __forceinline__ float tree8(const float (&v)[8]) {
    float a0 = v[0] + v[1], a1 = v[2] + v[3], a2 = v[4] + v[5], a3 = v[6] + v[7];
    return (a0 + a1) + (a2 + a3);
}

template <int ROWS>
__global__ void __launch_bounds__(32 * kWarps8) int8_gemv_exact_k2048_kernel(
    const uint16_t* __restrict__ a, const int8_t* __restrict__ w, const float* __restrict__ scale,
    uint16_t* __restrict__ out, int N, int K, int64_t stride_wn, int64_t stride_am, int64_t stride_om, int hc,
    int silu_cols) {
    __shared__ float red[kWarps8][ROWS];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n = blockIdx.x;
    float acc[ROWS][8];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][i] = 0.f;
    const int8_t* wrow = w + n * stride_wn;
    for (int kk = 0; kk < K; kk += 2048) {
        const int k0 = kk + warp * 256 + lane * 8;
        if (k0 >= K) continue;  // K is a multiple of 8: the 8 columns are all in or all out
        const uint2 wq = *reinterpret_cast<const uint2*>(wrow + k0);
        float wf[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
            wf[i] = static_cast<float>(static_cast<int8_t>((((i < 4) ? wq.x : wq.y) >> (8 * (i % 4))) & 0xFF));
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint4 av = *reinterpret_cast<const uint4*>(a + r * stride_am + k0);
            const uint32_t words[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[r][2 * i] += wf[2 * i] * bf16_to_f32(static_cast<uint16_t>(words[i] & 0xFFFFu));
                acc[r][2 * i + 1] += wf[2 * i + 1] * bf16_to_f32(static_cast<uint16_t>(words[i] >> 16));
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float t = tree8(acc[r]);
        t += __shfl_xor(t, 8);
        t += __shfl_xor(t, 4);
        t += __shfl_xor(t, 2);
        t += __shfl_xor(t, 1);
        t += __shfl_xor(t, 16);
        if (lane == 0) red[warp][r] = t;
    }
    __syncthreads();
    if (threadIdx.x < ROWS) {
        const int r = threadIdx.x;
        const float s = warps8_sum(red, r);
        uint16_t y = f32_to_bf16_rne(s * scale[n]);
        if (hc > 0 && n < silu_cols) {
            const float x = bf16_to_f32(y) / static_cast<float>(hc);
            y = f32_to_bf16_rne(x * sigmoid_triton(x));
        }
        out[r * stride_om + n] = y;
    }
}

// The 8-warp one-pass tile (BLOCK_N = 4, BLOCK_K = 4096; attention qkv, 6656 x 2560) at 1-8 rows: each thread 16
// columns (k = 512 w + 16 l), the same 16-value tree, the lanes as above, the warps pairing w with w + 4, + 2, + 1.
// A warp past K only adds zeros and is skipped.
// rows r0 .. r0 + RB - 1 (RB <= 4) of the 8-warp tile, into red[warp][4 r + n]
template <int RB>
__device__ __forceinline__ void n4w8_rows(const uint16_t* __restrict__ a, const int8_t* __restrict__ w, int n0, int N,
                                          int K, int64_t stride_wn, int64_t stride_am, int r0,
                                          float (&red)[kWarps8][32], int warp, int lane) {
    const int k0 = warp * 512 + lane * 16;
    const bool live = k0 < K;
    float av[RB][16];
#pragma unroll
    for (int r = 0; r < RB; ++r) {
        if (live) {
            const uint4* ap = reinterpret_cast<const uint4*>(a + (r0 + r) * stride_am + k0);
            uint4 lo = ap[0], hi = ap[1];
            const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                av[r][2 * i] = bf16_to_f32(static_cast<uint16_t>(words[i] & 0xFFFFu));
                av[r][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(words[i] >> 16));
            }
        }
    }
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
        const int n = n0 + nn;
        float t[RB];
#pragma unroll
        for (int r = 0; r < RB; ++r) t[r] = 0.f;
        if (live && n < N) {
            const uint4 wq = *reinterpret_cast<const uint4*>(w + n * stride_wn + k0);
            const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
            float wf[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) wf[i] = int8_at(ww[i / 4], i % 4);
#pragma unroll
            for (int r = 0; r < RB; ++r) {
                float p[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) p[i] = wf[i] * av[r][i];
                t[r] = tree16(p);
            }
        }
#pragma unroll
        for (int r = 0; r < RB; ++r) {
            float v = t[r];
            v += __shfl_xor(v, 8);
            v += __shfl_xor(v, 4);
            v += __shfl_xor(v, 2);
            v += __shfl_xor(v, 1);
            v += __shfl_xor(v, 16);
            if (lane == 0) red[warp][4 * (r0 + r) + nn] = v;
        }
    }
}

template <int ROWS>
__global__ void __launch_bounds__(32 * kWarps8) int8_gemv_exact_n4w8_kernel(
    const uint16_t* __restrict__ a, const int8_t* __restrict__ w, const float* __restrict__ scale,
    uint16_t* __restrict__ out, int N, int K, int64_t stride_wn, int64_t stride_am, int64_t stride_om) {
    __shared__ float red[kWarps8][ROWS * 4];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = blockIdx.x * 4;
    const int k0 = warp * 512 + lane * 16;
    float acc[ROWS][4];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int nn = 0; nn < 4; ++nn) acc[r][nn] = 0.f;
    if (warp * 512 < K) {
        const bool live = k0 < K;
        float av[ROWS][16];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (live) {
                const uint4* ap = reinterpret_cast<const uint4*>(a + r * stride_am + k0);
                uint4 lo = ap[0], hi = ap[1];
                const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    av[r][2 * i] = bf16_to_f32(static_cast<uint16_t>(words[i] & 0xFFFFu));
                    av[r][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(words[i] >> 16));
                }
            }
        }
#pragma unroll
        for (int nn = 0; nn < 4; ++nn) {
            const int n = n0 + nn;
            float t[ROWS];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) t[r] = 0.f;
            if (live && n < N) {
                const uint4 wq = *reinterpret_cast<const uint4*>(w + n * stride_wn + k0);
                const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
                float wf[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) wf[i] = int8_at(ww[i / 4], i % 4);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float p[16];
#pragma unroll
                    for (int i = 0; i < 16; ++i) p[i] = wf[i] * av[r][i];
                    t[r] = tree16(p);
                }
            }
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                float v = t[r];
                v += __shfl_xor(v, 8);
                v += __shfl_xor(v, 4);
                v += __shfl_xor(v, 2);
                v += __shfl_xor(v, 1);
                v += __shfl_xor(v, 16);
                acc[r][nn] = v;
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int nn = 0; nn < 4; ++nn) red[warp][4 * r + nn] = acc[r][nn];
    }
    __syncthreads();
    if (threadIdx.x < ROWS * 4) {
        const int i = threadIdx.x, r = i / 4, n = n0 + i % 4;
        if (n < N) {
            const float s = warps8_sum(red, i);
            out[r * stride_om + n] = f32_to_bf16_rne(s * scale[n]);
        }
    }
}

// 5-8 rows (two to four requests decoding): four rows a pass; the same sums for each row
template <int ROWS>
__global__ void __launch_bounds__(32 * kWarps8) int8_gemv_exact_n4w8_rows8_kernel(
    const uint16_t* __restrict__ a, const int8_t* __restrict__ w, const float* __restrict__ scale,
    uint16_t* __restrict__ out, int N, int K, int64_t stride_wn, int64_t stride_am, int64_t stride_om) {
    __shared__ float red[kWarps8][32];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n0 = blockIdx.x * 4;
    if (warp * 512 < K) {
        n4w8_rows<(ROWS < 4 ? ROWS : 4)>(a, w, n0, N, K, stride_wn, stride_am, 0, red, warp, lane);
        if constexpr (ROWS > 4) n4w8_rows<ROWS - 4>(a, w, n0, N, K, stride_wn, stride_am, 4, red, warp, lane);
    } else if (lane == 0) {
#pragma unroll
        for (int i = 0; i < ROWS * 4; ++i) red[warp][i] = 0.f;
    }
    __syncthreads();
    if (threadIdx.x < ROWS * 4) {
        const int i = threadIdx.x, r = i / 4, n = n0 + i % 4;
        if (n < N) {
            const float s = warps8_sum(red, i);
            out[r * stride_om + n] = f32_to_bf16_rne(s * scale[n]);
        }
    }
}

// gemv_int8_hc_mix (the hyper-connection's up projection, 4 streams x HD rows of K = 320, mixed into the streams) at
// 1-8 rows, its sums: per gate (stream, row) the 1-warp tile's 16-column tree and lanes (8, 4, 2, 1, 16), the gate
// rounded to bf16, then out[r, h] = bf16(sum over the streams in order of fma(sigmoid(gate), rn, acc) / 4). Each warp
// takes two hidden indices (one past 4 rows): their 32 gates are reduced together (each pair of lanes adds the same
// values as the xor tree, one gate a lane at the end) and their sigmoids computed once across the lanes instead of one
// after another.
template <int ROWS>
__global__ void __launch_bounds__(256) int8_hc_mix_exact_kernel(
    const uint16_t* __restrict__ a, const int8_t* __restrict__ w, const float* __restrict__ scale,
    const uint16_t* __restrict__ rn, uint16_t* __restrict__ out, int K, int HD, int64_t stride_wn,
    int64_t stride_am, int64_t stride_rn, int64_t stride_om) {
    constexpr int HC = 4;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int h0 = (blockIdx.x * 8 + warp) * 2;
    if (h0 >= HD) return;
    const bool live = lane * 16 < K;
    float av[ROWS][16];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int i = 0; i < 16; ++i) av[r][i] = 0.f;
        if (live) {
            const uint4* ap = reinterpret_cast<const uint4*>(a + r * stride_am + lane * 16);
            uint4 lo = ap[0], hi = ap[1];
            const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                av[r][2 * i] = bf16_to_f32(static_cast<uint16_t>(words[i] & 0xFFFFu));
                av[r][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(words[i] >> 16));
            }
        }
    }
    // v[j]: this lane's tree of gate j = hh * 16 + r * 4 + stream (rows past ROWS stay 0)
    float v[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) v[j] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
        for (int st = 0; st < HC; ++st) {
            const int n = st * HD + h0 + hh;
            if (live && h0 + hh < HD) {
                const uint4 wq = *reinterpret_cast<const uint4*>(w + n * stride_wn + lane * 16);
                const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
                float wf[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) wf[i] = int8_at(ww[i / 4], i % 4);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float p[16];
#pragma unroll
                    for (int i = 0; i < 16; ++i) p[i] = wf[i] * av[r][i];
                    v[hh * 16 + r * 4 + st] = tree16(p);
                }
            }
        }
    }
    // the xor tree's pairs (lane distance 8, 4, 2, 1, then 16), halving the gates a lane holds at each level: at
    // distance d a lane keeps the half selected by its bit d and adds its partner's values of that half
    float* cur = v;
    float buf[16];
#define FT_HALVE(D, COUNT, SRC, DST)                                                       \
    {                                                                                       \
        const bool upper = (lane & (D)) != 0;                                               \
        _Pragma("unroll") for (int j = 0; j < (COUNT) / 2; ++j) {                           \
            const float send = upper ? (SRC)[j] : (SRC)[j + (COUNT) / 2];                    \
            const float keep = upper ? (SRC)[j + (COUNT) / 2] : (SRC)[j];                    \
            (DST)[j] = keep + __shfl_xor(send, (D));                                        \
        }                                                                                   \
    }
    FT_HALVE(8, 32, cur, buf)    // gate bit 4 = lane bit 3
    FT_HALVE(4, 16, buf, cur)    // gate bit 3 = lane bit 2
    FT_HALVE(2, 8, cur, buf)     // gate bit 2 = lane bit 1
    FT_HALVE(1, 4, buf, cur)     // gate bit 1 = lane bit 0
    FT_HALVE(16, 2, cur, buf)    // gate bit 0 = lane bit 4
#undef FT_HALVE
    const float sum = buf[0];
    const int j = (((lane >> 3) & 1) << 4) | (((lane >> 2) & 1) << 3) | (((lane >> 1) & 1) << 2) | ((lane & 1) << 1) |
                  ((lane >> 4) & 1);
    const int hh = j >> 4, r = (j >> 2) & 3, st = j & 3;
    const int h = h0 + hh, n = st * HD + h;
    float sig = 0.f, x = 0.f;
    if (r < ROWS && h < HD) {
        const uint16_t g = f32_to_bf16_rne(sum * scale[n]);
        sig = sigmoid_triton(bf16_to_f32(g));
        x = bf16_to_f32(rn[r * stride_rn + n]);
    }
    // streams 0..3 of a (hh, r) sit at lanes b, b + 16, b + 1, b + 17 (b even, below 16)
    const float s1 = __shfl(sig, lane + 16), x1 = __shfl(x, lane + 16);
    const float s2 = __shfl(sig, lane + 1), x2 = __shfl(x, lane + 1);
    const float s3 = __shfl(sig, lane + 17), x3 = __shfl(x, lane + 17);
    if (st == 0 && r < ROWS && h < HD) {
        float acc = __builtin_fmaf(sig, x, 0.f);
        acc = __builtin_fmaf(s1, x1, acc);
        acc = __builtin_fmaf(s2, x2, acc);
        acc = __builtin_fmaf(s3, x3, acc);
        out[r * stride_om + h] = f32_to_bf16_rne(acc / static_cast<float>(HC));
    }
}

// 5-8 rows (two to four requests decoding): one hidden index a warp; the same sums for each row
template <int ROWS>
__global__ void __launch_bounds__(256) int8_hc_mix_exact_rows8_kernel(
    const uint16_t* __restrict__ a, const int8_t* __restrict__ w, const float* __restrict__ scale,
    const uint16_t* __restrict__ rn, uint16_t* __restrict__ out, int K, int HD, int64_t stride_wn,
    int64_t stride_am, int64_t stride_rn, int64_t stride_om) {
    constexpr int HC = 4;
    // two hidden indices a warp up to 4 rows, one past: 4 streams x RP rows x HH indices = 32 gates, one a lane
    constexpr int HH = ROWS <= 4 ? 2 : 1, RP = 8 / HH;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int h0 = (blockIdx.x * 8 + warp) * HH;
    if (h0 >= HD) return;
    const bool live = lane * 16 < K;
    float av[ROWS][16];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int i = 0; i < 16; ++i) av[r][i] = 0.f;
        if (live) {
            const uint4* ap = reinterpret_cast<const uint4*>(a + r * stride_am + lane * 16);
            uint4 lo = ap[0], hi = ap[1];
            const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                av[r][2 * i] = bf16_to_f32(static_cast<uint16_t>(words[i] & 0xFFFFu));
                av[r][2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(words[i] >> 16));
            }
        }
    }
    // v[j]: this lane's tree of gate j = (hh * RP + r) * 4 + stream (rows past ROWS stay 0)
    float v[32];
#pragma unroll
    for (int j = 0; j < 32; ++j) v[j] = 0.f;
#pragma unroll
    for (int hh = 0; hh < HH; ++hh) {
#pragma unroll
        for (int st = 0; st < HC; ++st) {
            const int n = st * HD + h0 + hh;
            if (live && h0 + hh < HD) {
                const uint4 wq = *reinterpret_cast<const uint4*>(w + n * stride_wn + lane * 16);
                const uint32_t ww[4] = {wq.x, wq.y, wq.z, wq.w};
                float wf[16];
#pragma unroll
                for (int i = 0; i < 16; ++i) wf[i] = int8_at(ww[i / 4], i % 4);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float p[16];
#pragma unroll
                    for (int i = 0; i < 16; ++i) p[i] = wf[i] * av[r][i];
                    v[(hh * RP + r) * 4 + st] = tree16(p);
                }
            }
        }
    }
    // the xor tree's pairs (lane distance 8, 4, 2, 1, then 16), halving the gates a lane holds at each level: at
    // distance d a lane keeps the half selected by its bit d and adds its partner's values of that half
    float* cur = v;
    float buf[16];
#define FT_HALVE(D, COUNT, SRC, DST)                                                       \
    {                                                                                       \
        const bool upper = (lane & (D)) != 0;                                               \
        _Pragma("unroll") for (int j = 0; j < (COUNT) / 2; ++j) {                           \
            const float send = upper ? (SRC)[j] : (SRC)[j + (COUNT) / 2];                    \
            const float keep = upper ? (SRC)[j + (COUNT) / 2] : (SRC)[j];                    \
            (DST)[j] = keep + __shfl_xor(send, (D));                                        \
        }                                                                                   \
    }
    FT_HALVE(8, 32, cur, buf)    // gate bit 4 = lane bit 3
    FT_HALVE(4, 16, buf, cur)    // gate bit 3 = lane bit 2
    FT_HALVE(2, 8, cur, buf)     // gate bit 2 = lane bit 1
    FT_HALVE(1, 4, buf, cur)     // gate bit 1 = lane bit 0
    FT_HALVE(16, 2, cur, buf)    // gate bit 0 = lane bit 4
#undef FT_HALVE
    const float sum = buf[0];
    const int j = (((lane >> 3) & 1) << 4) | (((lane >> 2) & 1) << 3) | (((lane >> 1) & 1) << 2) | ((lane & 1) << 1) |
                  ((lane >> 4) & 1);
    const int hh = j / (4 * RP), r = (j / 4) % RP, st = j & 3;
    const int h = h0 + hh, n = st * HD + h;
    float sig = 0.f, x = 0.f;
    if (r < ROWS && h < HD) {
        const uint16_t g = f32_to_bf16_rne(sum * scale[n]);
        sig = sigmoid_triton(bf16_to_f32(g));
        x = bf16_to_f32(rn[r * stride_rn + n]);
    }
    // streams 0..3 of a (hh, r) sit at lanes b, b + 16, b + 1, b + 17 (b even, below 16)
    const float s1 = __shfl(sig, lane + 16), x1 = __shfl(x, lane + 16);
    const float s2 = __shfl(sig, lane + 1), x2 = __shfl(x, lane + 1);
    const float s3 = __shfl(sig, lane + 17), x3 = __shfl(x, lane + 17);
    if (st == 0 && r < ROWS && h < HD) {
        float acc = __builtin_fmaf(sig, x, 0.f);
        acc = __builtin_fmaf(s1, x1, acc);
        acc = __builtin_fmaf(s2, x2, acc);
        acc = __builtin_fmaf(s3, x3, acc);
        out[r * stride_om + h] = f32_to_bf16_rne(acc / static_cast<float>(HC));
    }
}

}  // namespace

struct Int8GemvExact {
    static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView scale,
                    tvm::ffi::TensorView out) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto M = SymbolicSize{"rows"};
        auto N = SymbolicSize{"N"};
        auto K = SymbolicSize{"K"};
        auto xs = SymbolicSize{"x row stride"};
        auto ws = SymbolicSize{"w row stride"};
        auto os = SymbolicSize{"out row stride"};
        auto xd = SymbolicDType{};
        auto od = SymbolicDType{};
        TensorMatcher({M, K}).with_strides({xs, 1}).with_dtype(xd).with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({N, K}).with_strides({ws, 1}).with_dtype<int8_t>().with_device<kDLCUDA, kDLROCM>(device)
            .verify(w);
        TensorMatcher({N}).with_dtype<float>().with_device<kDLCUDA, kDLROCM>(device).verify(scale);
        TensorMatcher({M, N}).with_strides({os, 1}).with_dtype(od).with_device<kDLCUDA, kDLROCM>(device).verify(out);
        for (const auto dt : {xd.unwrap(), od.unwrap()}) {
            RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16 && dt.lanes == 1, "int8 exact GEMV: bfloat16 x and out");
        }
        const int m = static_cast<int>(M.unwrap()), n = static_cast<int>(N.unwrap()), k = static_cast<int>(K.unwrap());
        RuntimeCheck(m >= 1 && m <= 8, "int8 exact GEMV: 1-8 rows");
        RuntimeCheck(k % 16 == 0 && k <= 4096, "int8 exact GEMV: K a multiple of 16, at most 4096");
        auto launch = LaunchKernel((n + 1) / 2, kThreads, device.unwrap());
        auto* xa = static_cast<const uint16_t*>(x.data_ptr());
        auto* wa = static_cast<const int8_t*>(w.data_ptr());
        auto* sa = static_cast<const float*>(scale.data_ptr());
        auto* oa = static_cast<uint16_t*>(out.data_ptr());
        const int64_t sx = xs.unwrap(), sw = ws.unwrap(), so = os.unwrap();
        switch (m) {
            case 1: launch(int8_gemv_exact_kernel<1>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 2: launch(int8_gemv_exact_kernel<2>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 3: launch(int8_gemv_exact_kernel<3>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 4: launch(int8_gemv_exact_kernel<4>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 5: launch(int8_gemv_exact_rows8_kernel<5>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 6: launch(int8_gemv_exact_rows8_kernel<6>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 7: launch(int8_gemv_exact_rows8_kernel<7>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            default: launch(int8_gemv_exact_rows8_kernel<8>, xa, wa, sa, oa, n, k, sw, sx, so); break;
        }
    }
};

struct Int8GemvExactN4W8 {
    static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView scale,
                    tvm::ffi::TensorView out) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto M = SymbolicSize{"rows"};
        auto N = SymbolicSize{"N"};
        auto K = SymbolicSize{"K"};
        auto xs = SymbolicSize{"x row stride"};
        auto ws = SymbolicSize{"w row stride"};
        auto os = SymbolicSize{"out row stride"};
        auto xd = SymbolicDType{};
        auto od = SymbolicDType{};
        TensorMatcher({M, K}).with_strides({xs, 1}).with_dtype(xd).with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({N, K}).with_strides({ws, 1}).with_dtype<int8_t>().with_device<kDLCUDA, kDLROCM>(device)
            .verify(w);
        TensorMatcher({N}).with_dtype<float>().with_device<kDLCUDA, kDLROCM>(device).verify(scale);
        TensorMatcher({M, N}).with_strides({os, 1}).with_dtype(od).with_device<kDLCUDA, kDLROCM>(device).verify(out);
        for (const auto dt : {xd.unwrap(), od.unwrap()}) {
            RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16 && dt.lanes == 1, "int8 exact GEMV: bfloat16 x and out");
        }
        const int m = static_cast<int>(M.unwrap()), n = static_cast<int>(N.unwrap()), k = static_cast<int>(K.unwrap());
        RuntimeCheck(m >= 1 && m <= 8, "int8 exact GEMV: 1-8 rows");
        RuntimeCheck(k % 16 == 0 && k <= 4096, "int8 exact GEMV: K a multiple of 16, at most 4096");
        auto launch = LaunchKernel((n + 3) / 4, 32 * kWarps8, device.unwrap());
        auto* xa = static_cast<const uint16_t*>(x.data_ptr());
        auto* wa = static_cast<const int8_t*>(w.data_ptr());
        auto* sa = static_cast<const float*>(scale.data_ptr());
        auto* oa = static_cast<uint16_t*>(out.data_ptr());
        const int64_t sx = xs.unwrap(), sw = ws.unwrap(), so = os.unwrap();
        switch (m) {
            case 1: launch(int8_gemv_exact_n4w8_kernel<1>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 2: launch(int8_gemv_exact_n4w8_kernel<2>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 3: launch(int8_gemv_exact_n4w8_kernel<3>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 4: launch(int8_gemv_exact_n4w8_kernel<4>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 5: launch(int8_gemv_exact_n4w8_rows8_kernel<5>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 6: launch(int8_gemv_exact_n4w8_rows8_kernel<6>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            case 7: launch(int8_gemv_exact_n4w8_rows8_kernel<7>, xa, wa, sa, oa, n, k, sw, sx, so); break;
            default: launch(int8_gemv_exact_n4w8_rows8_kernel<8>, xa, wa, sa, oa, n, k, sw, sx, so); break;
        }
    }
};

struct Int8HcMixExact {
    static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView scale, tvm::ffi::TensorView rn,
                    tvm::ffi::TensorView out) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto M = SymbolicSize{"rows"};
        auto N = SymbolicSize{"N"};
        auto K = SymbolicSize{"K"};
        auto H = SymbolicSize{"HD"};
        auto xs = SymbolicSize{"x row stride"};
        auto ws = SymbolicSize{"w row stride"};
        auto rs = SymbolicSize{"rn row stride"};
        auto os = SymbolicSize{"out row stride"};
        auto xd = SymbolicDType{};
        auto rd = SymbolicDType{};
        auto od = SymbolicDType{};
        TensorMatcher({M, K}).with_strides({xs, 1}).with_dtype(xd).with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({N, K}).with_strides({ws, 1}).with_dtype<int8_t>().with_device<kDLCUDA, kDLROCM>(device)
            .verify(w);
        TensorMatcher({N}).with_dtype<float>().with_device<kDLCUDA, kDLROCM>(device).verify(scale);
        TensorMatcher({M, N}).with_strides({rs, 1}).with_dtype(rd).with_device<kDLCUDA, kDLROCM>(device).verify(rn);
        TensorMatcher({M, H}).with_strides({os, 1}).with_dtype(od).with_device<kDLCUDA, kDLROCM>(device).verify(out);
        for (const auto dt : {xd.unwrap(), rd.unwrap(), od.unwrap()}) {
            RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16 && dt.lanes == 1,
                         "int8 exact hc mix: bfloat16 x, rn, out");
        }
        const int m = static_cast<int>(M.unwrap()), n = static_cast<int>(N.unwrap()), k = static_cast<int>(K.unwrap());
        const int hd = static_cast<int>(H.unwrap());
        RuntimeCheck(m >= 1 && m <= 8, "int8 exact hc mix: 1-8 rows");
        RuntimeCheck(n == 4 * hd && k % 16 == 0 && k <= 512,
                     "int8 exact hc mix: 4 streams, K a multiple of 16 up to 512");
        auto launch = LaunchKernel(m <= 4 ? (hd + 15) / 16 : (hd + 7) / 8, 256, device.unwrap());
        auto* xa = static_cast<const uint16_t*>(x.data_ptr());
        auto* wa = static_cast<const int8_t*>(w.data_ptr());
        auto* sa = static_cast<const float*>(scale.data_ptr());
        auto* ra = static_cast<const uint16_t*>(rn.data_ptr());
        auto* oa = static_cast<uint16_t*>(out.data_ptr());
        const int64_t sx = xs.unwrap(), sw = ws.unwrap(), sr = rs.unwrap(), so = os.unwrap();
        switch (m) {
            case 1: launch(int8_hc_mix_exact_kernel<1>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
            case 2: launch(int8_hc_mix_exact_kernel<2>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
            case 3: launch(int8_hc_mix_exact_kernel<3>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
            case 4: launch(int8_hc_mix_exact_kernel<4>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
            case 5: launch(int8_hc_mix_exact_rows8_kernel<5>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
            case 6: launch(int8_hc_mix_exact_rows8_kernel<6>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
            case 7: launch(int8_hc_mix_exact_rows8_kernel<7>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
            default: launch(int8_hc_mix_exact_rows8_kernel<8>, xa, wa, sa, ra, oa, k, hd, sw, sx, sr, so); break;
        }
    }
};

struct Int8GemvExactK2048 {
    static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView scale,
                    tvm::ffi::TensorView out, int64_t hc, int64_t silu_cols) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto M = SymbolicSize{"rows"};
        auto N = SymbolicSize{"N"};
        auto K = SymbolicSize{"K"};
        auto xs = SymbolicSize{"x row stride"};
        auto ws = SymbolicSize{"w row stride"};
        auto os = SymbolicSize{"out row stride"};
        auto xd = SymbolicDType{};
        auto od = SymbolicDType{};
        TensorMatcher({M, K}).with_strides({xs, 1}).with_dtype(xd).with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({N, K}).with_strides({ws, 1}).with_dtype<int8_t>().with_device<kDLCUDA, kDLROCM>(device)
            .verify(w);
        TensorMatcher({N}).with_dtype<float>().with_device<kDLCUDA, kDLROCM>(device).verify(scale);
        TensorMatcher({M, N}).with_strides({os, 1}).with_dtype(od).with_device<kDLCUDA, kDLROCM>(device).verify(out);
        for (const auto dt : {xd.unwrap(), od.unwrap()}) {
            RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16 && dt.lanes == 1, "int8 exact GEMV: bfloat16 x and out");
        }
        const int m = static_cast<int>(M.unwrap()), n = static_cast<int>(N.unwrap()), k = static_cast<int>(K.unwrap());
        RuntimeCheck(m >= 1 && m <= 8, "int8 exact GEMV: 1-8 rows");
        RuntimeCheck(k % 8 == 0, "int8 exact GEMV: K a multiple of 8");
        auto launch = LaunchKernel(n, 32 * kWarps8, device.unwrap());
        auto* xa = static_cast<const uint16_t*>(x.data_ptr());
        auto* wa = static_cast<const int8_t*>(w.data_ptr());
        auto* sa = static_cast<const float*>(scale.data_ptr());
        auto* oa = static_cast<uint16_t*>(out.data_ptr());
        const int64_t sx = xs.unwrap(), sw = ws.unwrap(), so = os.unwrap();
        const int h = static_cast<int>(hc), sc = static_cast<int>(silu_cols);
        switch (m) {
            case 1: launch(int8_gemv_exact_k2048_kernel<1>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
            case 2: launch(int8_gemv_exact_k2048_kernel<2>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
            case 3: launch(int8_gemv_exact_k2048_kernel<3>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
            case 4: launch(int8_gemv_exact_k2048_kernel<4>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
            case 5: launch(int8_gemv_exact_k2048_kernel<5>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
            case 6: launch(int8_gemv_exact_k2048_kernel<6>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
            case 7: launch(int8_gemv_exact_k2048_kernel<7>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
            default: launch(int8_gemv_exact_k2048_kernel<8>, xa, wa, sa, oa, n, k, sw, sx, so, h, sc); break;
        }
    }
};
