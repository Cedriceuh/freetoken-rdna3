#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

// kernel/triton/router_shared.py's _router_shared_kernel at 1-8 rows as one HIP launch with the Triton kernel's sums
// (FREETOKEN_ROUTER_SHARED_HIP): a decode MoE layer's router GEMV (bf16, its (16, 128, 4) split-K tile) and its shared
// expert's gate | up GEMV with silu_and_mul (int8: NVFP4's (2, 512) K-blocked tile at one warp, EXL3's (1, 4096)
// one-pass tile at 4 warps, below). Products are exact in fp32 (bf16 x bf16, int8 x bf16), so only the order of the
// additions sets the bits; it is Triton's (gfx1100, read from the production kernels' compiled layouts):
//   router: a lane holds 8 adjacent columns of a row's 128-column block (threadsPerWarp [2, 16]): its 8 products as
//   a pairwise tree, the 16 lanes of the row as distances 8, 4, 2, 1; a split's 5 blocks added in order onto 0, the 4
//   splits added in order onto 0, rounded to bf16.
//   shared: a lane holds 16 adjacent columns of a 512-column block (threadsPerWarp [1, 32]): products accumulated
//   column by column over the 5 blocks onto 0, then the lane's 16 as a pairwise tree, the lanes as distances 8, 4, 2,
//   1, 16; gate and up rounded to bf16 after their scale, then silu(g) * u as activation.py computes it.
// Here a router row pair takes 4 warps (its splits, their partials added in split order in LDS: no atomic counter);
// the K-blocked gate | up one warp a weight row (at 3-4 rows two, two rows each; at 5-8 rows the first 4 rows and the
// others in blocks of their own, as the Triton kernel's two groups), the one-pass tile 4 warps a weight row; every
// weight load is issued before the sums.
namespace {

constexpr int kRouterRows = 512, kK = 2560, kSplit = 4, kSplitK = 640;

__device__ __forceinline__ float bf16f(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

// round to nearest even, a NaN as 0x7FFF: Triton's fp32 -> bf16 conversion on gfx11
__device__ __forceinline__ uint16_t to_bf16(float f) {
    if (f != f) return 0x7FFFu;
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

// lane ^ M within a 16-lane row (M = 1, 2, 4, 8) by DPP row_xmask, and lane ^ 16 by permlanex16: the moves Triton's
// reductions use (no LDS round trip as with ds_bpermute); the additions stay the same
template <int M>
__device__ __forceinline__ float dpp_xor(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + M, 0xF, 0xF, false));
}

__device__ __forceinline__ float xor16(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xFEDCBA98u, false, false));
}

// the lanes of a 32-lane sum as distances 8, 4, 2, 1, 16
__device__ __forceinline__ float lanes32_dpp(float t) {
    t += dpp_xor<8>(t);
    t += dpp_xor<4>(t);
    t += dpp_xor<2>(t);
    t += dpp_xor<1>(t);
    return t + xor16(t);
}

// the 16 lanes of a row as distances 8, 4, 2, 1
__device__ __forceinline__ float lanes16_dpp(float t) {
    t += dpp_xor<8>(t);
    t += dpp_xor<4>(t);
    t += dpp_xor<2>(t);
    return t + dpp_xor<1>(t);
}

__device__ __forceinline__ float tree8(const float (&v)[8]) {
    const float a0 = v[0] + v[1], a1 = v[2] + v[3], a2 = v[4] + v[5], a3 = v[6] + v[7];
    return (a0 + a1) + (a2 + a3);
}

__device__ __forceinline__ float tree16(const float (&v)[16]) {
    const float a0 = v[0] + v[1], a1 = v[2] + v[3], a2 = v[4] + v[5], a3 = v[6] + v[7];
    const float a4 = v[8] + v[9], a5 = v[10] + v[11], a6 = v[12] + v[13], a7 = v[14] + v[15];
    const float b0 = a0 + a1, b1 = a2 + a3, b2 = a4 + a5, b3 = a6 + a7;
    return (b0 + b1) + (b2 + b3);
}

__device__ __forceinline__ void unpack_bf16x8(const uint4 q, float (&v)[8]) {
    const uint32_t w[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = bf16f(static_cast<uint16_t>(w[i] & 0xFFFFu));
        v[2 * i + 1] = bf16f(static_cast<uint16_t>(w[i] >> 16));
    }
}

__device__ __forceinline__ float i8f(uint32_t word, int byte) {
    return static_cast<float>(static_cast<int8_t>((word >> (8 * byte)) & 0xFF));
}

// silu(g) * u as kernel/triton/activation.py: g / (1 + 2^(-g log2 e)) * u, the hardware exp2, a rounded division
__device__ __forceinline__ float silu_mul(float g, float u) {
    const float e = __builtin_amdgcn_exp2f(-g * 1.44269502162933349609375f);
    return g / (1.0f + e) * u;
}

// a router row pair at ROWS rows (Triton's tile is 16 rows; a row's sums do not depend on the rows beside it): warp %
// 4 = the split, warp / 4 = the row pair of the block, lanes 0-15 / 16-31 = its two rows; each split's partials into
// LDS, then added in split order onto 0
template <int ROWS, int W>
__device__ __forceinline__ void router_rows(const uint16_t* __restrict__ x, int64_t sx, const uint16_t* __restrict__ w,
                                            int64_t sw, uint16_t* __restrict__ logits, int64_t so, int b,
                                            float (&part)[W / 4][kSplit][ROWS][2]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int split = warp % 4, pair = warp / 4, half = lane / 16, g = lane % 16;
    const int row = (b * (W / 4) + pair) * 2 + half;
    const int k0 = split * kSplitK + g * 8;
    const uint16_t* wr = w + row * sw + k0;
    uint4 wq[kSplitK / 128];
#pragma unroll
    for (int kb = 0; kb < kSplitK / 128; ++kb) wq[kb] = *reinterpret_cast<const uint4*>(wr + kb * 128);
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    // 4 rows at a time (each row's sums in the same order): past 6 rows the loads of all of them hoisted spilled
#pragma unroll
    for (int r0 = 0; r0 < ROWS; r0 += 4) {
#pragma unroll
        for (int kb = 0; kb < kSplitK / 128; ++kb) {
            float wf[8];
            unpack_bf16x8(wq[kb], wf);
#pragma unroll
            for (int r = r0; r < (r0 + 4 < ROWS ? r0 + 4 : ROWS); ++r) {
                float av[8], p[8];
                unpack_bf16x8(*reinterpret_cast<const uint4*>(x + r * sx + k0 + kb * 128), av);
#pragma unroll
                for (int i = 0; i < 8; ++i) p[i] = wf[i] * av[i];
                acc[r] += lanes16_dpp(tree8(p));
            }
        }
        __builtin_amdgcn_sched_barrier(0);
    }
    if (g == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) part[pair][split][r][half] = acc[r];
    }
    __syncthreads();
    if (threadIdx.x < (W / 4) * 2 * ROWS) {
        const int pr = threadIdx.x / (2 * ROWS), r = (threadIdx.x / 2) % ROWS, hf = threadIdx.x % 2;
        float red = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < kSplit; ++s2) red += part[pr][s2][r][hf];
        logits[r * so + (b * (W / 4) + pr) * 2 + hf] = to_bf16(red);
    }
}

// the K-blocked gate | up tile (2, 512, one warp) at 1-2 rows: a warp = one weight row (warp % 2: gate, up) of column
// 2 b + warp / 2; products accumulated column by column over the 5 blocks onto 0, then the lane's 16 as a pairwise
// tree, the lanes as distances 8, 4, 2, 1, 16
template <int ROWS>
__device__ __forceinline__ void shared_kblocked(const uint16_t* __restrict__ x, int64_t sx,
                                                const int8_t* __restrict__ q, int64_t sq,
                                                const float* __restrict__ scale, uint16_t* __restrict__ act,
                                                int64_t sa, int D, int b, float (&red)[4][ROWS]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n = b * 2 + warp / 2, up = warp % 2;
    if (n < D) {
        const int8_t* wr = q + (up ? D + n : n) * sq + lane * 16;
        uint4 qw[kK / 512];
#pragma unroll
        for (int kb = 0; kb < kK / 512; ++kb) qw[kb] = *reinterpret_cast<const uint4*>(wr + kb * 512);
        float acc[ROWS][16];
#pragma unroll
        for (int r = 0; r < ROWS; ++r)
#pragma unroll
            for (int i = 0; i < 16; ++i) acc[r][i] = 0.f;
#pragma unroll
        for (int kb = 0; kb < kK / 512; ++kb) {
            const uint32_t w4[4] = {qw[kb].x, qw[kb].y, qw[kb].z, qw[kb].w};
            float wf[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) wf[i] = i8f(w4[i / 4], i % 4);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const uint4* ap = reinterpret_cast<const uint4*>(x + r * sx + kb * 512 + lane * 16);
                float lo[8], hi[8];
                unpack_bf16x8(ap[0], lo);
                unpack_bf16x8(ap[1], hi);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[r][i] += wf[i] * lo[i];
                    acc[r][8 + i] += wf[8 + i] * hi[i];
                }
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const float t = lanes32_dpp(tree16(acc[r]));
            if (lane == 0) red[warp][r] = t;
        }
    }
    __syncthreads();
    if (threadIdx.x < 2 * ROWS) {
        const int c = threadIdx.x / ROWS, r = threadIdx.x % ROWS, nn = b * 2 + c;
        if (nn < D) {
            const float gv = bf16f(to_bf16(red[2 * c][r] * scale[nn]));
            const float uv = bf16f(to_bf16(red[2 * c + 1][r] * scale[D + nn]));
            act[r * sa + nn] = to_bf16(silu_mul(gv, uv));
        }
    }
}

// rows r0 .. r0 + NR - 1 of one gate | up weight row (the K-blocked tile's sums), into red[r]: NR <= 2 keeps a warp's
// accumulators at 32 (4 rows in one warp spill)
template <int NR, int ROWS>
__device__ __forceinline__ void kblocked_rows(const uint16_t* __restrict__ x, int64_t sx, const int8_t* __restrict__ wr,
                                              int r0, float (&red)[ROWS]) {
    const int lane = threadIdx.x % 32;
    uint4 qw[kK / 512];
#pragma unroll
    for (int kb = 0; kb < kK / 512; ++kb) qw[kb] = *reinterpret_cast<const uint4*>(wr + kb * 512 + lane * 16);
    float acc[NR][16];
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[r][i] = 0.f;
#pragma unroll
    for (int kb = 0; kb < kK / 512; ++kb) {
        const uint32_t w4[4] = {qw[kb].x, qw[kb].y, qw[kb].z, qw[kb].w};
        float wf[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) wf[i] = i8f(w4[i / 4], i % 4);
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const uint4* ap = reinterpret_cast<const uint4*>(x + (r0 + r) * sx + kb * 512 + lane * 16);
            float lo[8], hi[8];
            unpack_bf16x8(ap[0], lo);
            unpack_bf16x8(ap[1], hi);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[r][i] += wf[i] * lo[i];
                acc[r][8 + i] += wf[8 + i] * hi[i];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float t = lanes32_dpp(tree16(acc[r]));
        if (lane == 0) red[r0 + r] = t;
    }
}

// the K-blocked tile at 3-4 rows: one column a block, warp / 2 = gate | up, warp % 2 = rows 0-1 | 2-3
template <int ROWS>
__device__ __forceinline__ void shared_kblocked_split(const uint16_t* __restrict__ x, int64_t sx,
                                                      const int8_t* __restrict__ q, int64_t sq,
                                                      const float* __restrict__ scale, uint16_t* __restrict__ act,
                                                      int64_t sa, int D, int n, float (&red)[2][ROWS]) {
    const int warp = threadIdx.x / 32;
    const int up = warp / 2;
    const int8_t* wr = q + (up ? D + n : n) * sq;
    if (warp % 2 == 0) {
        kblocked_rows<2, ROWS>(x, sx, wr, 0, red[up]);
    } else {
        kblocked_rows<ROWS - 2, ROWS>(x, sx, wr, 2, red[up]);
    }
    __syncthreads();
    if (threadIdx.x < ROWS) {
        const int r = threadIdx.x;
        const float gv = bf16f(to_bf16(red[0][r] * scale[n])), uv = bf16f(to_bf16(red[1][r] * scale[D + n]));
        act[r * sa + n] = to_bf16(silu_mul(gv, uv));
    }
}

// the one-pass gate | up tile (1, 4096, 4 warps: the EXL3 checkpoint's 768 rows a rank): warps 0-3 the gate row's
// tile, 4-7 the up row's. A lane holds 16 adjacent columns of each 2048-column half (k = 2048 h + 512 w + 16 l), its
// 32 products as a pairwise tree (a half past K adds +0), the lanes as distances 8, 4, 2, 1, 16, the tile's warps as
// (w0 + w2) + (w1 + w3)
template <int ROWS>
__device__ __forceinline__ void shared_one_pass(const uint16_t* __restrict__ x, int64_t sx,
                                                const int8_t* __restrict__ q, int64_t sq,
                                                const float* __restrict__ scale, uint16_t* __restrict__ act,
                                                int64_t sa, int D, int n, float (&red)[8][ROWS]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tw = warp % 4, up = warp / 4;
    const int8_t* wr = q + (up ? D + n : n) * sq;
    float hs[2][ROWS];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int k0 = h * 2048 + tw * 512 + lane * 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) hs[h][r] = 0.f;
        if (k0 >= kK) continue;  // K a multiple of 16: the lane's 16 columns all in or all out
        const uint4 qw = *reinterpret_cast<const uint4*>(wr + k0);
        const uint32_t w4[4] = {qw.x, qw.y, qw.z, qw.w};
        float wf[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) wf[i] = i8f(w4[i / 4], i % 4);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const uint4* ap = reinterpret_cast<const uint4*>(x + r * sx + k0);
            float lo[8], hi[8], p[16];
            unpack_bf16x8(ap[0], lo);
            unpack_bf16x8(ap[1], hi);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                p[i] = wf[i] * lo[i];
                p[8 + i] = wf[8 + i] * hi[i];
            }
            hs[h][r] = tree16(p);
            if (r % 4 == 3) __builtin_amdgcn_sched_barrier(0);  // 4 rows' loads in flight at most
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const float t = lanes32_dpp(hs[0][r] + hs[1][r]);
        if (lane == 0) red[warp][r] = t;
    }
    __syncthreads();
    if (threadIdx.x < ROWS) {
        const int r = threadIdx.x;
        const float tg = (red[0][r] + red[2][r]) + (red[1][r] + red[3][r]);
        const float tu = (red[4][r] + red[6][r]) + (red[5][r] + red[7][r]);
        const float gv = bf16f(to_bf16(tg * scale[n])), uv = bf16f(to_bf16(tu * scale[D + n]));
        act[r * sa + n] = to_bf16(silu_mul(gv, uv));
    }
}

// the shared expert's gate of one row, sigmoid(sum(x * w)), as kernel/triton/moe_shared_gate.py's
// _gate_sum_mul_add_kernel computes it at 16 warps (hidden 2560 in a 4096 block): 512 threads of 8 adjacent columns
// (past 2560: +0), each 8 products as a pairwise tree, the 32 lanes as distances 8, 4, 2, 1, 16, the 16 warps' sums as
// distances 8, 4, 2, 1, then 1 / (1 + exp2(x * -log2 e)). This block's W warps take 16 / W of those 16 warps each.
template <int W>
__device__ __forceinline__ void gate_row(const uint16_t* __restrict__ xr, const uint16_t* __restrict__ gw,
                                         float* __restrict__ gate, float (&ws)[16]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
    for (int j = 0; j < 16 / W; ++j) {
        const int vw = warp + j * W, c0 = (vw * 32 + lane) * 8;
        float p[8];
        if (c0 < kK) {
            float hv[8], wv[8];
            unpack_bf16x8(*reinterpret_cast<const uint4*>(xr + c0), hv);
            unpack_bf16x8(*reinterpret_cast<const uint4*>(gw + c0), wv);
#pragma unroll
            for (int i = 0; i < 8; ++i) p[i] = hv[i] * wv[i];
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) p[i] = 0.f;
        }
        const float t = lanes32_dpp(tree8(p));
        if (lane == 0) ws[vw] = t;
    }
    __syncthreads();
    if (warp == 0) {
        float v = lane < 16 ? ws[lane] : 0.f;
        v += dpp_xor<8>(v);
        v += dpp_xor<4>(v);
        v += dpp_xor<2>(v);
        v += dpp_xor<1>(v);
        if (lane == 0) {
            const float e = __builtin_exp2f(__fmul_rn(v, -1.44269502162933349609375f));
            *gate = __fdiv_rn(1.0f, __fadd_rn(e, 1.0f));
        }
    }
}

// W = 4 warps (the K-blocked gate | up: 2 columns a block) or 8 (the one-pass one: 1 column a block); the first
// 512 / (W / 2) blocks run the router's row pairs
template <int ROWS, int W>
__global__ void __launch_bounds__(32 * W) router_shared_exact_kernel(
    const uint16_t* __restrict__ x, int64_t sx, const uint16_t* __restrict__ rw, int64_t srw,
    uint16_t* __restrict__ logits, int64_t sl, const int8_t* __restrict__ q, int64_t sq,
    const float* __restrict__ scale, uint16_t* __restrict__ act, int64_t sa, int D, const uint16_t* __restrict__ gw,
    float* __restrict__ gate, int gate_from) {
    constexpr int kRouterBlocks = kRouterRows / (W / 2);
    const int b = blockIdx.x;
    if (gw != nullptr && b >= gate_from) {
        __shared__ float ws[16];
        gate_row<W>(x + (b - gate_from) * sx, gw, gate + (b - gate_from), ws);
        return;
    }
    if (b < kRouterBlocks) {
        __shared__ float part[W / 4][kSplit][ROWS][2];
        router_rows<ROWS, W>(x, sx, rw, srw, logits, sl, b, part);
        return;
    }
    if constexpr (W == 8) {
        __shared__ float red[W][ROWS];
        shared_one_pass<ROWS>(x, sx, q, sq, scale, act, sa, D, b - kRouterBlocks, red);
    } else if constexpr (ROWS > 4) {
        // rows 0-3 one column a block (D blocks), then rows 4 .. ROWS - 1 as they would run alone
        constexpr int R2 = ROWS - 4;
        const int bb = b - kRouterBlocks;
        if (bb < D) {
            __shared__ float red4[2][4];
            shared_kblocked_split<4>(x, sx, q, sq, scale, act, sa, D, bb, red4);
        } else if constexpr (R2 > 2) {
            __shared__ float red2[2][R2];
            shared_kblocked_split<R2>(x + 4 * sx, sx, q, sq, scale, act + 4 * sa, sa, D, bb - D, red2);
        } else {
            __shared__ float red1[W][R2];
            shared_kblocked<R2>(x + 4 * sx, sx, q, sq, scale, act + 4 * sa, sa, D, bb - D, red1);
        }
    } else if constexpr (ROWS > 2) {
        __shared__ float red2[2][ROWS];
        shared_kblocked_split<ROWS>(x, sx, q, sq, scale, act, sa, D, b - kRouterBlocks, red2);
    } else {
        __shared__ float red[W][ROWS];
        shared_kblocked<ROWS>(x, sx, q, sq, scale, act, sa, D, b - kRouterBlocks, red);
    }
}

}  // namespace

struct RouterSharedExact {
    static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView rw, tvm::ffi::TensorView logits,
                    tvm::ffi::TensorView q, tvm::ffi::TensorView scale, tvm::ffi::TensorView act, int64_t one_pass,
                    tvm::ffi::TensorView gw, tvm::ffi::TensorView gate, int64_t with_gate) {
        using namespace host;
        auto device = SymbolicDevice{};
        auto M = SymbolicSize{"rows"};
        auto RN = SymbolicSize{"router rows"};
        auto K = SymbolicSize{"K"};
        auto N2 = SymbolicSize{"gate | up rows"};
        auto D = SymbolicSize{"D"};
        auto xs = SymbolicSize{"x row stride"};
        auto rws = SymbolicSize{"router row stride"};
        auto ls = SymbolicSize{"logits row stride"};
        auto qs = SymbolicSize{"gate | up row stride"};
        auto as = SymbolicSize{"act row stride"};
        auto bf = SymbolicDType{};
        TensorMatcher({M, K}).with_strides({xs, 1}).with_dtype(bf).with_device<kDLCUDA, kDLROCM>(device).verify(x);
        TensorMatcher({RN, K}).with_strides({rws, 1}).with_dtype(bf).with_device<kDLCUDA, kDLROCM>(device).verify(rw);
        TensorMatcher({M, RN}).with_strides({ls, 1}).with_dtype(bf).with_device<kDLCUDA, kDLROCM>(device)
            .verify(logits);
        TensorMatcher({N2, K}).with_strides({qs, 1}).with_dtype<int8_t>().with_device<kDLCUDA, kDLROCM>(device)
            .verify(q);
        TensorMatcher({N2}).with_dtype<float>().with_device<kDLCUDA, kDLROCM>(device).verify(scale);
        TensorMatcher({M, D}).with_strides({as, 1}).with_dtype(bf).with_device<kDLCUDA, kDLROCM>(device).verify(act);
        const auto dt = bf.unwrap();
        RuntimeCheck(dt.code == kDLBfloat && dt.bits == 16 && dt.lanes == 1, "router | shared exact: bfloat16");
        const int m = static_cast<int>(M.unwrap()), d = static_cast<int>(D.unwrap());
        RuntimeCheck(m >= 1 && m <= 8, "router | shared exact: 1-8 rows");
        RuntimeCheck(RN.unwrap() == kRouterRows && K.unwrap() == kK && N2.unwrap() == 2 * D.unwrap(),
                     "router | shared exact: a 512 x 2560 router, a 2D x 2560 gate | up");
        // with_gate: one more block a row computes the shared expert's gate (gw [K] bf16 -> gate [rows] fp32)
        const uint16_t* gwa = nullptr;
        float* ga = nullptr;
        if (with_gate) {
            TensorMatcher({K}).with_dtype(bf).with_device<kDLCUDA, kDLROCM>(device).verify(gw);
            TensorMatcher({M}).with_dtype<float>().with_device<kDLCUDA, kDLROCM>(device).verify(gate);
            gwa = static_cast<const uint16_t*>(gw.data_ptr());
            ga = static_cast<float*>(gate.data_ptr());
        }
        // the K-blocked gate | up's blocks: one column a block at 3-4 rows, two at 1-2; past 4 rows both groups'
        const auto kb_blocks = [d](int rows) { return rows > 2 ? d : (d + 1) / 2; };
        const int from4 = kRouterRows / 2 + (m > 4 ? d + kb_blocks(m - 4) : kb_blocks(m)), from8 = kRouterRows / 4 + d;
        auto launch4 = LaunchKernel(from4 + (with_gate ? m : 0), 128, device.unwrap());
        auto launch8 = LaunchKernel(from8 + (with_gate ? m : 0), 256, device.unwrap());
        auto* xa = static_cast<const uint16_t*>(x.data_ptr());
        auto* ra = static_cast<const uint16_t*>(rw.data_ptr());
        auto* la = static_cast<uint16_t*>(logits.data_ptr());
        auto* qa = static_cast<const int8_t*>(q.data_ptr());
        auto* sa = static_cast<const float*>(scale.data_ptr());
        auto* aa = static_cast<uint16_t*>(act.data_ptr());
        const int64_t sx = xs.unwrap(), sr = rws.unwrap(), sl = ls.unwrap(), sq = qs.unwrap(), so = as.unwrap();
#define RS_CASE(R)                                                                                          \
    (one_pass                                                                                                   \
         ? launch8(router_shared_exact_kernel<R, 8>, xa, sx, ra, sr, la, sl, qa, sq, sa, aa, so, d, gwa, ga, from8) \
         : launch4(router_shared_exact_kernel<R, 4>, xa, sx, ra, sr, la, sl, qa, sq, sa, aa, so, d, gwa, ga, from4))
        switch (m) {
            case 1: RS_CASE(1); break;
            case 2: RS_CASE(2); break;
            case 3: RS_CASE(3); break;
            case 4: RS_CASE(4); break;
            case 5: RS_CASE(5); break;
            case 6: RS_CASE(6); break;
            case 7: RS_CASE(7); break;
            default: RS_CASE(8); break;
        }
#undef RS_CASE
    }
};
