// EXL3 experts on the CPU (exllamav3 trellis, mul1 codebook), behind MoeFormat as a format that runs its tasks whole:
// rotations in and out of the Hadamard domain bracket the two GEMV passes, so it brings its own phases (run_task).
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../format.h"

namespace freetoken::cpu_moe {
namespace {

inline float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t man = h & 0x3FFu;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) {
        man <<= 1;
        --exp;
      }
      man &= 0x3FFu;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1Fu) {
    f = sign | 0x7F800000u | (man << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

// =====================================================================================
// EXL3 experts (exllamav3 trellis, mul1 codebook, 2..4 bits per weight) on the host banks of
// layers/quantization/moe/exl3.py: the GPU kernels' math (kernel/triton/exl3.py). W = diag(suh) H Wq H diag(svh),
// H the normalized 128-point Hadamard per block; Wq in 16x16 tiles of 256 trellis states, a tail-biting MSB-first
// ring of 8K uint32 words per tile. mul1(s) = fp16(a * (1024 + bytesum(s * 0x83DCD12D)) + c): the GEMV accumulates
// x * bytesum and adds the affine part once per column, so a weight only misses its own fp16 rounding (relative
// error ~3e-4 on an expert's output against the reference codec).
struct Exl3Group {
  int w, wn, o;  // the group's 64-bit window: words w, wn and the bit offset of lane 0's state
};

#if CPU_MOE_X86
__attribute__((target("avx2,fma"))) static inline __m256 exl3_bytesum8(__m256i s) {
  const __m256i x = _mm256_mullo_epi32(s, _mm256_set1_epi32((int)0x83DCD12Du));
  const __m256i q = _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_set1_epi8(1)), _mm256_set1_epi16(1));
  return _mm256_cvtepi32_ps(q);
}

template <int BITS>
constexpr std::array<Exl3Group, 32> exl3_make_plan() {
  std::array<Exl3Group, 32> p{};
  for (int t = 0; t < 32; ++t) {
    const int s0 = (8 * t * BITS + BITS - 16 + 256 * BITS) % (256 * BITS);  // first bit of lane 0's window
    p[t] = Exl3Group{s0 >> 5, ((s0 >> 5) + 1) % (8 * BITS), s0 & 31};
  }
  return p;
}

// the 8 states of group T of a tile as bytesums, lanes in the order (0, 4, 1, 5, 2, 6, 3, 7) of j - 8T: one 64-bit
// window (K <= 4), its shifts compile-time constants
template <int BITS, int T>
__attribute__((target("avx2,fma"))) static inline __m256 exl3_group(const uint32_t* g) {
  constexpr Exl3Group p = exl3_make_plan<BITS>()[T];
  static_assert(p.o + 7 * BITS + 16 <= 64, "EXL3 state group does not fit 64 bits");
  const uint64_t w64 = ((uint64_t)g[p.w] << 32) | g[p.wn];
  const __m256i v = _mm256_set1_epi64x((long long)w64);
  const __m256i lo = _mm256_srlv_epi64(v, _mm256_setr_epi64x(48 - p.o, 48 - BITS - p.o, 48 - 2 * BITS - p.o,
                                                               48 - 3 * BITS - p.o));
  const __m256i hi = _mm256_srlv_epi64(v, _mm256_setr_epi64x(48 - 4 * BITS - p.o, 48 - 5 * BITS - p.o,
                                                               48 - 6 * BITS - p.o, 48 - 7 * BITS - p.o));
  const __m256i r = _mm256_and_si256(_mm256_blend_epi32(lo, _mm256_slli_epi64(hi, 32), 0xAA),
                                     _mm256_set1_epi32(0xFFFF));
  return exl3_bytesum8(r);
}

template <int BITS, int C0>
__attribute__((target("avx2,fma"))) static inline void exl3_col(const uint32_t* g, const __m256* xq, __m256* acc) {
  acc[C0] = _mm256_fmadd_ps(exl3_group<BITS, 4 * C0>(g), xq[0], acc[C0]);
  acc[C0] = _mm256_fmadd_ps(exl3_group<BITS, 4 * C0 + 1>(g), xq[1], acc[C0]);
  acc[C0] = _mm256_fmadd_ps(exl3_group<BITS, 4 * C0 + 2>(g), xq[2], acc[C0]);
  acc[C0] = _mm256_fmadd_ps(exl3_group<BITS, 4 * C0 + 3>(g), xq[3], acc[C0]);
}

// columns [16b, 16b + 16) of S = x @ bytesum(Wq) over K inputs; tiles [K/16][N/16][16 * BITS] int16. The trellis
// comes cold from host RAM in serving (a column of tiles is strided by N/16 tiles): prefetch kExl3Prefetch rows ahead
static constexpr int kExl3Prefetch = 6;

template <int BITS>
__attribute__((target("avx2,fma"))) static void exl3_gemv_cols_t(const int16_t* tr, int K, int N, int b,
                                                                const float* xpat, float* S16, int pf_tiles) {
  __m256 acc[8];
  for (int c = 0; c < 8; ++c) acc[c] = _mm256_setzero_ps();
  const int NB = N / 16, rows = K / 16;
  constexpr int tile = 16 * BITS;
  for (int a = 0; a < rows; ++a) {
    if (a + kExl3Prefetch < rows) {
      const char* nx = reinterpret_cast<const char*>(tr + ((size_t)(a + kExl3Prefetch) * NB + b) * tile);
      for (int off = 0; off < pf_tiles * tile * 2; off += 64) _mm_prefetch(nx + off, _MM_HINT_T0);
      _mm_prefetch(nx + pf_tiles * tile * 2 - 1, _MM_HINT_T0);
    }
    const uint32_t* g = reinterpret_cast<const uint32_t*>(tr + ((size_t)a * NB + b) * tile);
    const float* xp = xpat + a * 32;
    const __m256 xq[4] = {_mm256_loadu_ps(xp), _mm256_loadu_ps(xp + 8), _mm256_loadu_ps(xp + 16),
                          _mm256_loadu_ps(xp + 24)};
    exl3_col<BITS, 0>(g, xq, acc);
    exl3_col<BITS, 1>(g, xq, acc);
    exl3_col<BITS, 2>(g, xq, acc);
    exl3_col<BITS, 3>(g, xq, acc);
    exl3_col<BITS, 4>(g, xq, acc);
    exl3_col<BITS, 5>(g, xq, acc);
    exl3_col<BITS, 6>(g, xq, acc);
    exl3_col<BITS, 7>(g, xq, acc);
  }
  for (int c0 = 0; c0 < 8; ++c0) {
    alignas(32) float l[8];
    _mm256_store_ps(l, acc[c0]);
    S16[c0] = l[0] + l[2] + l[4] + l[6];
    S16[c0 + 8] = l[1] + l[3] + l[5] + l[7];
  }
}

// The same columns for nt <= 4 rows routed to the same expert: each tile row is decoded once into dec (L1) and every
// row accumulates from it, so a second row costs its FMAs instead of a second decode.
template <int BITS, int... T>
__attribute__((target("avx2,fma"))) static inline void exl3_dec_row(const uint32_t* g, float (*dec)[8],
                                                                   std::integer_sequence<int, T...>) {
  (_mm256_store_ps(dec[T], exl3_group<BITS, T>(g)), ...);
}

template <int BITS>
__attribute__((target("avx2,fma"))) static void exl3_gemv_cols_mt_t(const int16_t* tr, int K, int N, int b, int nt,
                                                                   const float* const* xpats, float* const* S16s,
                                                                   int pf_tiles) {
  alignas(32) float accm[4][8][8] = {};
  alignas(32) float dec[32][8];
  const int NB = N / 16, rows = K / 16;
  constexpr int tile = 16 * BITS;
  for (int a = 0; a < rows; ++a) {
    if (a + kExl3Prefetch < rows) {
      const char* nx = reinterpret_cast<const char*>(tr + ((size_t)(a + kExl3Prefetch) * NB + b) * tile);
      for (int off = 0; off < pf_tiles * tile * 2; off += 64) _mm_prefetch(nx + off, _MM_HINT_T0);
      _mm_prefetch(nx + pf_tiles * tile * 2 - 1, _MM_HINT_T0);
    }
    exl3_dec_row<BITS>(reinterpret_cast<const uint32_t*>(tr + ((size_t)a * NB + b) * tile), dec,
                       std::make_integer_sequence<int, 32>{});
    for (int t = 0; t < nt; ++t) {
      const float* xp = xpats[t] + a * 32;
      const __m256 x0 = _mm256_loadu_ps(xp), x1 = _mm256_loadu_ps(xp + 8), x2 = _mm256_loadu_ps(xp + 16),
                   x3 = _mm256_loadu_ps(xp + 24);
      for (int c0 = 0; c0 < 8; ++c0) {
        __m256 acc = _mm256_load_ps(accm[t][c0]);
        acc = _mm256_fmadd_ps(_mm256_load_ps(dec[4 * c0]), x0, acc);
        acc = _mm256_fmadd_ps(_mm256_load_ps(dec[4 * c0 + 1]), x1, acc);
        acc = _mm256_fmadd_ps(_mm256_load_ps(dec[4 * c0 + 2]), x2, acc);
        acc = _mm256_fmadd_ps(_mm256_load_ps(dec[4 * c0 + 3]), x3, acc);
        _mm256_store_ps(accm[t][c0], acc);
      }
    }
  }
  for (int t = 0; t < nt; ++t)
    for (int c0 = 0; c0 < 8; ++c0) {
      const float* l = accm[t][c0];
      S16s[t][c0] = l[0] + l[2] + l[4] + l[6];
      S16s[t][c0 + 8] = l[1] + l[3] + l[5] + l[7];
    }
}

static void exl3_gemv_cols_mt(const int16_t* tr, int K, int N, int b, int bits, int nt, const float* const* xpats,
                              float* const* S16s, int pf_tiles) {
  switch (bits) {
    case 2: exl3_gemv_cols_mt_t<2>(tr, K, N, b, nt, xpats, S16s, pf_tiles); break;
    case 3: exl3_gemv_cols_mt_t<3>(tr, K, N, b, nt, xpats, S16s, pf_tiles); break;
    default: exl3_gemv_cols_mt_t<4>(tr, K, N, b, nt, xpats, S16s, pf_tiles); break;
  }
}

static void exl3_gemv_cols(const int16_t* tr, int K, int N, int b, int bits, const float* xpat, float* S16,
                           int pf_tiles = 1) {
  switch (bits) {
    case 2: exl3_gemv_cols_t<2>(tr, K, N, b, xpat, S16, pf_tiles); break;
    case 3: exl3_gemv_cols_t<3>(tr, K, N, b, xpat, S16, pf_tiles); break;
    default: exl3_gemv_cols_t<4>(tr, K, N, b, xpat, S16, pf_tiles); break;
  }
}
#endif

// a tile row's inputs in exl3_group's lane order (0, 4, 1, 5, 2, 6, 3, 7) for its 4 row groups q: rows (r0, r0,
// r0+1, r0+1, r0+8, r0+8, r0+9, r0+9), r0 = 2q (tensor_core_perm: lanes 0-3 of a group are column c0, lanes 4-7
// column c0 + 8)
static void exl3_xpat(const float* x, int K, float* xpat) {
  for (int a = 0; a < K / 16; ++a)
    for (int q = 0; q < 4; ++q) {
      const float* xr = x + 16 * a;
      float* d = xpat + (a * 4 + q) * 8;
      const int r0 = 2 * q;
      d[0] = d[1] = xr[r0];
      d[2] = d[3] = xr[r0 + 1];
      d[4] = d[5] = xr[r0 + 8];
      d[6] = d[7] = xr[r0 + 9];
    }
}

#if CPU_MOE_X86
#define EXL3_TGT __attribute__((target("avx2,fma,f16c")))
// in-place normalized 128-point Hadamard: stages 8..64 between vectors, 1..4 inside them (the stages commute)
EXL3_TGT static void exl3_fwht128_avx2(float* p) {
  __m256 v[16];
  for (int i = 0; i < 16; ++i) v[i] = _mm256_loadu_ps(p + 8 * i);
  for (int i = 0; i < 16; ++i) {
    __m256 sw = _mm256_permute2f128_ps(v[i], v[i], 0x01);
    v[i] = _mm256_blend_ps(_mm256_add_ps(v[i], sw), _mm256_sub_ps(sw, v[i]), 0xF0);
    sw = _mm256_permute_ps(v[i], _MM_SHUFFLE(1, 0, 3, 2));
    v[i] = _mm256_blend_ps(_mm256_add_ps(v[i], sw), _mm256_sub_ps(sw, v[i]), 0xCC);
    sw = _mm256_permute_ps(v[i], _MM_SHUFFLE(2, 3, 0, 1));
    v[i] = _mm256_blend_ps(_mm256_add_ps(v[i], sw), _mm256_sub_ps(sw, v[i]), 0xAA);
  }
  for (int h = 1; h < 16; h <<= 1)
    for (int i = 0; i < 16; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        const __m256 a = v[j], b = v[j + h];
        v[j] = _mm256_add_ps(a, b);
        v[j + h] = _mm256_sub_ps(a, b);
      }
  const __m256 sc = _mm256_set1_ps(0.08838834764831845f);
  for (int i = 0; i < 16; ++i) _mm256_storeu_ps(p + 8 * i, _mm256_mul_ps(v[i], sc));
}

EXL3_TGT static inline __m256 exl3_ld_f16(const uint16_t* h) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(h)));
}

EXL3_TGT static inline __m256 exl3_ld_bf16(const uint16_t* b) {
  const __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b)));
  return _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
}

EXL3_TGT static inline float exl3_hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// exp for silu: 2^(x log2 e) split into an integer power and a degree-6 polynomial on [-0.5, 0.5] (rel err ~2e-7)
EXL3_TGT static inline __m256 exl3_exp(__m256 x) {
  x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.f)), _mm256_set1_ps(-88.f));
  const __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
  const __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  const __m256 f = _mm256_mul_ps(_mm256_sub_ps(t, n), _mm256_set1_ps(0.6931471805599453f));
  __m256 p = _mm256_set1_ps(1.f / 720.f);
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.f / 120.f));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.f / 24.f));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.f / 6.f));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.5f));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.f));
  p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.f));
  const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

// v[0..n) = x * scale (bf16 x, fp16 scale), rotated per 128-block; returns its sum
EXL3_TGT static float exl3_rotate_in(const uint16_t* x_bf16, const uint16_t* scale, int n, float* v) {
  __m256 acc = _mm256_setzero_ps();
  for (int k = 0; k < n; k += 8)
    _mm256_storeu_ps(v + k, _mm256_mul_ps(exl3_ld_bf16(x_bf16 + k), exl3_ld_f16(scale + k)));
  for (int blk = 0; blk < n / 128; ++blk) exl3_fwht128_avx2(v + blk * 128);
  for (int k = 0; k < n; k += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(v + k));
  return exl3_hsum(acc);
}

// a GEMV's raw sums to the output's rotated-out values: v = H(a * v + base) * svh (fp16), in place
EXL3_TGT static void exl3_rotate_out(float* v, int n, float a, float base, const uint16_t* svh) {
  const __m256 va = _mm256_set1_ps(a), vb = _mm256_set1_ps(base);
  for (int k = 0; k < n; k += 8) _mm256_storeu_ps(v + k, _mm256_fmadd_ps(va, _mm256_loadu_ps(v + k), vb));
  for (int blk = 0; blk < n / 128; ++blk) exl3_fwht128_avx2(v + blk * 128);
  for (int k = 0; k < n; k += 8) _mm256_storeu_ps(v + k, _mm256_mul_ps(_mm256_loadu_ps(v + k), exl3_ld_f16(svh + k)));
}

// m = silu(g) * u * suh (fp16), rotated per 128-block; returns its sum
EXL3_TGT static float exl3_mid(const float* g, const float* u, const uint16_t* suh, int n, float* m) {
  const __m256 one = _mm256_set1_ps(1.f);
  for (int k = 0; k < n; k += 8) {
    const __m256 gv = _mm256_loadu_ps(g + k);
    const __m256 sg = _mm256_div_ps(gv, _mm256_add_ps(one, exl3_exp(_mm256_sub_ps(_mm256_setzero_ps(), gv))));
    _mm256_storeu_ps(m + k, _mm256_mul_ps(_mm256_mul_ps(sg, _mm256_loadu_ps(u + k)), exl3_ld_f16(suh + k)));
  }
  for (int blk = 0; blk < n / 128; ++blk) exl3_fwht128_avx2(m + blk * 128);
  __m256 acc = _mm256_setzero_ps();
  for (int k = 0; k < n; k += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(m + k));
  return exl3_hsum(acc);
}

// acc[0..128) += v * w (one route's rotated-out block)
EXL3_TGT static inline void exl3_axpy128(float* acc, const float* v, float w) {
  const __m256 vw = _mm256_set1_ps(w);
  for (int k = 0; k < 128; k += 8)
    _mm256_storeu_ps(acc + k, _mm256_fmadd_ps(_mm256_loadu_ps(v + k), vw, _mm256_loadu_ps(acc + k)));
}
#endif

class Exl3Format final : public MoeFormat {
 public:
  Exl3Format(const MoeParams& p, const FormatArgs& a) : MoeFormat(p) {
#if CPU_MOE_X86
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma") || !__builtin_cpu_supports("f16c"))
      throw std::runtime_error("EXL3 CPU MoE needs AVX2 + FMA + F16C");
#else
    throw std::runtime_error("EXL3 CPU MoE is x86 (AVX2) only");
#endif
    bits_ = static_cast<int>(a.option("bits"));
    if (bits_ < 2 || bits_ > 4) throw std::runtime_error("EXL3 CPU MoE supports 2..4 bits per weight");
    if (p.H % 128 || p.I % 128) throw std::runtime_error("EXL3 CPU MoE needs H and I in 128-blocks");
    // Banks: gate_up trellis [E, H/16, 2I/16, 16K], gate_up_suh fp16 [E, 2, H], gate_up_svh fp16 [E, 2I],
    // down trellis [E, I/16, H/16, 16K], down_suh [E, I], down_svh [E, H].
    gate_up_ = a.bank("gate_up");
    gu_suh_ = a.bank("gate_up_suh");
    gu_svh_ = a.bank("gate_up_svh");
    down_ = a.bank("down");
    dn_suh_ = a.bank("down_suh");
    dn_svh_ = a.bank("down_svh");
    a_ = fp16_to_f32(0x1EEE);
    c_ = fp16_to_f32(0xC931);
    // the per-(token, route) scratch: x patterns and sums of the rotated inputs, gate|up products, down inputs,
    // down products
    const size_t R = (size_t)p.max_tokens * p.top_k;
    xpat_.assign(R * 2 * p.H * 2, 0.f);
    xsum_.assign(R * 2, 0.f);
    s1_.assign(R * 2 * p.I, 0.f);
    xpd_.assign(R * p.I * 2, 0.f);
    xsumd_.assign(R, 0.f);
    s2_.assign(R * p.H, 0.f);
  }

  std::string isa_name() const override { return "avx2+fma+f16c(exl3)"; }

  void gate_up(int, int, int, int, int, const bf16_t*, float, bf16_t*) const override {
    throw std::logic_error("EXL3 CPU MoE runs its tasks whole (run_task)");
  }
  void down(int, int, int, const DownRoute*, int, float*) const override {
    throw std::logic_error("EXL3 CPU MoE runs its tasks whole (run_task)");
  }

  bool runs_task() const override { return true; }

  void prepare_task(const TaskView& t) override {
    const int64_t R = (int64_t)t.num_tokens * p.top_k;
    const int64_t G = (int64_t)t.grp_start->size() - 1;  // the routes of each distinct expert, decoded once for all
    // tile columns per task: a block of adjacent columns, whose row segments lie together in RAM (each half of gate|up
    // and down must split evenly)
    const int nb1h = p.I / 16, nb2 = p.H / 16;
    cb1_ = cb2_ = 1;
    for (int c = cols_; c > 1; --c)
      if (nb1h % c == 0) { cb1_ = c; break; }
    for (int c = cols_; c > 1; --c)
      if (nb2 % c == 0) { cb2_ = c; break; }
    a_total_ = R;                         // rotate in, both projections of a route
    p1_total_ = G * (2 * nb1h / cb1_);    // gate|up column blocks
    mid_total_ = R;                       // mid
    p2_total_ = G * (nb2 / cb2_);         // down column blocks
    e_total_ = (int64_t)t.num_tokens * (p.H / 128);  // rotate out + routed sum, per 128-block
    a_next_.store(0, std::memory_order_relaxed);
    p1_next_.store(0, std::memory_order_relaxed);
    mid_next_.store(0, std::memory_order_relaxed);
    p2_next_.store(0, std::memory_order_relaxed);
    e_next_.store(0, std::memory_order_relaxed);
  }

  void run_task(const TaskView& t, const std::function<void()>& barrier) override {
#if CPU_MOE_X86
    const int top_k = p.top_k, H = p.H, I = p.I;
    const int L = t.layer, NB1 = 2 * I / 16, NB2 = H / 16, bits = bits_;
    const int16_t* gu = static_cast<const int16_t*>(tbl_at(gate_up_, L));
    const uint16_t* gu_suh = static_cast<const uint16_t*>(tbl_at(gu_suh_, L));
    const uint16_t* gu_svh = static_cast<const uint16_t*>(tbl_at(gu_svh_, L));
    const int16_t* dn = static_cast<const int16_t*>(tbl_at(down_, L));
    const uint16_t* dn_suh = static_cast<const uint16_t*>(tbl_at(dn_suh_, L));
    const uint16_t* dn_svh = static_cast<const uint16_t*>(tbl_at(dn_svh_, L));
    const size_t gu_e = (size_t)(H / 16) * NB1 * 16 * bits, dn_e = (size_t)(I / 16) * NB2 * 16 * bits;
    const float aff = c_ + 1024.f * a_;
    const int* grp_start = t.grp_start->data();
    const int* grp_routes = t.grp_routes->data();
    thread_local std::vector<float> tmp;
    if (tmp.size() < (size_t)std::max(H, 2 * I)) tmp.resize(std::max(H, 2 * I));
    // rotate in: H(x * suh) per projection, its sum and x pattern
    for (;;) {
      const int64_t r = a_next_.fetch_add(1, std::memory_order_relaxed);
      if (r >= a_total_) break;
      const int e = t.ids[r];
      if (e < 0 || e >= t.num_experts) continue;
      const bf16_t* x = t.x + (size_t)(r / top_k) * H;
      for (int pj = 0; pj < 2; ++pj) {
        xsum_[r * 2 + pj] = exl3_rotate_in(x, gu_suh + ((size_t)e * 2 + pj) * H, H, tmp.data());
        exl3_xpat(tmp.data(), H, xpat_.data() + ((size_t)r * 2 + pj) * H * 2);
      }
    }
    barrier();
    // gate|up: tile columns [0, I/16) are gate's, the rest up's (each with its own rotated input)
    for (;;) {
      const int64_t q = p1_next_.fetch_add(1, std::memory_order_relaxed);
      if (q >= p1_total_) break;
      const int64_t g = q / (NB1 / cb1_);
      const int b0 = (int)(q % (NB1 / cb1_)) * cb1_, proj = b0 >= NB1 / 2;
      const int* rs = grp_routes + grp_start[g];
      const int n = grp_start[g + 1] - grp_start[g], e = t.ids[rs[0]];
      for (int b = b0; b < b0 + cb1_; ++b) {
        const int pf = b == b0 ? cb1_ : 1;  // the first column brings the block's row segments in
        for (int j0 = 0; j0 < n; j0 += 4) {
          const int nt = std::min(4, n - j0);
          const float* xp[4];
          float* so[4];
          for (int j = 0; j < nt; ++j) {
            xp[j] = xpat_.data() + ((size_t)rs[j0 + j] * 2 + proj) * H * 2;
            so[j] = s1_.data() + (size_t)rs[j0 + j] * 2 * I + 16 * b;
          }
          if (nt == 1)
            exl3_gemv_cols(gu + (size_t)e * gu_e, H, 2 * I, b, bits, xp[0], so[0], pf);
          else
            exl3_gemv_cols_mt(gu + (size_t)e * gu_e, H, 2 * I, b, bits, nt, xp, so, pf);
        }
      }
    }
    barrier();
    // mid: rotate both out, * svh, silu(gate) * up, * down suh, rotate into down's domain
    for (;;) {
      const int64_t r = mid_next_.fetch_add(1, std::memory_order_relaxed);
      if (r >= mid_total_) break;
      const int e = t.ids[r];
      if (e < 0 || e >= t.num_experts) continue;
      float* gv = s1_.data() + (size_t)r * 2 * I;
      for (int pj = 0; pj < 2; ++pj)
        exl3_rotate_out(gv + pj * I, I, a_, aff * xsum_[r * 2 + pj], gu_svh + (size_t)e * 2 * I + pj * I);
      xsumd_[r] = exl3_mid(gv, gv + I, dn_suh + (size_t)e * I, I, tmp.data());
      exl3_xpat(tmp.data(), I, xpd_.data() + (size_t)r * I * 2);
    }
    barrier();
    // down: tile columns of H
    for (;;) {
      const int64_t q = p2_next_.fetch_add(1, std::memory_order_relaxed);
      if (q >= p2_total_) break;
      const int64_t g = q / (NB2 / cb2_);
      const int b0 = (int)(q % (NB2 / cb2_)) * cb2_;
      const int* rs = grp_routes + grp_start[g];
      const int n = grp_start[g + 1] - grp_start[g], e = t.ids[rs[0]];
      for (int b = b0; b < b0 + cb2_; ++b) {
        const int pf = b == b0 ? cb2_ : 1;
        for (int j0 = 0; j0 < n; j0 += 4) {
          const int nt = std::min(4, n - j0);
          const float* xp[4];
          float* so[4];
          for (int j = 0; j < nt; ++j) {
            xp[j] = xpd_.data() + (size_t)rs[j0 + j] * I * 2;
            so[j] = s2_.data() + (size_t)rs[j0 + j] * H + 16 * b;
          }
          if (nt == 1)
            exl3_gemv_cols(dn + (size_t)e * dn_e, I, H, b, bits, xp[0], so[0], pf);
          else
            exl3_gemv_cols_mt(dn + (size_t)e * dn_e, I, H, b, bits, nt, xp, so, pf);
        }
      }
    }
    barrier();
    // rotate out, * svh * router weight, summed over the token's routes, per 128-block of the output
    for (;;) {
      const int64_t q = e_next_.fetch_add(1, std::memory_order_relaxed);
      if (q >= e_total_) break;
      const int tok = (int)(q / (H / 128)), blk = (int)(q % (H / 128));
      alignas(32) float acc[128] = {0.f};
      alignas(32) float v[128];
      for (int k = 0; k < top_k; ++k) {
        const int64_t r = (int64_t)tok * top_k + k;
        const int e = t.ids[r];
        if (e < 0 || e >= t.num_experts) continue;
        std::memcpy(v, s2_.data() + (size_t)r * H + blk * 128, sizeof(v));
        exl3_rotate_out(v, 128, a_, aff * xsumd_[r], dn_svh + (size_t)e * H + blk * 128);
        exl3_axpy128(acc, v, t.w[r]);
      }
      bf16_t* y = t.y + (size_t)tok * H + blk * 128;
      for (int n = 0; n < 128; ++n) y[n] = f32_to_bf16(acc[n]);
    }
#else
    (void)t;
    (void)barrier;
#endif
  }

 private:
  int bits_ = 0;
  float a_ = 0.f, c_ = 0.f;
  const uint64_t *gate_up_ = nullptr, *gu_suh_ = nullptr, *gu_svh_ = nullptr;
  const uint64_t *down_ = nullptr, *dn_suh_ = nullptr, *dn_svh_ = nullptr;
  std::vector<float> xpat_, xsum_, s1_, xpd_, xsumd_, s2_;
  // FREETOKEN_CPU_MOE_EXL3_COLS=n: adjacent tile columns per task (their row segments are contiguous in RAM; the first
  // column of a block prefetches the whole segment, the others find it in the core's cache)
  const int cols_ = [] {
    const char* v = std::getenv("FREETOKEN_CPU_MOE_EXL3_COLS");
    return v ? std::max(1, std::atoi(v)) : 1;
  }();
  int cb1_ = 1, cb2_ = 1;
  alignas(64) std::atomic<int64_t> a_next_{0};
  alignas(64) std::atomic<int64_t> p1_next_{0};
  alignas(64) std::atomic<int64_t> mid_next_{0};
  alignas(64) std::atomic<int64_t> p2_next_{0};
  alignas(64) std::atomic<int64_t> e_next_{0};
  int64_t a_total_ = 0, p1_total_ = 0, mid_total_ = 0, p2_total_ = 0, e_total_ = 0;
};

}  // namespace

std::unique_ptr<MoeFormat> make_exl3_format(const MoeParams& p, const FormatArgs& a) {
  return std::make_unique<Exl3Format>(p, a);
}

}  // namespace freetoken::cpu_moe
