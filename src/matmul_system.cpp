#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if defined(GPROF_COMPUTE_ONLY)
  #include <sys/gmon.h>   // moncontrol
  extern "C" void moncontrol(int);
#endif


#ifdef _OPENMP
  #include <omp.h>
#endif

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

static inline double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
static inline double gflops(int N, double sec) {
  const double flops = 2.0 * double(N) * double(N) * double(N);
  return (flops / sec) / 1e9;
}

struct Args {
  int N = 1024;
  int iters = 5;
  int threads = 0; 
  std::string variant = "all"; // naive|jpi|blocked|packed_ic|packed_jc|all
  int BM = 128, BN = 128, BK = 128; // for "blocked"
  int MC = 640, NC = 768, KC = 256; // for "packed_*" (rounded internally)
  bool check = true;
  bool csv_header = true;
};

static void die(const std::string& msg) { std::cerr << "Error: " << msg << "\n"; std::exit(1); }

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto need = [&](const std::string& name) {
      if (i + 1 >= argc) die("Missing value after " + name);
      return std::string(argv[++i]);
    };
    if (k == "--n") a.N = std::stoi(need(k));
    else if (k == "--iters") a.iters = std::stoi(need(k));
    else if (k == "--threads") a.threads = std::stoi(need(k));
    else if (k == "--variant") a.variant = need(k);
    else if (k == "--bm") a.BM = std::stoi(need(k));
    else if (k == "--bn") a.BN = std::stoi(need(k));
    else if (k == "--bk") a.BK = std::stoi(need(k));
    else if (k == "--mc") a.MC = std::stoi(need(k));
    else if (k == "--nc") a.NC = std::stoi(need(k));
    else if (k == "--kc") a.KC = std::stoi(need(k));
    else if (k == "--check") a.check = (need(k) != "0");
    else if (k == "--csv_header") a.csv_header = (need(k) != "0");
    else if (k == "--help") {
      std::cout <<
        "Usage: ./matmul_system --n N --iters I --variant naive|jpi|blocked|packed_ic|packed_jc|all\n"
        "                    [--threads T]\n"
        "                    [--bm BM --bn BN --bk BK]     (blocked)\n"
        "                    [--mc MC --nc NC --kc KC]     (packed)\n"
        "                    [--check 0|1]\n"
        "Notes: FP32 only, column-major, compares fairly with OpenBLAS (also column-major).\n";
      std::exit(0);
    } else die("Unknown arg: " + k);
  }
  if (a.N <= 0 || a.iters <= 0) die("N and iters must be > 0");
  return a;
}

static float* aligned_alloc64(size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(float)) != 0) return nullptr;
  return (float*)p;
}
static void aligned_free64(float* p) { free(p); }

static void fill_rand(float* x, int N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  const size_t nn = size_t(N) * size_t(N);
  for (size_t i = 0; i < nn; i++) x[i] = dist(rng);
}
static void zero(float* x, int N) { std::memset(x, 0, size_t(N) * size_t(N) * sizeof(float)); }

static inline float& CM(float* X, int ld, int r, int c) { return X[c*ld + r]; }
static inline float  CMc(const float* X, int ld, int r, int c) { return X[c*ld + r]; }

static bool sampled_check(const float* A, const float* B, const float* C, int N,
                          int samples = 64, float rel_tol = 2e-2f) {
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, N - 1);
  for (int s = 0; s < samples; s++) {
    int i = dist(rng), j = dist(rng);
    double ref = 0.0;
    for (int p = 0; p < N; p++) ref += double(CMc(A, N, i, p)) * double(CMc(B, N, p, j));
    double got = double(CMc(C, N, i, j));
    double denom = std::max(1.0, std::abs(ref));
    double rel = std::abs(got - ref) / denom;
    if (rel > rel_tol) {
      std::cerr << "Check failed at ("<<i<<","<<j<<") ref="<<ref<<" got="<<got<<" rel="<<rel<<"\n";
      return false;
    }
  }
  return true;
}

// ===================== This is the main baseline =====================
static void matmul_naive_ijk(const float* A, const float* B, float* C, int N) {
#ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      float acc = 0.f;
      for (int p = 0; p < N; p++) {
        acc += CMc(A, N, i, p) * CMc(B, N, p, j);
      }
      CM(C, N, i, j) = acc;
    }
  }
}

// ===================== Variant 2: Making the loop order more efficient (j-p-i GEMM update) =====================
// C = A*B with i inner => contiguous A and C (column-major friendly)
static void matmul_jpi(const float* A, const float* B, float* C, int N) {
  zero(C, N);
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (int j = 0; j < N; j++) {
    float* Ccol = (float*)(C + j*N);
    for (int p = 0; p < N; p++) {
      const float b = B[j*N + p];          // B(p,j)
      const float* Acol = A + p*N;         // A(:,p)
      // i inner contiguous
      for (int i = 0; i < N; i++) Ccol[i] += Acol[i] * b;
    }
  }
}

// ===================== Variant 3: cache blocked (no packing) =====================
static void matmul_blocked(const float* A, const float* B, float* C, int N, int BM, int BN, int BK) {
  zero(C, N);
#ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int jc = 0; jc < N; jc += BN) {
    for (int pc = 0; pc < N; pc += BK) {
      const int jmax = std::min(N, jc + BN);
      const int pmax = std::min(N, pc + BK);

      for (int ic = 0; ic < N; ic += BM) {
        const int imax = std::min(N, ic + BM);

        for (int j = jc; j < jmax; j++) {
          float* Ccol = (float*)(C + j*N);
          for (int p = pc; p < pmax; p++) {
            const float b = B[j*N + p];
            const float* Acol = A + p*N;
            for (int i = ic; i < imax; i++) {
              Ccol[i] += Acol[i] * b;
            }
          }
        }
      }
    }
  }
}

// ===================== Packed SGEMM (16x6) — Improvement kernel.c  =====================
static inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

static inline void pack_panelB(const float* B, float* Bp, int nr, int kc, int ldB) {
  // layout: for p in [0..kc): write 6 floats (nr real + padding)
  for (int p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) *Bp++ = B[j*ldB + p];
    for (int j = nr; j < 6;  j++) *Bp++ = 0.f;
  }
}
static inline void pack_panelA(const float* A, float* Ap, int mr, int kc, int ldA) {
  // layout: for p in [0..kc): write 16 floats (mr real + padding)
  for (int p = 0; p < kc; p++) {
    for (int i = 0; i < mr; i++) *Ap++ = A[p*ldA + i];
    for (int i = mr; i < 16; i++) *Ap++ = 0.f;
  }
}

#if defined(__AVX2__)
alignas(32) static const int32_t MASK8[9][8] = {
  { 0, 0, 0, 0, 0, 0, 0, 0 },
  { -1, 0, 0, 0, 0, 0, 0, 0 },
  { -1, -1, 0, 0, 0, 0, 0, 0 },
  { -1, -1, -1, 0, 0, 0, 0, 0 },
  { -1, -1, -1, -1, 0, 0, 0, 0 },
  { -1, -1, -1, -1, -1, 0, 0, 0 },
  { -1, -1, -1, -1, -1, -1, 0, 0 },
  { -1, -1, -1, -1, -1, -1, -1, 0 },
  { -1, -1, -1, -1, -1, -1, -1, -1 }
};
static inline __m256i mask8(int active) {
  if (active < 0) active = 0;
  if (active > 8) active = 8;
  return _mm256_load_si256((const __m256i*)MASK8[active]);
}

// Full-tile kernel (16x6) — unrolled by 4 for speed
static inline void kernel_16x6_full(const float* __restrict__ Ap,
                                   const float* __restrict__ Bp,
                                   float* __restrict__ C,
                                   int kc, int ldc, bool loadC) {
  __m256 c0_0, c0_1, c1_0, c1_1, c2_0, c2_1, c3_0, c3_1, c4_0, c4_1, c5_0, c5_1;

  auto ldcol = [&](int col, __m256& lo, __m256& hi) {
    if (!loadC) { lo = _mm256_setzero_ps(); hi = _mm256_setzero_ps(); return; }
    lo = _mm256_loadu_ps(C + col*ldc + 0);
    hi = _mm256_loadu_ps(C + col*ldc + 8);
  };
  ldcol(0,c0_0,c0_1); ldcol(1,c1_0,c1_1); ldcol(2,c2_0,c2_1);
  ldcol(3,c3_0,c3_1); ldcol(4,c4_0,c4_1); ldcol(5,c5_0,c5_1);

  int p = 0;
  for (; p + 4 <= kc; p += 4) {
    #pragma unroll
    for (int u = 0; u < 4; u++) {
      const float* a = Ap + (p+u)*16;
      const float* b = Bp + (p+u)*6;

      __m256 a0 = _mm256_loadu_ps(a + 0);
      __m256 a1 = _mm256_loadu_ps(a + 8);

      __m256 b0 = _mm256_broadcast_ss(b + 0);
      __m256 b1 = _mm256_broadcast_ss(b + 1);
      __m256 b2 = _mm256_broadcast_ss(b + 2);
      __m256 b3 = _mm256_broadcast_ss(b + 3);
      __m256 b4 = _mm256_broadcast_ss(b + 4);
      __m256 b5 = _mm256_broadcast_ss(b + 5);

    #if defined(__FMA__)
      c0_0 = _mm256_fmadd_ps(a0,b0,c0_0); c0_1 = _mm256_fmadd_ps(a1,b0,c0_1);
      c1_0 = _mm256_fmadd_ps(a0,b1,c1_0); c1_1 = _mm256_fmadd_ps(a1,b1,c1_1);
      c2_0 = _mm256_fmadd_ps(a0,b2,c2_0); c2_1 = _mm256_fmadd_ps(a1,b2,c2_1);
      c3_0 = _mm256_fmadd_ps(a0,b3,c3_0); c3_1 = _mm256_fmadd_ps(a1,b3,c3_1);
      c4_0 = _mm256_fmadd_ps(a0,b4,c4_0); c4_1 = _mm256_fmadd_ps(a1,b4,c4_1);
      c5_0 = _mm256_fmadd_ps(a0,b5,c5_0); c5_1 = _mm256_fmadd_ps(a1,b5,c5_1);
    #else
      c0_0=_mm256_add_ps(c0_0,_mm256_mul_ps(a0,b0)); c0_1=_mm256_add_ps(c0_1,_mm256_mul_ps(a1,b0));
      c1_0=_mm256_add_ps(c1_0,_mm256_mul_ps(a0,b1)); c1_1=_mm256_add_ps(c1_1,_mm256_mul_ps(a1,b1));
      c2_0=_mm256_add_ps(c2_0,_mm256_mul_ps(a0,b2)); c2_1=_mm256_add_ps(c2_1,_mm256_mul_ps(a1,b2));
      c3_0=_mm256_add_ps(c3_0,_mm256_mul_ps(a0,b3)); c3_1=_mm256_add_ps(c3_1,_mm256_mul_ps(a1,b3));
      c4_0=_mm256_add_ps(c4_0,_mm256_mul_ps(a0,b4)); c4_1=_mm256_add_ps(c4_1,_mm256_mul_ps(a1,b4));
      c5_0=_mm256_add_ps(c5_0,_mm256_mul_ps(a0,b5)); c5_1=_mm256_add_ps(c5_1,_mm256_mul_ps(a1,b5));
    #endif
    }
  }
  for (; p < kc; p++) {
    const float* a = Ap + p*16;
    const float* b = Bp + p*6;
    __m256 a0 = _mm256_loadu_ps(a + 0);
    __m256 a1 = _mm256_loadu_ps(a + 8);
    __m256 b0 = _mm256_broadcast_ss(b + 0);
    __m256 b1 = _mm256_broadcast_ss(b + 1);
    __m256 b2 = _mm256_broadcast_ss(b + 2);
    __m256 b3 = _mm256_broadcast_ss(b + 3);
    __m256 b4 = _mm256_broadcast_ss(b + 4);
    __m256 b5 = _mm256_broadcast_ss(b + 5);
  #if defined(__FMA__)
    c0_0=_mm256_fmadd_ps(a0,b0,c0_0); c0_1=_mm256_fmadd_ps(a1,b0,c0_1);
    c1_0=_mm256_fmadd_ps(a0,b1,c1_0); c1_1=_mm256_fmadd_ps(a1,b1,c1_1);
    c2_0=_mm256_fmadd_ps(a0,b2,c2_0); c2_1=_mm256_fmadd_ps(a1,b2,c2_1);
    c3_0=_mm256_fmadd_ps(a0,b3,c3_0); c3_1=_mm256_fmadd_ps(a1,b3,c3_1);
    c4_0=_mm256_fmadd_ps(a0,b4,c4_0); c4_1=_mm256_fmadd_ps(a1,b4,c4_1);
    c5_0=_mm256_fmadd_ps(a0,b5,c5_0); c5_1=_mm256_fmadd_ps(a1,b5,c5_1);
  #else
    c0_0=_mm256_add_ps(c0_0,_mm256_mul_ps(a0,b0)); c0_1=_mm256_add_ps(c0_1,_mm256_mul_ps(a1,b0));
    c1_0=_mm256_add_ps(c1_0,_mm256_mul_ps(a0,b1)); c1_1=_mm256_add_ps(c1_1,_mm256_mul_ps(a1,b1));
    c2_0=_mm256_add_ps(c2_0,_mm256_mul_ps(a0,b2)); c2_1=_mm256_add_ps(c2_1,_mm256_mul_ps(a1,b2));
    c3_0=_mm256_add_ps(c3_0,_mm256_mul_ps(a0,b3)); c3_1=_mm256_add_ps(c3_1,_mm256_mul_ps(a1,b3));
    c4_0=_mm256_add_ps(c4_0,_mm256_mul_ps(a0,b4)); c4_1=_mm256_add_ps(c4_1,_mm256_mul_ps(a1,b4));
    c5_0=_mm256_add_ps(c5_0,_mm256_mul_ps(a0,b5)); c5_1=_mm256_add_ps(c5_1,_mm256_mul_ps(a1,b5));
  #endif
  }

  auto stcol = [&](int col, const __m256& lo, const __m256& hi) {
    _mm256_storeu_ps(C + col*ldc + 0, lo);
    _mm256_storeu_ps(C + col*ldc + 8, hi);
  };
  stcol(0,c0_0,c0_1); stcol(1,c1_0,c1_1); stcol(2,c2_0,c2_1);
  stcol(3,c3_0,c3_1); stcol(4,c4_0,c4_1); stcol(5,c5_0,c5_1);
}

// Tail kernel (mr<=16, nr<=6) using masks for mr and conditional cols for nr
static inline void kernel_16x6_tail(const float* __restrict__ Ap,
                                   const float* __restrict__ Bp,
                                   float* __restrict__ C,
                                   int mr, int nr, int kc, int ldc, bool loadC) {
  __m256 c0_0,c0_1,c1_0,c1_1,c2_0,c2_1,c3_0,c3_1,c4_0,c4_1,c5_0,c5_1;
  const __m256i m0 = mask8(std::min(mr, 8));
  const __m256i m1 = mask8(std::max(mr - 8, 0));

  auto ldcol = [&](int col, __m256& lo, __m256& hi) {
    if (col >= nr || !loadC) { lo=_mm256_setzero_ps(); hi=_mm256_setzero_ps(); return; }
    lo = _mm256_maskload_ps(C + col*ldc + 0, m0);
    hi = _mm256_maskload_ps(C + col*ldc + 8, m1);
  };
  ldcol(0,c0_0,c0_1); ldcol(1,c1_0,c1_1); ldcol(2,c2_0,c2_1);
  ldcol(3,c3_0,c3_1); ldcol(4,c4_0,c4_1); ldcol(5,c5_0,c5_1);

  for (int p = 0; p < kc; p++) {
    const float* a = Ap + p*16;
    const float* b = Bp + p*6;

    __m256 a0 = _mm256_loadu_ps(a + 0);
    __m256 a1 = _mm256_loadu_ps(a + 8);

    __m256 b0 = _mm256_broadcast_ss(b + 0);
    __m256 b1 = _mm256_broadcast_ss(b + 1);
    __m256 b2 = _mm256_broadcast_ss(b + 2);
    __m256 b3 = _mm256_broadcast_ss(b + 3);
    __m256 b4 = _mm256_broadcast_ss(b + 4);
    __m256 b5 = _mm256_broadcast_ss(b + 5);

  #if defined(__FMA__)
    if (nr>0){ c0_0=_mm256_fmadd_ps(a0,b0,c0_0); c0_1=_mm256_fmadd_ps(a1,b0,c0_1); }
    if (nr>1){ c1_0=_mm256_fmadd_ps(a0,b1,c1_0); c1_1=_mm256_fmadd_ps(a1,b1,c1_1); }
    if (nr>2){ c2_0=_mm256_fmadd_ps(a0,b2,c2_0); c2_1=_mm256_fmadd_ps(a1,b2,c2_1); }
    if (nr>3){ c3_0=_mm256_fmadd_ps(a0,b3,c3_0); c3_1=_mm256_fmadd_ps(a1,b3,c3_1); }
    if (nr>4){ c4_0=_mm256_fmadd_ps(a0,b4,c4_0); c4_1=_mm256_fmadd_ps(a1,b4,c4_1); }
    if (nr>5){ c5_0=_mm256_fmadd_ps(a0,b5,c5_0); c5_1=_mm256_fmadd_ps(a1,b5,c5_1); }
  #else
    if (nr>0){ c0_0=_mm256_add_ps(c0_0,_mm256_mul_ps(a0,b0)); c0_1=_mm256_add_ps(c0_1,_mm256_mul_ps(a1,b0)); }
    if (nr>1){ c1_0=_mm256_add_ps(c1_0,_mm256_mul_ps(a0,b1)); c1_1=_mm256_add_ps(c1_1,_mm256_mul_ps(a1,b1)); }
    if (nr>2){ c2_0=_mm256_add_ps(c2_0,_mm256_mul_ps(a0,b2)); c2_1=_mm256_add_ps(c2_1,_mm256_mul_ps(a1,b2)); }
    if (nr>3){ c3_0=_mm256_add_ps(c3_0,_mm256_mul_ps(a0,b3)); c3_1=_mm256_add_ps(c3_1,_mm256_mul_ps(a1,b3)); }
    if (nr>4){ c4_0=_mm256_add_ps(c4_0,_mm256_mul_ps(a0,b4)); c4_1=_mm256_add_ps(c4_1,_mm256_mul_ps(a1,b4)); }
    if (nr>5){ c5_0=_mm256_add_ps(c5_0,_mm256_mul_ps(a0,b5)); c5_1=_mm256_add_ps(c5_1,_mm256_mul_ps(a1,b5)); }
  #endif
  }

  auto stcol = [&](int col, const __m256& lo, const __m256& hi) {
    if (col >= nr) return;
    _mm256_maskstore_ps(C + col*ldc + 0, m0, lo);
    _mm256_maskstore_ps(C + col*ldc + 8, m1, hi);
  };
  stcol(0,c0_0,c0_1); stcol(1,c1_0,c1_1); stcol(2,c2_0,c2_1);
  stcol(3,c3_0,c3_1); stcol(4,c4_0,c4_1); stcol(5,c5_0,c5_1);
}
#endif

// Packed SGEMM strategy A: shared packB, parallel over ic (high locality, barriers)
static void matmul_packed_ic(const float* A, const float* B, float* C,
                            int N, int MC, int NC, int KC) {
  zero(C, N);
  const int MCp = round_up(MC, 16);
  const int NCp = round_up(NC, 6);
  const int KCp = KC;

  float* packB = aligned_alloc64(size_t(NCp) * size_t(KCp));
  if (!packB) die("packB alloc failed");
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    float* packA = aligned_alloc64(size_t(MCp) * size_t(KCp));
    if (!packA) die("packA alloc failed");

    for (int jc = 0; jc < N; jc += NCp) {
      int nc = std::min(NCp, N - jc);
      for (int pc = 0; pc < N; pc += KCp) {
        int kc = std::min(KCp, N - pc);

        
#ifdef _OPENMP
        #pragma 
#endif
        for (int j = 0; j < nc; j += 6) {
          int nr = std::min(6, nc - j);
          const float* Bj = &B[(jc + j)*N + pc]; 
          float* Bpj = packB + j*kc;             
          pack_panelB(Bj, Bpj, nr, kc, N);
        }

#ifdef _OPENMP
        #pragma 
        #pragma 
#endif
        for (int ic = 0; ic < N; ic += MCp) {
          int mc = std::min(MCp, N - ic);

          
          for (int i = 0; i < mc; i += 16) {
            int mr = std::min(16, mc - i);
            const float* Ai = &A[pc*N + (ic + i)];    
            float* Api = packA + i*kc;                
            pack_panelA(Ai, Api, mr, kc, N);
          }

          // Micro-kernel loops
          for (int jr = 0; jr < nc; jr += 6) {
            int nr = std::min(6, nc - jr);
            const float* Bp = packB + jr*kc;

            for (int ir = 0; ir < mc; ir += 16) {
              int mr = std::min(16, mc - ir);

              float* Ctile = &C[(jc + jr)*N + (ic + ir)];
              const float* Ap = packA + ir*kc;
              bool loadC = (pc != 0);

#if defined(__AVX2__)
              if (mr == 16 && nr == 6) kernel_16x6_full(Ap, Bp, Ctile, kc, N, loadC);
              else kernel_16x6_tail(Ap, Bp, Ctile, mr, nr, kc, N, loadC);
#else
              
              for (int j = 0; j < nr; j++) {
                for (int i = 0; i < mr; i++) {
                  float acc = loadC ? CMc(Ctile, N, i, j) : 0.f;
                  for (int p = 0; p < kc; p++) acc += Ap[p*16 + i] * Bp[p*6 + j];
                  CM(Ctile, N, i, j) = acc;
                }
              }
#endif
            }
          }
        }

#ifdef _OPENMP
        #pragma 
#endif
      }
    }

    aligned_free64(packA);
  }
  aligned_free64(packB);
}

// Packed SGEMM strategy B: parallelize outer jc blocks (simpler, fewer barriers)
static void matmul_packed_jc(const float* A, const float* B, float* C,
                            int N, int MC, int NC, int KC) {
  zero(C, N);
  const int MCp = round_up(MC, 16);
  const int NCp = round_up(NC, 6);
  const int KCp = KC;

#ifdef _OPENMP
  #pragma 
#endif
  for (int jc = 0; jc < N; jc += NCp) {
    int nc = std::min(NCp, N - jc);

    float* packB = aligned_alloc64(size_t(NCp) * size_t(KCp));
    float* packA = aligned_alloc64(size_t(MCp) * size_t(KCp));
    if (!packB || !packA) die("pack alloc failed");

    for (int pc = 0; pc < N; pc += KCp) {
      int kc = std::min(KCp, N - pc);

      
      for (int j = 0; j < nc; j += 6) {
        int nr = std::min(6, nc - j);
        const float* Bj = &B[(jc + j)*N + pc];
        float* Bpj = packB + j*kc;
        pack_panelB(Bj, Bpj, nr, kc, N);
      }

      for (int ic = 0; ic < N; ic += MCp) {
        int mc = std::min(MCp, N - ic);

        // packA for this ic
        for (int i = 0; i < mc; i += 16) {
          int mr = std::min(16, mc - i);
          const float* Ai = &A[pc*N + (ic + i)];
          float* Api = packA + i*kc;
          pack_panelA(Ai, Api, mr, kc, N);
        }

        for (int jr = 0; jr < nc; jr += 6) {
          int nr = std::min(6, nc - jr);
          const float* Bp = packB + jr*kc;

          for (int ir = 0; ir < mc; ir += 16) {
            int mr = std::min(16, mc - ir);

            float* Ctile = &C[(jc + jr)*N + (ic + ir)];
            const float* Ap = packA + ir*kc;
            bool loadC = (pc != 0);

#if defined(__AVX2__)
            if (mr == 16 && nr == 6) kernel_16x6_full(Ap, Bp, Ctile, kc, N, loadC);
            else kernel_16x6_tail(Ap, Bp, Ctile, mr, nr, kc, N, loadC);
#else
            for (int j = 0; j < nr; j++) {
              for (int i = 0; i < mr; i++) {
                float acc = loadC ? CMc(Ctile, N, i, j) : 0.f;
                for (int p = 0; p < kc; p++) acc += Ap[p*16 + i] * Bp[p*6 + j];
                CM(Ctile, N, i, j) = acc;
              }
            }
#endif
          }
        }
      }
    }

    aligned_free64(packB);
    aligned_free64(packA);
  }
}

// ===================== Benchmarking =====================
static void run_variant(const Args& args, const std::string& name,
                        const float* A, const float* B, float* C) {
  std::vector<double> t;
  t.reserve(args.iters);
  for (int it = 0; it < args.iters; it++) {
    double t0 = now_sec();
    if (name == "naive") matmul_naive_ijk(A, B, C, args.N);
    else if (name == "jpi") matmul_jpi(A, B, C, args.N);
    else if (name == "blocked") matmul_blocked(A, B, C, args.N, args.BM, args.BN, args.BK);
    else if (name == "packed_ic") matmul_packed_ic(A, B, C, args.N, args.MC, args.NC, args.KC);
    else if (name == "packed_jc") matmul_packed_jc(A, B, C, args.N, args.MC, args.NC, args.KC);
    else die("Unknown variant: " + name);
    double t1 = now_sec();
    t.push_back(t1 - t0);
  }
  double best = *std::min_element(t.begin(), t.end());
  double ms = best * 1000.0;
  double perf = gflops(args.N, best);

  int ok = 1;
  if (args.check) ok = sampled_check(A, B, C, args.N) ? 1 : 0;

  std::cout << args.N << "," << name << ","
            << (args.threads > 0 ? args.threads : -1) << ","
            << std::fixed << std::setprecision(3) << ms << ","
            << std::fixed << std::setprecision(2) << perf << ","
            << ok << "\n";

  if (!ok) die("Correctness failed for " + name);
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

#if defined(GPROF_COMPUTE_ONLY)
  moncontrol(0); // do not profile init (alloc + RNG)
#endif
#ifdef _OPENMP
  if (args.threads > 0) omp_set_num_threads(args.threads);
#endif

  if (args.csv_header) std::cout << "N,variant,threads,time_ms,gflops,ok\n";

  const size_t nn = size_t(args.N) * size_t(args.N);
  float* A = aligned_alloc64(nn);
  float* B = aligned_alloc64(nn);
  float* C = aligned_alloc64(nn);
  if (!A || !B || !C) die("Allocation failed");

  fill_rand(A, args.N, 1);
  fill_rand(B, args.N, 2);

#if defined(GPROF_COMPUTE_ONLY)
  moncontrol(1); // start profiling compute
#endif

  if (args.variant == "all") {
    run_variant(args, "naive", A, B, C);
    run_variant(args, "jpi", A, B, C);
    run_variant(args, "blocked", A, B, C);
    run_variant(args, "packed_ic", A, B, C);
    run_variant(args, "packed_jc", A, B, C);
  } else {
    run_variant(args, args.variant, A, B, C);
  }

#if defined(GPROF_COMPUTE_ONLY)
  moncontrol(0); // stop profiling
#endif

  aligned_free64(A); aligned_free64(B); aligned_free64(C);
  return 0;
}

