#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <cblas.h>

static inline double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
static inline double gflops(int N, double sec) {
  const double flops = 2.0 * double(N) * double(N) * double(N);
  return (flops / sec) / 1e9;
}
static void fill_rand(float* x, int N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  size_t nn = size_t(N) * size_t(N);
  for (size_t i = 0; i < nn; i++) x[i] = dist(rng);
}

int main(int argc, char** argv) {
  int N = 1024, iters = 5;
  for (int i=1;i<argc;i++) {
    std::string k = argv[i];
    auto need = [&](const std::string&){ return std::string(argv[++i]); };
    if (k=="--n") N = std::stoi(need(k));
    else if (k=="--iters") iters = std::stoi(need(k));
  }

  std::vector<float> A(size_t(N)*N), B(size_t(N)*N), C(size_t(N)*N);
  fill_rand(A.data(), N, 1);
  fill_rand(B.data(), N, 2);

  const float alpha = 1.f, beta = 0.f;
  double best = 1e100;
  for (int it=0; it<iters; it++) {
    std::fill(C.begin(), C.end(), 0.f);
    double t0 = now_sec();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, alpha, A.data(), N, B.data(), N, beta, C.data(), N);
    double t1 = now_sec();
    best = std::min(best, t1 - t0);
  }

  std::cout << "N,variant,threads,time_ms,gflops\n";
  std::cout << N << ",openblas,-1," << std::fixed << std::setprecision(3) << best*1000.0
            << "," << std::fixed << std::setprecision(2) << gflops(N,best) << "\n";
  return 0;
}
