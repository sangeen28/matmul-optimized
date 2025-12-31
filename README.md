# Optimized Square Matrix Multiplication

This repo contains a from-scratch single-precision GEMM implementation (square matrices) with:
- Baseline 3-loop implementation
- Blocking, packing, and an AVX2/FMA microkernel
- OpenMP parallel strategies
- OpenBLAS reference runner for performance comparison

## System used for the reported results
- CPU: Intel Core i5-6300U (2 cores / 4 threads)
- Cache: L2 512 KB, L3 3 MB
- OS: Windows + WSL2 (Ubuntu)


## License
MIT.
