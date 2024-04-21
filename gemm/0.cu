
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(16) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[1];
  for (int i_1_1 = 0; i_1_1 < 32; ++i_1_1) {
    for (int j_1_1 = 0; j_1_1 < 32; ++j_1_1) {
      for (int k = 0; k < 4096; ++k) {
        if (k == 0) {
          C_local[0] = 0.000000e+00f;
        }
        C_local[0] = (C_local[0] + (A[((((((int)blockIdx.x) * 524288) + (((int)threadIdx.x) * 131072)) + (i_1_1 * 4096)) + k)] * B[((((k * 4096) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 32)) + j_1_1)]));
      }
      C[((((((((int)blockIdx.x) * 524288) + (((int)threadIdx.x) * 131072)) + (i_1_1 * 4096)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 32)) + j_1_1)] = C_local[0];
    }
  }
}

