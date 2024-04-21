#include <cublas_v2.h>
#include <cstdio>
#include <iostream>

using namespace std;

#define CHECK_CUBLAS(Expr) { \
    int err = (Expr); \
    if (err != 0) { \
        printf("cuBLAS error %d at line %d\n", err, __LINE__); \
    } \
}

void gemm(cublasHandle_t handle,
          int m,
          int n,
          int k,
          const void *alpha,
          const void *beta,
          cudaDataType_t input_type,
          const void *A,
          const void *B,
          cudaDataType_t output_type,
          void *C,
#if __CUDACC_VER_MAJOR__ >= 11
          cublasComputeType_t compute_type,
#else
          cudaDataType_t compute_type,
#endif
          cublasGemmAlgo_t algo) {
    CHECK_CUBLAS(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
        alpha, B, input_type, n, A, input_type, k,
        beta, C, output_type, n, compute_type, algo));
}

int main(int args, char* argv[]) {

    // On RTX 6000 Ada
    // nvcc -o cublas_gemm cublas_gemm.cu -lcublas -code=sm_89 -arch=compute_89
    // m,n,k = 49152, 128, 64, mps=10%, latency=0.049252ms, 
    // at the mean time, juno latency = 59us
    // They can perfectly pipelined
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cudaDataType_t input_type = CUDA_R_16F;
    cudaDataType_t output_type = CUDA_R_16F;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
#else
    cudaDataType_t compute_type = CUDA_R_16F;
#endif
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    int iter = 100;

    void *A, *B, *C;
    cudaMalloc(&A, m * k * sizeof(__half));
    cudaMalloc(&B, k * n * sizeof(__half));
    cudaMalloc(&C, m * n * sizeof(__half));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
         output_type, C, compute_type, algo);

    cudaEventRecord(start);
    for (int i = 0; i < iter; ++i) {
        gemm(handle, m, n, k, &alpha, &beta, input_type, A, B,
             output_type, C, compute_type, algo);
    }
    cudaEventRecord(stop);

    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    long ops = (long)m * n * k * 2;
    double gops = ((double)ops / 1e9) / ((double)time_ms / iter / 1e3) / 1e3;
    printf("CBLAS - M : %d, N : %d, K : %d, %f ms, %f Tflops\n", m, n, k, (time_ms/iter), gops);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}