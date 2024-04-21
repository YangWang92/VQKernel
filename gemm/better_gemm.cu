#include "cuda.h"
#include "cuda_fp16.h"
#include "stdio.h"
#include "mma.h"
#include <iostream>
#include "cublas_v2.h"

#define TEST_VQ 0
#define COMPRESSION_RATIO 4
#define ENTRY 64
// nvcc -o better_gemm better_gemm.cu -arch=compute_89 -code=sm_89 -lcublas

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 1.0);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(norm_dist(rng));
        }
        else if constexpr(std::is_same<T, uint8_t>::value) {
            mat[i] = static_cast<uint8_t>((norm_dist(rng)) * (1.0 * ENTRY)) % ENTRY;
        }
    }
}

#define PROFILING 1

const int wmup = 50;
const int iter = 100;

#define M 2048
#define N 2048
#define K 2048

// ( 512,  512,  512) =  32.571
// (1024, 1024, 1024) =  92.098
// (2048, 2048, 2048) = 262.179
// (4096, 4096, 4096) = 253.452
// (8192, 8192, 8192) = 226.425

#define MAX_SHARED_MEMORY_USAGE 49152

#define WARP_SIZE 32
#define WARP_NUM 4

#define BLOCK_TILE_M 128
#define BLOCK_TILE_N 128
#define BLOCK_TILE_K 32


#define WARP_TILE_M 64
#define WARP_TILE_N 64
#define WARP_TILE_K 16

#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

__device__ void loadShmemA(half* shmem, half *A, int m, int k, int ko) {
    for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_K) / (WARP_SIZE * WARP_NUM); i++) {
        int row = i * ((WARP_SIZE * WARP_NUM) / BLOCK_TILE_K) + threadIdx.x / BLOCK_TILE_K;
        int col = threadIdx.x % BLOCK_TILE_K;
        shmem[(row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * WMMA_TILE_K) + (col / WMMA_TILE_K) * (WMMA_TILE_M * WMMA_TILE_K) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K] = 
        A[(blockIdx.x * BLOCK_TILE_M + row) * k + ko * BLOCK_TILE_K + col];
    }
}

__device__ void loadShmemB(half* shmem, half *B, int k, int n, int ko) {
    for (int i = 0; i < (BLOCK_TILE_K * BLOCK_TILE_N) / (WARP_SIZE * WARP_NUM); i++) {
        int row = i * ((WARP_SIZE * WARP_NUM) / BLOCK_TILE_N) + threadIdx.x / BLOCK_TILE_N;
        int col = threadIdx.x % BLOCK_TILE_N;
        shmem[(row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_K * WMMA_TILE_N) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % WMMA_TILE_N] = 
        B[(ko * BLOCK_TILE_K + row) * n + blockIdx.y * BLOCK_TILE_N + col];
    }
}

__device__ void storeShmemC(half *C, half* shmem, int m, int n) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    for (int i = 0; i < (BLOCK_TILE_M * BLOCK_TILE_N) / (WARP_SIZE * WARP_NUM); i++) {
        int row = i * ((WARP_SIZE * WARP_NUM) / BLOCK_TILE_M) + threadIdx.x / BLOCK_TILE_N;
        int col = threadIdx.x % BLOCK_TILE_N;
        C[(blockIdx.x * BLOCK_TILE_M + row) * n + (blockIdx.y * BLOCK_TILE_N + col)] = 
        shmem[(row / WMMA_TILE_M) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_M * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_M * WMMA_TILE_N) + (row % WMMA_TILE_M) * (WMMA_TILE_N) + col % WMMA_TILE_N];
    }
}

__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, nvcuda::wmma::row_major>* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    for (int i = 0; i < WARP_TILE_M / WMMA_TILE_M; i++) {
        int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M;
        int col = ki * WARP_TILE_K;
        nvcuda::wmma::load_matrix_sync(frag[i], shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * WMMA_TILE_K) + (col / WMMA_TILE_K) * (WMMA_TILE_M * WMMA_TILE_K), WMMA_TILE_M);
    }
}

__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, nvcuda::wmma::row_major>* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    for (int i = 0; i < WARP_TILE_N / WMMA_TILE_N; i++) {
        int row = ki * WARP_TILE_K;
        int col = warp_id_y * WARP_TILE_N + i * WMMA_TILE_N;
        nvcuda::wmma::load_matrix_sync(frag[i], shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_K * WMMA_TILE_N), WMMA_TILE_K);
    }
}

__device__ void storeFragC(half* shmem, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half>* frag) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    for (int i = 0; i < WARP_TILE_M / WMMA_TILE_M; i++) {
        for (int j = 0; j < WARP_TILE_N / WMMA_TILE_N; j++) {
            int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M;
            int col = warp_id_y * WARP_TILE_N + j * WMMA_TILE_N;
            nvcuda::wmma::store_matrix_sync(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_M * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_M * WMMA_TILE_N), frag[i * (WARP_TILE_N / WMMA_TILE_N) + j], WMMA_TILE_M, nvcuda::wmma::mem_row_major);
        }
    }
}

__device__ void loadCodebookB(half *codebook, half *codebook_buf, int entry) {
    // Every thread load 2 half since one bank.
}

__global__ void matmul(
    half* A,    // Row major
    half* B,    // Col major
    half* C,    // Row major
    uint32_t m, uint32_t n, uint32_t k
)
{
    extern __shared__ uint8_t shmem[];
    half *A_buf = reinterpret_cast<half*>(shmem);
    half *B_buf = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_K * sizeof(half));
    half *C_buf = reinterpret_cast<half*>(shmem);
    // half *codebook_buf = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_N * sizeof(half));

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, nvcuda::wmma::row_major> A_frag[WARP_TILE_M / WMMA_TILE_M];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, nvcuda::wmma::row_major> B_frag[WARP_TILE_N / WMMA_TILE_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half> C_frag[(WARP_TILE_M / WMMA_TILE_M) * (WARP_TILE_N / WMMA_TILE_N)];

    for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
        for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
            nvcuda::wmma::fill_fragment(C_frag[mm * (WARP_TILE_N / WMMA_TILE_N) + nn], 0.0);
        }
    }
    for (int ko = 0; ko < k / BLOCK_TILE_K; ko++) {
        loadShmemA(A_buf, A, m, k, ko);
        loadShmemB(B_buf, B, k, n, ko);
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA(A_frag, A_buf, ki);
            loadFragB(B_frag, B_buf, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    nvcuda::wmma::mma_sync(C_frag[mm * (WARP_TILE_N / WMMA_TILE_N) + nn], A_frag[mm], B_frag[nn], C_frag[mm * (WARP_TILE_N / WMMA_TILE_N) + nn]);
                }
            }
        }
    }
    storeFragC(C_buf, C_frag);
    __syncthreads();
    storeShmemC(C, C_buf, m, n);

}

__global__ void matmul_vq14(
    half* A,
    uint8_t B_q,
    half* codebook,
    half* C,
    uint32_t m, uint32_t n, uint32_t k
)
{
    extern __shared__ uint8_t shmem[];
    half *A_buf = reinterpret_cast<half*>(shmem);
    // Don't load quantized B into shared memory since only used once!
    // Decode them directly to the B_buf.
    half *B_buf = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_K * sizeof(half));
    half *C_buf = reinterpret_cast<half*>(shmem);
    // Load 64 entries, Reorder offline! 48kB shared memory.
    half *codebook_buf = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_N * sizeof(half));
    //  8k  8k  
    // | A | B | Codebook | 
    // |C reuse|
    //    32k       16k

    
}

#if TEST_VQ == 1
int main(int argc, char** argv) {
    half *h_A, *d_A;
    uint8_t *h_B, *d_B;
    half *h_C, *d_C;
    half *h_codebook, *d_codebook;
    h_A = new half [M * K];
    h_B = new uint8_t [K * N / COMPRESSION_RATIO];
    h_C = new half [M * N];
    h_codebook = new half [ENTRY * N];

    fill_matrix(h_A, M * K);
    fill_matrix(h_B, K * N / COMPRESSION_RATIO);
    fill_matrix(h_codebook, ENTRY * N);

    cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(half) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(uint8_t) * K * N / COMPRESSION_RATIO);
    cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_codebook), sizeof(half) * ENTRY * N);
    
    cudaMemcpy(reinterpret_cast<void*>(d_A), h_A, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_B), h_B, sizeof(uint8_t) * K * N / COMPRESSION_RATIO, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_codebook), h_codebook, sizeof(half) * ENTRY * N, cudaMemcpyHostToDevice);



    cudaMemcpy(h_C, reinterpret_cast<void*>(d_C), sizeof(half) * M * N, cudaMemcpyDeviceToHost);

    return 0;
}
#else 
int main(int argc, char** argv) {
    cudaFuncSetAttribute(matmul, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    
    half *h_A, *d_A;
    half *h_B, *d_B;
    half *h_C, *d_C;
    half *h_Cref, *d_Cref;
    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C = new half[M * N];
    h_Cref = new half[M * N];

    cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(half) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(half) * K * N);
    cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_Cref), sizeof(half) * M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = __float2half((1.0 * i + 2.0 * j) / 50000.0);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = __float2half(1.0 * (i + 1) * (j + 1) / 50000.0);
        }
    }
    cudaMemcpy(reinterpret_cast<void*>(d_A), h_A, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_B), h_B, sizeof(half) * K * N, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha = __float2half(1.0), beta = __float2half(0.0);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, CUDA_R_16F, M, d_B, CUDA_R_16F, K, &beta, d_Cref, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K, &beta, d_Cref, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, CUDA_R_16F, M, d_B, CUDA_R_16F, K, &beta, d_Cref, CUDA_R_16F, M, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K, &beta, d_Cref, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#if PROFILING == 1    
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << ((2.0 * M * N * K) / ((ms / (1.0 * iter)) / 1000.0)) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif
    cudaMemcpy(h_Cref, reinterpret_cast<void*>(d_Cref), sizeof(half) * M * N, cudaMemcpyDeviceToHost);

    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
    matmul<<<grid, WARP_NUM * WARP_SIZE, MAX_SHARED_MEMORY_USAGE>>>(
        d_A,
        d_B,
        d_C,
        M, N, K
    );
    }
    
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
    matmul<<<grid, WARP_NUM * WARP_SIZE, MAX_SHARED_MEMORY_USAGE>>>(
        d_A,
        d_B,
        d_C,
        M, N, K
    );
#if PROFILING == 1
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << ((2.0 * M * N * K) / ((ms / (1.0 * iter)) / 1000.0)) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif
    CHECK_LAST_CUDA_ERROR();
    cudaMemcpy(h_C, reinterpret_cast<void*>(d_C), sizeof(half) * M * N, cudaMemcpyDeviceToHost);

#if PROFILING == 0
    for (int i = 16; i < 32; i++) {
        for (int j = 16; j < 32; j++) {
            printf("%04.2f%c", __half2float(h_C[i * N + j]) - __half2float(h_Cref[j * N + i]), (j == 31) ? '\n' : ' ');
        }
    }
#endif
    return 0;
}
#endif