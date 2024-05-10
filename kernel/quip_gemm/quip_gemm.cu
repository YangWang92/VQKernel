#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <torch/extension.h>
#include "stdio.h"
#include <iostream>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include "mma.h"
#include <random>

#define PROFILING 1

#define WARP_NUM 4
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_NUM * WARP_SIZE)
#define PACKED_LOAD_WIDTH 8
#define ENTRY 256
#define RATIO 4
#define ENTRY_CENTRIC 0

#define BLOCK_TILE_M 128
#define BLOCK_TILE_N 128
#define BLOCK_TILE_K 32

#define WARP_TILE_M 64
#define WARP_TILE_N 64
#define WARP_TILE_K 16

#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

#define MMA_TILE_M 16
#define MMA_TILE_N 8
#define MMA_TILE_K 16

#define MAX_SHARED_MEMORY_USAGE 34816

__device__ __forceinline__ uint32_t shmem_uint32_t(const void* shmem_ptr) {
    uint32_t addr;
    asm volatile(
        "{.reg .u64 u64addr;\n"
        " cvta.to.shared.u64 u64addr, %1;\n"
        " cvt.u32.u64 %0, u64addr;}\n"
        : "=r"(addr)
        : "l"(shmem_ptr)
    );
    return addr;
}

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 5.0);
    std::exponential_distribution<float> exp_dist(0.01);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(norm_dist(rng));
        }
        else if constexpr(std::is_same<T, uint8_t>::value) {
            // mat[i] = static_cast<uint8_t>((norm_dist(rng)) * (1.0 * ENTRY)) % ENTRY;
            mat[i] = static_cast<uint8_t>(exp_dist(rng)) % ENTRY;
        }
    }
}

template <typename T, int len>
struct packed_vec{};

template <>
struct packed_vec<half, 8> {
    using Type = uint4;
};

template <>
struct packed_vec<half, 4> {
    using Type = uint64_t;
};

template <>
struct packed_vec<half, 2> {
    using Type = uint32_t;
};


__device__ void load_codebook(half* shmem, half* codebook) {
    // 256 entry, 4 half per entry.
    uint32_t rows_at_once = (BLOCK_SIZE * PACKED_LOAD_WIDTH) / RATIO;
    #pragma unroll
    for (int i = 0; i < (ENTRY * RATIO) / (PACKED_LOAD_WIDTH * BLOCK_SIZE); i++) {
        asm volatile (
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(shmem_uint32_t(shmem + i * (rows_at_once * RATIO) + threadIdx.x * PACKED_LOAD_WIDTH)), 
              "l"(&codebook[i * (rows_at_once * RATIO) + threadIdx.x * PACKED_LOAD_WIDTH]), 
              "r"((uint32_t)(PACKED_LOAD_WIDTH * 2))
        );
    }
}

__device__ void loadShmemA(half* shmem, half *A, int m, int k, int ko) {
    for (int i = 0; i < ((BLOCK_TILE_M * BLOCK_TILE_K) / BLOCK_SIZE) / 8; i++) {
        int row = i * 32 + threadIdx.x / 4;
        int col = 8 * (threadIdx.x % 4);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            ::
            "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K)), "l"(&A[(blockIdx.x * BLOCK_TILE_M + row) * k + ko * BLOCK_TILE_K + col])
        );
    }
}

__device__ void dequantToShmemB_EntryCentric(half* shmem, uint8_t* B_q, half* codebook, int k, int n, int ko) {
    uint8_t mask = 0, indices[8];
    half entry[RATIO];
    *(uint64_t*)(&indices[0]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K) * (n / RATIO) + blockIdx.y * (BLOCK_TILE_N / RATIO) + (threadIdx.x / 4) * (n / RATIO) + (threadIdx.x % 4) * 8]);
    for (uint8_t e = 0; e < ENTRY_CENTRIC; e++) {
        *(uint64_t*)(&entry[0]) = *(uint64_t*)(&codebook[((uint32_t)e) * RATIO]);
        for (int i = 0; i < 8; i++) {
            if (e == indices[i]) {
                *(uint64_t*)(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (2 * (threadIdx.x % 4) + (i / 4)) * (16 * 16) + ((threadIdx.x % 64) / 4) * (16) + (i % 4) * 4]) = *(uint64_t*)(&entry[0]);
                mask |= (0x1 << i);
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        if (mask & (0x1 << i)) continue;
        else {
            *(uint64_t*)(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (2 * (threadIdx.x % 4) + (i / 4)) * (16 * 16) + ((threadIdx.x % 64) / 4) * (16) + (i % 4) * 4]) = *(uint64_t*)(&codebook[((uint32_t) indices[i]) * RATIO]);
        }
    }
}

__device__ void dequantToRegB(uint32_t* frag, uint8_t* B_q, half* codebook, int k, int n, int ko, int ki) {
    // Output should be 16x64
    // Every warp do 16x8
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // for (int i = 0; i < 8; i++) {
    //     uint8_t id = B_q[((ko * BLOCK_TILE_K + ki * WARP_TILE_K) * (n / RATIO)) + blockIdx.y * (BLOCK_TILE_N / RATIO) + warp_id_y * (WARP_TILE_N / RATIO) + i * (MMA_TILE_N / RATIO) + (lane_id % 2) * 8 * (n / RATIO) + (lane_id / 4) * (n / RATIO) + ((lane_id % 4) / 2)];
    //     *(uint64_t*)(&frag[i * 2]) = *(uint64_t*)(&codebook[((uint32_t) id) * 4]);
    // }
    uint8_t ids[8];
    *(uint64_t*)(&ids[0]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K + ki * WARP_TILE_K) * (n / RATIO) + blockIdx.y * (BLOCK_TILE_N / RATIO) + warp_id_y * (WARP_TILE_N / RATIO) + (lane_id % 2) * 8 * (n / RATIO) + (lane_id / 4) * (n / RATIO) + ((lane_id % 4) / 2) * 8]);
    *(uint32_t*)(&ids[4]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&ids[4]), 2);
    *(uint16_t*)(&ids[2]) = __shfl_xor_sync(0xffffffff, *(uint16_t*)(&ids[2]), 2);
    *(uint16_t*)(&ids[6]) = __shfl_xor_sync(0xffffffff, *(uint16_t*)(&ids[6]), 2);
    *(uint8_t*)(&ids[1]) = __shfl_xor_sync(0xffffffff, *(uint8_t*)(&ids[1]), 2);
    *(uint8_t*)(&ids[3]) = __shfl_xor_sync(0xffffffff, *(uint8_t*)(&ids[3]), 2);
    *(uint8_t*)(&ids[5]) = __shfl_xor_sync(0xffffffff, *(uint8_t*)(&ids[5]), 2);
    *(uint8_t*)(&ids[7]) = __shfl_xor_sync(0xffffffff, *(uint8_t*)(&ids[7]), 2);

    *(uint64_t*)(&frag[0]) = *(uint64_t*)(&codebook[((uint32_t) ids[0]) * 4]);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        *(uint64_t*)(&frag[i * 2]) = *(uint64_t*)(&codebook[((uint32_t) ids[i]) * 4]);
        *(uint32_t*)(&frag[(i-1) * 2 + 1]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[(i-1) * 2 + 1]), 1);
    }
    *(uint32_t*)(&frag[15]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[15]), 1);
}

__device__ void loadFragA_mma(uint32_t* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < 4; i++) {       // Warp do 64x16, 16x16 a time, so 4 times
        for (int j = 0; j < 4; j++) {   // for every 16x16, every thread load 4 1x2 data
            int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M + (j / 2) * 8 + (lane_id / 4);
            int col = ki * WMMA_TILE_K + (j % 2) * 8 + (lane_id % 4) * 2;
            frag[i * 4 + j] = *(uint32_t*)(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K);
        }
        // int row = warp_id_x * WARP_TILE_M + i * 16 + (lane_id % 16);
        // int col = ki * WARP_TILE_K + (lane_id / 16) * 8;
        // asm volatile (
        //     "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        //     : "=r"(frag[i * 4]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
        //     : "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K))
        // );
    }
}

__device__ void loadFragB_mma(uint32_t* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < 8; i++) {       // Warp do 16x64, 16x8 a time, so 8 times
        for (int j = 0; j < 2; j++) {   // for every 16x8, every thread load 2 1x2 data
            int row = ki * WARP_TILE_K + j * 8 + (lane_id / 4);
            int col = warp_id_y * WARP_TILE_N + i * 8 + (lane_id % 4) * 2;
            frag[i * 2 + j] = *(uint32_t*)(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N));
        }
        // Can directly use ldmatrix.trans
        asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(frag[i * 2]) : "r"(frag[i * 2]));
        asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(frag[i * 2 + 1]) : "r"(frag[i * 2]));
    }
    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     int row = ki * WARP_TILE_K + (lane_id % 16);
    //     int col = warp_id_y * WARP_TILE_N + i * 16 + (lane_id / 16) * 8;
    //     asm volatile (
    //         "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    //         : "=r"(frag[i * 4]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
    //         : "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N)))
    //     );
    // }
}

__device__ void compute_mma(uint32_t* A, uint32_t* B, uint32_t* C) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1])
    );
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[2]), "r"(B[3]),
          "r"(C[2]), "r"(C[3])
    );
}

__device__ void storeFragC_mma(half* shmem, uint32_t* frag) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < 4; i++) {           // 4 rows
        for (int j = 0; j < 8; j++) {       // 8 cols
            for (int k = 0; k < 2; k++) {   // 2 frags
                int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M + k * 8 + (lane_id / 4);
                int col = warp_id_y * WARP_TILE_N + j * 8 + (lane_id % 4) * 2;
                *(uint32_t*)(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_M * WMMA_TILE_N) + (col / WMMA_TILE_N) * (WMMA_TILE_M * WMMA_TILE_N) + (row % WMMA_TILE_M) * (WMMA_TILE_N) + (col % WMMA_TILE_N)) = 
                frag[i * 8 * 2 + j * 2 + k];
            }
        }
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

__global__ void quip_gemm_kernel(
    half* _input,
    uint8_t* _w,
    half* _codebook,
    half* _o,
    int M, int N, int K
)
{
    extern __shared__ uint8_t shmem[];
    half *A_buf = reinterpret_cast<half*>(shmem);
    half *B_buf = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_K * sizeof(half));
    half *C_buf = reinterpret_cast<half*>(shmem);
    half *codebook_buf = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_N * sizeof(half));

    load_codebook(codebook_buf, _codebook);
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};

    for (int ko = 0; ko < K / BLOCK_TILE_K; ko++) {
        loadShmemA(A_buf, _input, M, K, ko);
        // dequantToShmemB_EntryCentric(B_buf, _w, codebook_buf, K, N, ko);
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A_buf, ki);
            // loadFragB_mma(B_frags, B_buf, ki);
            dequantToRegB(B_frags, _w, codebook_buf, K, N, ko, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
    }
    storeFragC_mma(C_buf, C_frags);
    __syncthreads();
    storeShmemC(_o, C_buf, M, N);  
}

// For rapid debugging
// int main(int argc, char** argv) {
//     cudaFuncSetAttribute(quip_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
//     half* h_input, *d_input;
//     half* h_codebook, *d_codebook;
//     uint8_t* h_w, *d_w;
//     half* h_o, *d_o;

//     cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 4096 * 4096);
//     cudaMalloc(reinterpret_cast<void**>(&d_codebook), sizeof(half) * 256 * 4);
//     cudaMalloc(reinterpret_cast<void**>(&d_w), sizeof(uint8_t) * 4096 * 1024);
//     cudaMalloc(reinterpret_cast<void**>(&d_o), sizeof(half) * 4096 * 4096);
//     dim3 grid(4096 / 128, 4096 / 128);
//         quip_gemm_kernel<<<grid, 128, MAX_SHARED_MEMORY_USAGE>>>(
//             d_input,
//             d_w,
//             d_codebook,
//             d_o,
//             4096,4096,4096
//         );
// }

torch::Tensor quip_gemm(
    torch::Tensor input,        // Seq_len * Hidden_dim             FP16
    torch::Tensor w,            // Hidden_dim * Hidden_dim / 4      UINT8
    torch::Tensor codebook      // 256 * 4                          FP16
)
{
#if PROFILING == 1
    const int wmup = 50;
    const int iter = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);
#endif
    cudaFuncSetAttribute(quip_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    
    auto seq_len = input.size(0);
    auto hidden_dim = input.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({seq_len, hidden_dim}, 0, options);

    uint8_t *h_dummy_w, *d_dummy_w;
    h_dummy_w = new uint8_t[hidden_dim * hidden_dim];
    fill_matrix(h_dummy_w, hidden_dim * hidden_dim);
    cudaMalloc(reinterpret_cast<void**>(&d_dummy_w), sizeof(uint8_t) * hidden_dim * hidden_dim);
    cudaMemcpy(reinterpret_cast<void*>(d_dummy_w), h_dummy_w, sizeof(uint8_t) * hidden_dim * hidden_dim, cudaMemcpyHostToDevice);

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>());
    half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    dim3 grid(seq_len / BLOCK_TILE_M, hidden_dim / BLOCK_TILE_N);
    dim3 block(BLOCK_SIZE);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        quip_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr,
            d_dummy_w,
            codebook_ptr,
            o_ptr,
            seq_len,
            hidden_dim,
            hidden_dim
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
    quip_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
        input_ptr,
        w_ptr,
        codebook_ptr,
        o_ptr,
        seq_len,
        hidden_dim,
        hidden_dim
    );
#if PROFILING == 1
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;
    std::cout << "TFLOPS : " << ((2.0 * hidden_dim * hidden_dim * seq_len) / ((ms / (1.0 * iter)) / (1000.0))) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif

    return o;
}