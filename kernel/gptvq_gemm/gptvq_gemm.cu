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

#define PROFILING 0
#define WARP_NUM 4
#define WARP_SIZE 32
#define BLOCK_SIZE (WARP_NUM * WARP_SIZE)
#define ENTRY 256
#define RATIO 2
#define RESIDUAL 1
#define ENTRY_CENTRIC 0
#define HOT 1
#define BUFFERED_ENTRY (ENTRY / HOT)
#define NUM_CODEBOOK_LOAD_AT_ONCE 16 // MAX: 64
#define NUM_CODEBOOK_LOAD_AT_ONCE_OUTTER 8 // MAX: NUM_CODEBOOK_LOAD_AT_ONCE / K_BLOCK

#define SPLIT_K 2048

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

#define MAX_SHARED_MEMORY_USAGE (16384 + NUM_CODEBOOK_LOAD_AT_ONCE * (ENTRY * RATIO * 2) / HOT)
#define MAX_SHARED_MEMORY_USAGE_OUTTER (16384 + NUM_CODEBOOK_LOAD_AT_ONCE_OUTTER * (ENTRY * RATIO * 2) / HOT)

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
    std::exponential_distribution<float> exp_dist(0.025);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(norm_dist(rng));
        }
        else if constexpr(std::is_same<T, uint8_t>::value) {
            // mat[i] = static_cast<uint8_t>((norm_dist(rng)) * (1.0 * ENTRY)) % ENTRY;
            mat[i] = static_cast<uint8_t>(exp_dist(rng)) % ENTRY;
        }
        else if constexpr(std::is_same<T, uint16_t>::value) {
            mat[i] = static_cast<uint16_t>(exp_dist(rng)) % ENTRY;
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

__device__ void load_codebook(
    half* shmem,
    half* codebook,
    int k
)
{
    uint32_t codebook_begin_row = (blockIdx.y / 2) * 64 + k / 2;

    uint32_t thread_to_load = min((uint32_t) BLOCK_SIZE, (uint32_t)((NUM_CODEBOOK_LOAD_AT_ONCE * ENTRY * RATIO / HOT) / 8));
    uint32_t iters_to_load = ((NUM_CODEBOOK_LOAD_AT_ONCE * ENTRY * RATIO / HOT) / 8) / thread_to_load;
    uint32_t load_cols = (ENTRY * RATIO / HOT) / 8;               // How many thread load a row
    uint32_t load_rows = thread_to_load / load_cols;              // How many rows a block load at one iter

    
    #pragma unroll
    for (int i = 0; i < iters_to_load; i++) {
        if (threadIdx.x < thread_to_load) {
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(shmem_uint32_t(&shmem[(i * load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO / HOT) + (threadIdx.x % load_cols) * 8])),
              "l"(&codebook[(codebook_begin_row + i * load_rows + threadIdx.x / load_cols) * ENTRY * RATIO + (threadIdx.x % load_cols) * 8])
            );
        }
    }
}

__device__ void load_codebook_outter(
    half* shmem,
    half* codebook,
    int k
)
{
    uint32_t codebook_begin_row = (blockIdx.y / 2) * 64 + k / 2;

    uint32_t thread_to_load = min((uint32_t) BLOCK_SIZE, (uint32_t)((NUM_CODEBOOK_LOAD_AT_ONCE_OUTTER * ENTRY * RATIO / HOT) / 8));
    uint32_t iters_to_load = ((NUM_CODEBOOK_LOAD_AT_ONCE_OUTTER * ENTRY * RATIO / HOT) / 8) / thread_to_load;
    uint32_t load_cols = (ENTRY * RATIO / HOT) / 8;               // How many thread load a row
    uint32_t load_rows = thread_to_load / load_cols;              // How many rows a block load at one iter

    
    #pragma unroll
    for (int i = 0; i < iters_to_load; i++) {
        if (threadIdx.x < thread_to_load) {
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(shmem_uint32_t(&shmem[(i * load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO / HOT) + (threadIdx.x % load_cols) * 8])),
            "l"(&codebook[(codebook_begin_row + i * load_rows + threadIdx.x / load_cols) * ENTRY * RATIO + (threadIdx.x % load_cols) * 8])
            );
        }
    }
}

__device__ void dequantToShmemB_EntryCentric(
    half* shmem,
    uint8_t* B_q,
    half* codebook_shmem,
    half* codebook_global,
    int k, 
    int n,
    int ko
)
{
    uint8_t buffered_entry = (uint8_t) ((uint32_t)ENTRY) / ((uint32_t)HOT);
    uint32_t prefill = 0x12345678;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        *(uint32_t*)(&shmem[(i * 2 + threadIdx.x / 64) * BLOCK_TILE_N + (threadIdx.x % 64) * 2]) = prefill;
    }
    uint32_t mask = 0;
    uint8_t indices[16];
    *(uint4*)(&indices[0]) = *(uint4*)(&B_q[(ko * BLOCK_TILE_K) * (n / RATIO) + blockIdx.y * (BLOCK_TILE_N / RATIO) + (threadIdx.x / 4) * (n / RATIO) + (threadIdx.x % 4) * 16]);
    for (uint8_t e = 0; e < ENTRY_CENTRIC; e++) {
        // Only for Entry_Centrid > 1 use: *(uint32_t*)(&entry[0]) = codebook[...]
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (e == indices[i]) {
                mask |= (0x1 << i);
            }
        }
    }
    for (int i = 0; i < 16; i++) {
        if (mask & (0x1 << i)) continue;
        else {
            if (indices[i] < buffered_entry) {
                // From shmem
                *(uint32_t*)(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 8) * (16 * 16) + ((threadIdx.x % 64) / 4) * (16) + (i % 8) * 2]) = *(uint32_t*)(&codebook_shmem[((ko % (2 * NUM_CODEBOOK_LOAD_AT_ONCE)) / 2) * (ENTRY * RATIO / HOT) + ((uint32_t) indices[i]) * RATIO]);
            }
            else {
                // From global
                *(uint32_t*)(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 8) * (16 * 16) + ((threadIdx.x % 64) / 4) * (16) + (i % 8) * 2]) = *(uint32_t*)(&codebook_global[((blockIdx.y / 2) * 64 + ko / 2) * ENTRY * RATIO + ((uint32_t) indices[i]) * RATIO]);
            }
        }
    }
}

__device__ void dequantToRegB(uint32_t* frag, uint8_t* B_q, half* codebook_shmem, half* codebook_global, int k, int n, int ko, int ki) {
    uint8_t buffered_entry = (uint8_t) ((uint32_t)ENTRY) / ((uint32_t)HOT);
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint8_t ids[16];

    *(uint64_t*)(&ids[0]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K + ki * WARP_TILE_K) * (n / RATIO) + blockIdx.y * (BLOCK_TILE_N / RATIO) + warp_id_y * (WARP_TILE_N / RATIO) + (lane_id / 4) * (n / RATIO) + (lane_id % 4) * 8]);
    *(uint64_t*)(&ids[8]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K + ki * WARP_TILE_K) * (n / RATIO) + blockIdx.y * (BLOCK_TILE_N / RATIO) + warp_id_y * (WARP_TILE_N / RATIO) + (lane_id / 4) * (n / RATIO) + (lane_id % 4) * 8 + 8 * (n / RATIO)]);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t id1 = ids[i], id2 = ids[i + 8];
        if (id1 < buffered_entry) {
            frag[i * 2 + 0] = *(uint32_t*)(&codebook_shmem[((ko % (2 * NUM_CODEBOOK_LOAD_AT_ONCE)) / 2) * (ENTRY * RATIO / HOT) + ((uint32_t) id1) * RATIO]);
        }
        else {
            frag[i * 2 + 0] = *(uint32_t*)(&codebook_global[((blockIdx.y / 2) * 64 + ko / 2) * ENTRY * RATIO + ((uint32_t) id1) * RATIO]);
        }
        if (id2 < buffered_entry) {
            frag[i * 2 + 1] = *(uint32_t*)(&codebook_shmem[((ko % (2 * NUM_CODEBOOK_LOAD_AT_ONCE)) / 2) * (ENTRY * RATIO / HOT) + ((uint32_t) id2) * RATIO]);
        }
        else {
            frag[i * 2 + 1] = *(uint32_t*)(&codebook_global[((blockIdx.y / 2) * 64 + ko / 2) * ENTRY * RATIO + ((uint32_t) id2) * RATIO]);
        }
    }
}

__device__ void loadFragA_mma(uint32_t* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < 4; i++) {       // Warp do 64x16, 16x16 a time, so 4 times
        // for (int j = 0; j < 4; j++) {   // for every 16x16, every thread load 4 1x2 data
        //     int row = warp_id_x * WARP_TILE_M + i * WMMA_TILE_M + (j / 2) * 8 + (lane_id / 4);
        //     int col = ki * WMMA_TILE_K + (j % 2) * 8 + (lane_id % 4) * 2;
        //     frag[i * 4 + j] = *(uint32_t*)(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K);
        // }
        int row = warp_id_x * WARP_TILE_M + i * 16 + (lane_id % 16);
        int col = ki * WARP_TILE_K + (lane_id / 16) * 8;
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag[i * 4]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
            : "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_M) * ((BLOCK_TILE_K / WMMA_TILE_K) * WMMA_TILE_M * (WMMA_TILE_K)) + (col / WMMA_TILE_K) * (WMMA_TILE_M * (WMMA_TILE_K)) + (row % WMMA_TILE_M) * (WMMA_TILE_K) + col % WMMA_TILE_K))
        );
    }
}

__device__ void loadFragB_mma(uint32_t* frag, half *shmem, int ki) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // for (int i = 0; i < 8; i++) {       // Warp do 16x64, 16x8 a time, so 8 times
    //     for (int j = 0; j < 2; j++) {   // for every 16x8, every thread load 2 1x2 data
    //         int row = ki * WARP_TILE_K + j * 8 + (lane_id / 4);
    //         int col = warp_id_y * WARP_TILE_N + i * 8 + (lane_id % 4) * 2;
    //         frag[i * 2 + j] = *(uint32_t*)(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N));
    //     }
    //     // Can directly use ldmatrix.trans
    //     asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(frag[i * 2]) : "r"(frag[i * 2]));
    //     asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(frag[i * 2 + 1]) : "r"(frag[i * 2]));
    // }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = ki * WARP_TILE_K + (lane_id % 16);
        int col = warp_id_y * WARP_TILE_N + i * 16 + (lane_id / 16) * 8;
        asm volatile (
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(frag[i * 4]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
            : "r"(shmem_uint32_t(shmem + (row / WMMA_TILE_K) * ((BLOCK_TILE_N / WMMA_TILE_N) * WMMA_TILE_K * (WMMA_TILE_N)) + (col / WMMA_TILE_N) * (WMMA_TILE_K * (WMMA_TILE_N)) + (row % WMMA_TILE_K) * (WMMA_TILE_N) + col % (WMMA_TILE_N)))
        );
    }
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

__device__ void storeC(half* C, uint32_t* frag, int m, int n) {
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            *(uint32_t*)(&C[(blockIdx.x * BLOCK_TILE_M + warp_id_x * WARP_TILE_M + i * MMA_TILE_M + (lane_id / 4) + 0) * n + (blockIdx.y * BLOCK_TILE_N + warp_id_y * WARP_TILE_N + j * MMA_TILE_N + (lane_id % 4) * 2)]) = 
            *(uint32_t*)(&frag[(i * 8 + j) * 2 + 0]);
            *(uint32_t*)(&C[(blockIdx.x * BLOCK_TILE_M + warp_id_x * WARP_TILE_M + i * MMA_TILE_M + (lane_id / 4) + 8) * n + (blockIdx.y * BLOCK_TILE_N + warp_id_y * WARP_TILE_N + j * MMA_TILE_N + (lane_id % 4) * 2)]) = 
            *(uint32_t*)(&frag[(i * 8 + j) * 2 + 1]);
        }
    }
}

__global__ void gptvq_gemm_kernel(
    half* _input,
    uint8_t* _w,
    half* _codebook,
    half* _o,
    int M, int N, int K
)
{
    extern __shared__ uint8_t shmem[];
    half *A_buf = reinterpret_cast<half*>(shmem);
    half *B_buf = reinterpret_cast<half*>(shmem + 8192);
    half *codebook_buf = reinterpret_cast<half*>(shmem + 16384);

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};

    for (int ko = 0; ko < K / BLOCK_TILE_K; ko++) {
        if (ko % (NUM_CODEBOOK_LOAD_AT_ONCE * 2) == 0) {
            // Load codebook
            load_codebook(codebook_buf, _codebook, ko);
        }
        loadShmemA(A_buf, _input, M, K, ko);
        // Wait until codebook is loaded, then dequant B to shmem
        // asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        dequantToShmemB_EntryCentric(B_buf, _w, codebook_buf, _codebook, K, N, ko);
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A_buf, ki);
            loadFragB_mma(B_frags, B_buf, ki);
            // dequantToRegB(B_frags, _w, codebook_buf, _codebook, K, N, ko, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
    }

    storeC(_o, C_frags, M, N);
}

// Split to 4 x (4096,1024,4096)
__global__ void gptvq_gemm_kernel_outter_product(
    half* _input,
    uint8_t* _w,
    half* _codebook,
    half* _o,
    int M, int N, int K
)
{
    extern __shared__ uint8_t shmem[];
    half *A_buf = reinterpret_cast<half*>(shmem);
    half *B_buf = reinterpret_cast<half*>(shmem + 8192);
    half *codebook_buf = reinterpret_cast<half*>(shmem + 16384);

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};

    for (int ko = 0; ko < K / BLOCK_TILE_K; ko++) {
        if (ko % (NUM_CODEBOOK_LOAD_AT_ONCE * 2) == 0) {
            load_codebook_outter(codebook_buf, _codebook, blockIdx.z * (SPLIT_K / BLOCK_TILE_K) + ko);
        }
        loadShmemA(A_buf, _input, M, 4096, blockIdx.z * (SPLIT_K / BLOCK_TILE_K) + ko);

        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        dequantToShmemB_EntryCentric(B_buf, _w, codebook_buf, _codebook, 4096, N, blockIdx.z * (SPLIT_K / BLOCK_TILE_K) + ko);
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A_buf, ki);
            loadFragB_mma(B_frags, B_buf, ki);

            // dequantToRegB(B_frags, _w, codebook_buf, _codebook, 4096, N, blockIdx.z * (SPLIT_K / BLOCK_TILE_K) + ko, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }        
    }
    storeC(&_o[blockIdx.z * M * N], C_frags, M, N);
}

torch::Tensor gptvq_gemm(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook
)
{
#if PROFILING == 1
    const int wmup = 50;
    const int iter = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);
#endif
    // cudaFuncSetAttribute(gptvq_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
    cudaFuncSetAttribute(gptvq_gemm_kernel_outter_product, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE_OUTTER);

    auto seq_len = input.size(0);
    auto hidden_dim = input.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({seq_len, hidden_dim}, 0, options);

    uint8_t *h_dummy_w, *d_dummy_w;
    h_dummy_w = new uint8_t[hidden_dim * hidden_dim / RATIO];
    fill_matrix(h_dummy_w, hidden_dim * hidden_dim / RATIO);
    cudaMalloc(reinterpret_cast<void**>(&d_dummy_w), sizeof(uint8_t) * hidden_dim * hidden_dim / RATIO);
    cudaMemcpy(reinterpret_cast<void*>(d_dummy_w), h_dummy_w, sizeof(uint8_t) * hidden_dim * hidden_dim / RATIO, cudaMemcpyHostToDevice);
    
    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>());
    half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    half* o_ptr_outter;
    cudaMalloc(reinterpret_cast<void**>(&o_ptr_outter), (hidden_dim / SPLIT_K) * seq_len * hidden_dim * sizeof(half));

    dim3 grid(seq_len / BLOCK_TILE_M, hidden_dim / BLOCK_TILE_N);
    dim3 grid_outter(seq_len / BLOCK_TILE_M, hidden_dim / BLOCK_TILE_N, hidden_dim / SPLIT_K);
    dim3 block(BLOCK_SIZE);

#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        gptvq_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr,
            d_dummy_w,
            codebook_ptr,
            o_ptr,
            seq_len,
            hidden_dim,
            hidden_dim
        );
        // gptvq_gemm_kernel_outter_product<<<grid_outter, block, MAX_SHARED_MEMORY_USAGE_OUTTER>>>(
        //     input_ptr,
        //     d_dummy_w,
        //     codebook_ptr,
        //     o_ptr_outter,
        //     seq_len,
        //     hidden_dim,
        //     SPLIT_K
        // );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
    gptvq_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
        input_ptr,
        d_dummy_w,
        codebook_ptr,
        o_ptr,
        seq_len,
        hidden_dim,
        hidden_dim
    );
        // gptvq_gemm_kernel_outter_product<<<grid_outter, block, MAX_SHARED_MEMORY_USAGE_OUTTER>>>(
        //     input_ptr,
        //     d_dummy_w,
        //     codebook_ptr,
        //     o_ptr_outter,
        //     seq_len,
        //     hidden_dim,
        //     SPLIT_K
        // );
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