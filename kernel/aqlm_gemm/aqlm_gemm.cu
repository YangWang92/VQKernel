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
#define ENTRY 65536
#define RATIO 8
#define RESIDUAL 2
#define ENTRY_CENTRIC 0
#define HOT 256

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

// #define MAX_SHARED_MEMORY_USAGE 16384
// #define MAX_SHARED_MEMORY_USAGE 24576
#define MAX_SHARED_MEMORY_USAGE 24576
// #define MAX_SHARED_MEMORY_USAGE 81920


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
    std::exponential_distribution<float> exp_dist(16);
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

__device__ void load_codebook(half* shmem, half* codebook) {
    // Load 0~1023 entries emprically!
    // Every thread load 8 row, 16*2 per row, ::L256 prefetch.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        asm volatile (
            "cp.async.ca.shared.global.L2::256B [%0], [%1], 16;\n"
            "cp.async.ca.shared.global          [%2], [%3], 16;\n"
            :
            : "r"(shmem_uint32_t(&shmem[(i * BLOCK_SIZE + threadIdx.x) * (RATIO * RESIDUAL) + 0 * RATIO])),
              "l"(&codebook[(i * BLOCK_SIZE + threadIdx.x) * (RATIO * RESIDUAL) + 0 * RATIO])
              "r"(shmem_uint32_t(&shmem[(i * BLOCK_SIZE + threadIdx.x) * (RATIO * RESIDUAL) + 1 * RATIO])),
              "l"(&codebook[(i * BLOCK_SIZE + threadIdx.x) * (RATIO * RESIDUAL) + 1 * RATIO])
        );
    }
    // for (int row = 0; row < 4; row++) {
    //     for (int col = 0; col < 8; col++) {
    //         asm volatile (
    //             "cp.async.ca.shared.global [%0], [%1], 4;\n"
    //             :
    //             : "r"(shmem_uint32_t(&shmem[row * 2048 + col * 256 + threadIdx.x * 2])),
    //               "l"(&codebook[((col % 4) * 128 + threadIdx.x) * 16 + (col / 4) * 8 + row * 2])
    //         );
    //     }
    // }
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

__device__ void element_wise_add(
    half* A,
    half* B,
    int len
)
{
    half areg[8];
    *(uint4*)(&areg[0]) = *(uint4*)(A);
    half breg[8];
    *(uint4*)(&breg[0]) = *(uint4*)(B);
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        *(__half2*)(&areg[i * 2]) = __hadd2(*(__half2*)(&areg[i * 2]), *(__half2*)(&breg[i * 2]));
    }
    *(uint4*)(A) = *(uint4*)(&areg[0]);
}

__device__ void element_wise_add_(
    half* A,
    half* B,
    int len
)
{
    half areg[8];
    *(uint4*)(&areg[0]) = *(uint4*)(A);
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        *(__half2*)(&areg[i * 2]) = __hadd2(*(__half2*)(&areg[i * 2]), *(__half2*)(&B[i * 2]));
    }
    *(uint4*)(A) = *(uint4*)(&areg[0]);
}

__device__ void element_wise_add_reg(
    half* A,
    half* B,
    int len
)
{
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        *(__half2*)(&A[i * 2]) = __hadd2(*(__half2*)(&A[i * 2]), *(__half2*)(&B[i * 2]));
    }
}

__device__ void dequantToShmemB(half* shmem, uint16_t* B_q, half* codebook, half* codebook_buf, int k, int n, int ko) {
    for (int i = 0; i < 32; i++) shmem[i * 128 + threadIdx.x] = __float2half(0.0);
    uint8_t mask = 0;
    uint16_t indices[8];
    *(uint64_t*)(&indices[0]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K) * (n / (RATIO / RESIDUAL)) + blockIdx.y * (BLOCK_TILE_N / (RATIO / RESIDUAL)) + (threadIdx.x / 4) * (n / (RATIO / RESIDUAL)) + (threadIdx.x % 4) * 4 +   0]);  // RQ0
    *(uint64_t*)(&indices[4]) = *(uint64_t*)(&B_q[(ko * BLOCK_TILE_K) * (n / (RATIO / RESIDUAL)) + blockIdx.y * (BLOCK_TILE_N / (RATIO / RESIDUAL)) + (threadIdx.x / 4) * (n / (RATIO / RESIDUAL)) + (threadIdx.x % 4) * 4 + 512]);  // RQ1
    half entry[RATIO];
    for (uint16_t e = 0; e < ENTRY_CENTRIC; e++) {
        // RQ0
        *(uint4*)(&entry[0]) = *(uint4*)(&codebook_buf[((uint32_t) e) * (RATIO * RESIDUAL) + 0 * RATIO]);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (e == indices[i]) {
                // element_wise_add_(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], entry, RATIO);
                mask |= (0x1 << i);
            }
        }
        // RQ1
        *(uint4*)(&entry[0]) = *(uint4*)(&codebook[((uint32_t) e) * (RATIO * RESIDUAL) + 1 * RATIO]);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (e == indices[i + 4]) {
                element_wise_add_(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], entry, RATIO);
                mask |= (0x1 << (i + 4));
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        if (mask & (0x1 << i)) continue;
        else {
            if ((uint32_t) indices[i] < (uint32_t) HOT) {
                *(uint4*)(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8]) = *(uint4*)(&codebook_buf[((uint32_t) indices[i]) * (RATIO * RESIDUAL) + 0 * RATIO]);
                // element_wise_add(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], &codebook_buf[((uint32_t) indices[i]) * (RATIO * RESIDUAL) + 0 * RATIO], RATIO);
                // element_wise_add_(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], &codebook_buf[((uint32_t) indices[i]) * 2 + 0 * 1024], RATIO);
            }
            else {    
                element_wise_add(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], &codebook[((uint32_t) indices[i]) * (RATIO * RESIDUAL) + 0 * RATIO], RATIO);
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        if (mask & (0x1 << (i + 4))) continue;
        else {
            if ((uint32_t) indices[i + 4] < (uint32_t) HOT) {
                element_wise_add(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], &codebook_buf[((uint32_t) indices[i + 4]) * (RATIO * RESIDUAL) + 1 * RATIO], RATIO);
                // element_wise_add_(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], &codebook_buf[((uint32_t) indices[i + 4]) * 2 + 1 * 1024], RATIO);
            }
            else {
                element_wise_add(&shmem[(threadIdx.x / 64) * (8 * 16 * 16) + (threadIdx.x % 4) * (2 * 16 * 16) + (i / 2) * (16 * 16) + ((threadIdx.x % 64) / 4) * 16 + (i % 2) * 8], &codebook[((uint32_t) indices[i + 4]) * (RATIO * RESIDUAL) + 1 * RATIO], RATIO);
            }
        }
    }
}

__device__ void dequantToRegB(uint32_t* frag, uint16_t* B_q, half* codebook, half* codebook_buf, int k, int n, int ko, int ki) {
    // Every warp do 16x64
    // Every time do 16x16
    uint32_t warp_id_x = (threadIdx.x / WARP_SIZE) / 2;
    uint32_t warp_id_y = (threadIdx.x / WARP_SIZE) % 2;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t lane_id_mod2 = lane_id % 2;
    uint32_t lane_id_mod4 = (lane_id % 4) / 2;
    uint8_t ids[8], _ids[8];
    // RQ0
    *(uint32_t*)(&ids[0]) = *(uint32_t*)(&B_q[(ko * BLOCK_TILE_K + ki * WARP_TILE_K) * (n / (RATIO / RESIDUAL)) + 0 * (n / RATIO) + blockIdx.y * (BLOCK_TILE_N / RATIO) + warp_id_y * (WARP_TILE_N / RATIO) + ((lane_id % 4) / 2) * 8 * (n / (RATIO / RESIDUAL)) + (lane_id / 4) * (n / (RATIO / RESIDUAL)) + (lane_id % 2) * 4]);
    *(uint32_t*)(&ids[4]) = *(uint32_t*)(&B_q[(ko * BLOCK_TILE_K + ki * WARP_TILE_K) * (n / (RATIO / RESIDUAL)) + 1 * (n / RATIO) + blockIdx.y * (BLOCK_TILE_N / RATIO) + warp_id_y * (WARP_TILE_N / RATIO) + ((lane_id % 4) / 2) * 8 * (n / (RATIO / RESIDUAL)) + (lane_id / 4) * (n / (RATIO / RESIDUAL)) + (lane_id % 2) * 4]);
    if (threadIdx.x & 0x1) {
        uint8_t tmp = ids[0];
        ids[0] = ids[1];
        ids[1] = tmp;

        tmp = ids[2];
        ids[2] = ids[3];
        ids[3] = tmp;

        tmp = ids[4];
        ids[4] = ids[5];
        ids[5] = tmp;

        tmp = ids[6];
        ids[6] = ids[7];
        ids[7] = tmp;
    }
    ids[1] = __shfl_xor_sync(0xffffffff, ids[1], 0x1);
    ids[3] = __shfl_xor_sync(0xffffffff, ids[3], 0x1);
    ids[5] = __shfl_xor_sync(0xffffffff, ids[5], 0x1);
    ids[7] = __shfl_xor_sync(0xffffffff, ids[7], 0x1);
    uint8_t tmp = ids[1];
    ids[1] = ids[2];
    ids[2] = tmp;
    tmp = ids[5];
    ids[5] = ids[6];
    ids[6] = tmp;
    if (threadIdx.x & 0x1) {
        uint16_t tmp = *(uint16_t*)(&ids[0]);
        *(uint16_t*)(&ids[0]) = *(uint16_t*)(&ids[2]);
        *(uint16_t*)(&ids[2]) = tmp;
        tmp = *(uint16_t*)(&ids[4]);
        *(uint16_t*)(&ids[4]) = *(uint16_t*)(&ids[6]);
        *(uint16_t*)(&ids[6]) = tmp;
    }
    // ids[0] = _ids[0];
    // ids[1] = _ids[2];
    // ids[2] = _ids[4];
    // ids[3] = _ids[6];
    // ids[4] = _ids[1];
    // ids[5] = _ids[3];
    // ids[6] = _ids[5];
    // ids[7] = _ids[7];
    // Try xchg
    // uint8_t tmp = ids[1];
    // ids[1] = ids[2];
    // ids[2] = tmp;
    // uint8_t tmp = ids[5];
    // ids[5] = ids[6];
    // ids[6] = tmp;

    // #pragma unroll
    // for (int i = 0; i < 4; i++) {
    //     *(uint4*)(&frag[i * 4]) = *(uint4*)(&codebook[((uint32_t) ids[i]) * 16 + 0]);
    // }

    // #pragma unroll
    for (int i = 0; i < 4; i++) {
        *(uint4*)(&frag[i * 4]) = *(uint4*)(&codebook[((uint32_t) ids[i]) * 16 + 0]);
        // *(uint4*)(&frag[i * 4]) = *(uint4*)(&codebook[((uint32_t) ids[i]) * 16 + 8]);
        element_wise_add_reg((half*)(&frag[i * 4]), &codebook[((uint32_t) ids[i + 4]) * 16 + 8], RATIO);
        if (threadIdx.x % 4 == 0) {
            uint32_t xchg = frag[i * 4];
            frag[i * 4] = frag[i * 4 + 1];
            frag[i * 4 + 1] = xchg;
        }
        if (threadIdx.x % 4 == 2) {
            uint32_t xchg = frag[i * 4 + 2];
            frag[i * 4 + 2] = frag[i * 4 + 3];
            frag[i * 4 + 3] = xchg;
        }
        if (threadIdx.x & 0x2) {
            uint64_t xchg = *(uint64_t*)(&frag[i * 4 + 2]);
            *(uint64_t*)(&frag[i * 4 + 2]) = *(uint64_t*)(&frag[i * 4 + 0]);
            *(uint64_t*)(&frag[i * 4 + 0]) = xchg;
        }
        *(uint32_t*)(&frag[i * 4 + 0]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[i * 4 + 0]), 1);

        if (threadIdx.x & 0x1) {
            uint32_t xchg = frag[i * 4 + 2];
            frag[i * 4 + 2] = frag[i * 4 + 3];
            frag[i * 4 + 3] = xchg;
        }
        *(uint32_t*)(&frag[i * 4 + 2]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[i * 4 + 2]), 2);
        *(uint32_t*)(&frag[i * 4 + 3]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[i * 4 + 3]), 3);
        // *(uint32_t*)(&frag[i * 4 + lane_id_mod4 * 2]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[i * 4 + lane_id_mod4 * 2]), 1);
        // *(uint32_t*)(&frag[i * 4 + intra_group ^ 2]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[i * 4 + intra_group ^ 2]), 2);
        // *(uint32_t*)(&frag[i * 4 + intra_group ^ 3]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&frag[i * 4 + intra_group ^ 3]), 3);
        if (threadIdx.x % 4 == 0) {
            uint32_t xchg = frag[i * 4 + 0];
            frag[i * 4 + 0] = frag[i * 4 + 1];
            frag[i * 4 + 1] = xchg;
        }
        if (threadIdx.x % 4 == 1) {
            uint32_t xchg = frag[i * 4 + 2];
            frag[i * 4 + 2] = frag[i * 4 + 3];
            frag[i * 4 + 3] = xchg;
        }
        if (threadIdx.x % 4 == 2) {
            uint32_t xchg = frag[i * 4 + 0];
            frag[i * 4 + 0] = frag[i * 4 + 1];
            frag[i * 4 + 1] = xchg;
            uint64_t xchg64 = *(uint64_t*)(&frag[i * 4 + 2]);
            *(uint64_t*)(&frag[i * 4 + 2]) = *(uint64_t*)(&frag[i * 4 + 0]);
            *(uint64_t*)(&frag[i * 4 + 0]) = xchg64;            
        }
        if (threadIdx.x % 4 == 3) {
            uint32_t xchg = frag[i * 4 + 2];
            frag[i * 4 + 2] = frag[i * 4 + 3];
            frag[i * 4 + 3] = xchg;
            uint64_t xchg64 = *(uint64_t*)(&frag[i * 4 + 2]);
            *(uint64_t*)(&frag[i * 4 + 2]) = *(uint64_t*)(&frag[i * 4 + 0]);
            *(uint64_t*)(&frag[i * 4 + 0]) = xchg64;               
        }
    }
}

// Vanilla, codebook --x--> shared memory
//          Residual no split no reduction
__global__ void aqlm_gemm_kernel(
    half* _input,
    uint16_t* _w,
    half* _codebook,
    half* _o,
    int M, int N, int K
)
{
    extern __shared__ uint8_t shmem[];
    half *A_buf = reinterpret_cast<half*>(shmem);
    half *B_buf = reinterpret_cast<half*>(shmem + BLOCK_TILE_M * BLOCK_TILE_K * sizeof(half));
    // half *C_buf = reinterpret_cast<half*>(shmem);
    half *codebook_buf = reinterpret_cast<half*>(shmem + 16384);
    // half *codebook_buf = reinterpret_cast<half*>(shmem);

    load_codebook(codebook_buf, _codebook);
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    uint32_t A_frags[16];
    uint32_t B_frags[16];
    uint32_t C_frags[64] = {0};

    for (int ko = 0; ko < K / BLOCK_TILE_K; ko++) {
        loadShmemA(A_buf, _input, M, K, ko);
        // Fill B_buf with entry[0]
        // for (int x = 0; x < 4; x++) {
        //     *(uint4*)(&B_buf[(threadIdx.x / 16) * 128 + (threadIdx.x % 16) * 8]) = 
        //     *(uint4*)(&codebook_buf[0]);
        // }
        // __syncthreads();
        // dequantToShmemB(B_buf, _w, _codebook, codebook_buf, K, N, ko);
        asm volatile("cp.async.wait_all;\n"::);
        __syncthreads();
        for (int ki = 0; ki < BLOCK_TILE_K / WARP_TILE_K; ki++) {
            loadFragA_mma(A_frags, A_buf, ki);
            // loadFragB_mma(B_frags, B_buf, ki);
            dequantToRegB(B_frags, _w, _codebook, codebook_buf, K, N, ko, ki);
            for (int mm = 0; mm < WARP_TILE_M / WMMA_TILE_M; mm++) {
                for (int nn = 0; nn < WARP_TILE_N / WMMA_TILE_N; nn++) {
                    compute_mma(&A_frags[mm * 4], &B_frags[nn * 4], &C_frags[(mm * 4 + nn) * 4]);
                }
            }
        }
    }
    // storeFragC_mma(C_buf, C_frags);
    // __syncthreads();
    // storeShmemC(_o, C_buf, M, N);  
    storeC(_o, C_frags, M, N);
}

// For rapid debugging
// int main(int argc, char** argv) {
//     cudaEvent_t st, ed;
//     cudaEventCreate(&st, NULL);
//     cudaEventCreate(&ed, NULL);

//     cudaFuncSetAttribute(aqlm_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);
//     half* h_input, *d_input;
//     half* h_codebook, *d_codebook;
//     uint16_t* h_w, *d_w;
//     half* h_o, *d_o;

//     cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 4096 * 4096);
//     cudaMalloc(reinterpret_cast<void**>(&d_codebook), sizeof(half) * 65536 * 8 * 2);
//     cudaMalloc(reinterpret_cast<void**>(&d_w), sizeof(uint16_t) * 4096 * 1024);
//     cudaMalloc(reinterpret_cast<void**>(&d_o), sizeof(half) * 4096 * 4096);
//     dim3 grid(4096 / 128, 4096 / 128);
//     cudaEventRecord(st);
//     for (int i = 0; i < 100; i++) {
//         aqlm_gemm_kernel<<<grid, 128, MAX_SHARED_MEMORY_USAGE>>>(
//             d_input,
//             d_w,
//             d_codebook,
//             d_o,
//             4096,4096,4096
//         );
//     }
//     cudaEventRecord(ed);
//     cudaEventSynchronize(ed);
//     float ms;
//     cudaEventElapsedTime(&ms, st, ed);
//     std::cout << ms / (1.0 * 100) << std::endl;
// }

torch::Tensor aqlm_gemm(
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
    cudaFuncSetAttribute(aqlm_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);

    auto seq_len = input.size(0);
    auto hidden_dim = input.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({seq_len, hidden_dim}, 0, options);

    uint16_t *h_dummy_w, *d_dummy_w;
    uint16_t *h_need_entry_ids, *d_need_entry_ids;
    h_dummy_w = new uint16_t [hidden_dim * hidden_dim / (RATIO / RESIDUAL)];
    h_need_entry_ids = new uint16_t[(seq_len / BLOCK_TILE_M) * (hidden_dim / BLOCK_TILE_N) * 32 * (128 / 8) * 2];
    for (int i = 0; i < hidden_dim; i+=32) {
        for (int j = 0; j < hidden_dim / RATIO; j+=16) {

        }
    }
    fill_matrix(h_dummy_w, hidden_dim * hidden_dim / (RATIO / RESIDUAL));
    cudaMalloc(reinterpret_cast<void**>(&d_dummy_w), sizeof(uint16_t) * hidden_dim * hidden_dim / (RATIO / RESIDUAL));
    cudaMemcpy(reinterpret_cast<void*>(d_dummy_w), h_dummy_w, sizeof(uint16_t) * hidden_dim * hidden_dim / (RATIO / RESIDUAL), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d_need_entry_ids), sizeof(uint16_t) * (seq_len / BLOCK_TILE_M) * (hidden_dim / BLOCK_TILE_N) * 32 * (128 / 8) * 2);
    cudaMemcpy(reinterpret_cast<void**>(d_need_entry_ids), h_need_entry_ids, sizeof(uint16_t) * (seq_len / BLOCK_TILE_M) * (hidden_dim / BLOCK_TILE_N) * 32 * (128 / 8) * 2, cudaMemcpyHostToDevice);

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    // uint16_t* w_ptr = reinterpret_cast<uint16_t*>(w.data_ptr<uint16_t>());
    half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    dim3 grid(seq_len / BLOCK_TILE_M, hidden_dim / BLOCK_TILE_N);
    dim3 block(BLOCK_SIZE);

#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        aqlm_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
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
    aqlm_gemm_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
        input_ptr,
        d_dummy_w,
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