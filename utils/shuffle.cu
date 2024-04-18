#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "stdio.h"
#define WARP_SIZE 32
#define WARP_NUM 16
#define BLOCK_LOAD_STRIDE (WARP_SIZE * WARP_NUM * 4)

#include <iostream>

#define WARP_SHUFFLE 1

#define COMPRESSION_RATIO 4

#define M 1024
#define N 1024

#define M_BLOCK 128
#define N_BLOCK 128

#define SHMEM_ITER (M_BLOCK * N_BLOCK / 4) / (WARP_SIZE * WARP_NUM)

#define M_WARP_TILE 16
#define N_WARP_TILE 128

#define M_WARP 16
#define N_WARP 8

#define WARP_PER_ROW (N_WARP_TILE / N_WARP)

// Every block shuffle 32x16, every warp shuffle 16x8, 128 blocks

__global__ void shuffle(half* _src, half* _dst) {
    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;    
    half data[4];


    for (int m = 0; m < M_BLOCK; m += M_WARP_TILE) {
        for (int n = 0; n < N_BLOCK; n += N_WARP_TILE) {
            uint32_t begin_idx = blockIdx.x * M_BLOCK * N + blockIdx.y * N_BLOCK + m * N + (warp_id / WARP_PER_ROW) * M_WARP_TILE * N + (warp_id % WARP_PER_ROW) * N_WARP;
            *(uint64_t*)(&data[0]) = *(uint64_t*)(&_src[begin_idx + (((lane_id / 4) % 2) * 8 + (lane_id / 8) * 2 + (lane_id % 2)) * N + ((lane_id % 4) / 2) * COMPRESSION_RATIO]);
        
            *(uint32_t*)(&data[2 * (((lane_id + 4) % 8) / 4)]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&data[2 * (((lane_id + 4) % 8) / 4)]), 4);

            if ((1 << lane_id) & 0x5a5a5a5a) {
                *(uint64_t*)(&data[0]) = __shfl_xor_sync(0xffffffff, *(uint64_t*)(&data[0]), 5);
            }

            *(uint32_t*)(&_dst[begin_idx + m * N + ((lane_id / 4) + 0) * N + (warp_id / WARP_PER_ROW) * M_WARP_TILE * N + n + (warp_id % WARP_PER_ROW) * N_WARP + (lane_id % 4) * 2]) = *(uint32_t*)(&data[0]);
            *(uint32_t*)(&_dst[begin_idx + m * N + ((lane_id / 4) + 8) * N + (warp_id / WARP_PER_ROW) * M_WARP_TILE * N + n + (warp_id % WARP_PER_ROW) * N_WARP + (lane_id % 4) * 2]) = *(uint32_t*)(&data[2]);            
        }
    }    

    // const uint32_t begin_idx = blockIdx.x * M_BLOCK * N + blockIdx.y * N_BLOCK + (warp_id / WARP_PER_ROW) * M_WARP * N + (warp_id % WARP_PER_ROW) * N_WARP;
    // *(uint64_t*)(&data[0]) = *(uint64_t*)(&_src[begin_idx + (((lane_id / 4) % 2) * 8 + (lane_id / 8) * 2 + (lane_id % 2)) * N + ((lane_id % 4) / 2) * COMPRESSION_RATIO]);

    // *(uint32_t*)(&data[2 * (((lane_id + 4) % 8) / 4)]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&data[2 * (((lane_id + 4) % 8) / 4)]), 4);

    // if ((1 << lane_id) & 0x5a5a5a5a) {
    //     *(uint64_t*)(&data[0]) = __shfl_xor_sync(0xffffffff, *(uint64_t*)(&data[0]), 5);
    // }

    // *(uint32_t*)(&_dst[begin_idx + (0 + lane_id / 4) * N + (lane_id % 4) * 2]) = *(uint32_t*)(&data[0]);
    // *(uint32_t*)(&_dst[begin_idx + (8 + lane_id / 4) * N + (lane_id % 4) * 2]) = *(uint32_t*)(&data[2]);

}

__global__ void shuffle_shmem(half* _src, half* _dst) {
    half __shared__ exchange[M_BLOCK * N_BLOCK];

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t begin_idx = blockIdx.x * M_BLOCK * N + blockIdx.y * N_BLOCK;
    
    for (int i = 0; i < SHMEM_ITER; i++) {
        *(uint64_t*)(&exchange[i * BLOCK_LOAD_STRIDE + threadIdx.x * 4]) = 
        *(uint64_t*)(&_src[begin_idx + i * BLOCK_LOAD_STRIDE + threadIdx.x * 4]);
    }
    __syncthreads();
    half data[4];

    for (int m = 0; m < M_BLOCK; m += M_WARP_TILE) {
        for (int n = 0; n < N_BLOCK; n += N_WARP_TILE) {
            *(uint32_t*)(&data[0]) = *(uint32_t*)(&exchange[m * N_BLOCK + ((lane_id / 4) + 0) * N_BLOCK + (warp_id / WARP_PER_ROW) * M_WARP_TILE * N_BLOCK + n + (warp_id % WARP_PER_ROW) * N_WARP + (lane_id % 4) * 2]);
            *(uint32_t*)(&data[2]) = *(uint32_t*)(&exchange[m * N_BLOCK + ((lane_id / 4) + 8) * N_BLOCK + (warp_id / WARP_PER_ROW) * M_WARP_TILE * N_BLOCK + n + (warp_id % WARP_PER_ROW) * N_WARP + (lane_id % 4) * 2]);

            *(uint32_t*)(&_dst[begin_idx + m * N + ((lane_id / 4) + 0) * N + (warp_id / WARP_PER_ROW) * M_WARP_TILE * N + n + (warp_id % WARP_PER_ROW) * N_WARP + (lane_id % 4) * 2]) = *(uint32_t*)(&data[0]);
            *(uint32_t*)(&_dst[begin_idx + m * N + ((lane_id / 4) + 8) * N + (warp_id / WARP_PER_ROW) * M_WARP_TILE * N + n + (warp_id % WARP_PER_ROW) * N_WARP + (lane_id % 4) * 2]) = *(uint32_t*)(&data[2]);
        }
    }

    // Load into shared 
    // *(uint64_t*)(&exchange[(threadIdx.x / 4) * 16 + (threadIdx.x % 4) * 4]) = *(uint64_t*)(&_src[begin_idx + (threadIdx.x / 4) * N + (threadIdx.x % 4) * 4]);
    // __syncthreads();

    // half data[4];
    // *(uint32_t*)(&data[0]) = *(uint32_t*)(&exchange[(warp_id / ROW_WARPS) * M_WARP * N_BLOCK + (warp_id % ROW_WARPS) * N_WARP + (lane_id / 4) * N_BLOCK + (lane_id % 4) * 2]);
    // *(uint32_t*)(&data[2]) = *(uint32_t*)(&exchange[(warp_id / ROW_WARPS) * M_WARP * N_BLOCK + (warp_id % ROW_WARPS) * N_WARP + (lane_id / 4) * N_BLOCK + (lane_id % 4) * 2 + 8 * N_BLOCK]);

    // *(uint32_t*)(&_dst[begin_idx + (0 + lane_id / 4) * N + (lane_id % 4) * 2]) = *(uint32_t*)(&data[0]);
    // *(uint32_t*)(&_dst[begin_idx + (8 + lane_id / 4) * N + (lane_id % 4) * 2]) = *(uint32_t*)(&data[2]);
    // *(uint64_t*)(&_dst[begin_idx + (threadIdx.x / 4) * N + (threadIdx.x % 4) * 4]) = *(uint64_t*)(&data[0]);
}

int main(int argc, char** argv) {
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);

    half *h_src, *h_dst;
    half *d_src, *d_dst;
    h_src = new half[M * N];
    h_dst = new half[M * N];
    cudaMalloc(reinterpret_cast<void**>(&d_src), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_dst), sizeof(half) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_src[i * N + j] = __float2half(1.0 * ((i % M_WARP) * N_WARP + (j % N_WARP)));
        }
    }

    // for (int i = 0; i < M_WARP; i++) {
    //     for (int j = 0; j < N_WARP; j++) {
    //         printf("%05.1f%c", __half2float(h_src[i * N + j]), (j == N_WARP - 1) ? '\n' : ' ');
    //     }
    // }

    cudaMemcpy(reinterpret_cast<void*>(d_src), h_src, sizeof(half) * M * N, cudaMemcpyHostToDevice);

    dim3 grid(M / M_BLOCK, N / N_BLOCK);
    // Warmup
    for (int i = 0; i < 50; i++) {
#if WARP_SHUFFLE == 1
        shuffle<<<grid, WARP_SIZE * WARP_NUM>>>(d_src, d_dst);
#else
        shuffle_shmem<<<grid, WARP_SIZE * WARP_NUM>>>(d_src, d_dst);
#endif
    }
    cudaEventRecord(st);
    for (int i = 0; i < 2500; i++) {
#if WARP_SHUFFLE == 1
        shuffle<<<grid, WARP_SIZE * WARP_NUM>>>(d_src, d_dst);
#else
        shuffle_shmem<<<grid, WARP_SIZE * WARP_NUM>>>(d_src, d_dst);
#endif
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);

    cudaMemcpy(h_dst, reinterpret_cast<void*>(d_dst), sizeof(half) * M * N, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < M_WARP; i++) {
    //     for (int j = 0; j < N_WARP; j++) {
    //         printf("%05.1f%c", __half2float(h_dst[i * N + j]), (j == N_WARP - 1) ? '\n' : ' ');
    //     }
    // }

    std::cout << ms / (2500.0) << std::endl;
    return 0;
}