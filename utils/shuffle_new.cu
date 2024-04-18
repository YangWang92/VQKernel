#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "stdio.h"
#include <iostream>
#include <random>

#define SHUFFLE 0

#define WARP_SIZE 32
#define WARP_NUM 8
#define BLOCK_SIZE (WARP_SIZE * WARP_NUM)

#define ENTRY 256

#define M 2048
#define N 2048

#define COMPRESSION_RATIO 4
#define SUB_SPACE_PER_BLOCK 2
template <typename T>
void random_array_gaussian(T* array, int len, float mean = 0.0, float std = 1.0) {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<float> dist{mean, std};
    for (int i = 0; i < len; i++) {
        if (std::is_same<T, half>::value) {
            array[i] = __float2half(dist(rng));
        }
        else if (std::is_same<T, uint8_t>::value) {
            array[i] = (uint8_t) (((uint32_t) (dist(rng) * 1000)) % 256);
        }
    }
}

__global__ void decode_shuffle(uint8_t* _in, half* _codebook, half* _out) {

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    // const uint32_t pred = ((lane_id % 8 == 1) || (lane_id % 8 == 3) || (lane_id % 8 == 4) || (lane_id % 8 == 6)) ? 1 : 0;
    // const uint32_t pred = (lane_id < 32);
    // const uint32_t one = 0x1;
    // asm volatile(
    //     ".reg .pred %pr_global;\n"
    //     "setp.eq.u32 %pr_global, %0, %1;\n"
    //     :
    //     : "r"(pred), "r"(one)
    // );
    // Load Codebook
    __shared__ half codebook_buf[SUB_SPACE_PER_BLOCK * ENTRY * COMPRESSION_RATIO];
    const uint32_t codebook_offset = blockIdx.x * SUB_SPACE_PER_BLOCK * ENTRY * COMPRESSION_RATIO;
    *(uint4*)(&codebook_buf[threadIdx.x * 8]) = *(uint4*)(&_codebook[codebook_offset + threadIdx.x * 8]);

    // Load compressed input
    __shared__ uint8_t in_buf[M * SUB_SPACE_PER_BLOCK];
    for (int tid = threadIdx.x; tid < M; tid += BLOCK_SIZE) {
        *(uint16_t*)(&in_buf[tid * SUB_SPACE_PER_BLOCK]) = *(uint16_t*)(&_in[tid * (N / COMPRESSION_RATIO) + blockIdx.x * SUB_SPACE_PER_BLOCK]);
    }
    __syncthreads();

#if SHUFFLE == 0
    __shared__ half exchange_buf[WARP_NUM][16][8];
    half data[4];
    for (int m = 0; m < M; m += 128) {
        *(uint64_t*)(&exchange_buf[warp_id][lane_id / 2][(lane_id % 2) * COMPRESSION_RATIO]) = 
        *(uint64_t*)(&codebook_buf[(lane_id % 2) * ENTRY * COMPRESSION_RATIO + (uint32_t) in_buf[(m + threadIdx.x / 2) * SUB_SPACE_PER_BLOCK + threadIdx.x % 2] * COMPRESSION_RATIO]);
        __syncthreads();
        *(uint32_t*)(&data[0]) = exchange_buf[warp_id][0 + lane_id / 4][(lane_id % 4) * 2];
        *(uint32_t*)(&data[2]) = exchange_buf[warp_id][8 + lane_id / 4][(lane_id % 4) * 2];

        *(uint64_t*)(&_out[(m + threadIdx.x / 2) * N + blockIdx.x * SUB_SPACE_PER_BLOCK * COMPRESSION_RATIO + (threadIdx.x % 2) * COMPRESSION_RATIO]) = *(uint64_t*)(&data[0]);
    }

#else
    half data[4];
    for (int m = 0; m < M; m += 128) {
        *(uint64_t*)(&data[0]) = *(uint64_t*)(&codebook_buf[(lane_id % 2) * ENTRY * COMPRESSION_RATIO + (uint32_t) in_buf[m * SUB_SPACE_PER_BLOCK + warp_id * 16 * SUB_SPACE_PER_BLOCK + (lane_id % 2) * 8 * SUB_SPACE_PER_BLOCK + (lane_id / 4) * SUB_SPACE_PER_BLOCK + (lane_id % 4) / 2] * COMPRESSION_RATIO]);
        *(uint32_t*)(&data[2 * ((lane_id % 2) + 1) % 2]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&data[2 * ((lane_id % 2) + 1) % 2]), 1);
        // *(uint64_t*)(&data[0]) = *(uint64_t*)(&codebook_buf[(lane_id % 2) * ENTRY * COMPRESSION_RATIO + (uint32_t) in_buf[m * SUB_SPACE_PER_BLOCK + warp_id * 16 * SUB_SPACE_PER_BLOCK + ((lane_id / 4) % 2) * 8 * SUB_SPACE_PER_BLOCK + (lane_id / 8) * 2 * SUB_SPACE_PER_BLOCK + (lane_id % 2) * SUB_SPACE_PER_BLOCK + ((lane_id % 4) / 2)] * COMPRESSION_RATIO]);
        // *(uint32_t*)(&data[2 * (((lane_id + 4) % 8) / 4)]) = __shfl_xor_sync(0xffffffff, *(uint32_t*)(&data[2 * (((lane_id + 4) % 8) / 4)]), 4);

        // // if (lane_id % 8 == 1 || lane_id % 8 == 3 || lane_id % 8 == 4 || lane_id % 8 == 6) {
        //     // *(uint64_t*)(&data[0]) = __shfl_xor_sync(0xffffffff, *(uint64_t*)(&data[0]), 5);
        // // }
        // asm volatile (
        //     "{.reg .pred %pr<2>;\n"
        //     "@%pr_global shfl.sync.bfly.b32 %0|%pr0, %2, 0x5, 0x1f, 0xffffffff;\n"
        //     "@%pr_global shfl.sync.bfly.b32 %1|%pr1, %3, 0x5, 0x1f, 0xffffffff;}\n"
        //     : "=r"(*(uint32_t*)(&data[0])), "=r"(*(uint32_t*)(&data[2]))
        //     : "r"(*(uint32_t*)(&data[0])), "r"(*(uint32_t*)(&data[2])), "r"(pred), "r"(one)
        // );
        *(uint64_t*)(&_out[(m + threadIdx.x / 2) * N + blockIdx.x * SUB_SPACE_PER_BLOCK * COMPRESSION_RATIO + (threadIdx.x % 2) * COMPRESSION_RATIO]) = *(uint64_t*)(&data[0]);
    
    }
#endif
}

int main(int argc, char** argv) {
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);

    uint8_t *h_in, *d_in;
    half *h_out, *d_out;
    half *h_codebook, *d_codebook;

    h_in = new uint8_t [M * N / COMPRESSION_RATIO];
    h_out = new half [M * N];
    h_codebook = new half [(N / COMPRESSION_RATIO) * ENTRY * COMPRESSION_RATIO];

    random_array_gaussian(h_in, M * N / COMPRESSION_RATIO);
    random_array_gaussian(h_codebook, (N / COMPRESSION_RATIO) * ENTRY * COMPRESSION_RATIO);

    cudaMalloc(reinterpret_cast<void**>(&d_in), sizeof(uint8_t) * M * N / COMPRESSION_RATIO);
    cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_codebook), sizeof(half) * (N / COMPRESSION_RATIO) * ENTRY * COMPRESSION_RATIO);

    cudaMemcpy(reinterpret_cast<void*>(d_in), h_in, sizeof(uint8_t) * M * N / COMPRESSION_RATIO, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_codebook), h_codebook, sizeof(half) * (N / COMPRESSION_RATIO) * ENTRY * COMPRESSION_RATIO, cudaMemcpyHostToDevice);
    for (int i = 0; i < 500; i++) {
        decode_shuffle<<<N / (SUB_SPACE_PER_BLOCK * COMPRESSION_RATIO), WARP_NUM * WARP_SIZE>>>(
            d_in,
            d_codebook,
            d_out
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < 2000; i++) {
        decode_shuffle<<<N / (SUB_SPACE_PER_BLOCK * COMPRESSION_RATIO), WARP_NUM * WARP_SIZE>>>(
            d_in,
            d_codebook,
            d_out
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << ms / (2000.0) << std::endl;

    cudaMemcpy(h_out, reinterpret_cast<void*>(d_out), sizeof(half) * M * N, cudaMemcpyDeviceToHost);
}