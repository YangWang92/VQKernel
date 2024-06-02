#include "cuda.h"
#include "cuda_fp16.h"
#include <iostream>
#include <map>
#include <algorithm>
#include <random>

// Implement a dequantization kernel to verify
// the inefficiency in motivation
// Configuration: 
//      Compression Ratio: 1 : 2
//      Codebook: Column wise, 256 entries
//      Dummy generator: exp(0.1)

#define WARP_SIZE 32
#define WARP_NUM 8
#define BLOCK_SIZE (WARP_SIZE * WARP_NUM)

#define ROWS 4096
#define COLS 4096
#define RATIO 4
#define ENTRY 256

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 5.0);
    std::exponential_distribution<float> exp_dist(0.01); // Match the profiled distribution well.
    // std::exponential_distribution<float> exp_dist(0.1); 
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(norm_dist(rng));
        }      
        else if constexpr(std::is_same<T, uint8_t>::value) {
            mat[i] = static_cast<uint8_t>(exp_dist(rng));
            // mat[i] = static_cast<uint8_t>((norm_dist(rng)) * (1.0 * ENTRY)) % ENTRY;
        } 
    }
}

#define HOT 256

#define SHMEM_SIZE (16 * HOT * 4 * 2)
/*
 4096 * 1024 dequant, every block do 256 rows, 32 cols, 16 subspaces 
 Shared memory: 32768 256
                24576 192
                16384 128
                 8192  64
*/
constexpr uint32_t hot = (uint32_t) HOT;
__global__ void dequant(
    half* out,
    uint8_t* in,
    half* codebook
)
{
    // Load codebook
    // 16 rows, each row 256 * 4 elements, need 128 threads to do packed load.
    extern __shared__ uint8_t shmem[];
    half* codebook_buf = reinterpret_cast<half*>(shmem);
    uint32_t iters = (16 * HOT * 4 / 8) / BLOCK_SIZE;
    uint32_t threads_per_row = HOT * 4 / 8;
    if (threadIdx.x < HOT) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            *(uint64_t*)(&codebook_buf[i * (HOT * 4) + threadIdx.x * 4]) = *(uint64_t*)(&codebook[(blockIdx.y * 16 + i) * ENTRY * 4 + threadIdx.x * 4]);
        }
    }
    __syncthreads();
    uint8_t __align__(16) ids[32];
    *(uint4*)(&ids[ 0]) = *(uint4*)(&in[(blockIdx.x * 256 + threadIdx.x) * (COLS / 4) + blockIdx.x * 32 +  0]);
    *(uint4*)(&ids[16]) = *(uint4*)(&in[(blockIdx.x * 256 + threadIdx.x) * (COLS / 4) + blockIdx.x * 32 + 16]);
    for (int i = 0; i < 32; i++) {
        if ((uint32_t) ids[i] < hot) {
            *(uint64_t*)(&out[(blockIdx.x * 256 + threadIdx.x) * COLS + blockIdx.y * 128 + i * 4]) = 
            *(uint64_t*)(&codebook_buf[(i / 2) * (HOT * 4) + ((uint32_t) ids[i]) * 4]);
        }
        else {
            *(uint64_t*)(&out[(blockIdx.x * 256 + threadIdx.x) * COLS + blockIdx.y * 128 + i * 4]) = 
            *(uint64_t*)(&codebook[(blockIdx.y * 16 + i / 2) * (ENTRY * 4) + ((uint32_t) ids[i]) * 4]);            
        }
    }
}


int main(int argc, char** argv) {
    cudaFuncSetAttribute(dequant, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SIZE);
    const int wmup = 50;
    const int iter = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);

    half *h_original, *d_original;
    half *h_codebook, *d_codebook;
    uint8_t *h_quantized, *d_quantized;

    h_original = new half[ROWS * COLS];
    h_codebook = new half[(COLS / RATIO) * ENTRY * RATIO];
    h_quantized = new uint8_t[ROWS * COLS / RATIO];

    fill_matrix(h_codebook, (COLS / RATIO) * ENTRY * RATIO);
    fill_matrix(h_quantized, ROWS * COLS / RATIO);

    cudaMalloc(reinterpret_cast<void**>(&d_codebook), sizeof(half) * (COLS / RATIO) * ENTRY * RATIO);
    cudaMalloc(reinterpret_cast<void**>(&d_quantized), sizeof(uint8_t) * ROWS * COLS / RATIO);
    cudaMemcpy(reinterpret_cast<void*>(d_codebook), h_codebook, sizeof(half) * (COLS / RATIO) * ENTRY * RATIO, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_quantized), h_quantized, sizeof(uint8_t) * ROWS * COLS / RATIO, cudaMemcpyHostToDevice);

    cudaMalloc(reinterpret_cast<void**>(&d_original), sizeof(half) * ROWS * COLS);
    
    dim3 grid(ROWS / 256, (ROWS / RATIO) / 32);
    dim3 block(BLOCK_SIZE); // 256
    for (int i = 0; i < wmup; i++) {
        dequant<<<grid, block, SHMEM_SIZE>>>(
            d_original,
            d_quantized,
            d_codebook
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
        dequant<<<grid, block, SHMEM_SIZE>>>(
            d_original,
            d_quantized,
            d_codebook
        );
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;

    cudaMemcpy(h_original, reinterpret_cast<void*>(d_original), sizeof(half) * ROWS * COLS, cudaMemcpyDeviceToHost);
    // Inspect h_quantized distribution.
    // std::map <uint32_t, uint32_t> mp;
    // for (int i = 0; i < ROWS * COLS / RATIO; i++) {
    //     uint32_t id = (uint32_t) h_quantized[i];
    //     auto iter = mp.find(id);
    //     if (iter == mp.end()) {
    //         mp[id] = 0;
    //     }
    //     mp[id]++;
    // }
    // std::vector <uint32_t> vec(ENTRY);

    // for (auto&& item : mp) {
    //     vec[item.first] = item.second;
    // }
    // sort(vec.begin(), vec.end(), [](const uint32_t& a, const uint32_t& b) {return a > b; });

    // int accus[6] = {0, 0, 0, 0, 0, 0}, accu = 0, sum = 0;
    // for (int i = 0; i < ENTRY; i++) {
    //     accu += vec[i];
    //     sum += vec[i];
    //     if (i == 8) accus[0] = accu;
    //     if (i == 16) accus[1] = accu;
    //     if (i == 32) accus[2] = accu;
    //     if (i == 64) accus[3] = accu;
    //     if (i == 128) accus[4] = accu;
    //     if (i == 192) accus[5] = accu;
    // }
    // for (int i = 0; i < 6; i++) {
    //     std::cout << (1.0 * accus[i]) / (1.0 * sum) << std::endl;
    // }


    return 0;
}