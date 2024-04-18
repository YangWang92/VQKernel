#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "stdio.h"
#include <iostream>
#include <random>
#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

#define WARP_SIZE 32
#define COMPUTE_WARP 8
#define LOAD_WARP 8
#define DEQUANT_WARP 8
#define WARP_NUM (COMPUTE_WARP + LOAD_WARP + DEQUANT_WARP)
#define BLOCK_SIZE (WARP_SIZE * WARP_NUM)
#define LOAD_BLOCK_SIZE 512

#define COMPUTE_THREAD_MASK 0x000
#define DEQUANT_THREAD_MASK 0x100
#define LOAD_THREAD_MASK    0x200
//   0 ~ 255 : Compute : 0000000000 ~ 0011111111
// 256 ~ 511 : Dequant : 0100000000 ~ 0111111111
// 512 ~ 767 : Load    : 1000000000 ~ 1011111111

#define M_BLOCK_TILE 64
#define N_BLOCK_TILE 64
#define K_BLOCK_TILE 64

#define M_WARP_TILE 16
#define N_WARP_TILE 32
#define K_WARP_TILE 64

#define M_MMA_TILE 16
#define N_MMA_TILE 8
#define K_MMA_TILE 16

#define _INV_HOT_ENTRY 1
#define DOUBLE_BUFFER 2

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

template <
    size_t INV_HOT_ENTRY,
    size_t PACKED_LOAD_WIDTH,
    size_t B32REG_LIMIT,
    size_t ENABLE_ENTRY_CENTRIC
>
void __global__ vq_gemm_kernel(
    half* _o,
    half* _h,
    uint8_t* _w,
    half* codebook,
    uint32_t _residuals, uint32_t _compression_ratio, uint32_t _entry,
    uint32_t _m, uint32_t _n, uint32_t _k
)
{
    const uint32_t THREAD_PACKED_LOAD_WIDTH = PACKED_LOAD_WIDTH / (8 * sizeof(half));
    const uint32_t thread_type = threadIdx.x >> 8;
    extern __shared__ uint8_t smem[];
    // | Codebook |    H     |    W     |    D     |
    // bar[ 0]: h0.ready,  bar[ 1]: h1.ready,  bar[ 2] : h2.ready
    // bar[ 3]: h0.filled, bar[ 4]: h1.filled, bar[ 5] : h2.filled
    // bar[ 6]: w0.ready,  bar[ 7]: w1.ready
    // bar[ 8]: w0.filled, bar[ 9]: w1.filled
    // bar[10]: d0.ready,  bar[11]: d1.ready
    // bar[12]: d0.filled, bar[13]: d1.filled

    // At most 512 threads load codebook
    barrier* bar = reinterpret_cast<barrier*>(smem);
    half *codebook_shmem = reinterpret_cast<half*>(smem + 14 * sizeof(barrier));
    const uint32_t codebook_to_access = (N_BLOCK_TILE / _compression_ratio) * _entry * _compression_ratio;
    const uint32_t codebook_offset = blockIdx.y * codebook_to_access;
    const uint32_t codebook_to_load = codebook_offset / INV_HOT_ENTRY;
    const uint32_t entry_to_load = _entry / INV_HOT_ENTRY;

    const uint32_t threads_need_to_load = min((uint32_t) LOAD_BLOCK_SIZE, (uint32_t) (codebook_to_load / THREAD_PACKED_LOAD_WIDTH));
    const uint32_t block_packed_load_width = threads_need_to_load * THREAD_PACKED_LOAD_WIDTH;
    const uint32_t load_iterations = max((uint32_t) 1, (uint32_t) (codebook_to_load / block_packed_load_width));
    const uint32_t subspace_stride = _entry * _compression_ratio;
    float subspaces_load_at_once = block_packed_load_width / (entry_to_load * _compression_ratio);
    const uint32_t iterations_for_one_subspace = max((uint32_t) 1, (uint32_t) (1.0 / subspaces_load_at_once));

    auto block = cooperative_groups::this_thread_block();
    if (threadIdx.x < 14) {
        init(bar + threadIdx.x, 256);
    }
    block.sync();
    
    if (threadIdx.x < 256) {
        // Ensure H and D are ready to be filled
        barrier::arrival_token token_h0 = bar[ 0 +  0].arrive();
        barrier::arrival_token token_h1 = bar[ 0 +  1].arrive();
        barrier::arrival_token token_h2 = bar[ 0 +  2].arrive();
        barrier::arrival_token token_d0 = bar[10 +  0].arrive();
        barrier::arrival_token token_d1 = bar[10 +  1].arrive();
        // Compute Warps
        for (int k = 0; k < _k / K_BLOCK_TILE; k++) {
            bar[ 3 + k % 3].arrive_and_wait();  // Wait for H[k % 3] is filled
            bar[12 + k % 2].arrive_and_wait();  // Wait for D[k % 2] is filled
            // Compute H * D
            if (threadIdx.x == 0) printf("[Compute] k = %d, Consuming H[%d] and D[%d]\n", k, k % 3, k % 2);
            barrier::arrival_token token_h_c = bar[ 0 + k % 3].arrive();  // Signal H[k % 3] is ready to be refilled
            barrier::arrival_token token_d_c = bar[10 + k % 2].arrive();  // Signal D[k % 2] is ready to be refilled
        }
    }
    else if (threadIdx.x >= 256 && threadIdx.x < 512) {
        // Ensure W is ready to be filled
        barrier::arrival_token token_w0 = bar[ 6 +  0].arrive();
        barrier::arrival_token token_w1 = bar[ 6 +  1].arrive();
        // Dequant Warps
        for (int k = 0; k < _k / K_BLOCK_TILE; k++) {
            bar[ 8 + k % 2].arrive_and_wait();  // Wait for W[k % 2] is filled
            bar[10 + k % 2].arrive_and_wait();  // Wait for D[k % 2] ready to be filled
            // D[k % 2] <- Dequant(W[k % 2])
            if (threadIdx.x == 256) printf("[Dequant] k = %d, Consuming W[%d], Producing D[%d]\n", k, k % 2, k % 2);
            barrier::arrival_token token_w_d = bar[ 6 + k % 2].arrive();  // Signal W[k % 2] is ready to be refilled
            barrier::arrival_token token_d_d = bar[12 + k % 2].arrive();  // Signal D[k % 2] is filled
        }
    }
    else if (threadIdx.x >= 512 && threadIdx.x < 768) {
        // Load Warps
        for (int k = 0; k < _k / K_BLOCK_TILE; k++) {
            bar[ 0 + k % 3].arrive_and_wait();   // Wait for H[k % 3] ready to be filled
            bar[ 6 + k % 2].arrive_and_wait();   // Wait for W[k % 2] ready to be filled
            // Produce (Load) data to H[k % 3] and W[k % 2] buffer
            if (threadIdx.x == 512) printf("[   Load] k = %d, Producing H[%d] and W[%d]\n", k, k % 3, k % 2);
            barrier::arrival_token token_h_l = bar[ 3 + k % 3].arrive();   // Signal H[k % 3] is filled
            barrier::arrival_token token_w_l = bar[ 8 + k % 2].arrive();   // Signal W[k % 2] is filled
        }
    }
}

int main(int argc, char** argv) {
    half *h_h, *d_h, *h_codebook, *d_codebook, *h_o, *d_o;
    uint8_t *h_w, *d_w;

    int m = 4096, n = 4096, k = 4096, compression_ratio = 4;
    h_h = new half [m * k];
    h_o = new half [m * n];
    h_w = new uint8_t [k * n / compression_ratio];
    h_codebook = new half [(n / compression_ratio) * 256 * compression_ratio];

    random_array_gaussian(h_h, m * k);
    random_array_gaussian(h_w, k * n / compression_ratio);
    random_array_gaussian(h_codebook, (n / compression_ratio) * 256 * compression_ratio);

    cudaMalloc(reinterpret_cast<void**>(&d_h), sizeof(half) * m * k);
    cudaMalloc(reinterpret_cast<void**>(&d_o), sizeof(half) * m * n);
    cudaMalloc(reinterpret_cast<void**>(&d_w), sizeof(uint8_t) * k * n / compression_ratio);
    cudaMalloc(reinterpret_cast<void**>(&d_codebook), sizeof(half) * (n / compression_ratio) * 256 * compression_ratio);

    cudaMemcpy(reinterpret_cast<void*>(d_h), h_h, sizeof(half) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w), h_w, sizeof(uint8_t) * k * n / compression_ratio, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_codebook), h_codebook, sizeof(half) * (n / compression_ratio) * 256 * compression_ratio, cudaMemcpyHostToDevice);
    
    int shared_memory_size = 0;
    /* Codebook */ shared_memory_size += 256 * N_BLOCK_TILE * sizeof(half) / _INV_HOT_ENTRY;
    /* H        */ shared_memory_size += M_BLOCK_TILE * K_BLOCK_TILE * sizeof(half) * (DOUBLE_BUFFER + 1);
    /* W        */ shared_memory_size += K_BLOCK_TILE * (N_BLOCK_TILE / compression_ratio) * sizeof(uint8_t) * DOUBLE_BUFFER;
    /* D        */ shared_memory_size += K_BLOCK_TILE * N_BLOCK_TILE * sizeof(half) * DOUBLE_BUFFER;

    auto kernel = vq_gemm_kernel<_INV_HOT_ENTRY, 128, 16, 0>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size + 1024);

    dim3 grid(m / M_BLOCK_TILE, n / N_BLOCK_TILE);
    kernel<<<1, BLOCK_SIZE, shared_memory_size + 1024>>>(
        d_o,
        d_h,
        d_w,
        d_codebook,
        1, compression_ratio, 256,
        m, n, k
    );

    cudaMemcpy(h_o, reinterpret_cast<void*>(d_o), sizeof(half) * m * n, cudaMemcpyDeviceToHost);
}