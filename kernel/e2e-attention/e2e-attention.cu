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

#define DIM_PER_BLOCK 32
#define RATIO 4
#define QUANTIZED_DIM_PER_BLOCK (DIM_PER_BLOCK / RATIO)

#define WARP_SIZE 32
#define WARP_NUM 32
#define BLOCK_SIZE (WARP_SIZE * WARP_NUM)


#define HEAD_NUM 32
#define HEAD_DIM 128
#define ENTRY 256

__device__ __forceinline__ uint32_t attn_shmem_uint32_t(const void* shmem_ptr) {
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

__device__ void load_codebook(
    half* k_codebook_buf, // 8 Rows, each rows have 256 * 4 half elements.
    half* v_codebook_buf,
    half* k_codebook,
    half* v_codebook
)
{
    // Every thread load one 16 byte.
    asm volatile (
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        "cp.async.ca.shared.global [%2], [%3], 16;\n"
        :
        : "r"(attn_shmem_uint32_t(&k_codebook_buf[(threadIdx.x / 128) * 1024 + (threadIdx.x % 128) * 8])), 
          "l"(&k_codebook[(blockIdx.x * 8 + threadIdx.x / 128) * 1024 + (threadIdx.x % 128) * 8]),
          "r"(attn_shmem_uint32_t(&v_codebook_buf[(threadIdx.x / 128) * 1024 + (threadIdx.x % 128) * 8])), 
          "l"(&v_codebook[(blockIdx.x * 8 + threadIdx.x / 128) * 1024 + (threadIdx.x % 128) * 8])
    );
}

__device__ half inner_product_halfx32(
    half* A,
    half* B
)
{
    half2 res = {__float2half(0.0), __float2half(0.0)};
    #pragma unroll
    for (int i = 0; i < 32; i+=2) {
        res = __hadd2(res, __hmul2(*(half2*)(&A[i]), *(half2*)(&B[i])));
    }
    return __hadd(res.x, res.y);
}

/*
    codebook : 2 * 256 * 32 * 2 = 32 kB
 */
__global__ void e2e_attention_decode_kernel(
    half* O,
    half* Q,
    half* K,
    half* V,
    uint8_t* K_CACHE,
    uint8_t* V_CACHE,
    half* K_CODEBOOK,
    half* V_CODEBOOK,
    half* K_CACHE_WINDOW,
    half* V_CACHE_WINDOW,
    int BATCH,
    int HIDDEN,
    int KV_CACHE_ROWS,
    int CNT,
    half* REDUCE
)
{
    __shared__ half k_codebook[8 * ENTRY * RATIO];
    __shared__ half v_codebook[8 * ENTRY * RATIO];
    __shared__ half q_shmem[8 * DIM_PER_BLOCK];
    __shared__ half softmax[32];
    __shared__ half2 o_reduce[32];
    __shared__ half final_scale;
    load_codebook(k_codebook, v_codebook, K_CODEBOOK, V_CODEBOOK);
    asm volatile ("cp.async.wait_all;\n" ::);
    __syncthreads();

    // // [ OK ] Verify codebook load correctness
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 0; i < 8; i++) {
    //         for (int j = 0; j < 32; j++) {
    //             printf("%6.3f%c", __half2float(k_codebook[i * 1024 + j]), (j == 31) ? '\n' : ' ');
    //         }
    //     }
    // }
    
    // Load all batches' q, 8 x 32 in total
    if (threadIdx.x < 32) {
        *(uint4*)(&q_shmem[(threadIdx.x / 4) * DIM_PER_BLOCK + (threadIdx.x % 4) * 8]) = *(uint4*)(&Q[(threadIdx.x / 4) * HIDDEN + blockIdx.x * DIM_PER_BLOCK + (threadIdx.x % 4) * 8]);
    }
    __syncthreads();

    // // [ OK ] Verify q load correctness
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     for (int i = 0; i < 8; i++) {
    //         for (int j = 0; j < 32; j++) {
    //             printf("%6.3f%c", __half2float(q_shmem[i * 32 + j]), (j == 31) ? '\n' : ' ');
    //         }
    //     }
    // }
    uint8_t __align__(8) indices[8]; // 32 / 4 = 8
    half __align__(8) dequantized[DIM_PER_BLOCK];
    for (int bid = 0; bid < BATCH; bid++) {
        half __align__(16) sum[8] = {__float2half(0.0)};
        half scale = __float2half(0.0);
        half res[DIM_PER_BLOCK] = {__float2half(0.0)};
        for (int row = threadIdx.x; row < KV_CACHE_ROWS; row+=blockDim.x) {
            // Load indices
            *(uint64_t*)(&indices[0]) = *(uint64_t*)(&K_CACHE[bid * (KV_CACHE_ROWS * HIDDEN / RATIO) + blockIdx.x * QUANTIZED_DIM_PER_BLOCK + row * (HIDDEN / RATIO)]);
            // [ OK ] Verify k dequantization correctness
            #pragma unroll
            for (int i = 0; i < 8; i++) *(uint64_t*)(&dequantized[i * 4]) = *(uint64_t*)(&k_codebook[i * 1024 + ((uint32_t) indices[i]) * 4]);
            // ----------------------------- Following contents are not verified yet ------------------------------------ //
            half partial_sum = inner_product_halfx32(dequantized, &q_shmem[bid * DIM_PER_BLOCK]);
            atomicAdd(&REDUCE[(blockIdx.x / 4) * 8192 + row], partial_sum);
        }

        for (int row = threadIdx.x; row < KV_CACHE_ROWS; row+=blockDim.x) {
            sum[row / blockDim.x] = hexp(REDUCE[(blockIdx.x / 4) * 8192 + row]);
        }
        #pragma unroll
        for (int i = 0; i < KV_CACHE_ROWS / BLOCK_SIZE; i++) {
            scale = __hadd(scale, sum[i]);
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            scale = __hadd(scale, __shfl_xor_sync(0xffffffff, scale, mask));
        }
        if (threadIdx.x % 32 == 0) softmax[threadIdx.x / 32] = scale;
        __syncthreads();
        if (threadIdx.x < 32) scale = softmax[threadIdx.x];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            scale = __hadd(scale, __shfl_xor_sync(0xffffffff, scale, mask));
        }
        if (threadIdx.x == 0) final_scale = scale;
        #pragma unroll
        for (int i = 0; i < KV_CACHE_ROWS / BLOCK_SIZE; i++) {
            sum[i] = __hdiv(sum[i], final_scale);
        }

        for (int row = threadIdx.x; row < KV_CACHE_ROWS; row+=blockDim.x) {
            *(uint64_t*)(&indices[0]) = *(uint64_t*)(&V_CACHE[bid * (KV_CACHE_ROWS * HIDDEN / RATIO) + blockIdx.x * QUANTIZED_DIM_PER_BLOCK + row * (HIDDEN / RATIO)]);
            #pragma unroll
            for (int i = 0; i < 8; i++) *(uint64_t*)(&dequantized[i * 4]) = *(uint64_t*)(&v_codebook[i * 1024 + ((uint32_t) indices[i]) * 4]);
            for (int d = 0; d < DIM_PER_BLOCK; d++) {
                res[d] = __hadd(res[d], __hmul(sum[row / blockDim.x], dequantized[d]));
            }
        }

        // Every threads now hold a [32] half array to be reduce.
        for (int d = 0; d < DIM_PER_BLOCK; d+=2) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                *(half2*)(&res[d]) = __hadd2(*(half2*)(&res[d]), __shfl_xor_sync(0xffffffff, *(half2*)(&res[d]), mask));
            }
            if (threadIdx.x % 32 == 0) o_reduce[threadIdx.x / 32] = *(half2*)(&res[d]);
            __syncthreads();
            if (threadIdx.x < 32) *(half2*)(&res[d]) = o_reduce[threadIdx.x];
            // 0~31 threads do reduce, get the final results, only thread 0's result is valid.
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                *(half2*)(&res[d]) = __hadd2(*(half2*)(&res[d]), __shfl_xor_sync(0xffffffff, *(half2*)(&res[d]), mask));
            }
        }

        if (threadIdx.x == 0) {
            for (int d = 0; d < DIM_PER_BLOCK; d+=8) {
                *(uint4*)(&O[bid * 1 * 4096 + blockIdx.x * DIM_PER_BLOCK + d]) = *(uint4*)(&res[d]);
            }
        }
    }
}

// Objective:
/*
RTX 4090-01

Batch = 1
Paged_Flash_decoding_kvcache fwd: 131.27 us
Flash_decoding_kvcache fwd: 44.53 us
Paged_Flash_attn_kvcache fwd: 134.55 us
Flash_attn_kvcache fwd: 247.91 us

Batch = 8
Paged_Flash_decoding_kvcache fwd: 576.08 us
Flash_decoding_kvcache fwd: 575.65 us
Paged_Flash_attn_kvcache fwd: 569.11 us
Flash_attn_kvcache fwd: 575.82 us

*/

torch::Tensor e2e_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_codebook,
    torch::Tensor v_codebook,
    torch::Tensor k_cache_window,
    torch::Tensor v_cache_window,
    int cnt
) 
{
#if PROFILING == 1
    const int wmup = 50;
    const int iter = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);
#endif

    auto batch = q.size(0);
    auto hidden = q.size(1);
    auto kv_cache_rows = k.size(1);
    std::cout << batch << " " << hidden << " " << kv_cache_rows << std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({batch, 1, hidden}, 0, options);
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    half *reduce_workspace;
    cudaMalloc(reinterpret_cast<void**>(&reduce_workspace), sizeof(half) * 32 * 8192);

    half* q_ptr = reinterpret_cast<half*>(q.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(k.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(v.data_ptr<at::Half>());
    uint8_t* k_cache_ptr = reinterpret_cast<uint8_t*>(k_cache.data_ptr<uint8_t>());
    uint8_t* v_cache_ptr = reinterpret_cast<uint8_t*>(v_cache.data_ptr<uint8_t>());
    half* k_codebook_ptr = reinterpret_cast<half*>(k_codebook.data_ptr<at::Half>());
    half* v_codebook_ptr = reinterpret_cast<half*>(v_codebook.data_ptr<at::Half>());
    half* k_cache_window_ptr = reinterpret_cast<half*>(k_cache_window.data_ptr<at::Half>());
    half* v_cache_window_ptr = reinterpret_cast<half*>(v_cache_window.data_ptr<at::Half>());
    std::cout << "Launch" << std::endl;
    dim3 grid(hidden / DIM_PER_BLOCK);
    dim3 block(BLOCK_SIZE);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        e2e_attention_decode_kernel<<<grid, block>>>(
            o_ptr,
            q_ptr, k_ptr, v_ptr,
            k_cache_ptr, v_cache_ptr,
            k_codebook_ptr, v_codebook_ptr,
            k_cache_window_ptr, v_cache_window_ptr,
            batch, hidden, kv_cache_rows,
            cnt,
            reduce_workspace
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        e2e_attention_decode_kernel<<<grid, block>>>(
            o_ptr,
            q_ptr, k_ptr, v_ptr,
            k_cache_ptr, v_cache_ptr,
            k_codebook_ptr, v_codebook_ptr,
            k_cache_window_ptr, v_cache_window_ptr,
            batch, hidden, kv_cache_rows,
            cnt,
            reduce_workspace
        );
#if PROFILING == 1
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;
#endif

    return o;
}