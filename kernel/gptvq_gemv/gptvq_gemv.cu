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
#define BATCH 1
#define PROFILING 1
#define WARP_SIZE 32
#define WARP_NUM 16
#define BLOCK_SIZE (WARP_SIZE * WARP_NUM)
#define ENTRY 256
#define RATIO 2
#define RESIDUAL 1
#define ENTRY_CENTRIC 0
#define HOT 1
#define HIDDEN_DIM 4096

#define ROW_PER_BLOCK 512    // Need reduction!
#define DIM_PER_BLOCK 16     // Dim on quantized data (4096*2048)
#define GROUP_Y (128 / DIM_PER_BLOCK)
#define GROUP_X (ROW_PER_BLOCK / 64)
#if ENTRY_CENTRIC > 0
#define MAX_SHARED_MEMORY_USAGE (ROW_PER_BLOCK * DIM_PER_BLOCK + GROUP_X * ENTRY * RATIO * 2 / HOT) 
#else
#define MAX_SHARED_MEMORY_USAGE (GROUP_X * ENTRY * RATIO * 2 / HOT) 

#endif
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

template <>
struct packed_vec<half, 1> {
    using Type = uint16_t;
};


__global__ void gptvq_gemv_kernel(
    half* _input,
    uint8_t* _w,
    half* _codebook,
    half* _o,
    int _hidden_dim
)
{
    uint8_t BUFFERED_ENTRY = (uint8_t) ((uint32_t) ENTRY / (uint32_t) HOT);
    extern __shared__ uint8_t shmem[];
    half* codebook_buf = reinterpret_cast<half*>(shmem);
    half* reduce_workspace = reinterpret_cast<half*>(shmem);

    half input_reg[8][ROW_PER_BLOCK / BLOCK_SIZE]; // Assume [] is <= 8
    // Load input into reg
    using INPUT_LOAD_TYPE = typename packed_vec<half, ROW_PER_BLOCK / BLOCK_SIZE>::Type;
    #pragma unroll
    for (int i = 0; i < BATCH; i++) {
        *(INPUT_LOAD_TYPE*)(&input_reg[i][0]) = *(INPUT_LOAD_TYPE*)(&_input[i * HIDDEN_DIM + blockIdx.x * ROW_PER_BLOCK + threadIdx.x * (ROW_PER_BLOCK / BLOCK_SIZE)]);
    }
    // Load codebook
    uint32_t codebook_begin_row = (blockIdx.y / GROUP_Y) * 64 + blockIdx.x * GROUP_X;
    uint32_t thread_to_load = min((uint32_t) BLOCK_SIZE, (uint32_t) ((GROUP_X * ENTRY * RATIO / HOT) / 8));
    uint32_t iters_to_load = ((GROUP_X * ENTRY * RATIO / HOT) / 8) / thread_to_load;
    uint32_t load_cols = (ENTRY * RATIO / HOT) / 8;
    uint32_t load_rows = thread_to_load / load_cols;
    #pragma unroll
    for (int i = 0; i < iters_to_load; i++) {
        if (threadIdx.x < thread_to_load) {
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(shmem_uint32_t(&codebook_buf[(i * load_rows + threadIdx.x / load_cols) * (ENTRY * RATIO / HOT) + (threadIdx.x % load_cols) * 8]))
              "l"(&_codebook[(codebook_begin_row + i * load_rows + threadIdx.x / load_cols) * ENTRY * RATIO + (threadIdx.x % load_cols) * 8])  
            );
        }
    }
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    // Store dequantized data.
    half w_reg[(ROW_PER_BLOCK / BLOCK_SIZE) * DIM_PER_BLOCK * RATIO];
    uint32_t mask[((ROW_PER_BLOCK / BLOCK_SIZE) * DIM_PER_BLOCK + 31) / 32];

#if ENTRY_CENTRIC > 0
    uint8_t* w_buf = reinterpret_cast<uint8_t*>(shmem + sizeof(half) * GROUP_X * ENTRY * RATIO / HOT);
    thread_to_load = min((uint32_t) BLOCK_SIZE, (uint32_t) (ROW_PER_BLOCK * DIM_PER_BLOCK / 16));
    iters_to_load = (ROW_PER_BLOCK * DIM_PER_BLOCK / 16) / thread_to_load;
    load_cols = DIM_PER_BLOCK / 16;
    load_rows = thread_to_load / load_cols;
    #pragma unroll
    for (int i = 0; i < iters_to_load; i++) {
        if (threadIdx.x < thread_to_load) {
            asm volatile ("cp.async.ca.shared.global [%0], [%1], 16;\n"
            :
            : "r"(shmem_uint32_t(&w_buf[(i * load_rows + threadIdx.x / load_cols) * DIM_PER_BLOCK + (threadIdx.x % load_cols) * 16])),
              "l"(&_w[(blockIdx.x * ROW_PER_BLOCK + i * load_rows + threadIdx.x / load_cols) * (HIDDEN_DIM / RATIO) + blockIdx.y * DIM_PER_BLOCK + (threadIdx.x % load_cols) * 16])
            );
        }
    }
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    uint32_t entry;
    for (uint8_t e = 0; e < ENTRY_CENTRIC; e++) {
        entry = *(uint32_t*)(&codebook_buf[(threadIdx.x * (ROW_PER_BLOCK / BLOCK_SIZE) / 64) * (ENTRY * RATIO / HOT) + ((uint32_t) e) * RATIO]);
        #pragma unroll
        for (int i = 0; i < (ROW_PER_BLOCK / BLOCK_SIZE) * DIM_PER_BLOCK; i++) {
            if (e == w_buf[(threadIdx.x * (ROW_PER_BLOCK / BLOCK_SIZE) + (i / DIM_PER_BLOCK)) * DIM_PER_BLOCK + (i % DIM_PER_BLOCK)]) {
                *(uint32_t*)(&w_reg[i * RATIO]) = entry;
                mask[i / 32] |= (0x1 << (i % 32));
            }
        }
    }
#endif
    for (int i = 0; i < (ROW_PER_BLOCK / BLOCK_SIZE) * DIM_PER_BLOCK; i+=4) {
        uint8_t ids[4];
#if ENTRY_CENTRIC > 0
        // We have w stored in shared
        *(uint32_t*)(&ids[0]) = *(uint32_t*)(&w_buf[(threadIdx.x * (ROW_PER_BLOCK / BLOCK_SIZE) + (i / DIM_PER_BLOCK)) * DIM_PER_BLOCK + (i % DIM_PER_BLOCK)]);
#else
        // We need to load from global
        *(uint32_t*)(&ids[0]) = *(uint32_t*)(&_w[(blockIdx.x * ROW_PER_BLOCK + threadIdx.x * (ROW_PER_BLOCK / BLOCK_SIZE) + (i / DIM_PER_BLOCK)) * (HIDDEN_DIM / RATIO) + blockIdx.y * DIM_PER_BLOCK + (i % DIM_PER_BLOCK)]);
#endif
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (!(mask[(i + j) / 32] & (0x1 << ((i + j) % 32)))) {
                if (ids[j] < BUFFERED_ENTRY) {
                    *(uint32_t*)(&w_reg[(i + j) * RATIO]) = *(uint32_t*)(&codebook_buf[(threadIdx.x * (ROW_PER_BLOCK / BLOCK_SIZE) / 64) * (ENTRY * RATIO / HOT) + ((uint32_t) ids[j]) * RATIO]);
                }
                else {
                    *(uint32_t*)(&w_reg[(i + j) * RATIO]) = *(uint32_t*)(&_codebook[((blockIdx.y / GROUP_Y) * 64 + (threadIdx.x * (ROW_PER_BLOCK / BLOCK_SIZE) / 64)) * (ENTRY * RATIO) + ((uint32_t) ids[j]) * RATIO]);
                }
            }
        }
    }
    _o[threadIdx.x] = w_reg[0];
    // for (int b = 0; b < BATCH; b++) {
    //     half partial[(ROW_PER_BLOCK / BLOCK_SIZE) * DIM_PER_BLOCK * RATIO];
    //     // Compute
    //     for (int i = 0; i < ROW_PER_BLOCK / BLOCK_SIZE; i++) {
    //         for (int j = 0; j < DIM_PER_BLOCK * RATIO; j++) {
    //             partial[i * DIM_PER_BLOCK + j] = __hmul(input_reg[b][i], w_reg[i * DIM_PER_BLOCK + j]);
    //         }
    //         if (i > 1) {
    //             for (int j = 0; j < DIM_PER_BLOCK * RATIO; j++) {
    //                 partial[j] = __hadd(partial[j], partial[i * DIM_PER_BLOCK + j]);
    //             }
    //         }
    //     }
    //     for (int i = 0; i < DIM_PER_BLOCK; i++) {
    //         #pragma unroll
    //         for (uint32_t mask = 16; mask > 0; mask >>= 1) {
    //             *(half2*)(&partial[i * 2]) = __hadd2(*(half2*)(&partial[i * 2]), __shfl_down_sync(0xffffffff, *(half2*)(&partial[i * 2]), mask));
    //         }
    //         if (threadIdx.x % WARP_SIZE == 0) {
    //             *(uint32_t*)(&reduce_workspace[(threadIdx.x / WARP_SIZE) * 2]) = *(uint32_t*)(&partial[i * 2]);
    //         }
    //         __syncthreads();
    //         if (threadIdx.x < WARP_SIZE) {
    //             if (threadIdx.x < WARP_NUM) {
    //                 *(uint32_t*)((&partial[i * 2])) = *(uint32_t*)(&reduce_workspace[threadIdx.x * 2]);
    //             }
    //             else {
    //                 *(uint32_t*)((&partial[i * 2])) = 0;
    //             }
    //         }
    //         if (threadIdx.x < WARP_SIZE) {
    //             #pragma unroll
    //             for (uint32_t mask = 8; mask > 0; mask >>= 1) {
    //                 *(half2*)((&partial[i * 2])) = __hadd2(*(half2*)((&partial[i * 2])), __shfl_down_sync(0xffffffff, *(half2*)((&partial[i * 2])), mask));
    //             }
    //         }
    //         if (threadIdx.x == 0) {
    //             atomicAdd((half2*)(&_o[b * HIDDEN_DIM + blockIdx.y * DIM_PER_BLOCK * RATIO + i * 2]), *(half2*)((&partial[i * 2])));
    //         }
    //     }
    // }

}

torch::Tensor gptvq_gemv(
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
    cudaFuncSetAttribute(gptvq_gemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE);

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

    dim3 grid(hidden_dim / ROW_PER_BLOCK, (hidden_dim / RATIO) / DIM_PER_BLOCK);
    dim3 block(BLOCK_SIZE);

#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        gptvq_gemv_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr,
            d_dummy_w,
            codebook_ptr,
            o_ptr,
            HIDDEN_DIM
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        gptvq_gemv_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE>>>(
            input_ptr,
            d_dummy_w,
            codebook_ptr,
            o_ptr,
            HIDDEN_DIM
        );
#if PROFILING == 1
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency (us): " << 1000.0 * (ms / (1.0 * iter)) << std::endl;
    // std::cout << "TFLOPS : " << ((2.0 * hidden_dim * hidden_dim * seq_len) / ((ms / (1.0 * iter)) / (1000.0))) / (1024.0 * 1024.0 * 1024.0 * 1024.0) << std::endl;
#endif

    return o;
}