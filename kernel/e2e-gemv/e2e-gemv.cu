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

#define DIM_PER_CODEBOOK 8
#define DIM_BLOCK 16 // 4 space as one block
#define ROW_BLOCK 1024
#define PROFILING 1
#define ENTRY 256
#define RATIO 4
#define RESIDUAL 2
#define HOT 1

#define WARP_SIZE 32
#define WARP_NUM 16
#define BLOCK_SIZE (WARP_SIZE * WARP_NUM)
#define ROW_PER_THREAD (ROW_BLOCK / BLOCK_SIZE)
#define QUANTIZED_DIM_PER_BLOCK (DIM_BLOCK / RATIO)
#define CODEBOOK_ONE_BLOCK ((DIM_BLOCK / DIM_PER_CODEBOOK) * (ENTRY * RATIO * 2) / HOT)
// #define MAX_SHARED_MEMORY_USAGE_R (RESIDUAL * CODEBOOK_ONE_BLOCK)
#define MAX_SHARED_MEMORY_USAGE_R 1024
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

template <>
struct packed_vec<uint8_t, 4> {
    using Type = uint32_t;
};

__device__ __forceinline__ uint32_t gemv_shmem_uint32_t(const void* shmem_ptr) {
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

__device__ void load_codebook_r(
    half* codebook_buf,
    half* codebook_r_buf,
    half* codebook,
    half* codebook_r
)
{
    //   0~255 Load R0
    // 256~511 Load R1
    uint32_t codebook_begin_row = blockIdx.y * DIM_BLOCK / DIM_PER_CODEBOOK;
    uint32_t iters_to_load = ((DIM_BLOCK / DIM_PER_CODEBOOK) * ENTRY * RATIO / 8) / (BLOCK_SIZE / 2);
    uint32_t load_cols = (ENTRY * RATIO) / 8;
    uint32_t load_rows = (BLOCK_SIZE / 2) / load_cols;
    uint32_t tid = threadIdx.x % 256;
    for (int i = 0; i < iters_to_load; i++) {
        if (threadIdx.x < 256) {
            asm volatile (
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :
                : "r"(gemv_shmem_uint32_t(&codebook_buf[(i * load_rows + tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])),
                "l"(&codebook[(codebook_begin_row + i * load_rows + tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])
            );
        }
        else {
            asm volatile (
                "cp.async.ca.shared.global [%0], [%1], 16;\n"
                :
                : "r"(gemv_shmem_uint32_t(&codebook_r_buf[(i * load_rows + tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])),
                "l"(&codebook_r[(codebook_begin_row + i * load_rows + tid / load_cols) * (ENTRY * RATIO) + (tid % load_cols) * 8])
            );            
        }
    }
}

__device__ void __hadd4(
    half* res,
    half* A,
    half* B
)
{
    half a[4], b[4];
    *(uint64_t*)a = *(uint64_t*)A;
    *(uint64_t*)b = *(uint64_t*)B;
    *(half2*)(&res[0]) = __hadd2(*(half2*)(&a[0]), *(half2*)(&b[0]));
    *(half2*)(&res[2]) = __hadd2(*(half2*)(&a[0]), *(half2*)(&b[0]));
}

__global__ void e2e_gemv_rq_kernel(
    half* __restrict__ _input,
    uint8_t* __restrict__ _w,
    half* __restrict__ _codebook,
    uint8_t* __restrict__ _w_r,
    half* __restrict__ _codebook_r,
    half* __restrict__ _o,
    int _batch, int _hidden
)
{
    extern __shared__ uint8_t shmem[];
    half2* reduce_workspace = reinterpret_cast<half2*>(shmem);
    // half *codebook   = reinterpret_cast<half*>(shmem);
    // half *codebook_r = reinterpret_cast<half*>(shmem + CODEBOOK_ONE_BLOCK);

    // // Load codebook
    // load_codebook_r(codebook, codebook_r, _codebook, _codebook_r);
    // asm volatile("cp.async.wait_all;\n"::);
    // __syncthreads();

    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;

    // Load quantized weight
    /* 2 */ uint8_t w_reg[ROW_PER_THREAD][QUANTIZED_DIM_PER_BLOCK];
    /* 2 */ uint8_t w_r_reg[ROW_PER_THREAD][QUANTIZED_DIM_PER_BLOCK];
    using w_load_type = typename packed_vec<uint8_t, QUANTIZED_DIM_PER_BLOCK>::Type;
    #pragma unroll
    for (int _ = 0; _ < ROW_PER_THREAD; _++) {
        *(w_load_type*)(w_reg[_])   = *(w_load_type*)(&_w[(blockIdx.x * ROW_BLOCK + threadIdx.x * ROW_PER_THREAD + _) * (_hidden / RATIO) + blockIdx.y * QUANTIZED_DIM_PER_BLOCK]);
        *(w_load_type*)(w_r_reg[_]) = *(w_load_type*)(&_w_r[(blockIdx.x * ROW_BLOCK + threadIdx.x * ROW_PER_THREAD + _) * (_hidden / RATIO) + blockIdx.y * QUANTIZED_DIM_PER_BLOCK]);
    }

    // Decode to reg
    /* 16 */ half w_dequantized_reg[ROW_PER_THREAD][DIM_BLOCK];
    /* 8 */ half psum[DIM_BLOCK];

    for (int i = 0; i < ROW_PER_THREAD * DIM_BLOCK; i++) psum[i] = __float2half(0.0);
    
    for (int i = 0; i < ROW_PER_THREAD; i++) {
        for (int j = 0; j < QUANTIZED_DIM_PER_BLOCK; j++) {
            __hadd4(&w_dequantized_reg[i][j * 4], 
                    // &codebook[(j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_reg[i][j]) * RATIO], 
                    &_codebook[(blockIdx.y * (DIM_BLOCK / DIM_PER_CODEBOOK) + j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_reg[i][j]) * RATIO],
                    // &codebook_r[(j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_r_reg[i][j]) * RATIO]
                    &_codebook_r[(blockIdx.y * (DIM_BLOCK / DIM_PER_CODEBOOK) + j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_reg[i][j]) * RATIO]
            );
        }
    }
    if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 0) {
        printf("%d : %6.3f, %6.3f, %6.3f, %6.3f\n", threadIdx.x, __half2float(w_dequantized_reg[0][0]), __half2float(w_dequantized_reg[0][1]), __half2float(w_dequantized_reg[0][2]), __half2float(w_dequantized_reg[0][3]));
    }
    /* 1 */ half input_reg[ROW_PER_THREAD];
    using input_load_type = typename packed_vec<half, ROW_PER_THREAD>::Type;

    for (int b = 0; b < _batch; b++) {
        *(input_load_type*)(&input_reg[0]) = *(input_load_type*)(&_input[b * _hidden + blockIdx.x * ROW_BLOCK + threadIdx.x * ROW_PER_THREAD]);
        // psum[i] = input_reg[:] * w_dequantized_reg[:][i]
        for (int row = 0; row < ROW_PER_THREAD; row++) {
            for (int dim = 0; dim < DIM_BLOCK; dim+=2) {
                // psum[dim] = __hadd(psum[dim], __hmul(input_reg[row], w_dequantized_reg[row][dim]));
                *(half2*)(&psum[dim]) = __hfma2({input_reg[row], input_reg[row]}, *(half2*)(&w_dequantized_reg[row][dim]), *(half2*)(&psum[dim]));
            }
        }
        // Accumulate psum @ block level
        for (int dim = 0; dim < DIM_BLOCK; dim+=2) {
            #pragma unroll
            for (int mask = 16; mask > 0; mask>>=1) {
                *(half2*)(&psum[dim]) = __hadd2(*(half2*)(&psum[dim]), __shfl_xor_sync(0xffffffff, *(half2*)(&psum[dim]), mask));
            }
            if (lane_id == 0) {
                reduce_workspace[warp_id] = *(half2*)(&psum[dim]);
            }
            __syncthreads();
            if (threadIdx.x < WARP_NUM) {
                *(half2*)(&psum[dim]) = reduce_workspace[threadIdx.x];
            }
            else *(half2*)(&psum[dim]) = {__float2half(0.0), __float2half(0.0)};
            if (warp_id == 0) {
                #pragma unroll
                for (int mask = 8; mask > 0; mask>>=1) {
                    *(half2*)(&psum[dim]) = __hadd2(*(half2*)(&psum[dim]), __shfl_xor_sync(0xffffffff, *(half2*)(&psum[dim]), mask));
                }
            }
            if (threadIdx.x == 0) {
                atomicAdd((half2*)&_o[b * _hidden + blockIdx.y * DIM_BLOCK + dim], *(half2*)(&psum[dim]));
            }
        }
    }
}

torch::Tensor e2e_gemv_rq(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook,
    torch::Tensor w_r,
    torch::Tensor codebook_r
)
{
#if PROFILING == 1
    const int wmup = 50;
    const int iter = 100;
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);
#endif
    cudaFuncSetAttribute(e2e_gemv_rq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE_R);
    // Assuming M is padded to 128, pad at torch level.

    auto BATCH = input.size(0);
    auto HIDDEN = input.size(1);
    std::cout << BATCH << " " << HIDDEN << std::endl;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({BATCH, HIDDEN}, 0, options);

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());

    uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>());
    uint8_t* w_r_ptr = reinterpret_cast<uint8_t*>(w_r.data_ptr<uint8_t>());
    half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
    half* codebook_r_ptr = reinterpret_cast<half*>(codebook_r.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

    dim3 grid(HIDDEN / ROW_BLOCK, HIDDEN / DIM_BLOCK);
    dim3 block(BLOCK_SIZE);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        e2e_gemv_rq_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE_R>>>(
            input_ptr, 
            w_ptr,
            codebook_ptr, 
            w_r_ptr,
            codebook_r_ptr,
            o_ptr,
            BATCH, HIDDEN
        );
    }
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
#endif
        e2e_gemv_rq_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE_R>>>(
            input_ptr, 
            w_ptr,
            codebook_ptr, 
            w_r_ptr,
            codebook_r_ptr,
            o_ptr,
            BATCH, HIDDEN
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