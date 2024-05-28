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
#define PROFILING 0
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

// __device__ void __hadd4(
//     half* res,
//     half* A,
//     half* B
// )
// {
//     half a[4], b[4];
//     *(uint64_t*)a = *(uint64_t*)A;
//     *(uint64_t*)b = *(uint64_t*)B;
//     *(half2*)(&res[0]) = __hadd2(*(half2*)(&a[0]), *(half2*)(&b[0]));
//     *(half2*)(&res[2]) = __hadd2(*(half2*)(&a[0]), *(half2*)(&b[0]));
// }

// __global__ void e2e_gemv_rq_kernel(
//     half* __restrict__ _input,
//     uint8_t* __restrict__ _w,
//     half* __restrict__ _codebook,
//     uint8_t* __restrict__ _w_r,
//     half* __restrict__ _codebook_r,
//     half* __restrict__ _o,
//     int _batch, int _hidden
// )
// {
//     extern __shared__ uint8_t shmem[];
//     half2* reduce_workspace = reinterpret_cast<half2*>(shmem);
//     // half *codebook   = reinterpret_cast<half*>(shmem);
//     // half *codebook_r = reinterpret_cast<half*>(shmem + CODEBOOK_ONE_BLOCK);

//     // // Load codebook
//     // load_codebook_r(codebook, codebook_r, _codebook, _codebook_r);
//     // asm volatile("cp.async.wait_all;\n"::);
//     // __syncthreads();

//     uint32_t lane_id = threadIdx.x % WARP_SIZE;
//     uint32_t warp_id = threadIdx.x / WARP_SIZE;

//     // Load quantized weight
//     /* 2 */ uint8_t w_reg[ROW_PER_THREAD][QUANTIZED_DIM_PER_BLOCK];
//     /* 2 */ uint8_t w_r_reg[ROW_PER_THREAD][QUANTIZED_DIM_PER_BLOCK];
//     using w_load_type = typename packed_vec<uint8_t, QUANTIZED_DIM_PER_BLOCK>::Type;
//     #pragma unroll
//     for (int _ = 0; _ < ROW_PER_THREAD; _++) {
//         *(w_load_type*)(w_reg[_])   = *(w_load_type*)(&_w[(blockIdx.x * ROW_BLOCK + threadIdx.x * ROW_PER_THREAD + _) * (_hidden / RATIO) + blockIdx.y * QUANTIZED_DIM_PER_BLOCK]);
//         *(w_load_type*)(w_r_reg[_]) = *(w_load_type*)(&_w_r[(blockIdx.x * ROW_BLOCK + threadIdx.x * ROW_PER_THREAD + _) * (_hidden / RATIO) + blockIdx.y * QUANTIZED_DIM_PER_BLOCK]);
//     }

//     // Decode to reg
//     /* 16 */ half w_dequantized_reg[ROW_PER_THREAD][DIM_BLOCK];
//     /* 8 */ half psum[DIM_BLOCK];

//     for (int i = 0; i < ROW_PER_THREAD * DIM_BLOCK; i++) psum[i] = __float2half(0.0);
    
//     for (int i = 0; i < ROW_PER_THREAD; i++) {
//         for (int j = 0; j < QUANTIZED_DIM_PER_BLOCK; j++) {
//             __hadd4(&w_dequantized_reg[i][j * 4], 
//                     // &codebook[(j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_reg[i][j]) * RATIO], 
//                     &_codebook[(blockIdx.y * (DIM_BLOCK / DIM_PER_CODEBOOK) + j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_reg[i][j]) * RATIO],
//                     // &codebook_r[(j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_r_reg[i][j]) * RATIO]
//                     &_codebook_r[(blockIdx.y * (DIM_BLOCK / DIM_PER_CODEBOOK) + j / (DIM_BLOCK / DIM_PER_CODEBOOK)) * ENTRY * RATIO + ((uint32_t) w_reg[i][j]) * RATIO]
//             );
//         }
//     }
//     // if (blockIdx.x == 0 && blockIdx.y == 0 && warp_id == 0) {
//     //     printf("%d : %6.3f, %6.3f, %6.3f, %6.3f\n", threadIdx.x, __half2float(w_dequantized_reg[0][0]), __half2float(w_dequantized_reg[0][1]), __half2float(w_dequantized_reg[0][2]), __half2float(w_dequantized_reg[0][3]));
//     // }
//     /* 1 */ half input_reg[ROW_PER_THREAD];
//     using input_load_type = typename packed_vec<half, ROW_PER_THREAD>::Type;

//     for (int b = 0; b < _batch; b++) {
//         *(input_load_type*)(&input_reg[0]) = *(input_load_type*)(&_input[b * _hidden + blockIdx.x * ROW_BLOCK + threadIdx.x * ROW_PER_THREAD]);
//         // psum[i] = input_reg[:] * w_dequantized_reg[:][i]
//         for (int row = 0; row < ROW_PER_THREAD; row++) {
//             for (int dim = 0; dim < DIM_BLOCK; dim+=2) {
//                 // psum[dim] = __hadd(psum[dim], __hmul(input_reg[row], w_dequantized_reg[row][dim]));
//                 *(half2*)(&psum[dim]) = __hfma2({input_reg[row], input_reg[row]}, *(half2*)(&w_dequantized_reg[row][dim]), *(half2*)(&psum[dim]));
//             }
//         }
//         // Accumulate psum @ block level
//         for (int dim = 0; dim < DIM_BLOCK; dim+=2) {
//             #pragma unroll
//             for (int mask = 16; mask > 0; mask>>=1) {
//                 *(half2*)(&psum[dim]) = __hadd2(*(half2*)(&psum[dim]), __shfl_xor_sync(0xffffffff, *(half2*)(&psum[dim]), mask));
//             }
//             if (lane_id == 0) {
//                 reduce_workspace[warp_id] = *(half2*)(&psum[dim]);
//             }
//             __syncthreads();
//             if (threadIdx.x < WARP_NUM) {
//                 *(half2*)(&psum[dim]) = reduce_workspace[threadIdx.x];
//             }
//             else *(half2*)(&psum[dim]) = {__float2half(0.0), __float2half(0.0)};
//             if (warp_id == 0) {
//                 #pragma unroll
//                 for (int mask = 8; mask > 0; mask>>=1) {
//                     *(half2*)(&psum[dim]) = __hadd2(*(half2*)(&psum[dim]), __shfl_xor_sync(0xffffffff, *(half2*)(&psum[dim]), mask));
//                 }
//             }


//             if (threadIdx.x == 0) {
//                 atomicAdd((half2*)&_o[b * _hidden + blockIdx.y * DIM_BLOCK + dim], *(half2*)(&psum[dim]));
//             }
//         }
//     }
// }

// torch::Tensor e2e_gemv_rq(
//     torch::Tensor input,
//     torch::Tensor w,
//     torch::Tensor codebook,
//     torch::Tensor w_r,
//     torch::Tensor codebook_r
// )
// {
// #if PROFILING == 1
//     const int wmup = 50;
//     const int iter = 100;
//     cudaEvent_t st, ed;
//     cudaEventCreate(&st, NULL);
//     cudaEventCreate(&ed, NULL);
// #endif
//     cudaFuncSetAttribute(e2e_gemv_rq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY_USAGE_R);
//     // Assuming M is padded to 128, pad at torch level.

//     auto BATCH = input.size(0);
//     auto HIDDEN = input.size(1);
//     std::cout << BATCH << " " << HIDDEN << std::endl;
//     auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
//     torch::Tensor o = torch::full({BATCH, HIDDEN}, 0, options);

//     half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());

//     uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>());
//     uint8_t* w_r_ptr = reinterpret_cast<uint8_t*>(w_r.data_ptr<uint8_t>());
//     half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
//     half* codebook_r_ptr = reinterpret_cast<half*>(codebook_r.data_ptr<at::Half>());
//     half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());

//     dim3 grid(HIDDEN / ROW_BLOCK, HIDDEN / DIM_BLOCK);
//     dim3 block(BLOCK_SIZE);
// #if PROFILING == 1
//     for (int i = 0; i < wmup; i++) {
//         e2e_gemv_rq_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE_R>>>(
//             input_ptr, 
//             w_ptr,
//             codebook_ptr, 
//             w_r_ptr,
//             codebook_r_ptr,
//             o_ptr,
//             BATCH, HIDDEN
//         );
//     }
//     cudaEventRecord(st);
//     for (int i = 0; i < iter; i++) {
// #endif
//         e2e_gemv_rq_kernel<<<grid, block, MAX_SHARED_MEMORY_USAGE_R>>>(
//             input_ptr, 
//             w_ptr,
//             codebook_ptr, 
//             w_r_ptr,
//             codebook_r_ptr,
//             o_ptr,
//             BATCH, HIDDEN
//         );
// #if PROFILING == 1
//     }
//     cudaEventRecord(ed);
//     cudaEventSynchronize(ed);
//     float ms;
//     cudaEventElapsedTime(&ms, st, ed);
//     std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;
// #endif
//     return o;
// }

// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------

// __device__ void __hadd4(
//     half* a,
//     half* b
// )
// {
//     *(half2*)(&a[0]) = __hadd2(*(half2*)(&a[0]), *(half2*)(&b[0]));
//     *(half2*)(&a[2]) = __hadd2(*(half2*)(&a[2]), *(half2*)(&b[2]));
// }

// __global__ void e2e_gemv_rq_kernel(
//     const half* _input,
//     const uint8_t* _w,
//     const half* _codebook,
//     const uint8_t* _w_r,
//     const half* _codebook_r,
//     half* _o,
//     int _batch, int _hidden   
// )
// {
//     uint32_t warp_id = threadIdx.x / 32;
//     uint32_t lane_id = threadIdx.x % 32;

//     const uint8_t* weight_ptr = (blockIdx.z == 0) ? _w : _w_r;
//     const half* codebook_ptr = (blockIdx.z == 0) ? _codebook : _codebook_r;

//     extern __shared__ uint8_t shmem[];
//     // half[1024][16]
//     half* dequanted_weight = reinterpret_cast<half*>(shmem);

//     // Quantized weight: uint8_t[1024][4]
//     // Dequant, every thread dequant 2
//     uint8_t ids0[4], ids1[4];
//     *(uint32_t*)(&ids0[0]) = *(uint32_t*)(&weight_ptr[(blockIdx.x * 1024 + threadIdx.x * 2 + 0) * 1024 + blockIdx.y * 4]);
//     *(uint32_t*)(&ids1[0]) = *(uint32_t*)(&weight_ptr[(blockIdx.x * 1024 + threadIdx.x * 2 + 1) * 1024 + blockIdx.y * 4]);

//     #pragma unroll
//     for (int i = 0; i < 4; i++) {
//         // Use cp.async?
//         *(uint64_t*)(&dequanted_weight[(threadIdx.x * 2 + 0) * 16 + i * 4]) = *(uint64_t*)(&codebook_ptr[blockIdx.y * 2 * 256 * 4 + (i / 2) * 256 * 4 + ((uint32_t) ids0[i]) * 4]);
//         *(uint64_t*)(&dequanted_weight[(threadIdx.x * 2 + 1) * 16 + i * 4]) = *(uint64_t*)(&codebook_ptr[blockIdx.y * 2 * 256 * 4 + (i / 2) * 256 * 4 + ((uint32_t) ids1[i]) * 4]);
//     }
//     __syncthreads();
//     half __align__(16) part_input[32];
//     half psum;
//     for (int b = 0; b < _batch; b++) {
//         #pragma unroll
//         for (int i = 0; i < 4; i++) {
//             *(uint4*)(&part_input[i * 8]) = *(uint4*)(&_input[b * 4096 + blockIdx.x * 1024 + lane_id * 32 + i * 8]);
//         }
//         // 1024 / warp_size = 32, every thread hold 32 element
//         half part_w[2] = {__float2half(0.0), __float2half(0.0)};
//         #pragma unroll
//         for (int i = 0; i < 32; i+=2) {
//             *(half2*)(&part_w) = __hfma2({part_input[i], part_input[i + 1]}, {dequanted_weight[(lane_id * 32 + i) * 16 + warp_id], dequanted_weight[(lane_id * 32 + i + 1) * 16 + warp_id]}, *(half2*)(&part_w));
//         }

//         psum = __hadd(part_w[0], part_w[1]);
//         #pragma unroll
//         for (int mask = 16; mask > 0; mask >>= 1) {
//             psum = __hadd(psum, __shfl_xor_sync(0xffffffff, psum, mask));
//         }
//         if (lane_id == 0) {
//             atomicAdd(&_o[b * 4096 + blockIdx.y * 16 + warp_id], psum);
//         }
//     }
// }

// int main(int argc, char** argv) {
//     half *h_input, *d_input;
//     uint8_t *h_w, *d_w;
//     uint8_t *h_w_r, *d_w_r;
//     half *h_codebook, *d_codebook;
//     half *h_codebook_r, *d_codebook_r;
//     half *h_o, *d_o;

//     h_o = new half[1 * 4096];

//     cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * 1 * 4096);
//     cudaMalloc(reinterpret_cast<void**>(&d_o), sizeof(half) * 1 * 4096);
//     cudaMalloc(reinterpret_cast<void**>(&d_w), sizeof(uint8_t) * 1024 * 4096);
//     cudaMalloc(reinterpret_cast<void**>(&d_w_r), sizeof(uint8_t) * 1024 * 4096);
//     cudaMalloc(reinterpret_cast<void**>(&d_codebook), sizeof(half) * 512 * 256 * 4);
//     cudaMalloc(reinterpret_cast<void**>(&d_codebook_r), sizeof(half) * 512 * 256 * 4);

//     dim3 grid(4096 / 2048, 4096 / 8, 2);
//     e2e_gemv_rq_kernel<<<grid, 512, 2048>>>(
//         d_input, d_w, d_codebook, d_w_r, d_codebook_r, d_o, 1, 4096
//     );
//     cudaDeviceSynchronize();
//     cudaMemcpy(h_o, reinterpret_cast<void*>(d_o), sizeof(half) * 1 * 4096, cudaMemcpyDeviceToHost);

// }

// torch::Tensor e2e_gemv_rq(
//     torch::Tensor input,
//     torch::Tensor w,
//     torch::Tensor codebook,
//     torch::Tensor w_r,
//     torch::Tensor codebook_r
// )
// {
// #if PROFILING == 1
//     const int wmup = 50;
//     const int iter = 100;
//     cudaEvent_t st, ed;
//     cudaEventCreate(&st, NULL);
//     cudaEventCreate(&ed, NULL);
// #endif
//     cudaFuncSetAttribute(e2e_gemv_rq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);

//     auto BATCH = input.size(0);
//     auto HIDDEN = input.size(1);
//     std::cout << sizeof(unsigned int) << " " << sizeof(unsigned long) << " " << sizeof(unsigned long long) << std::endl;
//     auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
//     torch::Tensor o = torch::full({BATCH, HIDDEN}, 0, options);

//     half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());

//     uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>());
//     uint8_t* w_r_ptr = reinterpret_cast<uint8_t*>(w_r.data_ptr<uint8_t>());
//     half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
//     half* codebook_r_ptr = reinterpret_cast<half*>(codebook_r.data_ptr<at::Half>());
//     half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    

//     // cudaStream_t stream;
//     // cudaStreamCreate(&stream);
//     // cudaStreamAttrValue stream_attr;
//     // stream_attr.accessPolicyWindow.base_ptr = codebook_ptr;
//     // stream_attr.accessPolicyWindow.num_bytes = 1048576;
//     // stream_attr.accessPolicyWindow.hitRatio = 1.0;
//     // stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
//     // stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
//     // cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    
//     dim3 grid(HIDDEN / 1024, HIDDEN / 16, RESIDUAL);
//     dim3 block(BLOCK_SIZE);
// #if PROFILING == 1
//     for (int i = 0; i < wmup; i++) {
//         e2e_gemv_rq_kernel<<<grid, block, 32768>>>(
//             input_ptr, 
//             w_ptr,
//             codebook_ptr, 
//             w_r_ptr,
//             codebook_r_ptr,
//             o_ptr,
//             BATCH, HIDDEN
//         );
//     }
//     cudaEventRecord(st);
//     for (int i = 0; i < iter; i++) {
// #endif
//         e2e_gemv_rq_kernel<<<grid, block, 32768>>>(
//             input_ptr, 
//             w_ptr,
//             codebook_ptr, 
//             w_r_ptr,
//             codebook_r_ptr,
//             o_ptr,
//             BATCH, HIDDEN
//         );
// #if PROFILING == 1
//     }
//     cudaEventRecord(ed);
//     cudaEventSynchronize(ed);
//     float ms;
//     cudaEventElapsedTime(&ms, st, ed);
//     std::cout << "Latency: " << ms / (1.0 * iter) << std::endl;
// #endif
//     return o;
// }

// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// --------------------------------------------------------------------


__device__ void accumulate(
    half2* sum,
    half2 multi,
    half* dequanted
)
{
    *sum = __hadd2(
        __hmul2({multi.x, multi.x}, *(half2*)(&dequanted[(threadIdx.x * 2 + 0) * 8 + threadIdx.y * 2])), 
        __hmul2({multi.y, multi.y}, *(half2*)(&dequanted[(threadIdx.x * 2 + 1) * 8 + threadIdx.y * 2]))
    );
}
__device__ void __forceinline__ dequant_to_shmem(
    half* dequanted_weight_buf,
    const uint8_t* weight,
    const half* codebook,
    int k
)
{
    uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t id = (uint32_t) weight[(k * 64 + tid / 2) * 1024 + blockIdx.y * 2 + tid % 2];
    *(uint64_t*)(&dequanted_weight_buf[threadIdx.y * 128 + threadIdx.x * 4]) = *(uint64_t*)(&codebook[blockIdx.y * 256 * 4 + id * 4]);
        __syncthreads();
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 0) {
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 8; j++) {
                    printf("% 6.3f%c", __half2float(dequanted_weight_buf[i * 8 + j]), (j == 7) ? '\n' : ' ');
                }
            }
        }
}

__global__ void e2e_gemv_rq_kernel(
    const half* _input,
    const uint8_t* _w,
    const half* _codebook,
    const uint8_t* _w_r,
    const half* _codebook_r,
    half* _o,
    int _batch, int _hidden   
)
{
    uint32_t batch_idx = blockIdx.x;
    extern __shared__ uint8_t shmem[];
    half* dequant_w0 = reinterpret_cast<half*>(shmem);
    const uint8_t* w_ptr = blockIdx.z == 0 ? _w : _w_r;
    const half* codebook_ptr = blockIdx.z == 0 ? _codebook : _codebook_r;

    half2 psum{__float2half(0.0), __float2half(0.0)};
    half2 input;
    for (int k = 0; k < 4096 / 64; k++) {
        dequant_to_shmem(dequant_w0, w_ptr, codebook_ptr, k);
        __syncthreads();
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 0) {
            printf("\n\n");
            for (int i = 0; i < 64; i++) {
                for (int j = 0; j < 8; j++) {
                    printf("% 6.3f%c", __half2float(dequant_w0[i * 8 + j]), (j == 7) ? '\n' : ' ');
                }
            }
        }
        
        accumulate(&psum, input, dequant_w0);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask>>=1) {
        psum = __hadd2(psum, __shfl_xor_sync(0xffffffff, psum, mask));
    }
    atomicAdd((half2*)&_o[batch_idx * 4096 + blockIdx.y * 8 + threadIdx.y * 2], psum);
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
    cudaFuncSetAttribute(e2e_gemv_rq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 2048);

    auto BATCH = input.size(0);
    auto HIDDEN = input.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({BATCH, HIDDEN}, 0, options);

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());

    uint8_t* w_ptr = reinterpret_cast<uint8_t*>(w.data_ptr<uint8_t>());
    uint8_t* w_r_ptr = reinterpret_cast<uint8_t*>(w_r.data_ptr<uint8_t>());
    half* codebook_ptr = reinterpret_cast<half*>(codebook.data_ptr<at::Half>());
    half* codebook_r_ptr = reinterpret_cast<half*>(codebook_r.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    

    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    // cudaStreamAttrValue stream_attr;
    // stream_attr.accessPolicyWindow.base_ptr = codebook_ptr;
    // stream_attr.accessPolicyWindow.num_bytes = 1048576;
    // stream_attr.accessPolicyWindow.hitRatio = 1.0;
    // stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    // stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    // cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    
    // Every block do 4096x8
    dim3 grid(BATCH, HIDDEN / 8, 2);
    // Every thread fo 128 row, 2 subspaces (32-bit)
    dim3 block(32, 4);
#if PROFILING == 1
    for (int i = 0; i < wmup; i++) {
        e2e_gemv_rq_kernel<<<grid, block, 2048>>>(
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
        e2e_gemv_rq_kernel<<<grid, block, 2048>>>(
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