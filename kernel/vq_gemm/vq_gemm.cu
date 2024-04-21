#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <torch/extension.h>
#include "stdio.h"
#include <iostream>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include "mma.h"
using namespace nvcuda;
using barrier = cuda::barrier<cuda::thread_scope_block>;

#define PROFILING 0

// #define CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
// void check(cudaError_t err, const char* const func, const char* const file,
//            const int line)
// {
//     if (err != cudaSuccess)
//     {
//         std::cerr << "CUDA Runtime Error at: " << file << ":" << line
//                   << std::endl;
//         std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
//         // We don't exit when we encounter CUDA errors in this example.
//         // std::exit(EXIT_FAILURE);
//     }
// }

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

using INPUT_TYPE = half;
using TORCH_INPUT_TYPE = at::Half;
using WEIGHT_TYPE = uint8_t;
using TORCH_WEIGHT_TYPE = uint8_t;

template <typename T, int len>
struct packed_vec {};

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
struct packed_vec<uint8_t, 1> {
    using Type = uint8_t;
};

template <>
struct packed_vec<uint8_t, 2> {
    using Type = uint16_t;
};

template <>
struct packed_vec<uint8_t, 4> {
    using Type = uint32_t;
};

template <>
struct packed_vec<uint8_t, 8> {
    using Type = uint64_t;
};

template <>
struct packed_vec<uint8_t, 16> {
    using Type = uint4;
};

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
#define TRIPLE_BUFFER 3

// Don't consider residual, since A * (B + C) = A*B + A*C
// We can add A*B and A*C with another kernel later. 
// Thus we do not need to decode residual codec in kernel.

template <
    size_t INV_HOT_ENTRY,
    size_t PACKED_LOAD_WIDTH,
    size_t B32REG_LIMIT,
    size_t ENABLE_ENTRY_CENTRIC
>
void __global__ vq_gemm_kernel(
    INPUT_TYPE* _o,
    INPUT_TYPE* _h,
    WEIGHT_TYPE* _w,
    INPUT_TYPE* _codebook,
    uint32_t _residuals, uint32_t _compression_ratio, uint32_t _entry,
    uint32_t _m, uint32_t _n, uint32_t _k
)
{
    const uint32_t THREAD_PACKED_LOAD_FP16_NUM = PACKED_LOAD_WIDTH / (8 * sizeof(INPUT_TYPE));
    const uint32_t THREAD_PACKED_LOAD_UINT8_NUM = PACKED_LOAD_WIDTH / (8 * sizeof(WEIGHT_TYPE));
    using PACKED_LOAD_TYPE_FP16 = typename packed_vec<INPUT_TYPE, THREAD_PACKED_LOAD_FP16_NUM>::Type;
    using PACKED_LOAD_TYPE_UINT8 = typename packed_vec<WEIGHT_TYPE, THREAD_PACKED_LOAD_UINT8_NUM>::Type;
    const uint32_t thread_type = threadIdx.x >> 8;
    __shared__ __restrict__ barrier bar[14];
    // bar[ 0]: h0.ready,  bar[ 1]: h1.ready,  bar[ 2] : h2.ready
    // bar[ 3]: h0.filled, bar[ 4]: h1.filled, bar[ 5] : h2.filled
    // bar[ 6]: w0.ready,  bar[ 7]: w1.ready
    // bar[ 8]: w0.filled, bar[ 9]: w1.filled
    // bar[10]: d0.ready,  bar[11]: d1.ready
    // bar[12]: d0.filled, bar[13]: d1.filled
    extern __shared__ __restrict__ uint8_t smem[];
    // | Codebook |    H     |    W     |    D     |

    // At most 512 threads load codebook
    INPUT_TYPE *codebook_shmem = reinterpret_cast<INPUT_TYPE*>(smem);
    const uint32_t codebook_to_access = (N_BLOCK_TILE / _compression_ratio) * _entry * _compression_ratio;
    const uint32_t codebook_offset = blockIdx.y * codebook_to_access;
    const uint32_t codebook_to_load = codebook_to_access / INV_HOT_ENTRY;
    const uint32_t entry_to_load = _entry / INV_HOT_ENTRY;

    const uint32_t threads_need_to_load = min((uint32_t) LOAD_BLOCK_SIZE, (uint32_t) (codebook_to_load / THREAD_PACKED_LOAD_FP16_NUM));
    const uint32_t block_packed_load_width = threads_need_to_load * THREAD_PACKED_LOAD_FP16_NUM;
    const uint32_t load_iterations = max((uint32_t) 1, (uint32_t) (codebook_to_load / block_packed_load_width));
    const uint32_t subspace_stride = _entry * _compression_ratio;
    float subspaces_load_at_once = block_packed_load_width / (entry_to_load * _compression_ratio);
    const uint32_t iterations_for_one_subspace = max((uint32_t) 1, (uint32_t) (1.0 / subspaces_load_at_once));

    uint32_t codebook_size = codebook_to_load * sizeof(INPUT_TYPE);
    uint32_t h_size = TRIPLE_BUFFER * sizeof(INPUT_TYPE) * M_BLOCK_TILE * K_BLOCK_TILE;
    uint32_t w_size = DOUBLE_BUFFER * sizeof(WEIGHT_TYPE) * K_BLOCK_TILE * (N_BLOCK_TILE / _compression_ratio);
    uint32_t d_size = DOUBLE_BUFFER * sizeof(INPUT_TYPE) * K_BLOCK_TILE * N_BLOCK_TILE;

    INPUT_TYPE *h_shmem = reinterpret_cast<INPUT_TYPE*>(smem + codebook_size);
    WEIGHT_TYPE *w_shmem = reinterpret_cast<WEIGHT_TYPE*>(smem + codebook_size + h_size);
    INPUT_TYPE *d_shmem = reinterpret_cast<INPUT_TYPE*>(smem + codebook_size + h_size + w_size);
    // barrier* bar = reinterpret_cast<barrier*>(d_shmem + sizeof(INPUT_TYPE) * DOUBLE_BUFFER * K_BLOCK_TILE * N_BLOCK_TILE);

    auto block = cooperative_groups::this_thread_block();
    if (threadIdx.x < LOAD_BLOCK_SIZE) {
        // Load Codebook
        if (threadIdx.x < threads_need_to_load) {
            const uint32_t load_thread_group_size = entry_to_load * _compression_ratio / THREAD_PACKED_LOAD_FP16_NUM;
            const uint32_t group_id  = threadIdx.x / load_thread_group_size;
            const uint32_t group_off = threadIdx.x % load_thread_group_size;
            for (int i = 0; i < load_iterations; i++) {
                *(PACKED_LOAD_TYPE_FP16*)(&codebook_shmem[i * block_packed_load_width + threadIdx.x * THREAD_PACKED_LOAD_FP16_NUM]) = 
                *(PACKED_LOAD_TYPE_FP16*)(&_codebook[codebook_offset + 
                                                (uint32_t) (i * subspaces_load_at_once) * subspace_stride + 
                                                group_id * subspace_stride + 
                                                group_off & THREAD_PACKED_LOAD_FP16_NUM + 
                                                (i % iterations_for_one_subspace) * block_packed_load_width
                                               ]);
            }
        }
    }
    else if (threadIdx.x >= LOAD_BLOCK_SIZE && threadIdx.x < LOAD_BLOCK_SIZE + 14) {
        init(bar + threadIdx.x - LOAD_BLOCK_SIZE, 256);
    }
    block.sync();
    
    if (threadIdx.x < 256) {
        // Compute Warps
        INPUT_TYPE* h_buf;
        INPUT_TYPE* d_buf;
        uint32_t tid = threadIdx.x;
        uint32_t warp_id = tid / WARP_SIZE;
        // Ensure H and D are ready to be filled
        barrier::arrival_token token_h0 = bar[ 0 +  0].arrive();
        barrier::arrival_token token_h1 = bar[ 0 +  1].arrive();
        barrier::arrival_token token_h2 = bar[ 0 +  2].arrive();
        barrier::arrival_token token_d0 = bar[10 +  0].arrive();
        barrier::arrival_token token_d1 = bar[10 +  1].arrive();
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, INPUT_TYPE, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, INPUT_TYPE, wmma::row_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, INPUT_TYPE> c0;
        wmma::fragment<wmma::accumulator, 16, 16, 16, INPUT_TYPE> c1;
        fill_fragment(c0, __float2half(0.0));
        fill_fragment(c1, __float2half(0.0));
        for (int k = 0; k < _k / K_BLOCK_TILE; k++) {
            bar[ 3 + k % 3].arrive_and_wait();  // Wait for H[k % 3] is filled
            bar[12 + k % 2].arrive_and_wait();  // Wait for D[k % 2] is filled
            // __syncthreads();
            // Compute H * D            
            h_buf = &h_shmem[(k % 3) * M_BLOCK_TILE * K_BLOCK_TILE];
            d_buf = &d_shmem[(k % 2) * K_BLOCK_TILE * N_BLOCK_TILE];

            // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) printf("[Compute] k = %d, Consuming H[%d] and D[%d]\n", k, k % 3, k % 2);
            // wmma matmul.
            for (int ik = 0; ik < K_BLOCK_TILE; ik += 16) {
                load_matrix_sync(a, (const INPUT_TYPE*) (&h_buf[(warp_id / 2) * 16 * K_BLOCK_TILE + ik]), K_BLOCK_TILE);
                load_matrix_sync(b, (const INPUT_TYPE*) (&d_buf[ik * N_BLOCK_TILE + (warp_id % 2) * (N_BLOCK_TILE / 2)]), N_BLOCK_TILE);
                mma_sync(c0, a, b, c0);
                load_matrix_sync(b, (const INPUT_TYPE*) (&d_buf[ik * N_BLOCK_TILE + (warp_id % 2) * (N_BLOCK_TILE / 2) + 16]), N_BLOCK_TILE);
                mma_sync(c1, a, b, c1);
            }


            barrier::arrival_token token_h_c = bar[ 0 + k % 3].arrive();  // Signal H[k % 3] is ready to be refilled
            barrier::arrival_token token_d_c = bar[10 + k % 2].arrive();  // Signal D[k % 2] is ready to be refilled
            __syncthreads();
        }
        store_matrix_sync((INPUT_TYPE*) (&_o[blockIdx.x * M_BLOCK_TILE * _n + blockIdx.y * N_BLOCK_TILE + (warp_id / 2) * 16 * _n + (warp_id % 2) * (N_BLOCK_TILE / 2)]), c0, _n, wmma::mem_row_major);
        store_matrix_sync((INPUT_TYPE*) (&_o[blockIdx.x * M_BLOCK_TILE * _n + blockIdx.y * N_BLOCK_TILE + (warp_id / 2) * 16 * _n + (warp_id % 2) * (N_BLOCK_TILE / 2) + 16]), c1, _n, wmma::mem_row_major);
    
    }
    else if (threadIdx.x >= 256 && threadIdx.x < 512) {
        // Dequant Warps
        WEIGHT_TYPE* w_buf;
        INPUT_TYPE* d_buf;
        uint32_t tid = threadIdx.x - 256;
        // Ensure W is ready to be filled
        barrier::arrival_token token_w0 = bar[ 6 +  0].arrive();
        barrier::arrival_token token_w1 = bar[ 6 +  1].arrive();
        __syncthreads();
        for (int k = 0; k < _k / K_BLOCK_TILE; k++) {
            bar[ 8 + k % 2].arrive_and_wait();  // Wait for W[k % 2] is filled
            bar[10 + k % 2].arrive_and_wait();  // Wait for D[k % 2] ready to be filled
            // __syncthreads();
            // D[k % 2] <- Dequant(W[k % 2])
            // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 256) printf("[Dequant] k = %d, Consuming W[%d], Producing D[%d]\n", k, k % 2, k % 2);
            w_buf = &w_shmem[(k % 2) * K_BLOCK_TILE * N_BLOCK_TILE / _compression_ratio];
            d_buf = &d_shmem[(k % 2) * K_BLOCK_TILE * N_BLOCK_TILE];

            uint32_t subspace_per_block = N_BLOCK_TILE / _compression_ratio;

            for (int t = tid; t < K_BLOCK_TILE * N_BLOCK_TILE / _compression_ratio; t += DEQUANT_WARP * WARP_SIZE) {
                switch (_compression_ratio) {
                    case 2:
                        *(uint32_t*)(&d_buf[t * _compression_ratio]) = *(uint32_t*)(&codebook_shmem[tid * _compression_ratio]);
                        break;
                    case 4:    
                        *(uint64_t*)(&d_buf[t * _compression_ratio]) = *(uint64_t*)(&codebook_shmem[(blockIdx.y * subspace_per_block + t % subspace_per_block) * subspace_stride + w_buf[t] * _compression_ratio]);
                        break;
                    case 8:
                        *(uint4*)(&d_buf[t * _compression_ratio]) = *(uint4*)(&codebook_shmem[(blockIdx.y * subspace_per_block + t % subspace_per_block) * subspace_stride + w_buf[t] * _compression_ratio]);
                        break;
                    default:
                        // Need for loop
                        break;
                }
            }

            barrier::arrival_token token_w_d = bar[ 6 + k % 2].arrive();  // Signal W[k % 2] is ready to be refilled
            barrier::arrival_token token_d_d = bar[12 + k % 2].arrive();  // Signal D[k % 2] is filled
            __syncthreads();
        }
    }
    else if (threadIdx.x >= 512 && threadIdx.x < 768) {
        // Load Warps
        INPUT_TYPE* h_buf;
        WEIGHT_TYPE* w_buf;
        uint32_t tid = threadIdx.x - 512;
        for (int k = 0; k < _k / K_BLOCK_TILE; k++) { 
            bar[ 0 + k % 3].arrive_and_wait();   // Wait for H[k % 3] ready to be filled
            bar[ 6 + k % 2].arrive_and_wait();   // Wait for W[k % 2] ready to be filled
            // __syncthreads();
            // Produce (Load) data to H[k % 3] and W[k % 2] buffer
            // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 512) printf("[   Load] k = %d, Producing H[%d] and W[%d]\n", k, k % 3, k % 2);
            h_buf = &h_shmem[(k % 3) * M_BLOCK_TILE * K_BLOCK_TILE];
            w_buf = &w_shmem[(k % 2) * K_BLOCK_TILE * N_BLOCK_TILE / _compression_ratio];

            uint32_t load_volume_fp16 = LOAD_WARP * WARP_SIZE * THREAD_PACKED_LOAD_FP16_NUM;
            uint32_t load_iterations = max((uint32_t) 1, (uint32_t) (M_BLOCK_TILE * K_BLOCK_TILE / load_volume_fp16));
            uint32_t thread_per_row = K_BLOCK_TILE / THREAD_PACKED_LOAD_FP16_NUM;
            uint32_t threads_need_to_load = min((uint32_t) (LOAD_WARP * WARP_SIZE), (uint32_t) (M_BLOCK_TILE * K_BLOCK_TILE / THREAD_PACKED_LOAD_FP16_NUM));
            for (int i = 0; i < load_iterations; i++) {
                if (tid < threads_need_to_load) {
                    *(PACKED_LOAD_TYPE_FP16*)(&h_buf[(i * (LOAD_WARP * WARP_SIZE / thread_per_row) + tid / thread_per_row) * K_BLOCK_TILE + (tid % thread_per_row) * THREAD_PACKED_LOAD_FP16_NUM]) = 
                    *(PACKED_LOAD_TYPE_FP16*)(&_h[blockIdx.x * M_BLOCK_TILE * _k + k * K_BLOCK_TILE + 
                                                  (i * (LOAD_WARP * WARP_SIZE / thread_per_row) + tid / thread_per_row) * _k + (tid % thread_per_row) * THREAD_PACKED_LOAD_FP16_NUM
                                                 ]);
                }
            }

            uint32_t load_volume_uint8 = LOAD_WARP * WARP_SIZE * THREAD_PACKED_LOAD_UINT8_NUM;
            load_iterations = max((uint32_t) 1, (uint32_t) (K_BLOCK_TILE * N_BLOCK_TILE / _compression_ratio) / load_volume_uint8);
            thread_per_row = (N_BLOCK_TILE / _compression_ratio) / THREAD_PACKED_LOAD_UINT8_NUM;
            threads_need_to_load = min((uint32_t) (LOAD_WARP * WARP_SIZE), (uint32_t) ((K_BLOCK_TILE * N_BLOCK_TILE / _compression_ratio) / THREAD_PACKED_LOAD_UINT8_NUM));
            for (int i = 0; i < load_iterations; i++) {
                if (tid < threads_need_to_load) {
                    *(PACKED_LOAD_TYPE_UINT8*)(&w_buf[(i * (LOAD_WARP * WARP_SIZE / thread_per_row) + tid / thread_per_row) * (N_BLOCK_TILE / _compression_ratio) + (tid % thread_per_row) * THREAD_PACKED_LOAD_UINT8_NUM]) = 
                    *(PACKED_LOAD_TYPE_UINT8*)(&_w[k * K_BLOCK_TILE * (_n) + blockIdx.y * N_BLOCK_TILE + 
                                                   (i * (LOAD_WARP * WARP_SIZE / thread_per_row) + tid / thread_per_row) * (_n) + (tid % thread_per_row) * THREAD_PACKED_LOAD_UINT8_NUM
                                                  ]);
                }
            }

            // __syncthreads();
            barrier::arrival_token token_h_l = bar[ 3 + k % 3].arrive();   // Signal H[k % 3] is filled
            barrier::arrival_token token_w_l = bar[ 8 + k % 2].arrive();   // Signal W[k % 2] is filled
            __syncthreads();
        }
    }
}

torch::Tensor vq_gemm(
    torch::Tensor h,
    torch::Tensor w,
    torch::Tensor codebook,
    int residual, int compression_ratio, int entry
)
{
    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);

    auto m = h.size(0);
    auto k = h.size(1);
    auto n = w.size(1);
    std::cout << m << " " << n * compression_ratio << " " << k << std::endl;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({m, n}, 0, options);

    int shared_memory_size = 0;
    /* Codebook */ shared_memory_size += entry * N_BLOCK_TILE * sizeof(INPUT_TYPE) / _INV_HOT_ENTRY;
    /* H        */ shared_memory_size += M_BLOCK_TILE * K_BLOCK_TILE * sizeof(INPUT_TYPE) * (TRIPLE_BUFFER);
    /* W        */ shared_memory_size += K_BLOCK_TILE * (N_BLOCK_TILE / compression_ratio) * sizeof(WEIGHT_TYPE) * DOUBLE_BUFFER;
    /* D        */ shared_memory_size += K_BLOCK_TILE * N_BLOCK_TILE * sizeof(INPUT_TYPE) * DOUBLE_BUFFER;
    /* Barrier  */ shared_memory_size += 14 * sizeof(barrier);

    auto h_ptr = reinterpret_cast<INPUT_TYPE*>(h.data_ptr<TORCH_INPUT_TYPE>());
    auto w_ptr = reinterpret_cast<WEIGHT_TYPE*>(w.data_ptr<TORCH_WEIGHT_TYPE>());
    auto codebook_ptr = reinterpret_cast<INPUT_TYPE*>(codebook.data_ptr<TORCH_INPUT_TYPE>());
    auto o_ptr = reinterpret_cast<INPUT_TYPE*>(o.data_ptr<TORCH_INPUT_TYPE>());

    dim3 grid(m / M_BLOCK_TILE, n / N_BLOCK_TILE);

    // dim3 grid_tmp(8, 8);
    auto kernel = vq_gemm_kernel<_INV_HOT_ENTRY, 128, 16, 0>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
#if PROFILING == 1
    for (int wm = 0; wm < 100; wm++) {
        kernel<<<grid, BLOCK_SIZE, shared_memory_size>>>(
            o_ptr,
            h_ptr,
            w_ptr,
            codebook_ptr,
            residual, compression_ratio, entry,
            m, n, k
        );
    cudaDeviceSynchronize();
    }
    cudaEventRecord(st);
    for (int iter = 0; iter < 500; iter++) {
#endif
    kernel<<<grid, BLOCK_SIZE, shared_memory_size>>>(
        o_ptr,
        h_ptr,
        w_ptr,
        codebook_ptr,
        residual, compression_ratio, entry,
        m, n, k
    );
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
#if PROFILING == 1
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << ms / 500.0 << std::endl;
#endif
    return o;
}