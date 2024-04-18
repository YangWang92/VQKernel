#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <torch/extension.h>
#include "stdio.h"
#include <iostream>

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

// #define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
// void checkLast(const char* const file, const int line)
// {
//     cudaError_t const err{cudaGetLastError()};
//     if (err != cudaSuccess)
//     {
//         std::cerr << "CUDA Runtime Error at: " << file << ":" << line
//                   << std::endl;
//         std::cerr << cudaGetErrorString(err) << std::endl;
//         // We don't exit when we encounter CUDA errors in this example.
//         // std::exit(EXIT_FAILURE);
//     }
// }

#define WARP_SIZE 32
#define WARP_NUM 16
#define MAX_SHARED_MEM 32768

// We assume the compressed data are:
// [ Residual 1 ][ Residual 2 ]...
// [Space1][Space2]...

// Every block conduct 4 supspaces from one residual
#define _COMPRESSED_DIMS_PER_BLOCK 4
#define _ROWS_BLOCK_DO_AT_ONCE 512

#define TRUE 1
#define FALSE 0

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

template <
    size_t BATCH_SIZE,  // <= 8
    size_t BLOCK_SIZE,
    size_t HIDDEN_DIM,
    size_t ROWS_BLOCK_DO_AT_ONCE,
    size_t COMPRESSED_DIMS_PER_BLOCK,
    size_t INV_HOT_ENTRY,
    size_t PACKED_LOAD_WIDTH,
    size_t B32REG_LIMIT,
    size_t ENABLE_ENTRY_CENTRIC
>
void __global__ vq_gemv_kernel(
    INPUT_TYPE* _o,
    INPUT_TYPE* _h,
    WEIGHT_TYPE* _w,
    INPUT_TYPE* _codebook,
    uint32_t _residuals, uint32_t _compression_ratio, uint32_t _entry,
    INPUT_TYPE* _verify
)
{
    const uint32_t THREAD_PACKED_LOAD_WIDTH = PACKED_LOAD_WIDTH / (8 * sizeof(INPUT_TYPE));
    const uint32_t COMPLETE_BLOCK_PACKED_LOAD_WIDTH = BLOCK_SIZE * THREAD_PACKED_LOAD_WIDTH;
    using PACKED_LOAD_TYPE = typename packed_vec<INPUT_TYPE, THREAD_PACKED_LOAD_WIDTH>::Type;
    using BATCHED_H_LOAD_TYPE = typename packed_vec<INPUT_TYPE, BATCH_SIZE>::Type;
    using COMPRESSED_WEIGHT_LOAD_TYPE = typename packed_vec<WEIGHT_TYPE, COMPRESSED_DIMS_PER_BLOCK>::Type;


    extern __shared__ uint8_t smem[];
    // codebook_shmem.shape = (COMPRESSED_DIMS_PER_BLOCK, _entry [/ INV], _compression_ratio)
    INPUT_TYPE *codebook_shmem = reinterpret_cast<INPUT_TYPE*>(smem);

    const uint32_t residual_id = blockIdx.x % _residuals;
    const uint32_t subspace_group_id = blockIdx.x / _residuals;
    const uint32_t codebook_offset = residual_id * (HIDDEN_DIM / _compression_ratio) * _entry * _compression_ratio + 
                                     subspace_group_id * COMPRESSED_DIMS_PER_BLOCK * _entry * _compression_ratio;
    const uint32_t codebook_to_access = COMPRESSED_DIMS_PER_BLOCK * _entry * _compression_ratio;
    const uint32_t codebook_to_load = codebook_to_access / INV_HOT_ENTRY;
    const uint32_t entry_to_load = _entry / INV_HOT_ENTRY;

    // Load codebook
    const uint32_t threads_need_to_load = min((uint32_t) BLOCK_SIZE, (uint32_t) (codebook_to_load / THREAD_PACKED_LOAD_WIDTH));
    const uint32_t block_packed_load_width = threads_need_to_load * THREAD_PACKED_LOAD_WIDTH;
    const uint32_t load_iterations = max((uint32_t) 1, (uint32_t) (codebook_to_load / block_packed_load_width));
    const uint32_t subspace_stride = _entry * _compression_ratio;
    float subspaces_load_at_once = block_packed_load_width / (entry_to_load * _compression_ratio);
    const uint32_t iterations_for_one_subspace = max((uint32_t) 1, (uint32_t) (1.0 / subspaces_load_at_once));

    if (threadIdx.x < threads_need_to_load) {
        const uint32_t load_thread_group_size = entry_to_load * _compression_ratio / THREAD_PACKED_LOAD_WIDTH;
        const uint32_t group_id = threadIdx.x / load_thread_group_size;
        const uint32_t group_off= threadIdx.x % load_thread_group_size;
        for (int i = 0; i < load_iterations; i++) {
            *(PACKED_LOAD_TYPE*)(&codebook_shmem[i * block_packed_load_width + threadIdx.x * THREAD_PACKED_LOAD_WIDTH]) = 
            *(PACKED_LOAD_TYPE*)(&_codebook[codebook_offset + 
                                            (uint32_t) (i * subspaces_load_at_once) * subspace_stride + 
                                            group_id * subspace_stride + 
                                            group_off * THREAD_PACKED_LOAD_WIDTH + 
                                            (i % iterations_for_one_subspace) * block_packed_load_width
                                           ]);
        }
    }
    __syncthreads();                  

    // Verify codebook load correctness.
    // if (threadIdx.x == 0 && blockIdx.x == 2) {
    //     for (int i = 0; i < 1; i++) {
    //         for (int j = 0; j < COMPRESSED_DIMS_PER_BLOCK; j++) {
    //             for (int k = 0; k < entry_to_load; k++) {
    //                 for (int x = 0; x < _compression_ratio; x++) {
    //                     printf("% 5.3f%c", __half2float(codebook_shmem[i * (COMPRESSED_DIMS_PER_BLOCK * entry_to_load * _compression_ratio) + j * (entry_to_load * _compression_ratio) + k * (_compression_ratio) + x]), (x == _compression_ratio - 1) ? '|' : ' ');
    //                 }
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    uint32_t set_mask[(ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE];
    #pragma unroll
    for (int _ = 0; _ < (ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE; _++) set_mask[_] = 0x00000000;
    
    INPUT_TYPE *h_shmem = reinterpret_cast<INPUT_TYPE*>(smem + sizeof(INPUT_TYPE) * codebook_to_load);
    WEIGHT_TYPE *w_compressed_ping = reinterpret_cast<WEIGHT_TYPE*>(h_shmem + sizeof(INPUT_TYPE) * HIDDEN_DIM);
    // WEIGHT_TYPE *w_compressed_pong = reinterpret_cast<WEIGHT_TYPE*>(w_compressed_ping + COMPRESSED_DIMS_PER_BLOCK * ROWS_BLOCK_DO_AT_ONCE * sizeof(WEIGHT_TYPE));
    half2 *reduce_workspace = reinterpret_cast<half2*>(w_compressed_ping + COMPRESSED_DIMS_PER_BLOCK * ROWS_BLOCK_DO_AT_ONCE * sizeof(WEIGHT_TYPE));

    INPUT_TYPE h_reg[(ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE][BATCH_SIZE];
    INPUT_TYPE w_dequant[(ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE][B32REG_LIMIT * (sizeof(uint32_t) / sizeof(INPUT_TYPE)) / ((ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE)];
    const uint32_t register_iter = max((uint32_t) 1, (uint32_t) ((_compression_ratio * COMPRESSED_DIMS_PER_BLOCK) / (B32REG_LIMIT * (sizeof(uint32_t) / sizeof(INPUT_TYPE)) / ((ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE))));
    // Load weight and dequant first, since all input vector use this weight
    for (int begin_row = 0; begin_row < HIDDEN_DIM; begin_row += ROWS_BLOCK_DO_AT_ONCE) {
        
        // TODO: Optimize: can double buffer here.
        #pragma unroll
        for (int tid = threadIdx.x; tid < ROWS_BLOCK_DO_AT_ONCE; tid += BLOCK_SIZE) {
            if (tid < HIDDEN_DIM) {
                // TODO: If COMPRSSED_DIMS_PER_BLOCK = 32, need load 256 bit, need for loop.
                *(COMPRESSED_WEIGHT_LOAD_TYPE*)(&w_compressed_ping[tid * COMPRESSED_DIMS_PER_BLOCK]) = 
                *(COMPRESSED_WEIGHT_LOAD_TYPE*)(&_w[(begin_row + tid) * ((HIDDEN_DIM / _compression_ratio) * _residuals) + residual_id * (HIDDEN_DIM / _compression_ratio) + subspace_group_id * COMPRESSED_DIMS_PER_BLOCK]);
            
                *(BATCHED_H_LOAD_TYPE*)(&h_reg[tid / BLOCK_SIZE][0]) = 
                *(BATCHED_H_LOAD_TYPE*)(&_h[(begin_row + tid) * BATCH_SIZE]);
            }
        }
        __syncthreads();

        INPUT_TYPE* entry_addr;
        for (int ri = 0; ri < register_iter; ri++) {
            // Dequant: Entry Centric
            for (int e = 0; e < ENABLE_ENTRY_CENTRIC; e++) {                                
                entry_addr = &codebook_shmem[(ri * (COMPRESSED_DIMS_PER_BLOCK / register_iter)) * entry_to_load * _compression_ratio + e * _compression_ratio];
                for (int tid = threadIdx.x; tid < ROWS_BLOCK_DO_AT_ONCE; tid += BLOCK_SIZE) {
                    uint8_t id;
                    if (tid < HIDDEN_DIM) {
                        for (int s = 0; s < COMPRESSED_DIMS_PER_BLOCK / register_iter; s++) {
                            id = w_compressed_ping[tid * COMPRESSED_DIMS_PER_BLOCK + ri * (COMPRESSED_DIMS_PER_BLOCK / register_iter) + s];
                            if (id == e) {
                                set_mask[tid / BLOCK_SIZE] |= (0x1 << s);
                                for (int _ = 0; _ < _compression_ratio; _ += 8) {
                                    if (_compression_ratio >= 8) {
                                        *(uint4*)(&w_dequant[tid / BLOCK_SIZE][s * _compression_ratio + _]) = 
                                        *(uint4*)(&entry_addr[_]);
                                    }
                                    else if (_compression_ratio == 4) {                                        
                                        *(uint64_t*)(&w_dequant[tid / BLOCK_SIZE][s * _compression_ratio + _]) = 
                                        *(uint64_t*)(&entry_addr[_]);
                                    }
                                    else if (_compression_ratio == 2) {
                                        *(uint32_t*)(&w_dequant[tid / BLOCK_SIZE][s * _compression_ratio + _]) = 
                                        *(uint32_t*)(&entry_addr[_]);
                                    }
                                }
                                entry_addr += (entry_to_load * _compression_ratio);
                            }
                        }
                    }
                }
            }

            // Dequant: Original
            for (int tid = threadIdx.x; tid < ROWS_BLOCK_DO_AT_ONCE; tid += BLOCK_SIZE) {
                if (tid < HIDDEN_DIM) {
                    for (int s = 0; s < COMPRESSED_DIMS_PER_BLOCK / register_iter; s++) {
                        if (!(set_mask[tid / BLOCK_SIZE] & (0x1 << s))) {
                            uint32_t id = (uint32_t) w_compressed_ping[tid * COMPRESSED_DIMS_PER_BLOCK + ri * (COMPRESSED_DIMS_PER_BLOCK / register_iter) + s];
                            entry_addr = &codebook_shmem[(ri * (COMPRESSED_DIMS_PER_BLOCK / register_iter) + s) * entry_to_load * _compression_ratio + id * _compression_ratio];                        
                            for (int _ = 0; _ < _compression_ratio; _ += 8) {
                                if (_compression_ratio >= 8) {
                                    *(uint4*)(&w_dequant[tid / BLOCK_SIZE][s * _compression_ratio + _]) = 
                                    *(uint4*)(&entry_addr[_]);
                                }
                                else if (_compression_ratio == 4) {                                        
                                    *(uint64_t*)(&w_dequant[tid / BLOCK_SIZE][s * _compression_ratio + _]) = 
                                    *(uint64_t*)(&entry_addr[_]);
                                }
                                else if (_compression_ratio == 2) {
                                    *(uint32_t*)(&w_dequant[tid / BLOCK_SIZE][s * _compression_ratio + _]) = 
                                    *(uint32_t*)(&entry_addr[_]);
                                }
                            }                        
                        }
                    }
                }
            }
        }
        
        // Verify the decoding correctness
        // if (threadIdx.x % 32 == 0 && blockIdx.x == 0) {
        //     for (int i = 0; i < COMPRESSED_DIMS_PER_BLOCK * _compression_ratio; i++) {
        //         printf("% 5.3f%c", __half2float(w_dequant[0][i]), (i == COMPRESSED_DIMS_PER_BLOCK * _compression_ratio - 1) ? '\n' : ' ');
        //     }
        // }
        for (int b = 0; b < BATCH_SIZE; b++) {

            // TODO: Optimize: __hadd2, __hmul2
            half2 h2_reg = __half2(h_reg[threadIdx.x / BLOCK_SIZE][b], h_reg[threadIdx.x / BLOCK_SIZE][b]);
            for (int d = 0; d < COMPRESSED_DIMS_PER_BLOCK * _compression_ratio; d+=2) {
                half2 reduce[(ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE];
                for (int tid = threadIdx.x; tid < ROWS_BLOCK_DO_AT_ONCE; tid += BLOCK_SIZE) {
                    if (tid < HIDDEN_DIM) {
                        reduce[tid / BLOCK_SIZE] = __hmul2(h2_reg, *(half2*)(&w_dequant[tid / BLOCK_SIZE][d]));
                        // if (blockIdx.x == 0 && b == 0 && d == 0) {
                        //     printf("%d : % 5.3f * %5.3f = % 5.3f\n", threadIdx.x, __half2float(h_reg[tid / BLOCK_SIZE][b]), __half2float(w_dequant[tid / BLOCK_SIZE][d]), __half2float(reduce[tid / BLOCK_SIZE]));
                        // }
                    }
                }
                // for (int i = 1; i < (ROWS_BLOCK_DO_AT_ONCE + (BLOCK_SIZE - 1)) / BLOCK_SIZE; i++) {
                //     reduce[0] = __hadd(reduce[0], reduce[i]);
                // }
                
                // Verify
                // print(h[114])
                // print(new_codebook[32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 : 32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 + 4])
                // print(h[114] * new_codebook[32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 + 2 : 32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 + 3])
                // if (threadIdx.x == 114 && blockIdx.x == 5 && d == 14) {
                //     printf("%d: % 5.3f % 5.3f % 5.3f\n", b, __half2float(reduce[0]), __half2float(h_reg[0][b]), __half2float(w_dequant[0][d]));
                // }

                // Blockwise reduce
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                    reduce[0] = __hadd2(reduce[0], __shfl_down_sync(0xffffffff, reduce[0], mask));
                }
                if (threadIdx.x % WARP_SIZE == 0) {
                    reduce_workspace[threadIdx.x / WARP_SIZE] = reduce[0];
                }
                __syncthreads();
                if (threadIdx.x < WARP_NUM) {
                    reduce[0] = reduce_workspace[threadIdx.x];
                }
                if (threadIdx.x < WARP_SIZE) {
                    #pragma unroll
                    for (int mask = WARP_NUM / 2; mask > 0; mask >>= 1) {
                        reduce[0] = __hadd2(reduce[0], __shfl_down_sync(0x0000ffff, reduce[0], mask));
                    }
                }
                if (threadIdx.x == 0) {
                    // If residual == 1, no atomic needed.
                    *(half2*)(&_o[b * HIDDEN_DIM + subspace_group_id * COMPRESSED_DIMS_PER_BLOCK * _compression_ratio + d]) = __hadd2(*(half2*)(&_o[b * HIDDEN_DIM + subspace_group_id * COMPRESSED_DIMS_PER_BLOCK * _compression_ratio + d]), *(half2*)(&reduce[0]));
                    // atomicAdd((half2*)&_o[b * HIDDEN_DIM + subspace_group_id * COMPRESSED_DIMS_PER_BLOCK * _compression_ratio + d], *(half2*)(&reduce[0]));
                }
            }
        }
    }
}

torch::Tensor vq_gemv(
    torch::Tensor h,
    torch::Tensor w,
    torch::Tensor codebook,
    int residual, int compression_ratio, int entry
)
{

    auto hidden_dim = h.size(0);
    auto batch = h.size(1);

    INPUT_TYPE* h_verify, *d_verify;
    h_verify = new INPUT_TYPE[residual * (hidden_dim / compression_ratio) * entry * compression_ratio];
    for (int i = 0; i < residual * (hidden_dim / compression_ratio) * entry * compression_ratio; i++) {
        h_verify[i] = __float2half(0.0);
    }
    cudaMalloc(reinterpret_cast<void**>(&d_verify), sizeof(INPUT_TYPE) * residual * (hidden_dim / compression_ratio) * entry * compression_ratio);

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({batch, hidden_dim}, 0, options);

    auto h_ptr = reinterpret_cast<INPUT_TYPE*>(h.data_ptr<TORCH_INPUT_TYPE>());
    auto w_ptr = reinterpret_cast<WEIGHT_TYPE*>(w.data_ptr<TORCH_WEIGHT_TYPE>());
    auto codebook_ptr = reinterpret_cast<INPUT_TYPE*>(codebook.data_ptr<TORCH_INPUT_TYPE>());
    auto o_ptr = reinterpret_cast<INPUT_TYPE*>(o.data_ptr<TORCH_INPUT_TYPE>());

    dim3 grid((residual * (hidden_dim / compression_ratio)) / _COMPRESSED_DIMS_PER_BLOCK);

    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);

    if ((batch == 8) && (hidden_dim == 4096)) {
        // ASSERTION: 1 * hidden_dim * sizeof(INPUT_TYPE) + _COMPRESSED_DIMS_PER_BLOCK * entry * compression_ratio * sizeof(INPUT_TYPE) + 2 * _ROWS_BLOCK_DO_AT_ONCE * _COMPRESSED_DIMS_PER_BLOCK * sizeof(WEIGHT_TYPE) < MAX_SHARED_MEM
        // hidden = 8192, _COMPRESSED_DIMS_PER_BLOCK = 4, _ROWS_BLOCK_DO_AT_ONCE = 1024 just okay!
        auto kernel = vq_gemv_kernel<8, WARP_NUM * WARP_SIZE, 4096, _ROWS_BLOCK_DO_AT_ONCE, _COMPRESSED_DIMS_PER_BLOCK, 1, 128, 16, 0>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEM);
#if PROFILING == 1
        for (int wm = 0; wm < 50; wm++) {
        kernel<<<grid, WARP_NUM * WARP_SIZE, MAX_SHARED_MEM>>>(
            o_ptr, 
            h_ptr,
            w_ptr,
            codebook_ptr, 
            residual, compression_ratio, entry,
            d_verify
        );
        }
        cudaEventRecord(st);
        for (int iter = 0; iter < 250; iter++) {
#endif
        kernel<<<grid, WARP_NUM * WARP_SIZE, MAX_SHARED_MEM>>>(
            o_ptr, 
            h_ptr,
            w_ptr,
            codebook_ptr, 
            residual, compression_ratio, entry,
            d_verify
        );
#if PROFILING == 1
        }
        cudaEventRecord(ed);
        cudaEventSynchronize(ed);
        float ms;
        cudaEventElapsedTime(&ms, st, ed);
        std::cout << ms / 250.0 << std::endl;
#endif
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_verify, reinterpret_cast<void*>(d_verify), sizeof(INPUT_TYPE) * residual * (hidden_dim / compression_ratio) * entry * compression_ratio, cudaMemcpyDeviceToHost);
    
    // CHECK_LAST_CUDA_ERROR();

    // INPUT_TYPE* codebook_host = new INPUT_TYPE[residual * (hidden_dim / compression_ratio) * entry * compression_ratio];
    // cudaMemcpy(codebook_host, reinterpret_cast<void*>(codebook_ptr), sizeof(INPUT_TYPE) * residual * (hidden_dim / compression_ratio) * entry * compression_ratio, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < residual; i++) {
    //     for (int j = 0; j < (hidden_dim / compression_ratio); j++) {
    //         for (int k = 0; k < entry; k++) {
    //             for (int x = 0; x < compression_ratio; x++) {
    //                 printf("% 5.3f%c", __half2float(codebook_host[i * (hidden_dim / compression_ratio) * entry * compression_ratio + j * entry * compression_ratio + k * compression_ratio + x]), (x == compression_ratio - 1) ? '|' : ' ');
    //             }
    //         }
    //         printf("\n");
    //     }
    // }

    return o;
}