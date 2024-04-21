#include "cuda.h"
#include "cuda_fp16.h"
#include "stdio.h"

#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define BLOCK_TILE_K 16

#define WARP_PER_BLOCK 8

#define WARP_TILE_M 32
#define WARP_TILE_N 16
#define WARP_TILE_K 16

#define WARP_NUM_M (BLOCK_TILE_M / WARP_TILE_M)
#define WARP_NUM_N (BLOCK_TILE_N / WARP_TILE_N)

#define MMA_TILE_M 16
#define MMA_TILE_N 8
#define MMA_TILE_K 8

#define WARP_SIZE 32
#define WARP_GROUP_SIZE 128

__device__ __forceinline__ uint32_t convert_shmem_ptr_to_uint32(const void* shmem_ptr) {
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

// A, B and C are all row-major
// shared memory: currently dont consider bank conflict
//               A: 64 * 16 * 2Byte/item * double buffer = 2KB * 2
//               B: 64 * 16 * 2Byte/item * double buffer = 2KB * 2
//               C: 64 * 64 * 2Byte/item                 = 8KB
//
// shared memory snapshot:
// [  A | ~A |  B | ~B |        C       ]
// 0    2    4    6    8                16 (KB)
// A : 0x 0000 0000 0000 0000 ~ 0x 0000 0111 1111 1111
//~A : 0x 0000 1000 0000 0000 ~ 0x 0000 1111 1111 1111
//        0000 x000 0000 0000      0000 x000 0000 0000   
//  
// B : 0x 0001 0000 0000 0000 ~ 0x 0001 0111 1111 1111
//~B : 0x 0001 1000 0000 0000 ~ 0x 0001 1111 1111 1111
//        0000 x000 0000 0000      0000 x000 0000 0000
//
// A_switch = 0x 0000 1000 0000 0000 = 0x0800
// B_switch = 0x 0000 1000 0000 0000 = 0x0800
__global__ void ada_hgemm_64x64x16(
    half* A,
    half* B,
    half* C,
    half* D,
    uint32_t m,
    uint32_t n,
    uint32_t k
)
{
    __shared__ __align__(4 * 1024) uint8_t smem[16 * 1024];
    half* a_smem = reinterpret_cast<half*>(smem);           // 64 * 16
    half* b_smem = reinterpret_cast<half*>(smem + 4 * 1024);// 16 * 64
    half* c_smem = reinterpret_cast<half*>(smem + 8 * 1024);// 64 * 64
    // mma.m16n8k8 register frag
    half A_frag[4];
    half B_frag[2];
    half C_frag[4];
    half D_frag[4];
    // for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            A_frag[j] = __float2half(0.0f);
            B_frag[j / 2] = __float2half(0.0f);
            C_frag[j] = __float2half(0.0f);
            D_frag[j] = __float2half(0.0f);
        }
    // }

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;
    const uint32_t groupid = threadIdx.x / WARP_GROUP_SIZE;
    
    // Fisrt, load one A and B part to the shared as the frist stage of pipeline
    // For A(64 * 16 half), thread 0~127 read 8 half each, 2 threads handle one row
    // For B(16 * 64 half), thread 0~127 read 8 half each, 8 threads handle one row
    // So, 128 threads from group0 read A, 128 threads from group1 read B, 
    // CAUSION: use ldgsts to avoid register round trip
    uint64_t global_A_load_start = (blockIdx.x * BLOCK_TILE_M + threadIdx.x / 2) * k + 8 * (threadIdx.x % 2);
    uint64_t global_B_load_start = ((warp_id % 4) * 4 * n + blockIdx.y * BLOCK_TILE_N + (lane_id / 8) * n + (lane_id % 8) * 8);
    half* global_load_ptr = (groupid == 0) ? A + global_A_load_start : B + global_B_load_start;
    uint32_t shared_A_produce_addr = (threadIdx.x / 2) * 16 + 8 * (threadIdx.x % 2);
    uint32_t shared_B_produce_addr = (warp_id % 4) * 4 * 64 + (lane_id / 8) * 64 + (lane_id % 8) * 8;
    volatile uint32_t shared_produce_addr = (groupid == 0) ? convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_addr) : convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_addr);
    
    asm volatile (
        "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"  // 8 * sizeof(half) = 16 byte
        : : "r"(shared_produce_addr), "l"(global_load_ptr)
    );

    // Commit the async load
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();
    // Switch shared memory double buffer
    shared_produce_addr ^= 0x0800;

    uint32_t shared_A_consume_addr = (warp_id / WARP_NUM_N) * WARP_TILE_M * BLOCK_TILE_K + lane_id * BLOCK_TILE_K;
    uint32_t shared_B_consume_addr = (warp_id % WARP_NUM_N) * WARP_TILE_N + lane_id * 64;
    uint32_t shared_C_writeback_addr = (warp_id / WARP_NUM_N) * WARP_TILE_M * BLOCK_TILE_N + (warp_id % WARP_NUM_N) * WARP_TILE_N;
    // Load A and B tile in shared memory to the register using ldmatrix
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(*(uint32_t*)(&A_frag[0])), "=r"(*(uint32_t*)(&A_frag[2]))
        : "r"(convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_addr))
    );
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
        : "=r"(*(uint32_t*)(&B_frag[0]))
        : "r"(convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_addr))
    );

    for (int ko = 0; ko < k; ko += BLOCK_TILE_K) {                

        // Async load next A B tile into the shared memory
        // shared_produce_addr = (groupid == 0) ? convert_shmem_ptr_to_uint32(a_smem_ + shared_A_produce_addr) : convert_shmem_ptr_to_uint32(b_smem_ + shared_B_produce_addr);
        global_A_load_start += BLOCK_TILE_K;
        global_B_load_start += BLOCK_TILE_K * n;
        global_load_ptr = (groupid == 0) ? A + global_A_load_start : B + global_B_load_start;
        
        asm volatile(
            "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
            : : "r"(shared_produce_addr), "l"(global_load_ptr)
        );
        for (int _k = 0; _k < WARP_TILE_K; _k += MMA_TILE_K) {
            for (int _m = 0; _m < WARP_TILE_M; _m += MMA_TILE_M) {
                for (int _n = 0; _n < WARP_TILE_N; _n += MMA_TILE_N) {
                    // Register level pipeline is accomplished based on the hardware itself
                    // the scheduler issue 1(or 2?) instruction per cycle, TENSOR and L/S 
                    // instruction can be hardware pipelined IF THE SCOREBOARD IS OKAY!!!
                    // Which means the data dependency is resolved, and ping-pong reg have
                    // no data dependency in nature. (SO, NO EXPLICIT SYNCHRONIZE IS NEEDED)
                                  
                    // if (((_m == 0) && (_n == 0) && (ko == 16) && (_k == 0)) && (threadIdx.x < WARP_SIZE)) {
                    //     printf("%2d:%8.3f,%8.3f,%8.3f,%8.3f\n", threadIdx.x, __half2float(A_frag[0]), __half2float(A_frag[1]), __half2float(A_frag[2]), __half2float(A_frag[3]));
                    // }
                    
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%6};\n" // Call mma to do the compute, occupy the TENSOR pipeline
                        //"stmatrix.sync.aligned.m8n8.x2.shared.b16 [%7], {%0,%1};\n" // ?? Only for SM90?
                        : "=r"(*(uint32_t*)(&D_frag[0])), "=r"(*(uint32_t*)(&D_frag[2]))
                        : "r"(*(uint32_t*)(&A_frag[0])), "r"(*(uint32_t*)(&A_frag[2])), "r"(*(uint32_t*)(&B_frag[0])), "r"(*(uint32_t*)(&C_frag[0])), "r"(*(uint32_t*)(&C_frag[2]))
                        //"r"(convert_shmem_ptr_to_uint32(c_smem + shared_C_writeback_addr))
                    ); 
                    // if (((_m == 0) && (_n == 0) && (ko == 16) && (_k == 8)) && (threadIdx.x < WARP_SIZE)) {
                    //     printf("%2d:%8.3f,%8.3f,%8.3f,%8.3f\n", threadIdx.x, __half2float(D_frag[0]), __half2float(D_frag[1]), __half2float(D_frag[2]), __half2float(D_frag[3]));
                    // }
                    if (!((_m == WARP_TILE_M - MMA_TILE_M) && (_n == WARP_TILE_N - MMA_TILE_N))) {
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(*(uint32_t*)(&B_frag[0]))
                            : "r"(convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_addr + (_n + MMA_TILE_N) % WARP_TILE_N + _k * 64))
                        );
                    }
                    else {
                        if (_k == WARP_TILE_K - MMA_TILE_K) {
                            shared_B_consume_addr ^= 0x400;
                        }
                        // Compute last frag, load next k's frag
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                            : "=r"(*(uint32_t*)(&B_frag[0]))
                            : "r"(convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_addr + (_n + MMA_TILE_N) % WARP_TILE_N + ((_k + MMA_TILE_K) % WARP_TILE_K) * 64))
                        );
                    }

                    // Store D_frag (partial sum) to shared memory
                    asm volatile(
                        "{.reg .f16x2 f<2>;\n"
                        " ld.shared.u32 f0, [%0];\n"
                        " ld.shared.u32 f1, [%0 + 1024];\n"
                        " add.f16x2 f0, f0, %1;\n"
                        " add.f16x2 f1, f1, %2;\n"
                        " st.shared.u32 [%0], f0;\n"
                        " st.shared.u32 [%0 + 1024], f1;}\n"
                        : 
                        : "r"(convert_shmem_ptr_to_uint32(c_smem + shared_C_writeback_addr + (lane_id / 4) * 64 + (lane_id % 4) * 2          + _m * 64 + _n + 0)),
                          "r"(*(uint32_t*)&(D_frag[0])), "r"(*(uint32_t*)&(D_frag[2]))

                    );
                }
                if (!(_m == WARP_TILE_M - MMA_TILE_M)) {
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(*(uint32_t*)(&A_frag[0])), "=r"(*(uint32_t*)(&A_frag[2]))
                        : "r"(convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_addr + ((_m + MMA_TILE_M) % WARP_TILE_M) * BLOCK_TILE_K + _k))
                    );
                }
                else {
                    if (_k == WARP_TILE_K - MMA_TILE_K) {
                        shared_A_consume_addr ^= 0x400;
                    }
                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(*(uint32_t*)(&A_frag[0])), "=r"(*(uint32_t*)(&A_frag[2]))
                        : "r"(convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_addr + ((_m + MMA_TILE_M) % WARP_TILE_M) * BLOCK_TILE_K + (_k + MMA_TILE_K) % WARP_TILE_K))
                    );
                }
            }
            if (_k == WARP_TILE_K - MMA_TILE_K) {
                asm volatile("cp.async.wait_all;\n"::);
                __syncthreads();
                shared_produce_addr ^= 0x0800;
            }
        }
    }
    *(uint4*)(&D[blockIdx.x * BLOCK_TILE_M * n + blockIdx.y * BLOCK_TILE_N + (warp_id / 4) * 32 * n + (warp_id % 4) * 16 + lane_id * n])      = *(uint4*)(&c_smem[(warp_id / 4) * 32 * 64 + (warp_id % 4) * 16 + lane_id * 64]);
    *(uint4*)(&D[blockIdx.x * BLOCK_TILE_M * n + blockIdx.y * BLOCK_TILE_N + (warp_id / 4) * 32 * n + (warp_id % 4) * 16 + lane_id * n + 8])  = *(uint4*)(&c_smem[(warp_id / 4) * 32 * 64 + (warp_id % 4) * 16 + lane_id * 64 + 8]);
}

int main(int argc, char** argv) {
    constexpr uint32_t M = 1024;
    constexpr uint32_t N = 1024;
    constexpr uint32_t K = 1024;

    half *h_A, *d_A;
    half *h_B, *d_B;
    half *h_C, *d_C;
    half *h_D, *d_D;
    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C = new half[M * N];
    h_D = new half[M * N];

    cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(half) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(half) * K * N);
    cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_D), sizeof(half) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = __float2half((1.0 * i + 2.0 * j) / 100.0);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = __float2half(1.0 * (i + 1) * (j + 1) / 100.0);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_C[i * M + j] = __float2half(0.0f);
        }
    }

    cudaMemcpy(reinterpret_cast<void*>(d_A), h_A, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_B), h_B, sizeof(half) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_C), h_C, sizeof(half) * M * N, cudaMemcpyHostToDevice);
    dim3 grid(M / BLOCK_TILE_M, N / BLOCK_TILE_N);
    ada_hgemm_64x64x16<<<grid, 256>>>(d_A, d_B, d_C, d_D, M, N, K);
    cudaMemcpy(h_D, reinterpret_cast<void*>(d_D), sizeof(half) * M * N, cudaMemcpyDeviceToHost);
    printf("\n");
    // for (int i = 0; i < 128; i++) {
    //     for (int j = 0; j < 128; j++) {
    //         // printf("%08.3f%c", __half2float(h_B[i * N + j]), (j == 7) ? '\n' : ' ');
    //         half res = __float2half(0.0);
    //         for (int k = 0; k < 64; k++) {
    //             auto tmp = __hmul(h_A[i * K + k], h_B[k * N + j]);
    //             res = __hadd(res, tmp);
    //         }
    //         printf("%d%c", ((abs(__half2float(h_D[i * N + j] - res)) / __half2float(res)) > 0.01) ? 1 : 0, (j == 127) ? '\n' : ' ');
    //         // printf("%12.3f%c", __half2float(res), (j == 7) ? '\n' : ' ');
    //         // printf("%12.3f%c", __half2float(h_D[i * N + j]), (j == 63) ? '\n' : ' ');
    //         // printf("%f%c", res, (j == 15) ? '\n' : ' ');
    //     }
    // }
    // for (int i = 0; i < 16; i++) {
    //     for (int j = 16; j < 24; j++) {
    //         printf("%12.3f%c", __half2float(h_A[i * K + j]), (j == 23) ? '\n' : ' ');
    //     }
    // }
    return 0;
}