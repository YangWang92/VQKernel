#include "cuda.h"
#include "cuda_fp16.h"
#include "stdio.h"

#define BLOCK_TILE_M 128
#define BLOCK_TILE_N 64
#define BLOCK_TILE_K 32

#define WARP_PER_BLOCK 16
#define WARP_SIZE 32
#define SHARED_MEM_OFFSET 8
#define DATA_SIZE 2 // sizeof(half) = 2 Byte
#define A_SHMEM_SIZE_IN_BYTE (BLOCK_TILE_M * (BLOCK_TILE_K + SHARED_MEM_OFFSET)) * DATA_SIZE
#define B_SHMEM_SIZE_IN_BYTE (BLOCK_TILE_K * (BLOCK_TILE_N + SHARED_MEM_OFFSET)) * DATA_SIZE

#define WARP_TILE_M 16
#define WARP_TILE_N 32
#define WARP_TILE_K 32

#define WARP_NUM_M (BLOCK_TILE_M / WARP_TILE_M)
#define WARP_NUM_N (BLOCK_TILE_N / WARP_TILE_N)

#define MMA_TILE_M 16
#define MMA_TILE_N 8
#define MMA_TILE_K 8

#define M 1024
#define N 1024
#define K 1024

constexpr int iter = 1;

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

__global__ void ada_hgemm_m128n64k32_fp16_sm89(
    half* A,
    half* B,
    half* C,
    half* D,
    uint32_t m,
    uint32_t n,
    uint32_t k
) 
{
    __shared__ __align__(1 * 1024) uint8_t smem[2 * (A_SHMEM_SIZE_IN_BYTE + B_SHMEM_SIZE_IN_BYTE)];
    half *a_smem = reinterpret_cast<half*>(smem);
    half *b_smem = reinterpret_cast<half*>(smem + 2 * A_SHMEM_SIZE_IN_BYTE);
    int32_t shmem_switch_A = 5120;
    int32_t shmem_switch_B = 2304;

    half A_frag[2][4];
    half B_frag[2][2];
    half C_frag[2][4];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            A_frag[i][j] = __float2half(0.0f);
            C_frag[i][j] = __float2half(0.0f);
            B_frag[i][j / 2] = __float2half(0.0f);
        }
    }

    const uint32_t warp_id = threadIdx.x / WARP_SIZE;
    const uint32_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t global_A_load_start = blockIdx.x * BLOCK_TILE_M * k + (threadIdx.x /  4) * k + (threadIdx.x %  4) * 8;
    uint32_t global_B_load_start = blockIdx.y * BLOCK_TILE_N     + (threadIdx.x / 16) * n + (threadIdx.x % 16) * 4;
    uint32_t shared_A_produce_start = (threadIdx.x /  4) * (BLOCK_TILE_K + SHARED_MEM_OFFSET) + (threadIdx.x %  4) * 8;
    uint32_t shared_B_produce_start = (threadIdx.x / 16) * (BLOCK_TILE_N + SHARED_MEM_OFFSET) + (threadIdx.x % 16) * 4;
    uint32_t shared_A_produce_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_start);
    uint32_t shared_B_produce_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_start);

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        "cp.async.ca.shared.global [%2], [%3], 8 ;\n"
        :
        : "r"(shared_A_produce_start_addr), "l"(A + global_A_load_start)
          "r"(shared_B_produce_start_addr), "l"(B + global_B_load_start)
    );
    asm volatile("cp.async.wait_all;\n"::);
    __syncthreads();

    shared_A_produce_start += shmem_switch_A;
    shared_B_produce_start += shmem_switch_B;

    uint32_t shared_A_consume_start = (warp_id % WARP_NUM_M) * WARP_TILE_M * (BLOCK_TILE_K + SHARED_MEM_OFFSET) + lane_id * (BLOCK_TILE_K + SHARED_MEM_OFFSET);
    uint32_t shared_B_consume_start = (warp_id / WARP_NUM_M) * WARP_TILE_N                                      + lane_id * (BLOCK_TILE_N + SHARED_MEM_OFFSET);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16       {%0, %1}, [%3];\n"
        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%2}    , [%4];\n"
        : "=r"(*(uint32_t*)(&A_frag[0][0])), "=r"(*(uint32_t*)(&A_frag[0][2])), "=r"(*(uint32_t*)((&B_frag[0][0])))
        : "r"(convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_start)), "r"(convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_start))
    );

    for (int ko = 0; ko < k; ko += BLOCK_TILE_K) {
        global_A_load_start += BLOCK_TILE_K;
        global_B_load_start += BLOCK_TILE_K * n;
        shared_A_produce_start_addr = convert_shmem_ptr_to_uint32(a_smem + shared_A_produce_start);
        shared_B_produce_start_addr = convert_shmem_ptr_to_uint32(b_smem + shared_B_produce_start);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 16;\n"
            "cp.async.ca.shared.global [%2], [%3], 8 ;\n"
            :
            : "r"(shared_A_produce_start_addr), "l"(A + global_A_load_start)
              "r"(shared_B_produce_start_addr), "l"(B + global_B_load_start)
        );

        for (int _m = 0; _m < WARP_TILE_M; _m += MMA_TILE_M) {
            for (int _n = 0; _n < WARP_TILE_N; _n += MMA_TILE_N) {
                int ping_pong = 0;
                #pragma unroll
                for (int _k = 0; _k < WARP_TILE_K; _k += MMA_TILE_K) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
                        : "=r"(*(uint32_t*)(&C_frag[ping_pong ^ 0x1][0])), 
                          "=r"(*(uint32_t*)(&C_frag[ping_pong ^ 0x1][2]))
                        : "r"(*(uint32_t*)(&A_frag[ping_pong][0])), 
                          "r"(*(uint32_t*)(&A_frag[ping_pong][2])), 
                          "r"(*(uint32_t*)(&B_frag[ping_pong][0])), 
                          "r"(*(uint32_t*)(&C_frag[ping_pong][0])), 
                          "r"(*(uint32_t*)(&C_frag[ping_pong][2]))
                    );
                    if ((_m == WARP_TILE_M - MMA_TILE_M) && (_n == WARP_TILE_N - MMA_TILE_N) && (_k == WARP_TILE_K - MMA_TILE_K)) {
                        asm volatile("cp.async.wait_all;\n"::);
                        __syncthreads();
                        shared_A_consume_start += shmem_switch_A;
                        shared_B_consume_start += shmem_switch_B;
                        shmem_switch_A *= -1;
                        shmem_switch_B *= -1;
                        shared_A_produce_start += shmem_switch_A;
                        shared_B_produce_start += shmem_switch_B;
                    }
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x2.shared.b16       {%0, %1}, [%3];\n"
                        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%2}    , [%4];\n"
                        : "=r"(*(uint32_t*)(&A_frag[ping_pong ^ 0x1][0])), 
                          "=r"(*(uint32_t*)(&A_frag[ping_pong ^ 0x1][2])), 
                          "=r"(*(uint32_t*)(&B_frag[ping_pong ^ 0x1][0]))
                        : "r"(convert_shmem_ptr_to_uint32(a_smem + shared_A_consume_start + ((_k + MMA_TILE_K) % WARP_TILE_K))), 
                          "r"(convert_shmem_ptr_to_uint32(b_smem + shared_B_consume_start + ((_k + MMA_TILE_K) % WARP_TILE_K) * (BLOCK_TILE_N + SHARED_MEM_OFFSET) + (_n + ((_k == WARP_TILE_K - MMA_TILE_K) ? MMA_TILE_N : 0)) % WARP_TILE_N))
                    );
                    ping_pong ^= 0x1;                    
                }
                *(uint32_t*)(&D[blockIdx.x * BLOCK_TILE_M * n + blockIdx.y * BLOCK_TILE_N + (warp_id % WARP_NUM_M) * WARP_TILE_M * n + (warp_id % WARP_NUM_M) * WARP_TILE_N + _m * n + _n + (lane_id / 4) * n + (lane_id % 4) * 2        ]) += *(uint32_t*)(&C_frag[ping_pong][0]);
                // *(uint32_t*)(&D[blockIdx.x * BLOCK_TILE_M * n + blockIdx.y * BLOCK_TILE_N + (warp_id % WARP_NUM_M) * WARP_TILE_M * n + (warp_id % WARP_NUM_M) * WARP_TILE_N + _m * n + _n + (lane_id / 4) * n + (lane_id % 4) * 2 + 8 * n]) += *(uint32_t*)(&C_frag[ping_pong][2]);
                C_frag[ping_pong][0] = __float2half(0.0f);
                C_frag[ping_pong][1] = __float2half(0.0f);
                C_frag[ping_pong][2] = __float2half(0.0f);
                C_frag[ping_pong][3] = __float2half(0.0f);
            }
        }
    }
}

int main(int argc, char** argv) {
    half *h_A, *d_A;
    half *h_B, *d_B;
    half *h_C, *d_C;
    half *h_D, *d_D;
    h_A = new half[M * K];
    h_B = new half[K * N];
    h_C = new half[M * N];
    h_D = new half[M * N];

    cudaEvent_t st, ed;
    cudaEventCreate(&st, NULL);
    cudaEventCreate(&ed, NULL);

    cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(half) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(half) * K * N);
    cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(half) * M * N);
    cudaMalloc(reinterpret_cast<void**>(&d_D), sizeof(half) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = __float2half((1.0 * i + 2.0 * j) / 50.0);
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = __float2half(1.0 * (i + 1) * (j + 1) / 50.0);
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
    cudaEventRecord(st);
    for (int i = 0; i < iter; i++) {
        ada_hgemm_m128n64k32_fp16_sm89<<<grid, WARP_PER_BLOCK * WARP_SIZE>>>(d_A, d_B, d_C, d_D, M, N, K);
    }
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms;
    cudaEventElapsedTime(&ms, st, ed);
    printf("%9.6f TFLOPS\n", ((2.0 * M * N * K) / ((ms / (1.0 * iter)) / 1000.0)) / (1024.0 * 1024.0 * 1024.0 * 1024.0));
    cudaMemcpy(h_D, reinterpret_cast<void*>(d_D), sizeof(half) * M * N, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < K; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%8.3f%c", __half2float(h_B[i * N + j]), (j == N - 1) ? '\n' : ' ');
    //     }
    // }
    // printf("\n\n\n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < N; j++) {
    //         half res = __float2half(0.0f);
    //         for (int k = 0; k < K; k++) {
    //             res += h_A[i * K + k] * h_B[k * N + j];
    //         }
    //         printf("%d%c", abs(__half2float(res) - __half2float(h_D[i * N + j])) < (0.05 * __half2float(res)) ? 0 : 1, (j == N - 1) ? '\n' : ' ');
    //         // printf("%08.3f%c", __half2float(res), (j == 8 - 1) ? '\n' : ' ');
    //         // printf("%08.3f%c", __half2float(h_D[i * N + j]), (j == 8 - 1) ? '\n' : ' ');
    //         // printf("%08.3f%c", __half2float(h_A[i * K + j]), (j == 23) ? '\n' : ' ');
    //     }
    // }

    // printf("\n\n\n");
    // for (int i = 64; i < 64 + 8; i++) {
    //     for (int j = 0; j < 8; j++) {
    //         // printf("%08.3f%c", abs(__half2float(res) - __half2float(h_D[i * N + j])), (j == N - 1) ? '\n' : ' ');
    //         printf("%08.3f%c", __half2float(h_B[i * 64 + j]), (j == 7) ? '\n' : ' ');
    //         // printf("%08.3f%c", __half2float(h_A[i * K + j]), (j == 23) ? '\n' : ' ');
    //     }
    // }
    return 0;
}