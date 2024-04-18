#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier ready[], barrier filled[], float* buffer, float* in, int N, int buffer_len) {
    for (int i = 0; i < (N / buffer_len); i++) {
        ready[i % 2].arrive_and_wait();
        // uint32_t lane_id = threadIdx.x % 32;
        // *(uint4*)(&buffer[(i % 2) * buffer_len + lane_id * 4]) = *(uint4*)(&in[i * buffer_len + lane_id * 4]);
        barrier::arrival_token token = filled[i % 2].arrive();
    }
}

__device__ void consumer(barrier ready[], barrier filled[], float* buffer, float* out, int N, int buffer_len) {
    barrier::arrival_token token1 = ready[0].arrive();
    barrier::arrival_token token2 = ready[1].arrive();
    for (int i = 0; i < (N / buffer_len); i++) {
        filled[i % 2].arrive_and_wait();
        // uint32_t lane_id = threadIdx.x % 32;
        // *(uint4*)(&out[i * buffer_len + lane_id * 4]) = *(uint4*)(&buffer[(i % 2) * buffer_len + lane_id * 4]);
        barrier::arrival_token token = ready[i % 2].arrive();
    }
}

__global__ void producer_consumer_kernel(int N, int buffer_len, float* in, float* out) {
    __shared__ extern float buffer[];
    __shared__ barrier bar[4];
    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() < 4) {
        init(bar + block.thread_rank(), block.size());
    }
    block.sync();

    if (block.thread_rank() < 32) {
        for (int i = 0; i < (N / buffer_len); i++) {
            bar[i % 2].arrive_and_wait();
            uint32_t lane_id = threadIdx.x % 32;
            *(uint4*)(&buffer[(i % 2) * buffer_len + lane_id * 4]) = *(uint4*)(&in[i * buffer_len + lane_id * 4]);
            barrier::arrival_token token = bar[2 + i % 2].arrive();
        }
    }
    else {
        barrier::arrival_token token1 = bar[0].arrive();
        barrier::arrival_token token2 = bar[1].arrive();
        for (int i = 0; i < (N / buffer_len); i++) {
            bar[2 + i % 2].arrive_and_wait();
            uint32_t lane_id = threadIdx.x % 32;
            *(uint4*)(&out[i * buffer_len + lane_id * 4]) = *(uint4*)(&buffer[(i % 2) * buffer_len + lane_id * 4]);
            barrier::arrival_token token = bar[i % 2].arrive();
        }
    }
}

int main(int argc, char** argv) {
    int len = 512;
    float *ha, *hb, *da, *db;
    ha = new float[len];
    hb = new float[len];
    for (int i = 0; i < len; i++) ha[i] = 1.0 * i;
    cudaMalloc(reinterpret_cast<void**>(&da), sizeof(float) * len);
    cudaMalloc(reinterpret_cast<void**>(&db), sizeof(float) * len);
    cudaMemcpy(reinterpret_cast<void*>(da), ha, sizeof(float) * len, cudaMemcpyHostToDevice);

    producer_consumer_kernel<<<1, 64, 4096>>>(len, 128, da, db);
    cudaMemcpy(hb, reinterpret_cast<void*>(db), sizeof(float) * len, cudaMemcpyDeviceToHost);
    for (int i = 0; i < len; i++) printf("%06.1f%c", hb[i], (i % 16 == 15) ? '\n' : ' ');
    return 0;
}