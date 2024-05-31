#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include <torch/extension.h>

// !!! ASSUMING RESIDUAL ALWAYS LESS THAN BLOCK_PER_HEAD !!!
// 1*8192 x 8192*16, PRQ16:1x2x8

#define WARP_SIZE 32
#define WARP_NUM 8
using QKV_TYPE = half;
using TORCH_QKV_TYPE = at::Half;
using KVCACHE_TYPE = uint8_t;
using TORCH_KVCACHE_TYPE = uint8_t;
using KVCONFIG_TYPE = int;
using TORCH_KVCONFIG_TYPE = int;
#define _TOKEN_BLOCK 256
#define _BLOCK_PER_HEAD 16
#define MAX_SHARED_MEM 65536
#define TRUE 1
#define FALSE 0
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

template <int len>
struct masking {};

template <>
struct masking<8> {
    using Type = uint8_t;
};

template <>
struct masking<16> {
    using Type = uint16_t;
};

template <>
struct masking<32> {
    using Type = uint32_t;
};

template <>
struct masking<64> {
    using Type = uint64_t;
};

template <>
struct masking<128> {
    using Type = uint4;
};

template <typename T>
__forceinline__ __device__ T _dot(
    T* a, T* b, int len
) 
{

}

// k_src_layout: row: (1, 1), col: (k's compression ratio, 1)
// v_src_layout: row: (1, 1), col: (v's compression ratio, 1)
// k_dst_layout: row: (1, 1), col: (dimension per block, 1)
// v_dst_layout: row: (1, 1), col: (dimension per block, 1)
// kv_cache_configs: [compression_ratio, entry_num, residual, codebook_offset]
// grid: (token_block_id, head_block_id)
template <
    size_t BATCH_SIZE,
    size_t BLOCK_SIZE,  // thread block size
    size_t HEAD_NUM,
    size_t HEAD_DIM,
    size_t BLOCK_PER_HEAD,
    size_t TOKEN_BLOCK, // flash attention token block size
    size_t K_THREAD_BINDING_GROUP,
    size_t V_THREAD_BINDING_GROUP,
    size_t K_INV_HOT_ENTRY,
    size_t V_INV_HOT_ENTRY,
    size_t PACKED_LOAD_WIDTH,
    size_t B32REG_LIMIT,
    size_t ENABLE_ENTRY_CENTRIC
>
void __global__ vq_attention_decoding_c4d8_kernel(
    QKV_TYPE* _o,
    QKV_TYPE* _q, 
    QKV_TYPE*       _k,                 QKV_TYPE*       _v,
    KVCACHE_TYPE*   _k_cache,           KVCACHE_TYPE*   _v_cache,
    QKV_TYPE*       _k_codebook,        QKV_TYPE*       _v_codebook,
    KVCONFIG_TYPE*  _k_cache_configs,   KVCONFIG_TYPE*  _v_cache_configs,
    uint32_t _seq_len,
    uint32_t _per_token_byte_k, uint32_t _per_token_byte_v,
    QKV_TYPE*       _reduce_workspace
)
{
    extern __shared__ uint8_t smem[];
    QKV_TYPE *k_codebook_buf = reinterpret_cast<QKV_TYPE*>(smem);

    const uint32_t thread_packed_load_width = PACKED_LOAD_WIDTH / (8 * sizeof(QKV_TYPE));                                                                               //add by JXGuo: 一次syscall load 128bit，此处计算出一次syscall load多少个QKV_TYPE
    using packed_load_type = typename packed_vec<QKV_TYPE, thread_packed_load_width>::Type;                                                                             //add by JXGuo: 可以一次被load出的类型

    const uint32_t batch_per_block = min((uint32_t) 1, (uint32_t) (BLOCK_SIZE / TOKEN_BLOCK));                                                                          // add by JXGuo: 当
    uint32_t batch_offset = threadIdx.x / TOKEN_BLOCK;
    uint32_t batch_rowid  = threadIdx.x % TOKEN_BLOCK;

    const uint32_t block_id = blockIdx.x;                               // Flash decoding block id
    const uint32_t head_id   = blockIdx.y / BLOCK_PER_HEAD;             // which head does block compute
    const uint32_t tb_offset = blockIdx.y % BLOCK_PER_HEAD;             // block id inside all blocks process one head add by JXGuo: 处理一个head的所有block中排第几
    const uint32_t k_residuals = _k_cache_configs[2 * 32 + head_id];    // How many residual of k  add by JXGuo: head num必须为32
    const uint32_t k_entry_num = _k_cache_configs[1 * 32 + head_id];
    const uint32_t k_entry_to_load = k_entry_num / K_INV_HOT_ENTRY;     // Load part of hot entry
    const uint32_t k_residual_id = tb_offset / k_residuals;                                                                                                             // add by JXGuo: 当前block在第几个residual
    const uint32_t k_subspace_id = tb_offset % k_residuals;                                                                                                             // add by JXGuo: 在当前residual下，当前block是所有block中的第几个
    const uint32_t k_compression_ratio = _k_cache_configs[0 * 32 + head_id];
    const uint32_t k_subspace_num = HEAD_DIM / k_compression_ratio;                                                                                                     //add by JXGuo: 实际subspace数量
    const uint32_t k_codeword_offset = _k_cache_configs[4 * 32 + head_id];
    // const uint32_t k_codeword_length = _k_cache_configs[5 * 32 + head_id];
    // For example: 16 block / head but only 8 subspace.
    // if (k_subspace_id >= k_subspace_num) return;

    const uint32_t k_subspace_per_block_to_compute = max((uint32_t) 1, (uint32_t) ((k_subspace_num * k_residuals) / BLOCK_PER_HEAD));                                   // add by JXGuo: subspace num是不考虑多个residual的
    const uint32_t k_segment_element_num = k_subspace_per_block_to_compute * k_compression_ratio;                                                                       // add by JXGuo: 每个block要处理的subspace数量 * subspace的维度，即实际要处理多少个channel(向compressrate对齐)

    // Load k codebook
    // Starting from codebook_offset, next k_subspace_per_block_to_compute rows.
    const uint32_t k_codebook_offset = _k_cache_configs[3 * 32 + head_id]                                                                                               // add by JXGuo: codebook offset
                                   + k_residual_id * k_subspace_num * k_entry_num * k_compression_ratio                                                                 // add by JXGuo: residual 造成的偏移
                                   + k_subspace_id * k_subspace_per_block_to_compute * k_entry_num * k_compression_ratio;
    const uint32_t codebook_to_access_k = k_subspace_per_block_to_compute * k_entry_num * k_compression_ratio; // codebook access begin of (residual id, subspace id)   // add by JXGuo: 当前block需要用到多大的codebook
    const uint32_t codebook_to_load_k = codebook_to_access_k / K_INV_HOT_ENTRY;                                                                                         // add by JXGuo: 实际需要cache在smem的codebook大小
    // Now load part ot hot codebook entries to shared memory
    // Every thread load 128 bits (8 half) element
    // We assume codebook entries are reordered with their frequency.

    uint32_t threads_need_to_load = min((uint32_t)BLOCK_SIZE, (uint32_t)(codebook_to_load_k / thread_packed_load_width)); // How many thread need to conduct the load   // add by JXGuo: 需要用到多少个thread
    uint32_t block_packed_load_width = threads_need_to_load * thread_packed_load_width;                                                                                 // add by JXGuo: 计算所有用到的thread一起一次可以load多少QKV_TYPE
    const uint32_t k_load_iterations = max((uint32_t)1, (uint32_t)(codebook_to_load_k / block_packed_load_width)); // How many iteration threads need to load all codebooks
    const uint32_t k_subspace_stride = k_entry_num * k_compression_ratio;                               // For hot entry use, stride to the next subspace
    const float k_subspace_load_at_once = block_packed_load_width / (k_entry_to_load * k_compression_ratio);                                                            // add by JXGuo: block内用到的所有thread一次可以load几个subspace所有的entry
    const uint32_t k_group_needed_for_one_subspace = max((uint32_t)1, (uint32_t)(1.0 / k_subspace_load_at_once));
    if (threadIdx.x < threads_need_to_load) {
        const uint32_t k_load_thread_group = k_entry_to_load * k_compression_ratio / thread_packed_load_width;                                                          // add by JXGuo: 一个subspace需要多少次ld128
        const uint32_t group_id = threadIdx.x / k_load_thread_group;                                                                                                    // add by JXGuo: 当前thread负责第几个subspace
        const uint32_t group_off = threadIdx.x % k_load_thread_group;                                                                                                   // add by JXGuo: 当前thread在负责的subspace中是第几个thread
        for (int i = 0; i < k_load_iterations; i++) {
            *(packed_load_type*)(&k_codebook_buf[i * block_packed_load_width + threadIdx.x * thread_packed_load_width]) = 
            *(packed_load_type*)(&_k_codebook[k_codebook_offset + 
                                              (uint32_t)(i * k_subspace_load_at_once) * k_subspace_stride +                                                             // add by JXGuo: 前几轮load产生的移动（如果一个subspace需要block load多次，那么这个值会多轮不变）
                                              group_id * k_subspace_stride + 
                                              group_off * thread_packed_load_width + 
                                              (i % k_group_needed_for_one_subspace) * block_packed_load_width                                                           // add by JXGuo: 若subspace需要block load多轮，则位移在此处体现
                                             ]);
        }
    }
    // Seems right, need to verify.
    // Codebook layout in shared memory: |Space0        |Space1        |......        |
    //                                  /                \
    //                                 /                   \
    //                                /                      \
    //                               |E0 |E1 |E2 |...|...|255|
    //                              /     \
    //                             /       \
    //                            /         \
    //                           | | | | | | |
    // Load V codebook anc v_cache
    const uint32_t v_residuals = _v_cache_configs[2 * 32 + head_id];    
    const uint32_t v_entry_num = _v_cache_configs[1 * 32 + head_id];
    const uint32_t v_entry_to_load = v_entry_num / V_INV_HOT_ENTRY;     
    const uint32_t v_residual_id = tb_offset % v_residuals;             
    const uint32_t v_subspace_id = tb_offset / v_residuals;             
    const uint32_t v_compression_ratio = _v_cache_configs[0 * 32 + head_id];
    const uint32_t v_subspace_num = HEAD_DIM / v_compression_ratio;
    const uint32_t v_codeword_offset = _v_cache_configs[4 * 32 + head_id];
    const uint32_t v_subspace_per_block_to_compute = max((uint32_t) 1, (uint32_t) ((v_subspace_num * v_residuals) / BLOCK_PER_HEAD));
    const uint32_t v_segment_element_num = v_subspace_per_block_to_compute * v_compression_ratio;

    const uint32_t v_codebook_offset = _v_cache_configs[3 * 32 + head_id] + 
                                       v_residual_id * v_subspace_num * v_entry_num * v_compression_ratio + 
                                       v_subspace_id * v_subspace_per_block_to_compute * v_entry_num * v_compression_ratio;
    const uint32_t codebook_to_access_v = v_subspace_per_block_to_compute * v_entry_num * v_compression_ratio;
    const uint32_t codebook_to_load_v = codebook_to_access_v / V_INV_HOT_ENTRY;

    threads_need_to_load = min((uint32_t) BLOCK_SIZE, (uint32_t)(codebook_to_load_v / thread_packed_load_width));
    block_packed_load_width = threads_need_to_load * thread_packed_load_width;

    const uint32_t v_load_iterations = max((uint32_t) 1, (uint32_t) (codebook_to_load_v / block_packed_load_width));
    const uint32_t v_subspace_stride = v_entry_num * v_compression_ratio;
    const float v_subspace_load_at_once = block_packed_load_width / (v_entry_to_load * v_compression_ratio);  // add by JXGuo: a bug here, should be v_compression_ratio
    const uint32_t v_group_needed_for_one_subspace = max((uint32_t) 1, (uint32_t) (1.0 / v_subspace_load_at_once));

    QKV_TYPE* q_shmem = reinterpret_cast<QKV_TYPE*>(smem + codebook_to_load_k * sizeof(QKV_TYPE));
    QKV_TYPE* k_shmem = reinterpret_cast<QKV_TYPE*>(q_shmem + BATCH_SIZE * k_segment_element_num * sizeof(QKV_TYPE));
    QKV_TYPE* v_shmem = reinterpret_cast<QKV_TYPE*>(k_shmem + BATCH_SIZE * k_segment_element_num * sizeof(QKV_TYPE));
    KVCACHE_TYPE* k_cache_shmem = reinterpret_cast<KVCACHE_TYPE*>(v_shmem + BATCH_SIZE * v_segment_element_num * sizeof(QKV_TYPE));
    KVCACHE_TYPE* v_cache_shmem = reinterpret_cast<KVCACHE_TYPE*>(k_cache_shmem + BATCH_SIZE * TOKEN_BLOCK * k_subspace_per_block_to_compute * sizeof(KVCACHE_TYPE));
    QKV_TYPE* attn_shmem = reinterpret_cast<QKV_TYPE*>(v_cache_shmem + BATCH_SIZE * TOKEN_BLOCK * v_subspace_per_block_to_compute * sizeof(KVCACHE_TYPE));
    QKV_TYPE* v_codebook_buf = reinterpret_cast<QKV_TYPE*>(attn_shmem + batch_per_block * TOKEN_BLOCK * sizeof(QKV_TYPE));
    QKV_TYPE* shuffle_workspace = reinterpret_cast<QKV_TYPE*>(v_codebook_buf + codebook_to_load_v * sizeof(QKV_TYPE));
    
    if (threadIdx.x < threads_need_to_load) {
        const uint32_t v_load_thread_group = v_entry_to_load * v_compression_ratio / thread_packed_load_width;
        const uint32_t group_id = threadIdx.x / v_load_thread_group;
        const uint32_t group_off = threadIdx.x % v_load_thread_group;
        for (int i = 0; i < v_load_iterations; i++) {
            *(packed_load_type*)(&v_codebook_buf[i * block_packed_load_width + threadIdx.x * thread_packed_load_width]) = 
            *(packed_load_type*)(&_v_codebook[v_codebook_offset + 
                                              (uint32_t)(i * v_subspace_load_at_once) * v_subspace_id + 
                                              group_id * v_subspace_stride + 
                                              group_off * thread_packed_load_width + 
                                              (i % v_group_needed_for_one_subspace) * block_packed_load_width
                                             ]);
        }
    }

    __syncthreads();
    // Load q and k segment, q follows compression ratio of k, k_segment_element_num elements.
    // qk_shmem shape = (BATCH, k_segment_element_num)
    if (threadIdx.x < BATCH_SIZE) {
        // One thread load one batch's q and k                                                        // add by JXGuo: 是否可优化？
        for (int _ = 0; _ < k_segment_element_num; _ += thread_packed_load_width) {
            *(packed_load_type*)(&q_shmem[threadIdx.x * k_segment_element_num + 
                                          _ * thread_packed_load_width
                                         ]) = 
            *(packed_load_type*)(&_q[threadIdx.x * (HEAD_NUM * HEAD_DIM) +
                                     head_id * HEAD_DIM + 
                                     k_subspace_id * k_segment_element_num + 
                                     _ * thread_packed_load_width
                                    ]);

            *(packed_load_type*)(&k_shmem[threadIdx.x * k_segment_element_num + 
                                          _ * thread_packed_load_width
                                         ]) = 
            *(packed_load_type*)(&_k[threadIdx.x * (HEAD_NUM * HEAD_DIM) + 
                                     head_id * HEAD_DIM +                                           // add by JXGuo: head_id 造成的位移
                                     k_subspace_id * k_segment_element_num +                        // add by JXGuo: subspace_id 造成的位移
                                     _ * thread_packed_load_width
                                    ]);
        }

        for (int _ = 0; _ < v_segment_element_num; _ += thread_packed_load_width) {
            *(packed_load_type*)(&v_shmem[threadIdx.x * v_segment_element_num + 
                                          _ * thread_packed_load_width
                                         ]) = 
            *(packed_load_type*)(&_v[threadIdx.x * (HEAD_NUM * HEAD_DIM) + 
                                     head_id * HEAD_DIM + 
                                     v_subspace_id * v_segment_element_num + 
                                     _ * thread_packed_load_width
                                    ]);
        }
    }

    // reg to store dequant K 
    const uint32_t token_per_thread = (TOKEN_BLOCK + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    QKV_TYPE* k_dequant_reg[token_per_thread][(B32REG_LIMIT / token_per_thread) * (4 / sizeof(QKV_TYPE))];  // add by JXGuo: 此处是否需要确保后面的这个值是整数？且大于等于1
    QKV_TYPE* v_dequant_reg[token_per_thread][(B32REG_LIMIT / token_per_thread) * (4 / sizeof(QKV_TYPE))];
    using MASK_TYPE = typename masking<(B32REG_LIMIT / token_per_thread) * (4 / sizeof(QKV_TYPE))>::Type;
    MASK_TYPE set_mask[token_per_thread];
    // Dequant k, Load k cache into shmem if entry centric dataflow is used
    // kcacheshmem.Shape = (BATCH_SIZE, TOKEN_BLOCK, k_subspace_per_block_to_compute)
    uint32_t _tid;
    if (ENABLE_ENTRY_CENTRIC) {                                                                             //
        for (int i = 0; i < BATCH_SIZE / batch_per_block; i++) {                                            // add by JXGuo: batch_per_block 每个block处理几个batch（用于token_block很小不足以充分利用时）
            _tid = batch_rowid;                                                                             // add by JXGuo: batch_rowid 当前token_block内的第几个token
            while (_tid < TOKEN_BLOCK) {
                for (int _s = 0; _s < k_subspace_per_block_to_compute; _s++) {
                    k_cache_shmem[i * batch_per_block * TOKEN_BLOCK * k_subspace_per_block_to_compute + 
                                  batch_offset * TOKEN_BLOCK * k_subspace_per_block_to_compute + 
                                  _tid * k_subspace_per_block_to_compute +                                  // add by JXGuo: 此处表示的是当前block要处理特定的几个subspace，_tid表明是哪个token的特定subspace
                                  _s
                                 ] = 
                    _k_cache[i * batch_per_block * _seq_len * _per_token_byte_k + 
                             batch_offset * _seq_len * _per_token_byte_k +                                  // add by JXGuo: 此处是batch引起的位移
                             block_id * TOKEN_BLOCK * _per_token_byte_k +
                             _tid * _per_token_byte_k +                                                     // add by JXGuo: seqlen上第几个token引发的偏移
                             k_codeword_offset +                                                            // add by JXGuo: token次序导致的offset
                             (k_subspace_id * k_subspace_per_block_to_compute + _s) * k_residuals +         // add by JXGuo: 是否意味着cache是优先同一个subspace 的 residual连续放在一起
                             k_residual_id
                            ];
                }
                for (int _s = 0; _s < v_subspace_per_block_to_compute; _s++) {
                    v_cache_shmem[i * batch_per_block * TOKEN_BLOCK * v_subspace_per_block_to_compute + 
                                  batch_offset * TOKEN_BLOCK * v_subspace_per_block_to_compute + 
                                  _tid * v_subspace_per_block_to_compute + 
                                  _s
                                 ] = 
                    _v_cache[i * batch_per_block * _seq_len * _per_token_byte_v + 
                             batch_offset * _seq_len * _per_token_byte_v + 
                             block_id * TOKEN_BLOCK * _per_token_byte_v + 
                             _tid * _per_token_byte_v + 
                             v_codeword_offset + 
                             (v_subspace_id * v_subspace_per_block_to_compute + _s) * v_residuals + 
                             v_residual_id
                            ];
                }
                _tid += blockIdx.x;                                                                          // add by JXGuo: warning 此处似乎有些问题，是否应为 _tid += blockDim.x，表示一轮结束
            }
        }
        __syncthreads();
    }


    QKV_TYPE attn_reg[token_per_thread];
    QKV_TYPE out_reg[token_per_thread];
    // Due to limited register resource, how many iteration need to be done
    uint32_t k_reg_iters = max((uint32_t) 1, (uint32_t)((k_subspace_per_block_to_compute * k_compression_ratio * sizeof(QKV_TYPE)) / (B32REG_LIMIT * sizeof(uint32_t))));   // add by JXGuo: 处理一个block所要处理的channel需要多少次使用所有的寄存器
    uint32_t v_reg_iters = max((uint32_t) 1, (uint32_t)((v_subspace_per_block_to_compute * v_compression_ratio * sizeof(QKV_TYPE)) / (B32REG_LIMIT * sizeof(uint32_t))));   // add by JXGuo: warning 此写法在 uint32_t 不能刚好放入整数个 QKV_TYPE 会出错
    QKV_TYPE* entry_addr;
    for (int i = 0; i < BATCH_SIZE / batch_per_block; i++) {
        for (int kri = 0; kri < k_reg_iters; kri++) {
            for (int _ = 0; _ < token_per_thread; _++) set_mask[_] = 0;
            for (uint8_t e = 0; e < ENABLE_ENTRY_CENTRIC; e++) {
                _tid = batch_rowid;
                while (_tid < TOKEN_BLOCK) {
                    uint8_t id;
                    for (int _s = 0; _s < k_subspace_per_block_to_compute / k_reg_iters; _s++) {
                        id = k_cache_shmem[i * batch_per_block * TOKEN_BLOCK * k_subspace_per_block_to_compute + 
                                           batch_offset * TOKEN_BLOCK * k_subspace_per_block_to_compute + 
                                           _tid * k_subspace_per_block_to_compute + 
                                           kri * (k_subspace_per_block_to_compute / k_reg_iters) + 
                                           _s
                                          ];
                        if (id == e) {
                            entry_addr = &k_codebook_buf[_s * k_entry_to_load * k_compression_ratio + 
                                                         kri * k_reg_iters * k_entry_to_load * k_compression_ratio + 
                                                         e * k_compression_ratio 
                                                        ];
                            set_mask[_tid / TOKEN_BLOCK] |= (0x1 << _s);
                            for (int _ = 0; _ < k_compression_ratio; _ += 8) {
                                if (k_compression_ratio >= 8) {
                                    *(uint4*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * k_compression_ratio + 
                                                                                _
                                                                                ]) = 
                                    *(uint4*) &entry_addr[_];
                                }
                                else if (k_compression_ratio == 4) {
                                    *(uint64_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * k_compression_ratio + 
                                                                                    _
                                                                                ]) = 
                                    *(uint64_t*) &entry_addr[_];
                                }
                                else if (k_compression_ratio == 2) {
                                    *(uint32_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * k_compression_ratio + 
                                                                                    _
                                                                                ]) = 
                                    *(uint32_t*) &entry_addr[_];   
                                }
                            }
                        }
                    }
                    _tid += blockDim.x;
                }
            }
            _tid = batch_rowid;
            while (_tid < TOKEN_BLOCK) {
                for (int _s = 0; _s < k_subspace_per_block_to_compute / k_reg_iters; _s++) {
                    if (set_mask[_tid / TOKEN_BLOCK] & (1 << _s) == 0x0) {
                        uint8_t id = 0;
                        if (ENABLE_ENTRY_CENTRIC) { // Steal k_cache_shmem
                            id = k_cache_shmem[i * batch_per_block * TOKEN_BLOCK * k_subspace_per_block_to_compute + 
                                               batch_offset * TOKEN_BLOCK * k_subspace_per_block_to_compute + 
                                               _tid * k_subspace_per_block_to_compute + 
                                               kri * (k_subspace_per_block_to_compute / k_reg_iters) + 
                                               _s
                                              ];
                        }
                        else {                      // Load from global
                            id = _k_cache[i * batch_per_block * _seq_len * _per_token_byte_k + 
                                          batch_offset * _seq_len * _per_token_byte_k + 
                                          block_id * TOKEN_BLOCK * _per_token_byte_k + 
                                          _tid * _per_token_byte_k + 
                                          k_codeword_offset + 
                                          (k_subspace_id * k_subspace_per_block_to_compute + kri * (k_subspace_per_block_to_compute / k_reg_iters) + _s) * k_residuals + 
                                          k_residual_id
                                         ];
                        }
                        if (id < k_entry_to_load) {
                            entry_addr = &k_codebook_buf[_s * k_entry_to_load * k_compression_ratio + 
                                                         kri * k_reg_iters * k_entry_to_load * k_compression_ratio + 
                                                         id * k_compression_ratio
                                                        ];
                        }
                        else {
                            entry_addr = &_k_codebook[k_codebook_offset + 
                                                      k_residual_id * k_subspace_num * k_entry_num * k_compression_ratio + 
                                                      k_subspace_id * k_subspace_per_block_to_compute * k_entry_num * k_compression_ratio + 
                                                      kri * k_reg_iters * k_entry_num * k_compression_ratio + 
                                                      _s * k_entry_num * k_compression_ratio + 
                                                      id * k_compression_ratio
                                                     ];
                        }
                        for (int _ = 0; _ < k_compression_ratio; _ += 8) {
                            if (k_compression_ratio >= 8) {
                                *(uint4*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * k_compression_ratio + 
                                                                             _
                                                                            ]) = 
                                *(uint4*) &entry_addr[_];
                            }
                            else if (k_compression_ratio == 4) {
                                *(uint64_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * k_compression_ratio + 
                                                                                _
                                                                               ]) = 
                                *(uint64_t*) &entry_addr[_];
                            }
                            else if (k_compression_ratio == 2) {
                                *(uint32_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * k_compression_ratio + 
                                                                                _
                                                                               ]) = 
                                *(uint32_t*) &entry_addr[_];   
                            }
                        }
                    }
                }
                _tid += blockDim.x;
            }

            // Compute q * k.T
            _tid = batch_rowid;
            while (_tid < TOKEN_BLOCK) {
                attn_reg[_tid / TOKEN_BLOCK] = _dot<QKV_TYPE>((QKV_TYPE*)(&q_shmem[i * batch_per_block + batch_offset]), (QKV_TYPE*)(&k_dequant_reg[_tid / TOKEN_BLOCK]), k_segment_element_num);
                _tid += blockDim.x;
            }
        }

        // kri accumulation (None cross block reduce)
        // One head do reduction, Just element-wisely add attn_reg of blocks from the same head!
        _tid = batch_rowid;
        while (_tid < TOKEN_BLOCK) {
            atomicAdd(&_reduce_workspace[(i * batch_per_block + batch_offset) * _seq_len * HEAD_NUM + (block_id * TOKEN_BLOCK + _tid) * HEAD_NUM + head_id], attn_reg[_tid / TOKEN_BLOCK]);
            _tid += blockDim.x;
        }

        // _reduce_workspace.shape = (batch, TOKEN_BLOCK, head),
        // Every block load (batch_per_block, TOKEN_BLOCK) data into shared
        _tid = batch_rowid;
        while (_tid < TOKEN_BLOCK) {
            attn_shmem[batch_offset * TOKEN_BLOCK + _tid] = _reduce_workspace[(i * batch_per_block + batch_offset) * _seq_len * HEAD_NUM + (block_id * TOKEN_BLOCK + _tid) * HEAD_NUM + head_id];
            _tid += blockDim.x;
        }
        __syncthreads();

        // Dequant V
        for (int vri = 0; vri < v_reg_iters; vri++) {
            for (int _ = 0; _ < token_per_thread; _++) set_mask[_] = 0;
            for (uint8_t e = 0; e < ENABLE_ENTRY_CENTRIC; e++) {
                _tid = batch_rowid;
                while (_tid < TOKEN_BLOCK) {
                    uint8_t id;
                    for (int _s = 0; _s < v_subspace_per_block_to_compute / v_reg_iters; _s++) {
                        id = v_cache_shmem[i * batch_per_block * TOKEN_BLOCK * v_subspace_per_block_to_compute + 
                                           batch_offset * TOKEN_BLOCK * v_subspace_per_block_to_compute + 
                                           _tid * v_subspace_per_block_to_compute + 
                                           vri * (v_subspace_per_block_to_compute / v_reg_iters) + 
                                           _s
                                          ];
                        if (id == e) {
                            entry_addr = &v_codebook_buf[_s * v_entry_to_load * v_compression_ratio + 
                                                         vri * v_reg_iters * v_entry_to_load * v_compression_ratio + 
                                                         e * v_compression_ratio 
                                                        ];
                            set_mask[_tid / TOKEN_BLOCK] |= (0x1 << _s);
                            for (int _ = 0; _ < v_compression_ratio; _ += 8) {
                                if (v_compression_ratio >= 8) {
                                    *(uint4*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * v_compression_ratio + 
                                                                                _
                                                                                ]) = 
                                    *(uint4*) &entry_addr[_];
                                }
                                else if (v_compression_ratio == 4) {
                                    *(uint64_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * v_compression_ratio + 
                                                                                    _
                                                                                ]) = 
                                    *(uint64_t*) &entry_addr[_];
                                }
                                else if (v_compression_ratio == 2) {
                                    *(uint32_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * v_compression_ratio + 
                                                                                    _
                                                                                ]) = 
                                    *(uint32_t*) &entry_addr[_];   
                                }
                            }
                        }
                    }
                    _tid += blockDim.x;
                }
            }
            _tid = batch_rowid;
            while (_tid < TOKEN_BLOCK) {
                for (int _s = 0; _s < v_subspace_per_block_to_compute / v_reg_iters; _s++) {
                    if (set_mask[_tid / TOKEN_BLOCK] & (1 << _s) == 0x0) {
                        uint8_t id = 0;
                        if (ENABLE_ENTRY_CENTRIC) {
                            id = v_cache_shmem[i * batch_per_block * TOKEN_BLOCK * v_subspace_per_block_to_compute + 
                                               batch_offset * TOKEN_BLOCK * v_subspace_per_block_to_compute + 
                                               _tid * v_subspace_per_block_to_compute + 
                                               vri * (v_subspace_per_block_to_compute / v_reg_iters) + 
                                               _s
                                              ];
                        }
                        else {
                            id = _v_cache[i * batch_per_block * _seq_len * _per_token_byte_v + 
                                          batch_offset * _seq_len * _per_token_byte_v + 
                                          block_id * TOKEN_BLOCK * _per_token_byte_v + 
                                          _tid * _per_token_byte_v + 
                                          v_codeword_offset + 
                                          (v_subspace_id * v_subspace_per_block_to_compute + vri * (v_subspace_per_block_to_compute / v_reg_iters) + _s) * v_residuals + 
                                          v_residual_id
                                         ];
                        }
                        if (id < v_entry_to_load) {
                            entry_addr = &v_codebook_buf[_s * v_entry_to_load * v_compression_ratio + 
                                                         vri * v_reg_iters * v_entry_to_load * v_compression_ratio + 
                                                         id * v_compression_ratio
                                                        ];
                        }
                        else {
                            entry_addr = &_v_codebook[v_codebook_offset + 
                                                      v_residual_id * v_subspace_num * v_entry_num * v_compression_ratio + 
                                                      v_subspace_id * v_subspace_per_block_to_compute * v_entry_num * v_compression_ratio + 
                                                      vri * v_reg_iters * v_entry_num * v_compression_ratio + 
                                                      _s * v_entry_num * v_compression_ratio + 
                                                      id * v_compression_ratio
                                                     ];
                        }
                        for (int _ = 0; _ < v_compression_ratio; _ += 8) {
                            if (v_compression_ratio >= 8) {
                                *(uint4*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * v_compression_ratio + 
                                                                             _
                                                                            ]) = 
                                *(uint4*) &entry_addr[_];
                            }
                            else if (v_compression_ratio == 4) {
                                *(uint64_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * v_compression_ratio + 
                                                                                _
                                                                               ]) = 
                                *(uint64_t*) &entry_addr[_];
                            }
                            else if (v_compression_ratio == 2) {
                                *(uint32_t*)(&k_dequant_reg[_tid / TOKEN_BLOCK][_s * v_compression_ratio + 
                                                                                _
                                                                               ]) = 
                                *(uint32_t*) &entry_addr[_];   
                            }
                        }
                    }
                }
                _tid += blockDim.x;
            }
        }
        // Now v_dequant_reg hold all v.
        for (int _v = 0; _v < v_subspace_per_block_to_compute * v_compression_ratio; _v++) {
            _tid = batch_rowid;
            while (_tid < TOKEN_BLOCK) {
                // TODO: half intrinsic
                // out_reg[_tid / TOKEN_BLOCK] += attn_shmem[batch_offset * TOKEN_BLOCK + _tid] * v_dequant_reg[_tid / TOKEN_BLOCK][_v];
                _tid += blockDim.x;
            }
            QKV_TYPE sum_all = __float2half(0.0);
            for (int t = 0; i < token_per_thread; i++) {
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) {
                // TODO: half intrinsic
                    // out_reg[t] += __shfl_down_sync(0xffffffff, out_reg[t], mask);
                }
                if (threadIdx.x % WARP_SIZE == 0) {
                    shuffle_workspace[threadIdx.x / WARP_SIZE] = out_reg[t];
                }
                __syncthreads();
                if (threadIdx.x < WARP_NUM) {
                    sum_all = shuffle_workspace[threadIdx.x];
                }
                #pragma unroll
                for (int mask = 4; mask > 0; mask >>= 1) {
                // TODO: half intrinsic
                    // sum_all += __shfl_down_sync(0xffffffff, sum_all, mask);
                }
            }
            if (threadIdx.x == 0) {
            atomicAdd(&_o[i * batch_per_block * HEAD_NUM * HEAD_DIM + 
                          batch_offset * HEAD_NUM * HEAD_DIM + 
                          head_id * HEAD_DIM + 
                          v_subspace_id * v_subspace_per_block_to_compute * v_compression_ratio + 
                          _v
                         ], sum_all);
            }
        }
    }

}

torch::Tensor vq_attention_decoding_c4d8(
    torch::Tensor q,                                                // [batch, 1, hidden_dim]    
    torch::Tensor k,                torch::Tensor v,
    torch::Tensor k_cache,          torch::Tensor v_cache,          // [batch, seq_len, token_byte]
    torch::Tensor k_codebook,       torch::Tensor v_codebook,       // []
    torch::Tensor k_cache_configs,  torch::Tensor v_cache_configs,  // [compression_ratio, entry_num, residual, codebook_offset]
    int head_num
)
{
    
    auto batch = q.size(0);
    auto seq_len = k_cache.size(1);
    auto hidden_dim = q.size(2);
    auto head_dim = hidden_dim / head_num;
    uint32_t per_token_byte_k = k_cache.size(2), per_token_byte_v = v_cache.size(2);
    
    auto q_ptr = reinterpret_cast<QKV_TYPE*>(q.data_ptr<TORCH_QKV_TYPE>());
    auto k_ptr = reinterpret_cast<QKV_TYPE*>(k.data_ptr<TORCH_QKV_TYPE>());
    auto v_ptr = reinterpret_cast<QKV_TYPE*>(v.data_ptr<TORCH_QKV_TYPE>());
    auto k_cache_ptr = reinterpret_cast<KVCACHE_TYPE*>(k_cache.data_ptr<TORCH_KVCACHE_TYPE>());
    auto v_cache_ptr = reinterpret_cast<KVCACHE_TYPE*>(v_cache.data_ptr<TORCH_KVCACHE_TYPE>());
    auto k_codebook_ptr = reinterpret_cast<QKV_TYPE*>(k_codebook.data_ptr<TORCH_QKV_TYPE>());
    auto v_codebook_ptr = reinterpret_cast<QKV_TYPE*>(v_codebook.data_ptr<TORCH_QKV_TYPE>());
    auto k_cache_configs_ptr = reinterpret_cast<KVCONFIG_TYPE*>(k_cache_configs.data_ptr<TORCH_KVCONFIG_TYPE>());
    auto v_cache_configs_ptr = reinterpret_cast<KVCONFIG_TYPE*>(v_cache_configs.data_ptr<TORCH_KVCONFIG_TYPE>());

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({batch, 1, hidden_dim}, 0, options);
    auto o_ptr = reinterpret_cast<QKV_TYPE*>(o.data_ptr<TORCH_QKV_TYPE>());
    
    QKV_TYPE* reduce_workspace;
    cudaMalloc(reinterpret_cast<void**>(&reduce_workspace), sizeof(QKV_TYPE) * batch * seq_len * head_num);

    dim3 grid((seq_len + (_TOKEN_BLOCK - 1)) / _TOKEN_BLOCK, head_num * _BLOCK_PER_HEAD);

    if ((batch == 8) && (head_num == 32) && (head_dim == 128)) {
        auto kernel = vq_attention_decoding_c4d8_kernel<8, WARP_NUM * WARP_SIZE, 32, 128, _TOKEN_BLOCK, _BLOCK_PER_HEAD, 1, 1, 1, 1, 128, 16, FALSE>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEM);
        kernel<<<grid, WARP_NUM * WARP_SIZE, MAX_SHARED_MEM>>>(
            o_ptr,
            q_ptr, 
            k_ptr, v_ptr,
            k_cache_ptr, v_cache_ptr,
            k_codebook_ptr, v_codebook_ptr,
            k_cache_configs_ptr, v_cache_configs_ptr,
            seq_len,
            per_token_byte_k, per_token_byte_v,
            reduce_workspace
        );
    }

    return o;
}