import torch
import faiss
from VQKernel import vq_attention_decoding
import numpy as np

BATCH = 8
SEQ_LEN = 4096
HIDDEN_DIM = 4096
HEAD_NUM = 32
HEAD_DIM = 128

q = torch.zeros((BATCH, 1, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
k = torch.zeros((BATCH, 1, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
v = torch.zeros((BATCH, 1, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
k_cache_configs = [
    [2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16], # Compression Ratio
    [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], # Entry num
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4], # Residual num
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # codebook offset
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # codeword offset
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # codeword length
]
v_cache_configs = [
    [2, 2, 2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16], # Compression Ratio
    [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256], # Entry num
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4], # Residual num
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # codebook offset
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # codeword offset
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # codeword length
]
k_token_byte, v_token_byte = 0, 0
k_codebook_offset, v_codebook_offset = 0, 0
k_codebook, v_codebook = None, None
for h in range(HEAD_NUM):
    k_cache_configs[4][h] = k_token_byte
    k_cache_configs[5][h] = (HEAD_DIM // k_cache_configs[0][h]) * k_cache_configs[2][h] * 1
    k_token_byte += (HEAD_DIM // k_cache_configs[0][h]) * k_cache_configs[2][h] * 1
    #               |_____ #compressed data               |_____ #residual        |_____ Padding to 1 byte, by default 256 entries
    v_cache_configs[4][h] = v_token_byte
    v_cache_configs[5][h] = (HEAD_DIM // v_cache_configs[0][h]) * v_cache_configs[2][h] * 1
    v_token_byte += (HEAD_DIM // v_cache_configs[0][h]) * v_cache_configs[2][h] * 1

k_cache = torch.zeros((BATCH, SEQ_LEN, k_token_byte)).type(torch.uint8).to("cuda:0")
v_cache = torch.zeros((BATCH, SEQ_LEN, v_token_byte)).type(torch.uint8).to("cuda:0")

for h in range(HEAD_NUM):
    dummy = torch.zeros((256, HEAD_DIM)).numpy()
    indexk = faiss.index_factory(
        HEAD_DIM,
        "PRQ%dx%dx%d" % (HEAD_DIM // k_cache_configs[0][h], k_cache_configs[2][h], 8)
    )
    indexk.train(dummy)
    quantizerk = indexk.prq
    k_cache_configs[3][h] = k_codebook_offset
    k_codebook_offset += quantizerk.total_codebook_size * k_cache_configs[0][h]
    print(type(faiss.vector_to_array(quantizerk.codebooks)))
    k_codebook = faiss.vector_to_array(quantizerk.codebooks) if k_codebook is None else np.concatenate((k_codebook, faiss.vector_to_array(quantizerk.codebooks)))

    indexv = faiss.index_factory(
        HEAD_DIM,
        "PRQ%dx%dx%d" % (HEAD_DIM // v_cache_configs[0][h], v_cache_configs[2][h], 8)
    )
    indexv.train(dummy)
    quantizerv = indexv.prq
    v_cache_configs[3][h] = v_codebook_offset
    v_codebook_offset += quantizerv.total_codebook_size * v_cache_configs[0][h]
    v_codebook = faiss.vector_to_array(quantizerv.codebooks) if v_codebook is None else np.concatenate((v_codebook, faiss.vector_to_array(quantizerv.codebooks)))

    # codebook shape: (residual, HEAD_DIM // compression_ratio, 256, (compression_ratio))
k_codebook = torch.from_numpy(k_codebook).type(torch.float16).to("cuda:0")
v_codebook = torch.from_numpy(v_codebook).type(torch.float16).to("cuda:0")
k_cache_configs = torch.from_numpy(np.array(k_cache_configs)).type(torch.int).to("cuda:0")
v_cache_configs = torch.from_numpy(np.array(v_cache_configs)).type(torch.int).to("cuda:0")

vq_attention_decoding(q, k, v, k_cache, v_cache, k_codebook, v_codebook, k_cache_configs, v_cache_configs, HEAD_NUM)
'''
k_original = torch.zeros((16, SEQ_LEN, 4096)).type(torch.float16).to("cuda:0")
k_compressed = torch.zeros((16, SEQ_LEN, 1024)).type(torch.uint8).to("cuda:0")
k_codebook = torch.zeros((2, 1024, 256, 4)).type(torch.float16).to("cuda:0")
v_original = torch.zeros((16, SEQ_LEN, 4096)).type(torch.float16).to("cuda:0")
v_compressed = torch.zeros((16, SEQ_LEN, 1024)).type(torch.uint8).to("cuda:0")
v_codebook = torch.zeros((2, 1024, 256, 4)).type(torch.float16).to("cuda:0")

# index = faiss.index_factory(4096, "PRQ1024x2x8")
# to_train = torch.flatten(k_original, start_dim=0, end_dim=1).to("cpu").numpy()
# index.train(to_train)
# quantizer = index.prq

# N_ratio = faiss.vector_to_array(quantizer.codebooks).shape[0] // quantizer.total_codebook_size
# N_space = quantizer.nsplits
# N_residual = quantizer.M // N_space
# N_entry = quantizer.total_codebook_size // (N_space * N_residual)

# codebook = faiss.vector_to_array(quantizer.codebooks).reshape(N_residual, N_space, N_entry, N_ratio)
# print(codebook.shape)
o = vq_attention_decoding(q, k_compressed, k_codebook, v_compressed, v_codebook)

'''