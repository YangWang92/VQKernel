import torch
import faiss
import numpy as np
from VQKernel import e2e_gemv_rq, e2e_gemm_rq
from faiss.contrib.inspect_tools import get_pq_centroids
from tqdm import tqdm
SEQ_LEN = 2048
COMPRESSION_RATIO = 4
ENTRY = 256
RESIDUAL = 1

HEAD_DIM = 128
HEAD_NUM = 32
KV_HEAD_NUM = 32

# torch.set_printoptions(profile="full")

# AQLM Original:
# 8 FP16 -> 1 UINT16, 65536 entries
# W           = 4096 * 4096 * sizeof(HALF) = 32 MB
# W_quantized = 4096 * (4096 / 8) * sizeof(UINT16) = 4 MB
# Codebook    = 65536 * 8 * sizeof(HALF) = 1 MB
# AQLM Fixed
# 4 FP16 -> 1 UINT8, 256 entries per 8 dimensions
# W_quantized = 4096 * (4096 / 4) * sizeof(UINT8) = 4 MB
# Codebook    = 256 * 4 * sizeof(HALF) * (4096 / 8) = 1 MB
# So, Train 512 (4096 / 8) codebooks, each with data from: W[d:d+8]
# w_original = torch.rand((HEAD_DIM * HEAD_NUM, HEAD_DIM * HEAD_NUM)).type(torch.float16).to("cuda:0")
# w_quantized = None
# w_dequantized = None
# codebook = None
# cnt = 0
# for d in tqdm(range(0, HEAD_DIM * HEAD_NUM, 8)):
#     data = torch.cat((w_original[:, d : d + 4], w_original[:, d + 4 : d + 8]), 0)
#     index = faiss.index_factory(COMPRESSION_RATIO, "PQ1x8")
#     index.train(data.to("cpu").numpy())
#     faiss.write_index(index, "/home/zhliu/workspace/VQKernel/test/e2e_weight/indices/index_%d_to_%d.index" % (d, d + 8))
#     cb = torch.from_numpy(get_pq_centroids(index.pq)[0]).type(torch.float16).to("cuda:0")
#     codebook = cb if codebook is None else torch.cat((codebook, cb), 0)

#     data_quantized = index.pq.compute_codes(data.to("cpu").numpy())
#     data_dequantized = index.pq.decode(data_quantized)
#     data_quantized = np.concatenate((data_quantized[0 : HEAD_DIM * HEAD_NUM, :], data_quantized[HEAD_DIM * HEAD_NUM : , :]), 1)
#     data_dequantized = np.concatenate((data_dequantized[0 : HEAD_DIM * HEAD_NUM, :], data_dequantized[HEAD_DIM * HEAD_NUM : , :]), 1)
#     data_quantized = torch.from_numpy(data_quantized).to("cuda:0")
#     data_dequantized = torch.from_numpy(data_dequantized).to("cuda:0")
#     w_quantized = data_quantized if w_quantized is None else torch.cat((w_quantized, data_quantized), 1)
#     w_dequantized = data_dequantized if w_dequantized is None else torch.cat((w_dequantized, data_dequantized), 1)
# codebook = codebook.type(torch.float16).to("cuda:0")
# w_quantized = w_quantized.type(torch.uint8).to("cuda:0")
# w_dequantized = w_dequantized.type(torch.float16).to("cuda:0")
# torch.save(w_original, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_original.pt")
# torch.save(codebook, "/home/zhliu/workspace/VQKernel/test/e2e_weight/codebook.pt")
# torch.save(w_quantized, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized.pt")
# torch.save(w_dequantized, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_dequantized.pt")
# print(w_original.shape, codebook.shape, w_quantized.shape, w_dequantized.shape)

w_original = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_original.pt", map_location="cuda:0")
w_quantized = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized.pt", map_location="cuda:0")
w_dequantized = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_dequantized.pt", map_location="cuda:0")
codebook = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/codebook.pt", map_location="cuda:0")

# Reorder!
# w_quantized_reordered = w_quantized.clone()
# codebook_reordered = codebook.clone()
# for d in range(0, HEAD_DIM * HEAD_NUM // COMPRESSION_RATIO, 2):
#     to_reorder = w_quantized[:, d : d + 2]
#     dummy = to_reorder.clone()
#     to_reorder_np = to_reorder.to("cpu").numpy().flatten()
#     value, count = np.unique(to_reorder_np, return_counts=True)
#     L = []
#     for i in range(len(value)):
#         L.append([value[i], count[i]])
#     L.sort(key=lambda x : x[1])
#     L = L[::-1]
#     for i in range(len(L)):
#         codebook_reordered[(d // 2) * 256 + i] = codebook[(d // 2) * 256 + int(L[i][0])]
#         w_quantized_reordered[:, d : d + 2] = torch.where(dummy == L[i][0], i, w_quantized_reordered[:, d : d + 2])
# torch.save(codebook_reordered, "/home/zhliu/workspace/VQKernel/test/e2e_weight/codebook_reordered.pt")
# torch.save(w_quantized_reordered, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized_reordered.pt")

codebook_reordered = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/codebook_reordered.pt")
w_quantized_reordered = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized_reordered.pt")

# residual = w_original - w_dequantized
# codebook_residual = None
# w_residual_quantized = None
# w_residual_dequantized = None
# for d in tqdm(range(0, HEAD_DIM * HEAD_NUM, 8)):
#     data = torch.cat((residual[:, d : d + 4], residual[:, d + 4 : d + 8]), 0)
#     index = faiss.index_factory(COMPRESSION_RATIO, "PQ1x8")
#     index.train(data.to("cpu").numpy())
#     faiss.write_index(index, "/home/zhliu/workspace/VQKernel/test/e2e_weight/indices/index_%d_to_%d_residual.index" % (d, d + 8))
#     cb = torch.from_numpy(get_pq_centroids(index.pq)[0]).type(torch.float16).to("cuda:0")
#     codebook_residual = cb if codebook_residual is None else torch.cat((codebook_residual, cb), 0)
    
#     data_quantized = index.pq.compute_codes(data.to("cpu").numpy())
#     data_dequantized = index.pq.decode(data_quantized)
#     data_quantized = np.concatenate((data_quantized[0 : HEAD_DIM * HEAD_NUM, :], data_quantized[HEAD_DIM * HEAD_NUM : , :]), 1)
#     data_dequantized = np.concatenate((data_dequantized[0 : HEAD_DIM * HEAD_NUM, :], data_dequantized[HEAD_DIM * HEAD_NUM : , :]), 1)
#     data_quantized = torch.from_numpy(data_quantized).to("cuda:0")
#     data_dequantized = torch.from_numpy(data_dequantized).to("cuda:0")
#     w_residual_quantized = data_quantized if w_residual_quantized is None else torch.cat((w_residual_quantized, data_quantized), 1)
#     w_residual_dequantized = data_dequantized if w_residual_dequantized is None else torch.cat((w_residual_dequantized, data_dequantized), 1)
# codebook_residual = codebook_residual.type(torch.float16).to("cuda:0")
# w_residual_quantized = w_residual_quantized.type(torch.uint8).to("cuda:0")
# w_residual_dequantized = w_residual_dequantized.type(torch.float16).to("cuda:0")
# torch.save(codebook_residual, "/home/zhliu/workspace/VQKernel/test/e2e_weight/codebook_residual.pt")
# torch.save(w_residual_quantized, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_residual_quantized.pt")
# torch.save(w_residual_dequantized, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_residual_dequantized.pt")

w_residual_quantized = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_residual_quantized.pt", map_location="cuda:0")
w_residual_dequantized = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_residual_dequantized.pt", map_location="cuda:0")
codebook_residual = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/codebook_residual.pt", map_location="cuda:0")


hidden = torch.rand((4096, HEAD_DIM * HEAD_NUM)).type(torch.float16).to("cuda:0") - 0.5
# hidden1d = torch.rand((8, HEAD_DIM * HEAD_NUM)).type(torch.float16).to("cuda:0") - 0.5

ori = torch.matmul(hidden, w_original)
ref = torch.matmul(hidden, w_dequantized + w_residual_dequantized)
res = e2e_gemm_rq(hidden, w_quantized, codebook, w_residual_quantized, codebook_residual)

# print(w_quantized[0:64, 0:2])

# print(w_dequantized[64:128,0:8])
# print(hidden1d[0][64:128])
# print(torch.matmul(hidden1d[0][:], w_dequantized[:, 0]))
# print(ref)

# print(res)
# ref = torch.matmul(hidden, w_dequantized)
# res = e2e_gemm(hidden, w_quantized_reordered, codebook_reordered)
# print("Origin VS Reference Error:%5.2f" % (np.median(np.abs(((ori - ref) / ori).to("cpu").numpy())) * 100), "%")
# print("Reference VS Custom Error:%5.2f" % (np.median(np.abs(((ref - res) / ref).to("cpu").numpy())) * 100), "%")
# print("Origin VS    Custom Error:%5.2f" % (np.median(np.abs(((ori - res) / ori).to("cpu").numpy())) * 100), "%")



# Plot cdf
# tmp = w_quantized_reordered.to("cpu").numpy().flatten()
# v, c = np.unique(tmp, return_counts=True)
# l = []
# for i in range(len(v)):
#     l.append([v[i], c[i]])
# l.sort(key=lambda x : x[1])
# l = l[::-1]
# cdf = [l[0][1]]
# for i in range(1, len(l)):
#     cdf.append(l[i][1] + cdf[i - 1])

# import matplotlib.pyplot as plt 
# plt.plot(cdf)
# plt.savefig("cdf.png")