import torch
import faiss
import numpy as np
from VQKernel import vq_gemm

BATCH = 1
PROMPT_LEN = 4096
HIDDEN_DIM = 4096
COMPRESSION_RATIO = 2
RESIDUALS = 1
ENTRY = 256

# torch.manual_seed(1889)
# dummy = torch.rand((HIDDEN_DIM, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
# h = torch.rand((BATCH * PROMPT_LEN, HIDDEN_DIM)).type(torch.float16).to("cuda:0")

# index = faiss.index_factory(
#     HIDDEN_DIM,
#     "PRQ%dx%dx8" % (HIDDEN_DIM // COMPRESSION_RATIO, RESIDUALS)
# )

# index.train(dummy.to("cpu").numpy())
# codebook = torch.from_numpy(faiss.vector_to_array(index.prq.codebooks)).type(torch.float16).to("cuda:0")
# new_codebook = torch.zeros(codebook.shape).type(torch.float16).to("cuda:0")
# w = torch.from_numpy(index.prq.compute_codes(dummy.to("cpu").numpy())).type(torch.uint8).to("cuda:0")
# new_w = torch.zeros(w.shape).type(torch.uint8).to("cuda:0")

# for d in range((HIDDEN_DIM // COMPRESSION_RATIO) * RESIDUALS):
#     tmp = w[:, d].to("cpu").numpy()
#     ID, FREQ = np.unique(tmp, return_counts=True)
#     tmp_w = []
#     for i in range(len(ID)):
#         tmp_w.append([ID[i], FREQ[i]])
#     for i in range(256):
#         if i not in ID:
#             tmp_w.append([i, 0])
#     tmp_w.sort(key=lambda x : x[1])
#     tmp_w = tmp_w[::-1]

#     for i in range(len(tmp_w)):
#         src = tmp_w[i][0]
#         new_codebook[(d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + i * COMPRESSION_RATIO : (d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + i * COMPRESSION_RATIO + COMPRESSION_RATIO] = codebook[(d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + src * COMPRESSION_RATIO : (d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + src * COMPRESSION_RATIO + COMPRESSION_RATIO]
#         new_w[:, d][tmp == src] = i
# # Reorder the residual layout of compressed tensor
# wr = []
# for r in range(RESIDUALS):
#     wr.append(new_w[:, [RESIDUALS * _ + r for _ in range((HIDDEN_DIM // COMPRESSION_RATIO))]])
# new_new_w = None
# for r in range(RESIDUALS):
#     new_new_w = wr[r] if new_new_w is None else torch.cat([new_new_w, wr[r]], dim=-1)

# print("Begin")
# torch.save(h, "/home/zhliu/workspace/VQKernel/kernel/hidden.pt")
# torch.save(new_new_w, "/home/zhliu/workspace/VQKernel/kernel/weight.pt")
# torch.save(new_codebook, "/home/zhliu/workspace/VQKernel/kernel/codebook.pt")
h = torch.load("/home/zhliu/workspace/VQKernel/kernel/hidden.pt").type(torch.float16).to("cuda:0")
new_new_w = torch.load("/home/zhliu/workspace/VQKernel/kernel/weight.pt").type(torch.uint8).to("cuda:0")
new_codebook = torch.load("/home/zhliu/workspace/VQKernel/kernel/codebook.pt").type(torch.float16).to("cuda:0")
o = vq_gemm(h, new_new_w, new_codebook, RESIDUALS, COMPRESSION_RATIO, ENTRY)