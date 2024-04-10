import torch
import faiss
import numpy as np
from VQKernel import vq_gemv

# BATCH <= 16, > 16 -> vq_gemm
BATCH = 8
HIDDEN_DIM = 4096
COMPRESSION_RATIO = 2
RESIDUALS = 1
ENTRY = 256

torch.manual_seed(1889)

dummy = torch.rand((HIDDEN_DIM, HIDDEN_DIM)).type(torch.float16).to("cuda:0")

h = torch.rand((BATCH, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
h = h.permute(1, 0).contiguous()
w = ((torch.rand((HIDDEN_DIM, (HIDDEN_DIM // COMPRESSION_RATIO) * RESIDUALS)) * 1000) % 256).type(torch.uint8).to("cuda:0")
new_w = torch.zeros(w.shape).type(torch.uint8).to("cuda:0")
index = faiss.index_factory(
    HIDDEN_DIM,
    "PRQ%dx%dx8" % (HIDDEN_DIM // COMPRESSION_RATIO, RESIDUALS)
)
index.train(dummy.to("cpu").numpy())
codebook = torch.from_numpy(faiss.vector_to_array(index.prq.codebooks)).type(torch.float16).to("cuda:0")
new_codebook = torch.zeros(codebook.shape).type(torch.float16).to("cuda:0")
# print(new_w.shape)
w = torch.from_numpy(index.prq.compute_codes(dummy.to("cpu").numpy())).type(torch.uint8).to("cuda:0")
# reorder w
for d in range((HIDDEN_DIM // COMPRESSION_RATIO) * RESIDUALS):
    tmp = w[:, d].to("cpu").numpy()
    ID, FREQ = np.unique(tmp, return_counts=True)
    tmp_w = []
    for i in range(len(ID)):
        tmp_w.append([ID[i], FREQ[i]])
    for i in range(256):
        if i not in ID:
            tmp_w.append([i, 0])
    tmp_w.sort(key=lambda x : x[1])
    tmp_w = tmp_w[::-1]

    for i in range(len(tmp_w)):
        src = tmp_w[i][0]
        new_codebook[(d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + i * COMPRESSION_RATIO : (d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + i * COMPRESSION_RATIO + COMPRESSION_RATIO] = codebook[(d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + src * COMPRESSION_RATIO : (d % RESIDUALS) * (HIDDEN_DIM // COMPRESSION_RATIO) * 256 * COMPRESSION_RATIO + (d // RESIDUALS) * 256 * COMPRESSION_RATIO + src * COMPRESSION_RATIO + COMPRESSION_RATIO]
        new_w[:, d][tmp == src] = i
# Reorder the residual layout of compressed tensor
wr = []
for r in range(RESIDUALS):
    wr.append(new_w[:, [RESIDUALS * _ + r for _ in range((HIDDEN_DIM // COMPRESSION_RATIO))]])
new_new_w = None
for r in range(RESIDUALS):
    new_new_w = wr[r] if new_new_w is None else torch.cat([new_new_w, wr[r]], dim=-1)

# Important!!!!!!!!!!!!
print(h.is_contiguous())
# # print(codebook)
o = vq_gemv(h, new_new_w, new_codebook, RESIDUALS, COMPRESSION_RATIO, ENTRY)


# # print(o)

# print(h[114])
# print(new_codebook[32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 : 32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 + 4])
# print(h[114] * new_codebook[32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 + 2 : 32*256*4 + 2 * 4 * 256 * 4 + 3 * 256 * 4 + int(new_new_w[114, 32 + 8 + 3]) * 4 + 3])

w_decoded = index.prq.decode(w.to("cpu").numpy())
# print(w_decoded[:, 0])
# print(h[:, 0])


# print(torch.dot(h[:, 0], torch.from_numpy(w_decoded[:, 0]).type(torch.float16).to("cuda:0")))
# print(o[0, 0])
# print(h[:, 0])
# print(h[:, 1])
# print(torch.from_numpy(w_decoded[:, 0]).type(torch.float16).to("cuda:0"))
# for item in new_new_w[:, 0]:
#     print(new_codebook[int(item) * 4])
# print(new_codebook[int(new_new_w[:, 0])])

# print(torch.dot(h[:, 3], torch.from_numpy(w_decoded[:, 114]).type(torch.float16).to("cuda:0")))
# print(o[3, 114])