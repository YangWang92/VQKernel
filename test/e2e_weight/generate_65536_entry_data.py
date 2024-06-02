import torch
import faiss

# tensor = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_original.pt", map_location="cuda:0")
# data = None

# for i in range(0, 4096, 8):
#     data = tensor[:, i : i + 8] if data is None else torch.cat((data, tensor[:, i : i + 8]), 0)
# data = data.to("cpu").numpy()

# index = faiss.index_factory(8, "PQ1x16np")
# index.train(data)
# faiss.write_index(index, "/home/zhliu/workspace/VQKernel/test/e2e_weight/index65536.index")
# index = faiss.read_index("/home/zhliu/workspace/VQKernel/test/e2e_weight/index65536.index")
# compressed = index.pq.compute_codes(data)
# decompressed = index.pq.decode(compressed)
# compressed = torch.from_numpy(compressed)
# decompressed = torch.from_numpy(decompressed)
# tensor_compressed, tensor_decompressed = None, None
# for i in range(0, 2097152, 4096):
#     tensor_compressed = compressed[i : i + 4096, : ] if tensor_compressed is None else torch.cat((tensor_compressed, compressed[i : i + 4096, : ]), 1)
#     tensor_decompressed = decompressed[i : i + 4096, : ] if tensor_decompressed is None else torch.cat((tensor_decompressed, decompressed[i : i + 4096, : ]), 1)
    
# print(tensor_compressed.shape, tensor_decompressed.shape)
# tensor_compressed = tensor_compressed.to("cuda:0")
# tensor_decompressed = tensor_decompressed.to("cuda:0")
# torch.save(tensor_compressed, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized_65536.pt")
# torch.save(tensor_decompressed, "/home/zhliu/workspace/VQKernel/test/e2e_weight/w_dequantized_65536.pt")
import sys
import numpy as np
tensor_compressed = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized_65536.pt")
BLOCK = int(sys.argv[1])
SIZE = int(sys.argv[2])
ENTRY = (SIZE * 1024) // 16
avg = []
# for row in range(0, 4096, BLOCK):
#     for col in range(0, 1024, 32):
#         data = tensor_compressed[row : row + BLOCK, col : col + 32]
#         data_fixed = []
#         for i in range(0, BLOCK):
#             for j in range(0, 32, 2):
#                 data_fixed.append(int(data[i, j + 1]) * 256 + int(data[i, j]))
#         value, count = np.unique(data_fixed, return_counts=True)
#         L = []
#         for i in range(65536):
#             if i not in value:
#                 L.append([i, 0])
#             else:
#                 idx = np.where(value == i)
#                 L.append([i, count[idx][0]])
#         L.sort(key=lambda x : x[1])
#         L = L[::-1]
#         freq = [x[1] for x in L]
#         avg.append(np.sum(freq[0 : ENTRY]) / np.sum(freq))
#         # print(len(freq), np.sum(freq[0 : ENTRY]), np.sum(freq), np.sum(freq[0 : ENTRY]) / np.sum(freq))
# print(BLOCK * 16 // (32 * 16), np.median(avg))

fixed = None
for i in range(0, 1024, 2):
    tmp1, tmp2 = tensor_compressed[:, i : i + 1], tensor_compressed[:, i + 1 : i + 2]
    tmp = tmp2.type(torch.int32) * 256 + tmp1.type(torch.int32)
    # print(tmp1, tmp2, tmp)
    fixed = tmp if fixed is None else torch.cat((fixed, tmp), 1)
    if i == 16: break

fixed = fixed.to("cpu").numpy().flatten()

value, count = np.unique(fixed, return_counts=True)
L = []
for i in range(65536):
    if i not in value:
        L.append([i, 0])
    else:
        idx = np.where(value == i)
        L.append([i, count[idx][0]])
L.sort(key=lambda x : x[1])
L = L[::-1]
freq = [x[1] for x in L]
cdf = [freq[0]]
for i in range(1, 65536):
    cdf.append(cdf[-1] + freq[i])
import matplotlib.pyplot as plt 
plt.plot(cdf)
plt.savefig("tmp.png")