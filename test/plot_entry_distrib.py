import torch
import numpy as np
import matplotlib.pyplot as plt 
t = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized_reordered.pt")
import sys
X = int(sys.argv[1])
data = t.to("cpu").numpy().flatten()
value, count = np.unique(data, return_counts=True)
L = []
for i in range(256):
    if i not in value:
        L.append([i, 0])
    else:
        idx = np.where(value == i)
        L.append([i, count[idx]])
L.sort(key=lambda x : x[1])
L = L[::-1]
L = [x[1] for x in L]
cdf = [L[0]]
for i in range(1, len(L)):
    cdf.append(cdf[-1] + L[i])
L = [x[0] for x in L]
L /= np.min(L)
for i in L:
    print(i)
plt.ylim(4096, 32768)
plt.plot(L)
plt.yscale('log', base=2)
plt.savefig("tmp1.png")
print(np.sum(L[0 : 0 + X]) / np.sum(L[256 - X : 256]))

# BLOCK = int(sys.argv[1])
# SIZE = int(sys.argv[2])
# ENTRY = (SIZE * 1024) // (16 * 4 * 2)

# # 每个BLOCK*32 x 2看freq，然后求平均
# avg = []
# for row in range(0, 4096, BLOCK * 32):
#     for col in range(0, 1024, 2):
#         data = t[row : row + BLOCK * 32, col : col + 2].to("cpu").numpy().flatten()
#         value, count = np.unique(data, return_counts=True)
#         L = []
#         for i in range(256):
#             if i not in value:
#                 L.append([i, 0])
#             else:
#                 idx = np.where(value == i)
#                 L.append([i, count[idx][0]])
#         L.sort(key=lambda x : x[1])
#         L = L[::-1]
#         freq = [x[1] for x in L]
#         # print(freq)
#         # print(np.sum(freq[0 : ENTRY]), np.sum(freq))
#         avg.append(np.sum(freq[0 : ENTRY]) / np.sum(freq))
# print(np.mean(avg))


