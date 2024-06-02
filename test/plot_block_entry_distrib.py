import torch
import numpy as np
import matplotlib.pyplot as plt 
t = torch.load("/home/zhliu/workspace/VQKernel/test/e2e_weight/w_quantized_reordered.pt")
import sys
Avg = []
cnt = 0
for i in range(0, 4096, 128):
    for j in range(0, 1024, 32):
        data = t[i : i + 128, j : j + 32].to("cpu").numpy().flatten()
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
        Avg.append(L)

MIN, MAX, MEAN = [], [], []
for i in range(256):
    tmp = [x[i] for x in Avg]
    MIN.append(np.min(tmp))
    MAX.append(np.max(tmp))
    MEAN.append(np.mean(tmp))
MEAN /= np.min(MEAN)
for i in MEAN:
    print(i)
# plt.fill_between(range(256), MIN, MAX)
plt.plot(MEAN)
plt.savefig("All.png")