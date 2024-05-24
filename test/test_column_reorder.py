import faiss
import numpy as np
res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()
t = np.random.normal(0, 0.01, (4096, 4096))

import matplotlib.pyplot as plt
# count, bins = np.histogram(t, bins=1000)
# plt.stairs(count, bins)
# plt.xlim(-0.3,0.3)
# plt.savefig("tmp.png")

t_new = None
for i in range(0, 4096, 8):
    t_new = t[:, i:i+8] if t_new is None else np.concatenate((t_new, t[:, i:i+8]))

index = faiss.index_factory(8, "PRQ1x2x16", faiss.METRIC_INNER_PRODUCT)
index.prq.max_beam_size = 1
# index = faiss.index_cpu_to_gpu(res, 0, index, co)
index.train(t_new[0:65536])
faiss.write_index(index, "aqlm_1_8_rq2.index")
print("Training Finished")
# index = faiss.read_index("aqlm_1_8.index")
# index = faiss.index_cpu_to_gpu(res, 0, index, co)
t_compressed = index.prq.compute_codes(t_new)
print(t_new.shape, t_compressed.shape, t_compressed.dtype)
# t_compressed_fixed = np.zeros((2097152, ), dtype=np.uint16)
# for i in range(len(t_compressed)):
#     t_compressed_fixed[i] = int(t_compressed[i][0]) * 256 + int(t_compressed[i][1])
# value, count = np.unique(t_compressed_fixed, return_counts=True)
# count.sort()
# count = count[::-1]
# plt.plot(count)
# plt.savefig("count_pdf.png")
# plt.clf()
# count_cdf = [count[0]]
# for i in range(1, len(count)):
#     count_cdf.append(count[i] + count_cdf[-1])
# plt.plot(count_cdf)
# plt.savefig("count_cdf.png")

# t_compressed_fixed = np.reshape(t_compressed_fixed, (4096, 512))
# # 16 columns -> a group, calculate the number of unique ids in this 4096x16 tensor
# for i in range(0, 512, 16):
#     tmp = t_compressed_fixed[:, i:i+16]
#     value, count = np.unique(tmp, return_counts=True)
#     count.sort()
#     count = count[::-1]
#     plt.plot(count)
#     plt.savefig("count_%d.png" % (i / 16))
#     plt.clf()

# columns = []
# for i in range(512):
#     columns.append(t_compressed_fixed[:, i])

# new_t_compressed_fixed = []
# # new_t_compressed_fixed.append(columns[0])

# for i in range(0, 512):
#     # print(i, (i//16)*16, (i//16)*16+(i%16))
#     if i % 16 == 0: 
#         new_t_compressed_fixed.append(columns[i])
#     else:
#         columns[i:].sort(key=lambda x: len(np.setdiff1d(np.array(x), np.array(new_t_compressed_fixed[(i//16)*16:(i//16)*16+(i%16)]).flatten())))
#         new_t_compressed_fixed.append(columns[i])

# new_t_compressed_fixed = np.array(new_t_compressed_fixed).transpose()
# print(new_t_compressed_fixed.shape)

# for i in range(0, 512, 16):
#     tmp = t_compressed_fixed[:, i:i+16]
#     value, count = np.unique(tmp, return_counts=True)
#     count.sort()
#     count = count[::-1]
#     plt.plot(count)
#     plt.savefig("count_reordered_%d.png" % (i / 16))
#     plt.clf()
