import torch
import faiss
import numpy as np
from VQKernel import gptvq_gemm

SEQ_LEN = 128
HIDDEN_DIM = 4096
COMPRESSION_RATIO = 2
ENTRY = 256
RESIDUAL = 1

torch.manual_seed(1889)
INPUT = torch.rand((SEQ_LEN, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
WEIGHT = torch.rand((HIDDEN_DIM, HIDDEN_DIM // COMPRESSION_RATIO)).type(torch.uint8).to("cuda:0")

# 64 rows, 16 columns, 256 * 2
# Organize to 1024 rows, 512 half per row
CODEBOOK = torch.rand((1024, 512)).type(torch.float16).to("cuda:0")

OUTPUT = gptvq_gemm(INPUT, WEIGHT, CODEBOOK)
print(OUTPUT[0])

