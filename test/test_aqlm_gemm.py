import torch
import faiss
import numpy as np
from VQKernel import aqlm_gemm

SEQ_LEN = 4096
HIDDEN_DIM = 4096
COMPRESSION_RATIO = 8
ENTRY = 65536
RESIDUAL = 2

torch.manual_seed(1889)
INPUT = torch.rand((SEQ_LEN, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
WEIGHT = torch.rand((HIDDEN_DIM, HIDDEN_DIM // (COMPRESSION_RATIO // RESIDUAL))).type(torch.short).to("cuda:0")
CODEBOOK = torch.rand((ENTRY, COMPRESSION_RATIO * RESIDUAL)).type(torch.float16).to("cuda:0")

OUTPUT = aqlm_gemm(INPUT, WEIGHT, CODEBOOK)
print(OUTPUT[0])