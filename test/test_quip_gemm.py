import torch
import faiss
import numpy as np
from VQKernel import quip_gemm

SEQ_LEN = 4096
HIDDEN_DIM = 4096
COMPRESSION_RATIO = 4
ENTRY = 256

torch.manual_seed(1889)
INPUT = torch.rand((SEQ_LEN, HIDDEN_DIM)).type(torch.float16).to("cuda:0")
WEIGHT = torch.rand((HIDDEN_DIM, HIDDEN_DIM // COMPRESSION_RATIO)).type(torch.uint8).to("cuda:0")
CODEBOOK = torch.rand((ENTRY, COMPRESSION_RATIO)).type(torch.float16).to("cuda:0")

OUTPUT = quip_gemm(INPUT, WEIGHT, CODEBOOK)
print(OUTPUT[0])