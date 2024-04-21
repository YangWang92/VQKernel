import tvm
from tvm.script import tir as T
import numpy as np
import os

M = 4096
N = 4096
K = 4096
BLOCK_TILE_M = 128
BLOCK_TILE_N = 128
WARP_TILE_M = 32
WARP_TILE_N = 32
BLOCK_TILE_K = 64
# Warp do 64x64x64 mma
_dtype = 'float32'

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.script.ir_module
class TirGeMM:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype=_dtype)
        B = T.match_buffer(b, [K, N], dtype=_dtype)
        C = T.match_buffer(c, [M, N], dtype=_dtype)
        for i, j, k in T.grid(M, N, K):
            with T.block("tb"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

ir_module = TirGeMM
sch = tvm.tir.Schedule(ir_module, debug_mask="all")
block = sch.get_block("tb")
(i, j, k) = sch.get_loops(block)
io, ii = sch.split(i, [None, BLOCK_TILE_M])
jo, ji = sch.split(j, [None, BLOCK_TILE_N])
iio, iii = sch.split(ii, [None, WARP_TILE_M])
jio, jii = sch.split(ji, [None, WARP_TILE_N])

# ko, ki = sch.split(k, [None, BLOCK_TILE_K])
sch.reorder(io, jo, iio, jio, iii, jii)
sch.bind(io, "blockIdx.x")
sch.bind(jo, "blockIdx.y")
sch.bind(iio, "threadIdx.x")
sch.bind(jio, "threadIdx.y")
# jo, ji = sch.split(j, [N // 256, 256])
# sch.bind(i, "blockIdx.x")
# sch.bind(ji, "threadIdx.x")
tb_cl = sch.cache_write(block, 0, "local")
sch.reverse_compute_at(tb_cl, jii)

ctx = tvm.cuda(0)
matmul = tvm.build(sch.mod, target='cuda')
write_code(matmul.imported_modules[0].get_source(), '0.cu')
a = np.arange(M * K).reshape((M, K)).astype(_dtype)
b = np.arange(K * N).reshape((K, N)).astype(_dtype)
c = np.zeros((M, N)).astype(_dtype)
c_ref = np.mat(a) * np.mat(b)
cuda_a = tvm.nd.array(a, ctx)
cuda_b = tvm.nd.array(b, ctx)
cuda_c = tvm.nd.array(c, ctx)
matmul(cuda_a, cuda_b, cuda_c)
print((c_ref.A1 - cuda_c.numpy().flatten()) / c_ref.A1)
num_flops = 2 * M * N * K
num_runs = 10
timer = matmul.time_evaluator(matmul.entry_name, ctx, number=num_runs)
t = timer(cuda_a, cuda_b, cuda_c).mean

TFLOPS = (num_flops / t) / (1024 * 1024 * 1024 * 1024)
print(TFLOPS)


