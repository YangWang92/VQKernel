import numpy as np
import tvm
from tvm import te
from tvm import meta_schedule as ms
from tvm.meta_schedule.builder import LocalBuilder
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule
from tvm._ffi import register_func
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.tir.tensor_intrin import cuda
import tvm.testing

import tempfile

def initializer():
    @register_func("meta_schedule.builder.async_build")
    def async_build(mod, target, _params):  # pylint: disable=unused-variable, unused-argument
        # pylint: disable=import-outside-toplevel
        from tvm.driver import build as tvm_build
        from tvm.tir.transform import RemoveWeightLayoutRewriteBlock

        # re-import here for local builder to register index_map_m16n8k8_matrixC
        # pylint: disable=import-outside-toplevel, unused-import
        from tvm.tir.tensor_intrin import cuda

        mod = RemoveWeightLayoutRewriteBlock(skip_ndarray_rewrite=True)(mod)
        with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
            rt_mod = tvm_build(mod, target=target)
        return rt_mod

def matmul_fp16(M, N, K):
    a = te.placeholder((M, K), name='A', dtype="float16")
    b = te.placeholder((K, N), name='B', dtype="float16")
    k = te.reduce_axis((0, K), name='K')
    c = te.compute(
        (M, N),
        lambda i, j: te.sum(a[i][k] * b[k][j], axis=[k]),
        name='C',
    )
    return (a, b, c)

# def multi_level_tiling_mma():
#     return ms.schedule_rule.MultiLevelTilingTensorCore(
#         intrin_groups=[
#             {
#                 "init":     "mma_init_m16n8k8_f16",
#                 "load_a":   "mma_load_m16n8k8_f16_A_shared_dyn",
#                 "load_b":   "mma_load_m16n8k8_f16_B_shared_dyn",
#                 "compute":  "mma_sync_m16n8k8_f16f16f16",
#                 "store":    "mma_store_m16n8k8_f16_global",
#             },
#         ],
#         structure="SSSRRSRS",
#         tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
#         max_innermost_factor=4,
#         vector_load_lens=[1, 2, 3, 4, 8, 16],
#         reuse_read=ms.schedule_rule.ReuseType(
#             req="must",
#             levels=[4],
#             scope="shared.dyn",
#         ),
#         reuse_write=ms.schedule_rule.ReuseType(
#             req="no",
#             levels=[2],
#             scope="shared.dyn",
#         ),
#         use_software_pipeline=True,
#     )
M, N, K = 1024, 1024, 1024
blockM, blockN, blockK = 64, 64, 64
cK = 512
_dtype = "float16"
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

@tvm.script.ir_module
class CompressedGeMM:
    @T.prim_func
    def main(a: T.handle, b: T.handle, codebook_a: T.handle, codebook_b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, cK], dtype='int8')
        B = T.match_buffer(b, [N, cK], dtype='int8')
        CA = T.match_buffer(codebook_a, [256, K], dtype='float16')
        CB = T.match_buffer(codebook_b, [256, K], dtype='float16')
        C = T.match_buffer(c, [M, N], dtype='float16')
        for i, j, k in T.grid(M, N, cK):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + (CA[A[vi, vk], vk * 2 + 0] * CB[B[vi, vk], vk * 2 + 0]) + (CA[A[vi, vk], vk * 2 + 1] * CB[B[vi, vk], vk * 2 + 1])

@tvm.script.ir_module
class CompressedTiledGeMM:
    @T.prim_func
    def main(a: T.handle, b: T.handle, codebook_a: T.handle, codebook_b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M // blockM, K // blockK, blockM, blockK // 2], dtype='uint8')
        B = T.match_buffer(b, [K // blockK, N // blockN, blockK // 2, blockN], dtype='uint8')
        CA = T.match_buffer(codebook_a, [256, K], dtype='float16')
        CB = T.match_buffer(codebook_b, [256, K], dtype='float16')
        A_ = T.alloc_buffer([M // blockM, K // blockK, blockM, blockK], dtype='float16')
        B_ = T.alloc_buffer([K // blockK, M // blockN, blockK, blockN], dtype='float16')
        C = T.match_buffer(c, [M // blockM, N // blockN, blockM, blockN], dtype='float16')
        for ii, jj, kk, i, j, k in T.grid(M // blockM, N // blockN, K // blockK, blockM, blockN, blockK):
            with T.block("A_"):
                vii, vkk, vi, vk = T.axis.remap("SSSS", [ii, kk, i, k])
                A_[vii, vkk, vi, vk] = CA[A[vii, vkk, vi, vk // 2], vk % 2]
            with T.block("B_"):
                vkk, vjj, vk, vj = T.axis.remap("SSSS", [kk, jj, k, j])
                B_[vkk, vjj, vk, vj] = CB[B[vkk, vjj, vk // 2, vj], vk % 2]
        for ii, jj, kk, i, j, k in T.grid(M // blockM, N // blockN, K // blockK, blockM, blockN, blockK):
            with T.block("C"):
                vii, vjj, vkk, vi, vj, vk = T.axis.remap("SSRSSR", [ii, jj, kk, i, j, k])
                with T.init():
                    C[vii, vjj, vi, vj] = 0
                C[vii, vjj, vi, vj] = C[vii, vjj, vi, vj] + A_[vii, vkk, vi, vk] * B_[vkk, vjj, vk, vj]
        # for ii, jj, kk in T.grid(M // blockM, N // blockN, K // blockK):
        #     with T.block("A_"):
                





                    



@tvm.testing.requires_tensorcore
@tvm.testing.requires_cublas
def mma_tune():
    arch = tvm.contrib.nvcc.get_target_compute_version()
    major, minor = tvm.contrib.nvcc.parse_compute_version(arch)
    print(major, minor)
    if major < 8:
        return
    from tvm.contrib import cublas

    def tune():
        M, N, K = 1024, 1024, 1024
        target = Target("nvidia/geforce-rtx-4090")
        # func = te.create_prim_func(matmul_fp16(N=N, M=M, K=K)).with_attr(
        #     {"global_symbol": "main"}
        # )
        # mod = tvm.IRModule({"main": func})
        mod = CompressedTiledGeMM
        with tempfile.TemporaryDirectory() as work_dir:
            db = ms.tir_integration.tune_tir(
                mod=mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=8,
                builder=LocalBuilder(
                    f_build="meta_schedule.builder.async_build", initializer=initializer
                ),
                # space=ms.space_generator.PostOrderApply(
                #     sch_rules=[multi_level_tiling_mma()],
                # ),
            )
            sch = db.query_schedule(mod, target=target, workload_name="main")
            with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
                rt_mod = tvm.build(sch.mod, target=target)
        print(rt_mod.imported_modules[0].get_source())
        a_np = np.random.uniform(0, 1, size=(M, K)).astype("float16")
        b_np = np.random.uniform(0, 1, size=(K, N)).astype("float16")
        A_cublas = te.placeholder((M, K), name="A", dtype="float16")
        B_cublas = te.placeholder((K, N), name="B", dtype="float16")
        C_cublas = cublas.matmul(A_cublas, B_cublas, dtype="float16")
        s = te.create_schedule(C_cublas.op)
        dev = tvm.cuda(0)
        f_cublas = tvm.build(s, [A_cublas, B_cublas, C_cublas], target)
        a_cublas = tvm.nd.array(a_np.astype("float16"), dev)
        b_cublas = tvm.nd.array(b_np.astype("float16"), dev)
        c_cublas = tvm.nd.array(np.zeros((M, N), dtype="float16"), dev)
        f_cublas(a_cublas, b_cublas, c_cublas)
        a_tvm = tvm.nd.array(a_np, device=tvm.cuda(0))
        b_tvm = tvm.nd.array(b_np, device=tvm.cuda(0))
        c_tvm = tvm.nd.array(np.empty((M, N)).astype("float16"), device=tvm.cuda(0))
        rt_mod(a_tvm, b_tvm, c_tvm)
        # print(c_tvm.numpy())
        # print(c_cublas.numpy())
        assert np.allclose(c_tvm.numpy(), c_cublas.numpy(), rtol=1e-2)

    tune()

mma_tune()