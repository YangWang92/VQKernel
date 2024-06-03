from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

extra_compile_args = {
    "cxx": [
        "-std=c++17",
        "-O2"
    ],
    "nvcc": [
        "-O2",
        "-std=c++17",
        "-arch=compute_80",
        "-code=sm_80",
        # "-lineinfo",
    ],
}

setup(
    name="VQKernel",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="VQKernel",
            sources=[
                "pybind.cpp",
                # "vq_attention/vq_attention_decoding.cu",
                # "vq_gemv/vq_gemv.cu",
                # "vq_gemm/vq_gemm.cu",
                # "quip_gemm/quip_gemm.cu",
                # "aqlm_gemm/aqlm_gemm.cu",
                # "gptvq_gemm/gptvq_gemm.cu",
                # "gptvq_gemv/gptvq_gemv.cu",
                "e2e-gemm/e2e-gemm.cu",
                "e2e-gemv/e2e-gemv.cu",
                "e2e-attention/e2e-attention.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
