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
                "vq_attention/vq_attention_decoding.cu",
                "vq_gemv/vq_gemv.cu",
                "vq_gemm/vq_gemm.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
