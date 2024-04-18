#pragma once
#include <torch/extension.h>

torch::Tensor vq_gemm(
    torch::Tensor h,
    torch::Tensor w,
    torch::Tensor codebook,
    int residual, int compression_ratio, int entry
);