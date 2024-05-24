#pragma once
#include <torch/extension.h>

torch::Tensor e2e_gemm(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook
);

torch::Tensor e2e_gemm_rq(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook,
    torch::Tensor w_r,
    torch::Tensor codebook_r
);