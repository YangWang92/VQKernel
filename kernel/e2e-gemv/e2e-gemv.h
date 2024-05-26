#pragma once
#include <torch/extension.h>

// torch::Tensor e2e_gemv(
//     torch::Tensor input,
//     torch::Tensor w,
//     torch::Tensor codebook
// );

torch::Tensor e2e_gemv_rq(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook,
    torch::Tensor w_r,
    torch::Tensor codebook_r
);