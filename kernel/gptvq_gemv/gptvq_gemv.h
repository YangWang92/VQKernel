#pragma once
#include <torch/extension.h>

torch::Tensor gptvq_gemv(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook
);