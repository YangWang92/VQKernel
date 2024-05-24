#pragma once
#include <torch/extension.h>

/*  Compression ratio   == 2
 *  Residual            == 1
 *  Codebook shared      : 64x256 group
 */

 torch::Tensor gptvq_gemm(
    torch::Tensor input,
    torch::Tensor w,
    torch::Tensor codebook
 );