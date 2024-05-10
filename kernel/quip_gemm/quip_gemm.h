#pragma once
#include <torch/extension.h>

/*  Compression ratio == 4
 *  Codebook shared among tensor
 */
torch::Tensor quip_gemm(
    torch::Tensor input,        // Seq_len * Hidden_dim             FP16
    torch::Tensor w,            // Hidden_dim * Hidden_dim / 4      UINT8
    torch::Tensor codebook      // 256 * 4                          FP16
);