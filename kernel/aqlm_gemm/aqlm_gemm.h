#pragma once
#include <torch/extension.h>

/*  Compression ratio == 8
 *  Residual          == 2
 *  Codebook shared among tensor
 */
torch::Tensor aqlm_gemm(
    torch::Tensor input,        // Seq_len * Hidden_dim                     FP16
    torch::Tensor w,            // Hidden_dim * Hidden_dim / (8 / 2)        UINT16
    torch::Tensor codebook      // 65536 * (8 * 2)                          FP16
);

// Original W:      32MB
// Compressed W:    8MB
// Codebook:        2MB
//  