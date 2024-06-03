#pragma once
#include <torch/extension.h>

torch::Tensor e2e_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k_codebook,
    torch::Tensor v_codebook,
    torch::Tensor k_cache_window,
    torch::Tensor v_cache_window,
    int cnt
);