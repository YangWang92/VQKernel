#pragma once
#include <torch/extension.h>

torch::Tensor vq_attention_decoding(
    torch::Tensor q,        
    torch::Tensor k,                torch::Tensor v,
    torch::Tensor k_cache,          torch::Tensor v_cache,
    torch::Tensor k_codebook,       torch::Tensor v_codebook,
    torch::Tensor k_cache_configs,  torch::Tensor v_cache_configs,
    int head_num
);