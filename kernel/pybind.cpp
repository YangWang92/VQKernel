#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "vq_attention/vq_attention_decoding.h"
#include "vq_gemv/vq_gemv.h"
#include "vq_gemm/vq_gemm.h"
#include "quip_gemm/quip_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vq_attention_decoding", &vq_attention_decoding, "");
    m.def("vq_gemv", &vq_gemv, "");
    m.def("vq_gemm", &vq_gemm, "");
    m.def("quip_gemm", &quip_gemm, "");
}