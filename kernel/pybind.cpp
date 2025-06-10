#include <pybind11/pybind11.h>
#include <torch/extension.h>
// #include "vq_attention/vq_attention_decoding.h"
#include "vq_gemv/vq_gemv.h"
#include "vq_gemm/vq_gemm.h"
// #include "quip_gemm/quip_gemm.h"
// #include "aqlm_gemm/aqlm_gemm.h"
// #include "gptvq_gemm/gptvq_gemm.h"
// #include "gptvq_gemv/gptvq_gemv.h"
#include "e2e-gemm/e2e-gemm.h"
#include "e2e-gemv/e2e-gemv.h"
#include "e2e-attention/e2e-attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("vq_attention_decoding", &vq_attention_decoding, "");
    m.def("vq_gemv", &vq_gemv, "");
    m.def("vq_gemm", &vq_gemm, "");
    // m.def("quip_gemm", &quip_gemm, "");
    // m.def("aqlm_gemm", &aqlm_gemm, "");
    // m.def("gptvq_gemm", &gptvq_gemm, "");
    // m.def("gptvq_gemv", &gptvq_gemv, "");
    m.def("e2e_gemm", &e2e_gemm, "");
    m.def("e2e_gemm_rq", &e2e_gemm_rq, "");
    // m.def("e2e_gemv", &e2e_gemv, "");
    m.def("e2e_gemv_rq", &e2e_gemv_rq, "");
    m.def("e2e_attention", &e2e_attention, "");
}