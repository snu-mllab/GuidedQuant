#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gemv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("anyprec_gemv", &anyprec_gemv, "ANYPREC GEMV");
	m.def("lutgemm_gemv", &lutgemm_gemv, "LUTGEMM GEMV");
}
