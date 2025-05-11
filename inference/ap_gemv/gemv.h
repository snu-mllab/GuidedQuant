#ifndef GEMV_CUH
#define GEMV_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>

#include <torch/extension.h>
#include <cuda_runtime.h>

void anyprec_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
);

void lutgemm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size
);

#endif 
