#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "lutgemm.h"
#include "typetraits.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>

#define K_TILE_SIZE 32
#define NUM_THREADS 256
#define M_TILE_SIZE 2048


__global__ void nqmv_bias(
    uint32_t* W, // quantized weights, W[kSize/32][nb][mSize]
    __half* alpha, // alpha[num_groups][nb][mSize]
    __half* q_bias, // q_bias[num_groups][mSize]
    __half* input, // input[kSize]
    __half* output, // output[mSize]
    int M, // mSize
    int K, // kSize
    int NUM_BITS, // nb
    int group_size // group_size
) {
    __shared__ __half lut[K_TILE_SIZE / 8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE / 8);

    int lut_y = threadIdx.x / lut_x_size;
    int lut_x = threadIdx.x % lut_x_size;

    __half* _inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];

    __half base =    + __float2half((2 * ((lut_x>>0) & 1) - 1)) * _inp[0]
                     + __float2half((2 * ((lut_x>>1) & 1) - 1)) * _inp[1]
                     + __float2half((2 * ((lut_x>>2) & 1) - 1)) * _inp[2]
                     + __float2half((2 * ((lut_x>>3) & 1) - 1)) * _inp[3]
                     + __float2half((2 * ((lut_x>>4) & 1) - 1)) * _inp[4]
                     + __float2half((2 * ((lut_x>>5) & 1) - 1)) * _inp[5]
                     + __float2half((2 * ((lut_x>>6) & 1) - 1)) * _inp[6]
                     + __float2half((2 * ((lut_x>>7) & 1) - 1)) * _inp[7];

    lut[lut_y][lut_x] = base;

    int s = (lut_x_size == 1)
                ? 0
                : (lut_x_size == 2)
                ? 1
                : (lut_x_size == 4)
                ? 2
                : (lut_x_size == 8)
                ? 3
                : (lut_x_size == 16)
                ? 4
                : (lut_x_size == 32)
                ? 5
                : (lut_x_size == 64)
                ? 6
                : (lut_x_size == 128)
                ? 7
                : 8;

    for (; s < 8; s++) {
        __half iValue = __float2half(2) * _inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] = lut[lut_y][i + lut_x - (1 << s)] + iValue;
        }
    }
    __syncthreads();

    int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x * 2;
    int m_end = (blockIdx.x + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = blockDim.x * 2;

    uint32_t* bW = &W[blockIdx.y * K_TILE_SIZE / 32 * NUM_BITS * M];
    int group_idx = (blockIdx.y * K_TILE_SIZE) / group_size;
    for (int m = m_start; m < m_end; m += m_step) {
        __half reg_o0 = 0;
        __half reg_o1 = 0;

        {
            __half reg_a0 = q_bias[group_idx * M + m + 0];
            __half reg_a1 = q_bias[group_idx * M + m + 1];
            __half reg_t_o0 = 0;
            __half reg_t_o1 = 0;
            for (int kt = 0; kt < K_TILE_SIZE / 32; kt++) {
                reg_t_o0 += +lut[kt * 4 + 0][255];
                reg_t_o0 += +lut[kt * 4 + 1][255];
                reg_t_o0 += +lut[kt * 4 + 2][255];
                reg_t_o0 += +lut[kt * 4 + 3][255];

                reg_t_o1 += +lut[kt * 4 + 0][255];
                reg_t_o1 += +lut[kt * 4 + 1][255];
                reg_t_o1 += +lut[kt * 4 + 2][255];
                reg_t_o1 += +lut[kt * 4 + 3][255];
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
        }

        __half reg_a0 = alpha[group_idx * NUM_BITS * M + m + 0];
        __half reg_a1 = alpha[group_idx * NUM_BITS * M + m + 1];
        for (int b = 0; b < NUM_BITS; b++) {
            __half reg_t_o0 = 0;
            __half reg_t_o1 = 0;

            //__half reg_a0 = alpha[group_idx * NUM_BITS * M + b * M + m + 0];
            //__half reg_a1 = alpha[group_idx * NUM_BITS * M + b * M + m + 1];
            for (int kt = 0; kt < K_TILE_SIZE / 32; kt++) {
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m];
                int reg_w0 = (reg_w >> 8 * 0) & 255;
                reg_t_o0 += +lut[kt * 4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;
                reg_t_o0 += +lut[kt * 4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;
                reg_t_o0 += +lut[kt * 4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;
                reg_t_o0 += +lut[kt * 4 + 3][reg_w3];

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1];
                reg_w0 = (reg_w >> 8 * 0) & 255;
                reg_t_o1 += +lut[kt * 4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;
                reg_t_o1 += +lut[kt * 4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;
                reg_t_o1 += +lut[kt * 4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;
                reg_t_o1 += +lut[kt * 4 + 3][reg_w3];
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
	     
	    reg_a0 = reg_a0 * __int2half_rn(2);
	    reg_a1 = reg_a1 * __int2half_rn(2);
	    
        }
	atomicAdd((half2*)&output[m], __halves2half2(reg_o0, reg_o1));
    }
}

