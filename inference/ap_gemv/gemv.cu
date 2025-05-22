#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <torch/extension.h>
#include "gemv.h"
#include "anyprec.h"
#include "lutgemm.h"
#include "datatype.h"

// LUTGEMM
#define K_TILE_SIZE 32
#define NUM_THREADS 256
#define M_TILE_SIZE 2048

void cudaError(cudaError_t errCode, const char * filename, int linenum) {
    if(errCode != cudaSuccess) {
        printf("Error : %s (%s : %d)\n", cudaGetErrorString(errCode), filename, linenum);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (cudaError(err, __FILE__, __LINE__))

////////////////////////////////////////////////////////////////////////////////
//                                     ANYPREC
////////////////////////////////////////////////////////////////////////////////

void anyprec_gemv_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    uint32_t M = input.size(0);
    uint32_t N = output.size(2);
    uint32_t K = input.size(2);

    anyprec_matmul(
        (__half*)input.data_ptr<at::Half>(),
        (__half*)output.data_ptr<at::Half>(),
        (uint32_t*)qweight.data_ptr<int>(),
        (__half*)lut.data_ptr<at::Half>(),
        M, N, K,
        bitwidth,
        stream
    );
}

void anyprec_gemv_stream(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth,
    cudaStream_t stream
) {
    TORCH_CHECK(bitwidth >= 2 && bitwidth <= 8, "Bitwidth must be between 2 and 8.");
    TORCH_CHECK(input.scalar_type() == lut.scalar_type() && input.scalar_type() == output.scalar_type(), 
                "Mismatched data types between input, lut, and output tensors.");
    TORCH_CHECK(qweight.scalar_type() == at::kInt, "qweight tensor must be of type int.");
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, hidden_size).");
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, hidden_size).");

    // lut is of shape (output_feat, 2 ** bitwidth)
    TORCH_CHECK(lut.dim() == 2 && lut.size(1) == (1 << bitwidth) && lut.size(0) == output.size(2),
    "lut tensor must be of shape (output_feat, 2 ** bitwidth). Expected (", output.size(2), ", ", 1 << bitwidth, "), got (", lut.size(0), ", ", lut.size(1), ").");

    // qweight is of shape (bitwidth, output_feat, input_feat / 32)
    TORCH_CHECK(qweight.dim() == 3 && qweight.size(0) == bitwidth && qweight.size(2) == input.size(2) / 32 && qweight.size(1) == output.size(2),
    "qweight tensor must be of shape (bitwidth, output_feat, input_feat / 32). Expected (", bitwidth, ", ", output.size(2), ", ", input.size(2) / 32, "), got (", qweight.size(0), ", ", qweight.size(1), ", ", qweight.size(2), ").");

    // Check that sequence length is 1
    TORCH_CHECK(input.size(1) == 1, "Only sequence length of 1 is supported.");
    TORCH_CHECK(output.size(1) == 1, "Only sequence length of 1 is supported.");

    // Check that input and output are both on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(qweight.is_contiguous(), "qweight tensor must be contiguous.");
    TORCH_CHECK(lut.is_contiguous(), "lut tensor must be contiguous.");

    auto dtype = input.scalar_type();
    anyprec_gemv_templated(input, output, qweight, lut, bitwidth, stream);
}

void anyprec_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
) {
    HANDLE_ERROR(cudaSetDevice(qweight.device().index()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    anyprec_gemv_stream(input, output, qweight, lut, bitwidth, stream);
}

torch::Tensor anyprec_dequant(
    torch::Tensor qweight,
    torch::Tensor lut,
    int bitwidth
) {
    HANDLE_ERROR(cudaSetDevice(qweight.device().index()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(qweight.device());
    at::Tensor weight = torch::empty({N, K}, options);

    anyprec_dequant_kbit(
        (uint32_t*) qweight.data_ptr<int32_t>(),
        N, K,
        (__half*) lut.data_ptr<at::Half>(),
        (__half*) weight.data_ptr<at::Half>(),
        bitwidth,
        stream
    );

    return weight;
}

////////////////////////////////////////////////////////////////////////////////
//                                     LUTGEMM
////////////////////////////////////////////////////////////////////////////////

void lutgemm_gemv_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size,
    cudaStream_t stream
) {
    uint32_t kSize = input.size(2);
    uint32_t mSize = output.size(2);

    dim3 grid((mSize + M_TILE_SIZE - 1) / M_TILE_SIZE,
              (kSize + K_TILE_SIZE - 1) / K_TILE_SIZE);
    dim3 block(NUM_THREADS);

    nqmv_bias<<<grid, block, 0, stream>>>(
        (uint32_t*) q_weight.data_ptr<int32_t>(),
        (__half*) alpha.data_ptr<at::Half>(),
        (__half*) q_bias.data_ptr<at::Half>(),
        (__half*) input.data_ptr<at::Half>(),
        (__half*) output.data_ptr<at::Half>(),
        mSize, kSize, bitwidth, group_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
}

void lutgemm_gemv_stream(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size,
    cudaStream_t stream
) {
    TORCH_CHECK(bitwidth >= 1 && bitwidth <= 8, "Bitwidth must be between 1 and 8.");
    TORCH_CHECK(input.scalar_type() == alpha.scalar_type() && input.scalar_type() == q_bias.scalar_type() && input.scalar_type() == output.scalar_type(), "Mismatched data types between input, alpha, q_bias, and output tensors.");
    // Check that input is of shape (batch_size, seq_len, input_feat)
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, input_feat).");
    // Check that output is of shape (batch_size, seq_len, output_feat)
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, output_feat).");

    // Only allow single batch size and sequence length
    TORCH_CHECK(input.size(0) == 1, "Batch size must be 1 for input tensor.");
    TORCH_CHECK(input.size(1) == 1, "Sequence length must be 1 for input tensor.");
    TORCH_CHECK(output.size(0) == 1, "Batch size must be 1 for output tensor.");
    TORCH_CHECK(output.size(1) == 1, "Sequence length must be 1 for output tensor.");

    // Check that input and output are both on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(q_weight.is_contiguous(), "q_weight tensor must be contiguous.");
    TORCH_CHECK(alpha.is_contiguous(), "alpha tensor must be contiguous.");
    TORCH_CHECK(q_bias.is_contiguous(), "q_bias tensor must be contiguous.");

    uint32_t kSize = input.size(2);
    uint32_t mSize = output.size(2);
    uint32_t num_groups = kSize / group_size;

    // check that q_weight is of shape (input_feat / 32, bitwidth, output_feat)
    TORCH_CHECK(q_weight.dim() == 3 && q_weight.size(0) == kSize / 32 && q_weight.size(1) == bitwidth && q_weight.size(2) == mSize, "q_weight tensor must be of shape (input_feat / 32, bitwidth, output_feat). Expected (", kSize / 32, ", ", bitwidth, ", ", mSize, "), got (", q_weight.size(0), ", ", q_weight.size(1), ", ", q_weight.size(2), ").");
    // check that alpha is of shape (num_groups, bitwidth, mSize)
    TORCH_CHECK(alpha.dim() == 3 && alpha.size(0) == num_groups && alpha.size(1) == bitwidth && alpha.size(2) == mSize, 
                "alpha tensor must be of shape (num_groups, bitwidth, output_feat). Expected (", num_groups, ", ", bitwidth, ", ", mSize, "), got (", alpha.size(0), ", ", alpha.size(1), ", ", alpha.size(2), ").");

    auto dtype = input.scalar_type();
    lutgemm_gemv_templated(input, output, q_weight, alpha, q_bias, bitwidth, group_size, stream);
}

void lutgemm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor q_bias,
    int bitwidth,
    int group_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lutgemm_gemv_stream(input, output, q_weight, alpha, q_bias, bitwidth, group_size, stream);
}
