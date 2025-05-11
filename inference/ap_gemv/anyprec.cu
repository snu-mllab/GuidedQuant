#include <cuda_fp16.h>
#include <cstdint>
#include <cassert>
#include "anyprec.h"
#include "typetraits.h"
#include "datatype.h"

#define ANYPREC_NUM_ROWS 4
#define num_rows 4
#define DIV_ROUND_UP(x, y) (((x)+(y)-1)/(y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))


template<int, bool>
__device__ __forceinline__ void dequant(const uint32_t q[], uint32_t q_w[]);

template <>
__device__ __forceinline__ void dequant<2, true>(const uint32_t q[2], uint32_t q_w[2]) {
        constexpr uint32_t mask0 = 0xAAAAAAAA;
        constexpr uint32_t mask1 = 0x55555555;

        q_w[0] = ((q[0]&mask0)) | ((q[1]&mask0) >> 1); 
        q_w[1] = ((q[0]&mask1) << 1) | (q[1]&mask1);
}

template <>
__device__ __forceinline__ void dequant<2, false>(const uint32_t q[2], uint32_t q_w[8]) {
        constexpr uint32_t mask0 = 0xAAAAAAAA;
        constexpr uint32_t mask1 = 0x55555555;

        q_w[0] = ((q[0]&mask0)) | ((q[1]&mask0) >> 1); 
        q_w[1] = ((q[0]&mask1) << 1) | (q[1]&mask1);

        constexpr uint32_t mask = 0x03030303;
        q_w[2] = (q_w[0] >> 4) & mask;
        q_w[3] = (q_w[1] >> 4) & mask;

        q_w[4] = (q_w[0] >> 2) & mask;
        q_w[5] = (q_w[1] >> 2) & mask;
    
        q_w[6] = (q_w[0] >> 0) & mask;
        q_w[7] = (q_w[1] >> 0) & mask;
    
        q_w[0] = (q_w[0] >> 6) & mask;
        q_w[1] = (q_w[1] >> 6) & mask;
}

template<>
__device__ __forceinline__ void dequant<3, true>(const uint32_t q[3], uint32_t q_w[4]) {
    constexpr
    uint32_t mask0 = 0x88888888;
    constexpr
    uint32_t mask1 = 0x44444444;
    constexpr
    uint32_t mask2 = 0x22222222;
    constexpr
    uint32_t mask3 = 0x11111111;

    // fast transpose
    q_w[0] = (((q[0] & mask0)) | ((q[1] & mask0) >> 1) | ((q[2] & mask0) >> 2)) >> 1;
    q_w[1] = ((q[0] & mask1)) | ((q[1] & mask1) >> 1) | ((q[2] & mask1) >> 2);
    q_w[2] = ((q[0] & mask2) << 1) | ((q[1] & mask2)) | ((q[2] & mask2) >> 1);
    q_w[3] = ((q[0] & mask3) << 2) | ((q[1] & mask3) << 1) | ((q[2] & mask3));

    // table lookup merge
    #pragma unroll
    for (int i = 0; i < 4; i++)
        q_w[i] = (q_w[i] & 0x0f0f0f0f) | ((q_w[i] & 0xf0f0f0f0) >> 1);
}

template<>
__device__ __forceinline__ void dequant<3, false>(const uint32_t q[3], uint32_t q_w[8]) {
    constexpr
    uint32_t mask0 = 0x88888888;
    constexpr
    uint32_t mask1 = 0x44444444;
    constexpr
    uint32_t mask2 = 0x22222222;
    constexpr
    uint32_t mask3 = 0x11111111;

    q_w[0] = (((q[0] & mask0)) | ((q[1] & mask0) >> 1) | ((q[2] & mask0) >> 2)) >> 1;
    q_w[1] = ((q[0] & mask1)) | ((q[1] & mask1) >> 1) | ((q[2] & mask1) >> 2);
    q_w[2] = ((q[0] & mask2) << 1) | ((q[1] & mask2)) | ((q[2] & mask2) >> 1);
    q_w[3] = ((q[0] & mask3) << 2) | ((q[1] & mask3) << 1) | ((q[2] & mask3));

    constexpr
    uint32_t mask = 0x0f0f0f0f;
    q_w[4] = q_w[0] & mask;
    q_w[5] = q_w[1] & mask;
    q_w[6] = q_w[2] & mask;
    q_w[7] = q_w[3] & mask;

    q_w[0] = (q_w[0] >> 4) & mask;
    q_w[1] = (q_w[1] >> 4) & mask;
    q_w[2] = (q_w[2] >> 4) & mask;
    q_w[3] = (q_w[3] >> 4) & mask;
}

template<>
__device__ __forceinline__ void dequant<4, true>(const uint32_t q[4], uint32_t q_w[4]) {
    constexpr
    uint32_t mask0 = 0x88888888;
    constexpr
    uint32_t mask1 = 0x44444444;
    constexpr
    uint32_t mask2 = 0x22222222;
    constexpr
    uint32_t mask3 = 0x11111111;

    q_w[0] = ((q[0] & mask0)) | ((q[1] & mask0) >> 1) | ((q[2] & mask0) >> 2) | ((q[3] & mask0) >> 3);
    q_w[1] = ((q[0] & mask1) << 1) | (q[1] & mask1) | ((q[2] & mask1) >> 1) | ((q[3] & mask1) >> 2);
    q_w[2] = ((q[0] & mask2) << 2) | ((q[1] & mask2) << 1) | (q[2] & mask2) | ((q[3] & mask2) >> 1);
    q_w[3] = ((q[0] & mask3) << 3) | ((q[1] & mask3) << 2) | ((q[2] & mask3) << 1) | (q[3] & mask3);
}

template<>
__device__ __forceinline__ void dequant<4, false>(const uint32_t q[4], uint32_t q_w[8]) {
    constexpr
    uint32_t mask0 = 0x88888888;
    constexpr
    uint32_t mask1 = 0x44444444;
    constexpr
    uint32_t mask2 = 0x22222222;
    constexpr
    uint32_t mask3 = 0x11111111;

    q_w[0] = ((q[0] & mask0)) | ((q[1] & mask0) >> 1) | ((q[2] & mask0) >> 2) | ((q[3] & mask0) >> 3);
    q_w[1] = ((q[0] & mask1) << 1) | (q[1] & mask1) | ((q[2] & mask1) >> 1) | ((q[3] & mask1) >> 2);
    q_w[2] = ((q[0] & mask2) << 2) | ((q[1] & mask2) << 1) | (q[2] & mask2) | ((q[3] & mask2) >> 1);
    q_w[3] = ((q[0] & mask3) << 3) | ((q[1] & mask3) << 2) | ((q[2] & mask3) << 1) | (q[3] & mask3);

    constexpr
    uint32_t mask = 0x0f0f0f0f;
    q_w[4] = q_w[0] & mask;
    q_w[5] = q_w[1] & mask;
    q_w[6] = q_w[2] & mask;
    q_w[7] = q_w[3] & mask;

    q_w[0] = (q_w[0] >> 4) & mask;
    q_w[1] = (q_w[1] >> 4) & mask;
    q_w[2] = (q_w[2] >> 4) & mask;
    q_w[3] = (q_w[3] >> 4) & mask;
}

template<>
__device__ __forceinline__ void dequant<8, false>(const uint32_t q[8], uint32_t q_w[8]) {
    constexpr
    uint32_t mask0 = 0x80808080;
    constexpr
    uint32_t mask1 = 0x40404040;
    constexpr
    uint32_t mask2 = 0x20202020;
    constexpr
    uint32_t mask3 = 0x10101010;
    constexpr
    uint32_t mask4 = 0x08080808;
    constexpr
    uint32_t mask5 = 0x04040404;
    constexpr
    uint32_t mask6 = 0x02020202;
    constexpr
    uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0] & mask0) >> 0) | ((q[1] & mask0) >> 1) | ((q[2] & mask0) >> 2) | ((q[3] & mask0) >> 3) |
             ((q[4] & mask0) >> 4) | ((q[5] & mask0) >> 5) | ((q[6] & mask0) >> 6) | ((q[7] & mask0) >> 7);
    q_w[1] = ((q[0] & mask1) << 1) | ((q[1] & mask1) >> 0) | ((q[2] & mask1) >> 1) | ((q[3] & mask1) >> 2) |
             ((q[4] & mask1) >> 3) | ((q[5] & mask1) >> 4) | ((q[6] & mask1) >> 5) | ((q[7] & mask1) >> 6);
    q_w[2] = ((q[0] & mask2) << 2) | ((q[1] & mask2) << 1) | ((q[2] & mask2) >> 0) | ((q[3] & mask2) >> 1) |
             ((q[4] & mask2) >> 2) | ((q[5] & mask2) >> 3) | ((q[6] & mask2) >> 4) | ((q[7] & mask2) >> 5);
    q_w[3] = ((q[0] & mask3) << 3) | ((q[1] & mask3) << 2) | ((q[2] & mask3) << 1) | ((q[3] & mask3) >> 0) |
             ((q[4] & mask3) >> 1) | ((q[5] & mask3) >> 2) | ((q[6] & mask3) >> 3) | ((q[7] & mask3) >> 4);
    q_w[4] = ((q[0] & mask4) << 4) | ((q[1] & mask4) << 3) | ((q[2] & mask4) << 2) | ((q[3] & mask4) << 1) |
             ((q[4] & mask4) >> 0) | ((q[5] & mask4) >> 1) | ((q[6] & mask4) >> 2) | ((q[7] & mask4) >> 3);
    q_w[5] = ((q[0] & mask5) << 5) | ((q[1] & mask5) << 4) | ((q[2] & mask5) << 3) | ((q[3] & mask5) << 2) |
             ((q[4] & mask5) << 1) | ((q[5] & mask5) >> 0) | ((q[6] & mask5) >> 1) | ((q[7] & mask5) >> 2);
    q_w[6] = ((q[0] & mask6) << 6) | ((q[1] & mask6) << 5) | ((q[2] & mask6) << 4) | ((q[3] & mask6) << 3) |
             ((q[4] & mask6) << 2) | ((q[5] & mask6) << 1) | ((q[6] & mask6) >> 0) | ((q[7] & mask6) >> 1);
    q_w[7] = ((q[0] & mask7) << 7) | ((q[1] & mask7) << 6) | ((q[2] & mask7) << 5) | ((q[3] & mask7) << 4) |
             ((q[4] & mask7) << 3) | ((q[5] & mask7) << 2) | ((q[6] & mask7) << 1) | ((q[7] & mask7) >> 0);
}

template<>
__device__ __forceinline__ void dequant<7, false>(const uint32_t q[7], uint32_t q_w[8]) {
    constexpr
    uint32_t mask0 = 0x80808080;
    constexpr
    uint32_t mask1 = 0x40404040;
    constexpr
    uint32_t mask2 = 0x20202020;
    constexpr
    uint32_t mask3 = 0x10101010;
    constexpr
    uint32_t mask4 = 0x08080808;
    constexpr
    uint32_t mask5 = 0x04040404;
    constexpr
    uint32_t mask6 = 0x02020202;
    constexpr
    uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0] & mask0) >> 1) | ((q[1] & mask0) >> 2) | ((q[2] & mask0) >> 3) | ((q[3] & mask0) >> 4) |
             ((q[4] & mask0) >> 5) | ((q[5] & mask0) >> 6) | ((q[6] & mask0) >> 7);
    q_w[1] = ((q[0] & mask1) >> 0) | ((q[1] & mask1) >> 1) | ((q[2] & mask1) >> 2) | ((q[3] & mask1) >> 3) |
             ((q[4] & mask1) >> 4) | ((q[5] & mask1) >> 5) | ((q[6] & mask1) >> 6);
    q_w[2] = ((q[0] & mask2) << 1) | ((q[1] & mask2) >> 0) | ((q[2] & mask2) >> 1) | ((q[3] & mask2) >> 2) |
             ((q[4] & mask2) >> 3) | ((q[5] & mask2) >> 4) | ((q[6] & mask2) >> 5);
    q_w[3] = ((q[0] & mask3) << 2) | ((q[1] & mask3) << 1) | ((q[2] & mask3) >> 0) | ((q[3] & mask3) >> 1) |
             ((q[4] & mask3) >> 2) | ((q[5] & mask3) >> 3) | ((q[6] & mask3) >> 4);
    q_w[4] = ((q[0] & mask4) << 3) | ((q[1] & mask4) << 2) | ((q[2] & mask4) << 1) | ((q[3] & mask4) >> 0) |
             ((q[4] & mask4) >> 1) | ((q[5] & mask4) >> 2) | ((q[6] & mask4) >> 3);
    q_w[5] = ((q[0] & mask5) << 4) | ((q[1] & mask5) << 3) | ((q[2] & mask5) << 2) | ((q[3] & mask5) << 1) |
             ((q[4] & mask5) >> 0) | ((q[5] & mask5) >> 1) | ((q[6] & mask5) >> 2);
    q_w[6] = ((q[0] & mask6) << 5) | ((q[1] & mask6) << 4) | ((q[2] & mask6) << 3) | ((q[3] & mask6) << 2) |
             ((q[4] & mask6) << 1) | ((q[5] & mask6) >> 0) | ((q[6] & mask6) >> 1);
    q_w[7] = ((q[0] & mask7) << 6) | ((q[1] & mask7) << 5) | ((q[2] & mask7) << 4) | ((q[3] & mask7) << 3) |
             ((q[4] & mask7) << 2) | ((q[5] & mask7) << 1) | ((q[6] & mask7) >> 0);
}

template<>
__device__ __forceinline__ void dequant<6, false>(const uint32_t q[6], uint32_t q_w[8]) {
    constexpr
    uint32_t mask0 = 0x80808080;
    constexpr
    uint32_t mask1 = 0x40404040;
    constexpr
    uint32_t mask2 = 0x20202020;
    constexpr
    uint32_t mask3 = 0x10101010;
    constexpr
    uint32_t mask4 = 0x08080808;
    constexpr
    uint32_t mask5 = 0x04040404;
    constexpr
    uint32_t mask6 = 0x02020202;
    constexpr
    uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0] & mask0) >> 2) | ((q[1] & mask0) >> 3) | ((q[2] & mask0) >> 4) | ((q[3] & mask0) >> 5) |
             ((q[4] & mask0) >> 6) | ((q[5] & mask0) >> 7);
    q_w[1] = ((q[0] & mask1) >> 1) | ((q[1] & mask1) >> 2) | ((q[2] & mask1) >> 3) | ((q[3] & mask1) >> 4) |
             ((q[4] & mask1) >> 5) | ((q[5] & mask1) >> 6);
    q_w[2] = ((q[0] & mask2) >> 0) | ((q[1] & mask2) >> 1) | ((q[2] & mask2) >> 2) | ((q[3] & mask2) >> 3) |
             ((q[4] & mask2) >> 4) | ((q[5] & mask2) >> 5);
    q_w[3] = ((q[0] & mask3) << 1) | ((q[1] & mask3) >> 0) | ((q[2] & mask3) >> 1) | ((q[3] & mask3) >> 2) |
             ((q[4] & mask3) >> 3) | ((q[5] & mask3) >> 4);
    q_w[4] = ((q[0] & mask4) << 2) | ((q[1] & mask4) << 1) | ((q[2] & mask4) >> 0) | ((q[3] & mask4) >> 1) |
             ((q[4] & mask4) >> 2) | ((q[5] & mask4) >> 3);
    q_w[5] = ((q[0] & mask5) << 3) | ((q[1] & mask5) << 2) | ((q[2] & mask5) << 1) | ((q[3] & mask5) >> 0) |
             ((q[4] & mask5) >> 1) | ((q[5] & mask5) >> 2);
    q_w[6] = ((q[0] & mask6) << 4) | ((q[1] & mask6) << 3) | ((q[2] & mask6) << 2) | ((q[3] & mask6) << 1) |
             ((q[4] & mask6) >> 0) | ((q[5] & mask6) >> 1);
    q_w[7] = ((q[0] & mask7) << 5) | ((q[1] & mask7) << 4) | ((q[2] & mask7) << 3) | ((q[3] & mask7) << 2) |
             ((q[4] & mask7) << 1) | ((q[5] & mask7) << 0);
}

template<>
__device__ __forceinline__ void dequant<5, false>(const uint32_t q[5], uint32_t q_w[8]) {
    constexpr
    uint32_t mask0 = 0x80808080;
    constexpr
    uint32_t mask1 = 0x40404040;
    constexpr
    uint32_t mask2 = 0x20202020;
    constexpr
    uint32_t mask3 = 0x10101010;
    constexpr
    uint32_t mask4 = 0x08080808;
    constexpr
    uint32_t mask5 = 0x04040404;
    constexpr
    uint32_t mask6 = 0x02020202;
    constexpr
    uint32_t mask7 = 0x01010101;

    q_w[0] = ((q[0] & mask0) >> 3) | ((q[1] & mask0) >> 4) | ((q[2] & mask0) >> 5) | ((q[3] & mask0) >> 6) |
             ((q[4] & mask0) >> 7);
    q_w[1] = ((q[0] & mask1) >> 2) | ((q[1] & mask1) >> 3) | ((q[2] & mask1) >> 4) | ((q[3] & mask1) >> 5) |
             ((q[4] & mask1) >> 6);
    q_w[2] = ((q[0] & mask2) >> 1) | ((q[1] & mask2) >> 2) | ((q[2] & mask2) >> 3) | ((q[3] & mask2) >> 4) |
             ((q[4] & mask2) >> 5);
    q_w[3] = ((q[0] & mask3) >> 0) | ((q[1] & mask3) >> 1) | ((q[2] & mask3) >> 2) | ((q[3] & mask3) >> 3) |
             ((q[4] & mask3) >> 4);
    q_w[4] = ((q[0] & mask4) << 1) | ((q[1] & mask4) >> 0) | ((q[2] & mask4) >> 1) | ((q[3] & mask4) >> 2) |
             ((q[4] & mask4) >> 3);
    q_w[5] = ((q[0] & mask5) << 2) | ((q[1] & mask5) << 1) | ((q[2] & mask5) >> 0) | ((q[3] & mask5) >> 1) |
             ((q[4] & mask5) >> 2);
    q_w[6] = ((q[0] & mask6) << 3) | ((q[1] & mask6) << 2) | ((q[2] & mask6) << 1) | ((q[3] & mask6) >> 0) |
             ((q[4] & mask6) >> 1);
    q_w[7] = ((q[0] & mask7) << 4) | ((q[1] & mask7) << 3) | ((q[2] & mask7) << 2) | ((q[3] & mask7) << 1) |
             ((q[4] & mask7) >> 0);
}

template <int bits>
__global__ void dequant_kbit_store(
	const uint32_t * W,
	const uint32_t N, const uint32_t K,
	const half * C, half * O
) {
	static_assert(bits >= 2 && bits <= 8);
	constexpr int num_centroids = 1 << bits, warp_size = 32;

	const uint32_t row_idx = blockIdx.x * num_rows + threadIdx.y;
	const int centroid_idx = threadIdx.y * num_centroids;

	__shared__ half shC[num_rows * num_centroids];

	if constexpr (bits < 6) {
		if (threadIdx.x < num_centroids)
			shC[centroid_idx + threadIdx.x] = C[num_centroids * row_idx + threadIdx.x];
	} else if constexpr (bits == 6) {
		((half2 *)shC)[centroid_idx / 2 + threadIdx.x] = ((half2 *)C)[num_centroids * row_idx / 2 + threadIdx.x];
	} else if constexpr (bits == 7) {
		((float2 *)shC)[centroid_idx / 4 + threadIdx.x] = ((float2 *)C)[num_centroids * row_idx / 4 + threadIdx.x];
	} else if constexpr (bits == 8) {
		((float4 *)shC)[centroid_idx / 8 + threadIdx.x] = ((float4 *)C)[num_centroids * row_idx / 8 + threadIdx.x];
	}
	__syncthreads();

	int eff_warp_size = warp_size;
	uint32_t q[bits], q_w[8];
	half2 dq_w[16];

	const uint32_t maxi = DIV_ROUND_UP(K, 32 * warp_size);
	for (int i = 0; i < maxi; i++) {
		if (i == K / (32 * warp_size)) {
			eff_warp_size = (K % (32 * warp_size)) / 32;
			if (threadIdx.x >= eff_warp_size) break;
		}

		// load quantized weight
		#pragma unroll
		for (int j = 0; j < bits; j++) {
			const int k = (j * N + row_idx) * (K / 32) + i * 32 + threadIdx.x;
			q[j] = W[k];
		}

		// dequantize
		dequant<bits, false>(q, q_w);

		// lookup
		#pragma unroll
		for (int j = 3; j >= 0; j--) {
			#pragma unroll
			for (int k = 0; k < 4; k++) {
				const half x = shC[centroid_idx | (q_w[k*2+0] & 0xff)];
				const half y = shC[centroid_idx | (q_w[k*2+1] & 0xff)];
				dq_w[j * 4 + k] = make_half2(x, y);
			}
			#pragma unroll
			for (int k = 0; k < 8; k++)
				q_w[k] >>= 8;
		}

		#pragma unroll
		for (int j = 0; j < 4; j++)
			((float4 *)O)[(row_idx*K + 8*eff_warp_size*j + i*warp_size*32 + 8*threadIdx.x)/8] = ((float4 *)dq_w)[j];
	}
}


/* warp-wide sum with tree-reduction */
__device__ __forceinline__ half warp_reduce_sum(
        half sum
) {
    #pragma unroll
    for (int i = 4; i >= 0; i--)
        sum += __shfl_down_sync(0xffffffff, sum, 1 << i);
    return sum;
}

template <int maxm, int bits, bool use_ksplit>
__global__ void matmul_kbit_32(
	const half * I, const uint32_t * W,
	const uint32_t M, const uint32_t N, const uint32_t K,
	const half * C, half * O
) {
	static_assert(maxm >= 1 && bits >= 2 && bits <= 8);
	static_assert(!use_ksplit || maxm == 1);
	constexpr bool use_half2_centroid = (bits <= 3 || (bits == 4 && maxm > 1));
	constexpr int multi_row = (maxm == 1 ? 1 : 4);

	constexpr int num_centroids = 1 << bits, warp_size = 32;
	constexpr int shC_siz = (use_half2_centroid ? num_centroids * num_centroids * 2 : num_centroids);
	constexpr int q_w_siz = (use_half2_centroid && bits == 2) ? 2 : (use_half2_centroid ? 4 : 8);

	const uint32_t row_idx_base = blockIdx.x * num_rows * multi_row + threadIdx.y;
	const int centroid_idx_base = threadIdx.y * (use_half2_centroid ? num_centroids * num_centroids : num_centroids);

	__shared__ half shC[num_rows * multi_row * shC_siz];

	if (!use_ksplit || threadIdx.z == 0) {
		#pragma unroll
		for (int h = 0; h < multi_row; h++) {
			const uint32_t row_idx = row_idx_base + h * num_rows;
			const int centroid_idx = centroid_idx_base + h * num_rows * (use_half2_centroid ? num_centroids * num_centroids : num_centroids);
			if constexpr (use_half2_centroid) {
				// num_centroids * num_centroids is smaller than warp size when bits == 2
				if constexpr(bits == 2){
					if (threadIdx.x >= num_centroids * num_centroids) break;
				}
				const int xx = threadIdx.x % num_centroids, yy = threadIdx.x / num_centroids;
				const half fragCX = C[row_idx * num_centroids | xx];
				#pragma unroll
				for (int i = 0; i < MAX(1, shC_siz / warp_size / 2); i++) {
					const int yidx = yy | (i * warp_size / num_centroids);
					const half fragCY = C[row_idx * num_centroids | yidx];
					((half2 * )shC)[centroid_idx | (yidx * num_centroids) | xx] = make_half2(fragCY, fragCX);
				}
			} else if constexpr (bits < 6) {
				if (threadIdx.x < num_centroids)
					shC[centroid_idx + threadIdx.x] = C[num_centroids * row_idx + threadIdx.x];
			} else if constexpr (bits == 6) {
				((half2 *)shC)[centroid_idx / 2 + threadIdx.x] = ((half2 *)C)[num_centroids * row_idx / 2 + threadIdx.x];
			} else if constexpr (bits == 7) {
				((float2 *)shC)[centroid_idx / 4 + threadIdx.x] = ((float2 *)C)[num_centroids * row_idx / 4 + threadIdx.x];
			} else if constexpr (bits == 8) {
				((float4 *)shC)[centroid_idx / 8 + threadIdx.x] = ((float4 *)C)[num_centroids * row_idx / 8 + threadIdx.x];
			}
		}
	}
	__syncthreads();

	int eff_warp_size = warp_size;
	half partial_sum[maxm * multi_row] = {0, };
	uint32_t q[bits], q_w[q_w_siz];
	half2 dq_w[16];

	int mini = (use_ksplit ? threadIdx.z * 4 : 0);
	int maxi = DIV_ROUND_UP(K, 32 * warp_size);
	if (use_ksplit && maxi > mini + 4) maxi = mini + 4;
	for (int i = mini; i < maxi; i++) {
		if (i == K / (32 * warp_size)) {
			eff_warp_size = (K % (32 * warp_size)) / 32;
			if (threadIdx.x >= eff_warp_size) break;
		}

		#pragma unroll
		for (int h = 0; h < multi_row; h++) {
			const uint32_t row_idx = row_idx_base + h * num_rows;
			const int centroid_idx = centroid_idx_base + h * num_rows * (use_half2_centroid ? num_centroids * num_centroids : num_centroids);

			// load quantized weight
			#pragma unroll
			for (int j = 0; j < bits; j++) {
				const int k = (j * N + row_idx) * (K / 32) + i * 32 + threadIdx.x;
				q[j] = W[k];
			}

			// dequantize
			dequant<bits, use_half2_centroid>(q, q_w);

			// lookup
			if (use_half2_centroid && bits == 2) {
				#pragma unroll
				for (int j = 7; j >= 0; j--) {
					if constexpr (use_half2_centroid) {
						const half2 x = ((half2 * )shC)[centroid_idx | (q_w[0] & 0xf)];
						const half2 y = ((half2 * )shC)[centroid_idx | (q_w[1] & 0xf)];
						dq_w[j * 2] = make_half2(x.x, y.x);
						dq_w[j * 2 + 1] = make_half2(x.y, y.y);
					}
					for (int k = 0; k < q_w_siz; k++)
						q_w[k] >>= 4;
				}
			}
			else {
				#pragma unroll
				for (int j = 3; j >= 0; j--) {
					if constexpr (use_half2_centroid) {
						#pragma unroll
						for (int k = 0; k < 2; k++) {
							const half2 x = ((half2 * )shC)[centroid_idx | (q_w[k*2+0] & 0xff)];
							const half2 y = ((half2 * )shC)[centroid_idx | (q_w[k*2+1] & 0xff)];
							dq_w[j * 4 + k + 0] = make_half2(x.x, y.x);
							dq_w[j * 4 + k + 2] = make_half2(x.y, y.y);
						}
					} else {
						#pragma unroll
						for (int k = 0; k < 4; k++) {
							const half x = shC[centroid_idx | (q_w[k*2+0] & 0xff)];
							const half y = shC[centroid_idx | (q_w[k*2+1] & 0xff)];
							dq_w[j * 4 + k] = make_half2(x, y);
						}
					}
					#pragma unroll
					for (int k = 0; k < q_w_siz; k++)
						q_w[k] >>= 8;
				}
			}

			// accumulate
			#pragma unroll
			for (int l = 0; l < maxm; l++) {
				half2 sum = make_half2(0.0, 0.0);
				#pragma unroll
				for (int j = 3; j >= 0; j--) {
					const int idx = (l*K/8 + eff_warp_size*j) + i*warp_size*4 + threadIdx.x;
					float4 in_buf = ((float4 *)I)[idx];
					half2 * in_half = (half2 *)&in_buf;
					#pragma unroll
					for (int k = 0; k < 4; k++)
						sum = __hfma2(dq_w[j * 4 + k], in_half[k], sum);
				}
				partial_sum[l + h * maxm] += sum.x + sum.y;
			}
		}
	}

	#pragma unroll
	for (int i = 0; i < maxm * multi_row; i++)
		partial_sum[i] = warp_reduce_sum(partial_sum[i]);

	if constexpr (use_ksplit) {
		__shared__ half shO[maxm * multi_row * num_rows];
		if (threadIdx.x == 0 && threadIdx.z == 0)
			#pragma unroll
			for (int j = 0; j < multi_row; j++)
				shO[j + threadIdx.y * multi_row] = 0;
		__syncthreads();
		if (threadIdx.x == 0)
			#pragma unroll
			for (int j = 0; j < multi_row; j++)
				atomicAdd(shO + j + threadIdx.y * multi_row, partial_sum[j]);
		__syncthreads();
		if (threadIdx.x == 0 && threadIdx.z == 0)
			#pragma unroll
			for (int j = 0; j < multi_row; j++)
				partial_sum[j] = shO[j + threadIdx.y * multi_row];
	}

	if (threadIdx.x == 0 && (!use_ksplit || threadIdx.z == 0)) {
		#pragma unroll
		for (int i = 0; i < maxm; i++) {
			#pragma unroll
			for (int j = 0; j < multi_row; j++) {
				const uint32_t row_idx = row_idx_base + j * num_rows;
				O[i * N + row_idx] = partial_sum[i + j * maxm];
			}
		}
	}
}


using matmul_func = void (*)(
        const __half *, const uint32_t *,
        const uint32_t, const uint32_t, const uint32_t,
        const __half *, __half *
);

template<int s, int e>
struct get_matmul_func {
    void operator()(matmul_func func[][9][2]) const {
        if constexpr(s <= e)
        {
            func[s][1][0] = matmul_kbit_32<1, s, false>;
            func[s][1][1] = matmul_kbit_32<1, s, true>;
            func[s][2][0] = matmul_kbit_32<2, s, false>;
            func[s][3][0] = matmul_kbit_32<3, s, false>;
            func[s][4][0] = matmul_kbit_32<4, s, false>;
            func[s][5][0] = matmul_kbit_32<5, s, false>;
            func[s][6][0] = matmul_kbit_32<6, s, false>;
            func[s][7][0] = matmul_kbit_32<7, s, false>;
            func[s][8][0] = matmul_kbit_32<8, s, false>;
            get_matmul_func<s + 1, e>()(func);
        }
    }
};

using dequant_func = void (*)(
        const uint32_t *,
        const uint32_t, const uint32_t,
        const half *, half *
);

template<int s, int e>
struct get_dequant_func {
    void operator()(dequant_func func[]) const {
        if constexpr(s <= e)
        {
            func[s] = dequant_kbit_store<s>;
            get_dequant_func<s + 1, e>()(func);
        }
    }
};

bool matmul_initialized = false;

matmul_func matmul_functions[9][9][2] = {nullptr};

void anyprec_matmul(
    __half *in,        // [M, K]
    __half *out,       // [M, N]
    uint32_t *qweight,   // [w_bits, N, K/32]
    __half *lut,       // [out_size, num_centroids]
    uint32_t M,           // batch size
    uint32_t N,           // output size
    uint32_t K,           // input size
    int w_bits,            // bit width
    cudaStream_t stream
) {
    assert(M >= 1 && M <= 8 && w_bits >= 2 && w_bits <= 8);
    // Initialize the function pointers if they haven't been initialized for this type
    if (!matmul_initialized) {
    get_matmul_func<2, 8>()(matmul_functions);
    matmul_initialized = true;
    }

    // Compute grid and block dimensions
    const int multi_row = (M == 1 ? 1 : 4);
    const int use_ksplit = M == 1 && K > 4096 && w_bits >= 7;
    const int num_ksplit = (use_ksplit ? DIV_ROUND_UP(K, 4096) : 1);

    dim3 grid(N / (ANYPREC_NUM_ROWS * multi_row)), block(32, ANYPREC_NUM_ROWS, num_ksplit);

    // Use the initialized function pointers for the kernel launch
    matmul_functions[w_bits][M][use_ksplit]<<<grid, block, 0, stream>>>(
        in, qweight, M, N, K, lut, out
    );
}

bool dequant_initalized = false;

dequant_func dequant_functions[9] = {nullptr};


void anyprec_dequant_kbit(
    const uint32_t *qweight,
    const uint32_t N, const uint32_t K,
    const __half *lut, __half *weight,
    int w_bits,
    cudaStream_t stream
) {
    assert(w_bits >= 2 && w_bits <= 8);

    if (!dequant_initalized) {
        get_dequant_func<2, 8>()(dequant_functions);
        dequant_initalized = true;
    }

    dim3 grid(N / ANYPREC_NUM_ROWS), block(32, ANYPREC_NUM_ROWS);
    dequant_functions[w_bits]<<<grid, block, 0, stream>>>(
        qweight, N, K, lut, weight
    );
}

