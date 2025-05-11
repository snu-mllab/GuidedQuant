#ifndef TYPE_TRAITS_CUH
#define TYPE_TRAITS_CUH

#include <cmath>
#include "datatype.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>

template<DataType DT>
struct TypeTraits;

// Specialization for FP32
template<>
struct TypeTraits<DataType::FP32> {
    using fp_dtype = float;
    using aten = float;
    static constexpr bool is_halfsized = false;

#ifdef __CUDACC__
    // CUDA-only functions
    using fp_dtype2 = float2;

    static __device__ __forceinline__ float to_dtype(float x) { return x; }
    static __device__ __forceinline__ float to_float(float x) { return x; }
    static __device__ __forceinline__ float abs(float x) { return std::abs(x); }

    static __device__ __forceinline__ void atomicAddWrapper(float *address, float value) {
        atomicAdd(address, value);
    }
    static __device__ __forceinline__ float2 to_dtype2(float x, float y) { return {x, y}; }
    static __device__ __forceinline__ float2 hfma2(const float2 &a, const float2 &b, const float2 &c) {
        return {fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)};
    }
#endif
};

// Specialization for FP16
template<>
struct TypeTraits<DataType::FP16> {
    using fp_dtype = __half;
    using aten = at::Half;
    static constexpr bool is_halfsized = true;

#ifdef __CUDACC__
    // CUDA-only functions
    using fp_dtype2 = __half2;

    static __device__ __forceinline__ __half to_dtype(float x) { return __float2half(x); }
    static __device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
    static __device__ __forceinline__ __half abs(__half x) { return __habs(x); }

    static __device__ __forceinline__ void atomicAddWrapper(__half *address, __half value) {
        const __half next_col_sum = __shfl_down_sync(0xFFFFFFFF, value, 1);
        if (threadIdx.x % 2 == 0) {
            atomicAdd(reinterpret_cast<__half2 *>(address), __half2(value, next_col_sum));
        }
    }
    static __device__ __forceinline__ __half2 to_dtype2(__half x, __half y) { return {x, y}; }
    static __device__ __forceinline__ __half2 hfma2(const __half2 &a, const __half2 &b, const __half2 &c) {
        return __hfma2(a, b, c);
    }
#endif
};

// Specialization for BF16
template<>
struct TypeTraits<DataType::BF16> {
    using fp_dtype = __nv_bfloat16;
    using aten = at::BFloat16;
    static constexpr bool is_halfsized = true;

#ifdef __CUDACC__
    // CUDA-only functions
    using fp_dtype2 = __nv_bfloat162;

    static __device__ __forceinline__ __nv_bfloat16 to_dtype(float x) { return __float2bfloat16(x); }
    static __device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
    static __device__ __forceinline__ __nv_bfloat16 abs(__nv_bfloat16 x) { return __habs(x); }

    static __device__ __forceinline__ void atomicAddWrapper(__nv_bfloat16 *address, __nv_bfloat16 value) {
        const __nv_bfloat16 next_col_sum = __shfl_down_sync(0xFFFFFFFF, value, 1);
        if (threadIdx.x % 2 == 0) {
            atomicAdd(reinterpret_cast<__nv_bfloat162 *>(address), __nv_bfloat162(value, next_col_sum));
        }
    }
    static __device__ __forceinline__ __nv_bfloat162 to_dtype2(__nv_bfloat16 x, __nv_bfloat16 y) { return {x, y}; }
    static __device__ __forceinline__ __nv_bfloat162 hfma2(const __nv_bfloat162 &a, const __nv_bfloat162 &b, const __nv_bfloat162 &c) {
        return __hfma2(a, b, c);
    }
#endif
};

// Macros for easy access to TypeTraits members
#define FP_DTYPE(DT) typename TypeTraits<DT>::fp_dtype
#define FP_DTYPE2(DT) typename TypeTraits<DT>::fp_dtype2
#define ATEN_DTYPE(DT) typename TypeTraits<DT>::aten
#define IS_HALFSIZED(DT) TypeTraits<DT>::is_halfsized

#ifdef __CUDACC__
#define TO_DTYPE(DT, x) TypeTraits<DT>::to_dtype(x)
#define TO_FLOAT(DT, x) TypeTraits<DT>::to_float(x)
#define ABS(DT, x) TypeTraits<DT>::abs(x)
#define ATOMIC_ADD(DT, address, value) TypeTraits<DT>::atomicAddWrapper(address, value)
#define TO_DTYPE2(DT, x, y) TypeTraits<DT>::to_dtype2(x, y)
#define HFMA2(DT, a, b, c) TypeTraits<DT>::hfma2(a, b, c)
#endif

#endif // TYPE_TRAITS_CUH
