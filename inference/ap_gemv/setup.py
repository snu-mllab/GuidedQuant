import os
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="ap_gemv",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="ap_gemv", 
            sources=["bindings.cpp", "gemv.cu", "anyprec.cu", "lutgemm.cu"],
            extra_compile_args={
                'cxx': ["-O3", "-DENABLE_BF16"],
                'nvcc': [
                    '-lineinfo', 
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_HALF2_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "-gencode=arch=compute_90,code=sm_90",   # H100
                    "-gencode=arch=compute_89,code=sm_89",   # Ada
                    "-gencode=arch=compute_86,code=sm_86",   # GA10x
                    "-gencode=arch=compute_80,code=sm_80",   # A100
                ]
            },
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
)
