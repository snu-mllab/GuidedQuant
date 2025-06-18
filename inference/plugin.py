import torch
import ap_gemv

"""
Any-Precision
"""
@torch.library.custom_op("plugin::anyprec_gemv", mutates_args={"output"})
def anyprec_gemv(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, output:torch.Tensor, bitwidth:int) -> None:
    ap_gemv.anyprec_gemv(x, output, q_weight, lut, bitwidth)

@anyprec_gemv.register_fake
def _(x, q_weight, lut, output, bitwidth):
    return None

#@torch.library.custom_op("plugin::anyprec_dequant", mutates_args=())
def anyprec_dequant(q_weight: torch.Tensor, lut: torch.Tensor, bitwidth:int) -> torch.Tensor:
    weight = ap_gemv.anyprec_dequant(q_weight, lut, bitwidth)
    return weight

@torch.library.custom_op("plugin::lutgemm_gemv", mutates_args={"output"})
def lutgemm_gemv(x: torch.Tensor, output: torch.Tensor, q_weight: torch.Tensor, alpha: torch.Tensor, q_bias: torch.Tensor, bitwidth: int, group_size: int) -> None:
    ap_gemv.lutgemm_gemv(x, output, q_weight, alpha, q_bias, bitwidth, group_size)

@lutgemm_gemv.register_fake
def _(x, output, q_weight, alpha, q_bias, bitwidth, group_size):
    return None
