import torch
import torch.nn as nn

try:
    import ap_gemv
except:
    ap_gemv = None

@torch.library.custom_op("plugin::anyprec_gemv", mutates_args={"output"})
def anyprec_gemv(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, output:torch.Tensor, bitwidth:int) -> None:
    ap_gemv.anyprec_gemv(x, output, q_weight, lut, bitwidth)

@anyprec_gemv.register_fake
def _(x, q_weight, lut, output, bitwidth):
    return None

class AnyPrecisionLinear(nn.Module):
    def __init__(self, in_features, out_features, supported_bits, bias=True, precisions=None, device=None,
                 dtype=None):
        super().__init__()
        if ap_gemv is None:
            raise ModuleNotFoundError("ap_gemv is not installed. Please install `ap_gemv` kernel.")
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError('supported_bits must be a list of integers.')
        if dtype is not None and dtype != torch.float16:
            raise RuntimeError('Only float16 is supported for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = max(self.precisions)
        self.supported_bits = supported_bits

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, self.out_features), dtype=torch.float16, device='cuda')

    def prune_precisions(self):
        self.qweight = self.qweight[:max(self.precisions)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                delattr(self, f'lut{bit}')

    def forward(self, x, **kwargs):
        if 'precision' in kwargs:
            w_bits = kwargs['precision']
        else:
            w_bits = self.precision

        if x.numel() // x.shape[-1] > 1:
            weight = ap_gemv.anyprec_dequant(self.qweight, self._buffers[f'lut{w_bits}'].to(torch.float16), w_bits).to(x.dtype)
            x = torch.matmul(x, weight.T)
        else:
            anyprec_gemv(x.to(torch.float16), self.qweight, self._buffers[f'lut{w_bits}'].to(torch.float16), self.output, w_bits)
            x = self.output.to(x.dtype)

        if self.bias is not None:
            x += self.bias

        return x.clamp_(torch.finfo(x.dtype).min * (1.0 - 5e-3), torch.finfo(x.dtype).max * (1.0 - 5e-3))
        # return x

    def set_precision(self, precision):
        if precision not in self.precisions:
            raise RuntimeError(f"{self.precisions}-bit precisions are supported but {precision}-bit was specified.")

        self.precision = precision

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
