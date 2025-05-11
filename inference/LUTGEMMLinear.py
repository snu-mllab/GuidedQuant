import torch
import torch.nn as nn
from plugin import *

class LUTGEMMLinear(nn.Module):
    def __init__(self, in_features, out_features, bitwidth, group_size, bias=False, dtype=torch.half):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bitwidth = bitwidth
        if group_size == -1:
            self.group_size = in_features
        else:
            self.group_size = group_size
        self.dtype = dtype

        self.register_buffer(
            'qweight',
            torch.empty((in_features//32, bitwidth, out_features), dtype=torch.int32, device='cuda')
        )

        self.register_buffer(
            'alpha',
            torch.empty((in_features // self.group_size, bitwidth, out_features), dtype=self.dtype, device='cuda')
        )
        
        self.register_buffer(
            'q_bias',
            torch.empty((in_features // self.group_size, out_features), dtype=self.dtype, device='cuda')
        )
        

        ### random init
        """
        self.qweight = torch.randint(
                low=-2147483648, high=2147483647,
                size=(in_features//32, bitwidth, out_features), 
                dtype=torch.int32,
                device='cuda'
        )
        
        self.alpha = torch.rand(
                in_features // group_size, bitwidth, out_features,
                dtype=torch.half,
                device='cuda'
        )
        
        self.q_bias = torch.rand(
                in_features // group_size, out_features,
                dtype=torch.half,
                device='cuda'
        )
        """


       
        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype, device='cuda')
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, out_features), dtype=self.dtype, device='cuda')

    def forward(self, x, **kwargs):

        assert(x.shape[0] == 1)
        assert(x.shape[1] == 1)

        # clear the output
        self.output.zero_()

        lutgemm_gemv(x, self.output, self.qweight, self.alpha, self.q_bias, self.bitwidth, self.group_size)
        
        if self.bias is not None:
            self.output += self.bias

        return self.output
