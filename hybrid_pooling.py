import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from Hartley_Spectral_Pooling import *


# def hybrid_pool_layer_valid(x, pool_size=(2, 2)):
#     channel = int(x.shape[-1])
#     kernel_size = int(x.shape[1])
#
#     max_pool_output = F.max_pool2d(x, kernel_size=pool_size)
#
#     spectral_pool_output = SpectralPool2d(scale_factor=(0.5, 0.5))(x)
#
#     output = torch.cat([max_pool_output, spectral_pool_output], dim=-1)
#
#     return output
class HybridPoolLayerValid(nn.Module):
    def __init__(self, pool_size=2):
        super(HybridPoolLayerValid, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        # Max pooling
        max_pool_output = F.max_pool2d(x, kernel_size=self.pool_size,stride=self.pool_size)

        # Spectral pooling
        #spectral_pool = SpectralPoolingFunction.apply(x, x.size(-2)//2, x.size(-1)//2)
        spectral_pool_layer = SpectralPool2d(scale_factor=((1/self.pool_size),(1/self.pool_size)))
        spectral_pool_output =spectral_pool_layer(x)

        final_output = torch.add(max_pool_output, spectral_pool_output)
        final_output = torch.div(final_output, 2)

        return final_output

