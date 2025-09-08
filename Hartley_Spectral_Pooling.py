import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair


def _spectral_crop(input, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            bottom_right = input[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            top_left = input[:, :, :cutoff_freq_h, :cutoff_freq_w]
            top_right = input[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            bottom_left = input[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            bottom_right = input[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    top_combined = torch.cat((top_left, top_right), dim=-1)
    bottom_combined = torch.cat((bottom_left, bottom_right), dim=-1)
    all_together = torch.cat((top_combined, bottom_combined), dim=-2)

    return all_together


def _spectral_pad(input, output, oheight, owidth):
    cutoff_freq_h = math.ceil(oheight / 2)
    cutoff_freq_w = math.ceil(owidth / 2)

    pad = torch.zeros_like(input)

    if oheight % 2 == 1:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -(cutoff_freq_w - 1):] = output[:, :, -(cutoff_freq_h - 1):,
                                                                      -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w] = output[:, :, -(cutoff_freq_h - 1):, :cutoff_freq_w]
            pad[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:] = output[:, :, -(cutoff_freq_h - 1):, -cutoff_freq_w:]
    else:
        if owidth % 2 == 1:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):] = output[:, :, :cutoff_freq_h, -(cutoff_freq_w - 1):]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):] = output[:, :, -cutoff_freq_h:, -(cutoff_freq_w - 1):]
        else:
            pad[:, :, :cutoff_freq_h, :cutoff_freq_w] = output[:, :, :cutoff_freq_h, :cutoff_freq_w]
            pad[:, :, :cutoff_freq_h, -cutoff_freq_w:] = output[:, :, :cutoff_freq_h, -cutoff_freq_w:]
            pad[:, :, -cutoff_freq_h:, :cutoff_freq_w] = output[:, :, -cutoff_freq_h:, :cutoff_freq_w]
            pad[:, :, -cutoff_freq_h:, -cutoff_freq_w:] = output[:, :, -cutoff_freq_h:, -cutoff_freq_w:]

    return pad


def DiscreteHartleyTransform(input):
    #fft = torch.rfft(input, 2, normalized=True, onesided=False)

    output = torch.fft.fft2(input, dim=(-2, -1))
    output1 = torch.stack((output.real, output.imag), -1)
    output1=output1/torch.max(output1)
    #dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    dht = output1[:, :, :, :, -2] - output1[:, :, :, :, -1]
    return dht
# def DiscreteHartleyTransform(input):
#     output_complex = torch.fft.fft2(input, dim=(-2, -1))
#     output_real = output_complex.real
#     output_imag = output_complex.imag
#
#     # 归一化实部和虚部
#     max_real = torch.max(output_real)
#     max_imag = torch.max(output_imag)
#     output_real_normalized = output_real / max_real
#     output_imag_normalized = output_imag / max_imag
#
#     # 重新组合成复数形式
#     output_normalized_complex = output_real_normalized + 1j * output_imag_normalized
#
#     # 计算 Hartley 变换
#     dht = torch.stack((output_normalized_complex[:,:, :, :, -2] - output_normalized_complex[:,:, :, :, -1]), dim=-1)
#     return dht


class SpectralPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, oheight, owidth):
        ctx.oh = oheight
        ctx.ow = owidth
        ctx.save_for_backward(input)

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(input)

        # frequency cropping
        all_together = _spectral_crop(dht, oheight, owidth)
        # inverse Hartley transform
        dht = DiscreteHartleyTransform(all_together)
        return dht

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables

        # Hartley transform by RFFT
        dht = DiscreteHartleyTransform(grad_output)
        # frequency padding
        grad_input = _spectral_pad(input, dht, ctx.oh, ctx.ow)
        # inverse Hartley transform
        grad_input = DiscreteHartleyTransform(grad_input)
        return grad_input, None, None


# class SpectralPool2d(nn.Module):
#     def __init__(self, scale_factor):
#         super(SpectralPool2d, self).__init__()
#         self.scale_factor = _pair(scale_factor)
#
#     def forward(self, input):
#         H, W = input.size(-2), input.size(-1)
#         h, w = math.ceil(H * self.scale_factor[0]), math.ceil(W * self.scale_factor[1])
#         return SpectralPoolingFunction.apply(input, h, w)


class SpectralPool2d(nn.Module):
    def __init__(self, scale_factor):
        super(SpectralPool2d, self).__init__()
        self.scale_factor = _pair(scale_factor)

    def forward(self, input):
        H, W = input.size(-2), input.size(-1)
        h, w = math.ceil(H * self.scale_factor[0]), math.ceil(W * self.scale_factor[1])

        # 使用 SpectralPoolingFunction 进行池化
        pooled_input = SpectralPoolingFunction.apply(input, h, w)

        return pooled_input