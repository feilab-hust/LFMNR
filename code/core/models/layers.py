import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope) * scale

    else:
        return F.leaky_relu(input, negative_slope=negative_slope) * scale

class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate.

    During the forward pass the weights are scaled by the inverse of the He constant (i.e. sqrt(in_dim)) to
    prevent vanishing gradients and accelerate training. This constant only works for ReLU or LeakyReLU
    activation functions.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    bias: bool
        Use bias term.
    bias_init: float
        Initial value for the bias.
    lr_mul: float
        Learning rate multiplier. By scaling weights and the bias we can proportionally scale the magnitude of
        the gradients, effectively increasing/decreasing the learning rate for this layer.
    activate: bool
        Apply leakyReLU activation.

    """
    def __init__(self, in_channel, out_channel, bias=True, bias_init=0, lr_mul=1, activate=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

        self.activate = activate
        self.scale = (1 / math.sqrt(in_channel)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activate:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"

class ModulationLinear(nn.Module):
    """Linear modulation layer.

    This layer is inspired by the modulated convolution layer from StyleGAN2, but adapted to linear layers.

    Args:
    ----
    in_channel: int
        Input channels.
    out_channel: int
        Output channels.
    z_dim: int
        Latent dimension.
    demodulate: bool
        Demudulate layer weights.
    activate: bool
        Apply LeakyReLU activation to layer output.
    bias: bool
        Add bias to layer output.

    """
    def __init__(
        self,
        in_channel,
        out_channel,
        z_dim,
        demodulate=True,
        activate=True,
        bias=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.z_dim = z_dim
        self.demodulate = demodulate

        self.scale = 1 / math.sqrt(in_channel)
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel))
        self.modulation = EqualLinear(z_dim, in_channel, bias_init=1, activate=False)

        if activate:
            # FusedLeakyReLU includes a bias term
            self.activate = FusedLeakyReLU(out_channel, bias=bias)
        elif bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channel))

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, z_dim={self.z_dim})'

    def forward(self, input, z):
        # feature modulation
        gamma = self.modulation(z)  # B, in_ch
        input = input * gamma

        weight = self.weight * self.scale

        if self.demodulate:
            # weight is out_ch x in_ch
            # here we calculate the standard deviation per input channel
            demod = torch.rsqrt(weight.pow(2).sum([1]) + self.eps)
            weight = weight * demod.view(-1, 1)

            # also normalize inputs
            input_demod = torch.rsqrt(input.pow(2).sum([1]) + self.eps)
            input = input * input_demod.view(-1, 1)

        out = F.linear(input, weight)

        if hasattr(self, 'activate'):
            out = self.activate(out)

        if hasattr(self, 'bias'):
            out = out + self.bias

        return out
