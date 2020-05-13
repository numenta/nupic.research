# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Variational dropout layers

Based on:
[1] Kingma, Diederik P., Tim Salimans, and Max Welling.
    "Variational dropout and the local reparameterization trick." NIPS (2015).
[2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov.
    "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter

from nupic.research.frameworks.pytorch.modules import MaskedConv2d


class VDropLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_normal_(self.weight, mode="fan_out")

        self.w_logvar = Parameter(torch.Tensor(out_features, in_features))
        self.w_logvar.data.fill_(-10)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias.data.fill_(0)
        else:
            self.bias = None

        self.threshold = 3
        self.epsilon = 1e-8
        self.tensor_constructor = (torch.FloatTensor
                                   if not torch.cuda.is_available()
                                   else torch.cuda.FloatTensor)

    def constrain_parameters(self):
        self.w_logvar.data.clamp_(min=-10., max=10.)

    def compute_mask(self):
        w_logalpha = self.w_logvar - (self.weight ** 2 + self.epsilon).log()
        return (w_logalpha < self.threshold).float()

    def forward(self, x):
        if self.training:
            return vdrop_linear_forward(x,
                                        lambda: self.weight,
                                        lambda: self.w_logvar.exp(),
                                        self.bias, self.tensor_constructor,
                                        self.epsilon)
        else:
            return F.linear(x, self.weight * self.compute_mask(), self.bias)

    def regularization(self):
        w_logalpha = self.w_logvar - (self.weight ** 2 + self.epsilon).log()
        return vdrop_regularization(w_logalpha).sum()

    def get_inference_nonzeros(self):
        return self.compute_mask().int().sum(dim=1)

    def count_inference_flops(self):
        # For each unit, multiply with its n inputs then do n - 1 additions.
        # To capture the -1, subtract it, but only in cases where there is at
        # least one weight.
        nz_by_unit = self.get_inference_nonzeros()
        multiplies = torch.sum(nz_by_unit)
        adds = multiplies - torch.sum(nz_by_unit > 0)
        return multiplies.item(), adds.item()

    def weight_size(self):
        return self.weight.size()


class VDropConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.groups = groups

        self.weight = Parameter(torch.Tensor(out_channels,
                                             in_channels // groups,
                                             *self.kernel_size))
        init.kaiming_normal_(self.weight, mode="fan_out")

        self.w_logvar = Parameter(torch.Tensor(out_channels,
                                               in_channels // groups,
                                               *self.kernel_size))
        self.w_logvar.data.fill_(-10)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(0)
        else:
            self.bias = None

        self.input_shape = None

        self.threshold = 3
        self.epsilon = 1e-8
        self.tensor = (torch.FloatTensor if not torch.cuda.is_available()
                       else torch.cuda.FloatTensor)

    def extra_repr(self):
        s = (f"{self.in_channels}, {self.out_channels}, "
             f"kernel_size={self.kernel_size}, stride={self.stride}")
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return s

    def constrain_parameters(self):
        self.w_logvar.data.clamp_(min=-10., max=10.)

    def compute_mask(self):
        w_logalpha = self.w_logvar - (self.weight ** 2 + self.epsilon).log()
        return (w_logalpha < self.threshold).float()

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.size()

        if self.training:
            return vdrop_conv_forward(x,
                                      lambda: self.weight,
                                      lambda: self.w_logvar.exp(),
                                      self.bias, self.stride, self.padding,
                                      self.dilation, self.groups,
                                      self.tensor_constructor, self.epsilon)
        else:
            return F.conv2d(x, self.weight * self.compute_mask(), self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)

    def regularization(self):
        w_logalpha = self.w_logvar - (self.weight ** 2 + self.epsilon).log()
        return vdrop_regularization(w_logalpha).sum()

    def get_inference_nonzeros(self):
        mask = self.compute_mask().int()
        return mask.sum(dim=tuple(range(1, len(mask.shape))))

    def count_inference_flops(self):
        # For each unit, multiply with its n inputs then do n - 1 additions.
        # To capture the -1, subtract it, but only in cases where there is at
        # least one weight.
        nz_by_unit = self.get_inference_nonzeros()
        multiplies_per_instance = torch.sum(nz_by_unit)
        adds_per_instance = multiplies_per_instance - torch.sum(nz_by_unit > 0)

        # for rows
        instances = (
            (self.input_shape[-2] - self.kernel_size[0]
             + 2 * self.padding[0]) / self.stride[0]) + 1
        # multiplying with cols
        instances *= (
            (self.input_shape[-1] - self.kernel_size[1] + 2 * self.padding[1])
            / self.stride[1]) + 1

        multiplies = multiplies_per_instance * instances
        adds = adds_per_instance * instances

        return multiplies.item(), adds.item()

    def weight_size(self):
        return self.weight.size()


class FixedVDropConv2d(MaskedConv2d):
    """
    This is designed to be used with snapshots generated by other classes, e.g.
    VDropConv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, alpha,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 mask_mode="channel_to_channel"):
        """
        @param alpha (float)
        Defined as w_var / w_mu**2. Weights are multiplied with noise sampled
        from distribution N(1,alpha).
        """
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, mask_mode=mask_mode)
        self.alpha = alpha
        self.tensor_constructor = (torch.FloatTensor
                                   if not torch.cuda.is_available()
                                   else torch.cuda.FloatTensor)
        self.epsilon = 1e-8

    def extra_repr(self):
        return f"alpha={self.alpha}"

    def forward(self, x):
        if self.training:
            return vdrop_conv_forward(
                x,
                lambda: self.weight * self.weight_mask,
                lambda: self.alpha * (self.weight.pow(2) * self.weight_mask),
                self.bias, self.stride, self.padding, self.dilation, self.groups,
                self.tensor_constructor, self.epsilon)
        else:
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding,
                self.dilation, self.groups
            )


def vdrop_linear_forward(x, get_w_mu, get_w_var, bias, tensor_constructor,
                         epsilon=1e-8):
    """
    Rather than sampling weights from gaussian distribution N(w_mu, w_var), use
    the "local reparameterization trick", using w_mu and w_var to compute y_mu
    and y_var, the distribution of the downstream unit's activation, and sample
    that distribution. (As described in [1], this enables us to simulate
    sampling different weights for every item in the batch, while still getting
    the hardware performance benefits of batching.)

    This computes y_mu + y_sigma*noise, carefully reusing buffers to avoid
    unnecessary memory usage. It takes in functions get_w_mu and get_w_var
    rather than taking in actual tensors so that those tensors won't need to use
    memory any longer than necessary.

    @param get_w_mean (function)
    Returns each weight's mean.

    @param get_w_var (function)
    Returns each weight's variance.
    """
    # Compute y_var
    y = F.linear(x.pow(2), get_w_var())

    # If any values are 0, we'll divide by zero on the backward pass.
    # Note that clamping y rather than y.data would use much more memory.
    y.data.clamp_(epsilon)

    # Compute y_stdev
    y = y.sqrt_()

    # Convert to the additive noise.
    # (Can't do in-place update after sqrt_.)
    y = y * tensor_constructor(y.size()).normal_()

    # Add y_mu
    y += F.linear(x, get_w_mu(), bias)
    return y


def vdrop_conv_forward(x, get_w_mu, get_w_var, bias, stride, padding, dilation,
                       groups, tensor_constructor, epsilon=1e-8):
    """
    Rather than sampling weights from gaussian distribution N(w_mu, w_var), use
    the "local reparameterization trick", using w_mu and w_var to compute y_mu
    and y_var, the distribution of the downstream unit's activation, and sample
    that distribution. (As described in [1], this enables us to simulate
    sampling different weights for every item in the batch, while still getting
    the hardware performance benefits of batching.)

    This computes y_mu + y_sigma*noise, carefully reusing buffers to avoid
    unnecessary memory usage. It takes in functions get_w_mu and get_w_var
    rather than taking in actual tensors so that those tensors won't need to use
    memory any longer than necessary.

    @param get_w_mean (function)
    Returns each weight's mean.

    @param get_w_var (function)
    Returns each weight's variance.
    """
    # Compute y_var.
    y = F.conv2d(
        x.pow(2), get_w_var(), None, stride, padding, dilation, groups
    )

    # This handles two possible issues:
    # - It's possible some values are negative, which will lead to NaN
    #   on the forward pass.
    #   https://github.com/pytorch/pytorch/issues/30934
    # - If any values are 0, we'll get NaN on the backward pass.
    # Note that clamping y rather than y.data would use much more memory.
    y.data.clamp_(epsilon)

    # Compute y_stdev
    y = y.sqrt_()

    # Convert to the additive noise.
    # (Can't do in-place update after sqrt_.)
    y = y * tensor_constructor(y.size()).normal_()

    # Add y_mu.
    y += F.conv2d(x, get_w_mu(), bias, stride, padding,
                  dilation, groups)
    return y


def vdrop_regularization(logalpha):
    """
    alpha is defined as w_var / w_mu**2

    @param logalpha (Tensor)
    """
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    return -(k1 * torch.sigmoid(k2 + k3 * logalpha)
             - 0.5 * F.softplus(-logalpha) - k1)


__all__ = [
    "VDropLinear",
    "VDropConv2d",
    "FixedVDropConv2d",
]
