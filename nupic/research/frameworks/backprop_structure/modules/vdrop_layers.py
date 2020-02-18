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
"""

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter


class VDropLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.w_mu = Parameter(torch.Tensor(out_features, in_features))
        init.kaiming_normal_(self.w_mu, mode="fan_out")

        self.w_logsigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.w_logsigma2.data.fill_(-10)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias.data.fill_(0)
        else:
            self.bias = None

        self.threshold = 3
        self.epsilon = 1e-8
        self.tensor = (torch.FloatTensor if not torch.cuda.is_available()
                       else torch.cuda.FloatTensor)

    def compute_mask(self):
        w_logalpha = self.w_logsigma2 - (self.w_mu ** 2 + self.epsilon).log()
        return (w_logalpha < self.threshold).float()

    def forward(self, x):
        if self.training:
            y_mu = F.linear(x, self.w_mu, self.bias)

            w_mu2 = self.w_mu ** 2
            w_alpha = self.w_logsigma2.exp() / (w_mu2 + self.epsilon)

            # Avoid sqrt(0), otherwise a divide-by-zero occurs during backprop.
            y_sigma = F.linear(x ** 2,
                               w_alpha * w_mu2).clamp(self.epsilon).sqrt()

            rv = self.tensor(y_mu.size()).normal_()
            return y_mu + (rv * y_sigma)
        else:
            return F.linear(x, self.w_mu * self.compute_mask(), self.bias)

    def regularization(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        w_logalpha = self.w_logsigma2 - (self.w_mu ** 2 + self.epsilon).log()

        return -(k1 * torch.sigmoid(k2 + k3 * w_logalpha)
                 - 0.5 * F.softplus(-w_logalpha) - k1).sum()

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
        return self.w_mu.size()


class VDropConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups

        self.w_mu = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                           *self.kernel_size))
        init.kaiming_normal_(self.w_mu, mode="fan_out")

        self.w_logsigma2 = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                                  *self.kernel_size))
        self.w_logsigma2.data.fill_(-10)

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

    def compute_mask(self):
        w_logalpha = self.w_logsigma2 - (self.w_mu ** 2 + self.epsilon).log()
        return (w_logalpha < self.threshold).float()

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.size()

        if self.training:
            y_mu = F.conv2d(x, self.w_mu, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)

            w_mu2 = self.w_mu ** 2
            w_alpha = self.w_logsigma2.exp() / (w_mu2 + self.epsilon)

            # Avoid sqrt(0), otherwise a divide-by-zero occurs during backprop.
            y_sigma = F.conv2d(x ** 2, w_alpha * w_mu2, None, self.stride,
                               self.padding, self.dilation, self.groups).clamp(
                                   self.epsilon).sqrt()

            rv = self.tensor(y_mu.size()).normal_()
            return y_mu + (rv * y_sigma)
        else:
            return F.conv2d(x, self.w_mu * self.compute_mask(), self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)

    def regularization(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        w_logalpha = self.w_logsigma2 - (self.w_mu ** 2 + self.epsilon).log()

        return -(k1 * torch.sigmoid(k2 + k3 * w_logalpha)
                 - 0.5 * F.softplus(-w_logalpha) - k1).sum()

    def get_inference_nonzeros(self):
        mask = self.compute_mask().int()
        return mask.sum(dim=tuple(range(1, len(mask.shape))))

    def count_inference_flops(self):
        # For each unit, multiply with its n inputs then do n - 1 additions.
        # To capture the -1, subtract it, but only in cases where there is at
        # least one weight.
        nz_by_unit = self.get_inference_nonzeros()
        multiplies = torch.sum(nz_by_unit)
        adds = multiplies - torch.sum(nz_by_unit > 0)
        return multiplies.item(), adds.item()

    def weight_size(self):
        return self.w_mu.size()
