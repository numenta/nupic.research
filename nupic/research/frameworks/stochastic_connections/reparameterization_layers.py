# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
Adapted from https://github.com/AMLab-Amsterdam/L0_regularization, modified
to learn gates for every weight rather than using shared gates.
"""

import math

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter

LIMIT_A = -.1
LIMIT_B = 1.1
EPSILON = 1e-6


class HardConcreteGatedLinear(Module):
    """
    Linear layer with stochastic connections, as in
    https://arxiv.org/abs/1712.01312
    """
    def __init__(self, in_features, out_features, l0_strength=1.,
                 l2_strength=1., bias=True, droprate_init=0.5,
                 temperature=(2 / 3), **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param l2_strength: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized
                              to
        :param temperature: Temperature of the concrete distribution
        :param l0_strength: Strength of the L0 penalty
        """
        super(HardConcreteGatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.l0_strength = l0_strength
        self.l2_strength = l2_strength
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.loga = Parameter(torch.Tensor(out_features, in_features))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.use_bias = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available()
                            else torch.cuda.FloatTensor)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode="fan_out")
        self.loga.data.normal_(math.log(1 - self.droprate_init)
                               - math.log(self.droprate_init),
                               1e-2)
        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - LIMIT_A) / (LIMIT_B - LIMIT_A)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.loga).clamp(
            min=EPSILON, max=1 - EPSILON)

    def quantile_concrete(self, x):
        """
        Implements the quantile, aka inverse CDF, of the 'stretched' concrete
        distribution
        """
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.loga) / self.temperature)
        return y * (LIMIT_B - LIMIT_A) + LIMIT_A

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """

        weight_decay_ungated = .5 * self.l2_strength * self.weight.pow(2)
        weight_l2_l0 = torch.sum(
            (weight_decay_ungated + self.l0_strength) * (1 - self.cdf_qz(0)))
        bias_l2 = (0 if not self.use_bias
                   else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
        return -weight_l2_l0 - bias_l2

    def count_expected_flops_and_l0(self):
        """
        Measures the expected floating point operations (FLOPs) and the expected
        L0 norm
        """
        # dim_in multiplications and dim_in - 1 additions for each output unit
        # for the weights # + the bias addition for each unit
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(EPSILON, 1 - EPSILON)
        eps = Variable(eps)
        return eps

    def sample_weight(self):
        if self.training:
            z = self.quantile_concrete(
                self.get_eps(self.floatTensor(self.loga.size())))
            mask = F.hardtanh(z, min_val=0, max_val=1)
        else:
            pi = torch.sigmoid(self.loga)
            mask = F.hardtanh(pi * (LIMIT_B - LIMIT_A) + LIMIT_A, min_val=0,
                              max_val=1)

        return mask * self.weight

    def forward(self, x):
        return F.linear(x, self.sample_weight(),
                        (self.bias if self.use_bias else None))

    def get_expected_nonzeros(self):
        expected_gates = 1 - self.cdf_qz(0)
        return expected_gates.sum(
            dim=tuple(range(1, len(expected_gates.shape)))).detach()

    def get_inference_nonzeros(self):
        inference_gates = F.hardtanh(
            torch.sigmoid(self.loga) * (LIMIT_B - LIMIT_A) + LIMIT_A,
            min_val=0, max_val=1)
        return (inference_gates > 0).sum(
            dim=tuple(range(1, len(inference_gates.shape)))).detach()

    def __repr__(self):
        s = ("{name}({in_features} -> {out_features}, "
             "droprate_init={droprate_init}, l0_strength={l0_strength}, "
             "temperature={temperature}, l2_strength={l2_strength}, ")
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class HardConcreteGatedConv2d(Module):
    """
    Convolutional layer with stochastic connections, as in
    https://arxiv.org/abs/1712.01312
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, droprate_init=0.5,
                 temperature=(2 / 3), l2_strength=1., l0_strength=1., **kwargs):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the L0 gates will be initialized
                              to
        :param temperature: Temperature of the concrete distribution
        :param l2_strength: Strength of the L2 penalty
        :param l0_strength: Strength of the L0 penalty
        """
        super(HardConcreteGatedConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.l2_strength = l2_strength
        self.l0_strength = l0_strength
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.temperature = temperature
        self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available()
                            else torch.cuda.FloatTensor)
        self.use_bias = False
        self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                             *self.kernel_size))
        self.loga = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                           *self.kernel_size))
        self.dim_z = out_channels
        self.input_shape = None

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.use_bias = True

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode="fan_in")

        self.loga.data.normal_(
            math.log(1 - self.droprate_init) - math.log(self.droprate_init),
            1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - LIMIT_A) / (LIMIT_B - LIMIT_A)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.loga).clamp(
            min=EPSILON, max=1 - EPSILON)

    def quantile_concrete(self, x):
        """
        Implements the quantile, aka inverse CDF, of the 'stretched' concrete
        distribution
        """
        y = torch.sigmoid(
            (torch.log(x) - torch.log(1 - x) + self.loga) / self.temperature)
        return y * (LIMIT_B - LIMIT_A) + LIMIT_A

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """
        weight_decay_ungated = .5 * self.l2_strength * self.weight.pow(2)
        weight_l2_l0 = torch.sum(
            (weight_decay_ungated + self.l0_strength) * (1 - self.cdf_qz(0)))
        bias_l2 = (0 if not self.use_bias
                   else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
        return -weight_l2_l0 - bias_l2

    def count_expected_flops_and_l0(self):
        """
        Measures the expected floating point operations (FLOPs) and the expected
        L0 norm
        """
        ppos = torch.sum(1 - self.cdf_qz(0))
        # vector_length
        n = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        # (n: multiplications and n-1: additions)
        flops_per_instance = n + (n - 1)

        # for rows
        num_instances_per_filter = (
            (self.input_shape[1] - self.kernel_size[0]
             + 2 * self.padding[0]) / self.stride[0]) + 1
        # multiplying with cols
        num_instances_per_filter *= (
            (self.input_shape[2] - self.kernel_size[1] + 2 * self.padding[1])
            / self.stride[1]) + 1

        flops_per_filter = num_instances_per_filter * flops_per_instance
        # multiply with number of filters
        expected_flops = flops_per_filter * ppos
        expected_l0 = n * ppos

        if self.use_bias:
            # since the gate is applied to the output we also reduce the bias
            # computation
            expected_flops += num_instances_per_filter * ppos
            expected_l0 += ppos

        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(EPSILON, 1 - EPSILON)
        eps = Variable(eps)
        return eps

    def sample_weight(self):
        if self.training:
            z = self.quantile_concrete(
                self.get_eps(self.floatTensor(self.loga.size())))
            mask = F.hardtanh(z, min_val=0, max_val=1)
        else:
            pi = torch.sigmoid(self.loga)
            mask = F.hardtanh(pi * (LIMIT_B - LIMIT_A) + LIMIT_A, min_val=0,
                              max_val=1)

        return mask * self.weight

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.size()
        return F.conv2d(x, self.sample_weight(),
                        (self.bias if self.use_bias else None),
                        self.stride, self.padding, self.dilation,
                        self.groups)

    def get_expected_nonzeros(self):
        expected_gates = 1 - self.cdf_qz(0)
        return expected_gates.sum(
            dim=tuple(range(1, len(expected_gates.shape)))).detach()

    def get_inference_nonzeros(self):
        inference_gates = F.hardtanh(
            torch.sigmoid(self.loga) * (LIMIT_B - LIMIT_A) + LIMIT_A,
            min_val=0, max_val=1)
        return (inference_gates > 0).sum(
            dim=tuple(range(1, len(inference_gates.shape)))).detach()

    def __repr__(self):
        s = ("{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, "
             "stride={stride}, droprate_init={droprate_init}, "
             "temperature={temperature}, l2_strength={l2_strength}, "
             "l0_strength={l0_strength}")
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)
