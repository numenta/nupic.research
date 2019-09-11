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
Layers with stochastic binary gates for each synapse.

The gate parameters are learned using the gradient of the log probability with
respect to those parameters, times the difference in loss due to the gate's
current value compared to the expected loss across all possible values for this
gate. The expected loss is computed using weighted average of the current loss
and the loss that would have occurred if the gate in question had adopted the
other value. This latter loss is estimated using dL/dz where z is the value of
the gate. All of this simplifies to the ratio of P(z = 0) and P(z = 1).
"""

import math

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter


class BinaryGatedLinear(Module):
    """
    Linear layer with stochastic binary gates
    """
    def __init__(self, in_features, out_features, l0_strength=1.,
                 l2_strength=1., learn_weight=True, bias=True, droprate_init=0.5,
                 **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param l2_strength: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the gates will be initialized to
        :param l0_strength: Strength of the L0 penalty
        """
        super(BinaryGatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.l0_strength = l0_strength
        self.l2_strength = l2_strength
        self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available()
                            else torch.cuda.FloatTensor)
        self.weight = self.floatTensor(out_features, in_features)
        if learn_weight:
            self.weight = Parameter(self.weight)
        self.logit_p1 = Parameter(torch.Tensor(out_features, in_features))
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.use_bias = False
        if bias:
            self.bias = self.floatTensor(out_features)
            if learn_weight:
                self.bias = Parameter(self.bias)
            self.use_bias = True
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode="fan_out")
        self.logit_p1.data.normal_(math.log(1 - self.droprate_init)
                                   - math.log(self.droprate_init), 1e-2)
        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        pass

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """
        p1 = torch.sigmoid(self.logit_p1)
        weight_decay_ungated = .5 * self.l2_strength * self.weight.pow(2)
        weight_l2_l0 = torch.sum((weight_decay_ungated + self.l0_strength) * p1)
        bias_l2 = (0 if not self.use_bias
                   else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
        return -weight_l2_l0 - bias_l2

    def sample_weight(self):
        u = self.floatTensor(self.logit_p1.size()).uniform_(0, 1)

        p1 = torch.sigmoid(self.logit_p1)
        mask = p1 > u

        def cc_to_p1(grad):
            ratio = p1 / (1 - p1)
            p1.backward(grad * torch.where(mask, 1 / ratio, ratio))
            return grad

        z = mask.float()
        z.requires_grad_()
        z.register_hook(cc_to_p1)

        return self.weight * z

    def forward(self, x):
        return F.linear(x, self.sample_weight(),
                        (self.bias if self.use_bias else None))

    def get_expected_nonzeros(self):
        expected_gates = torch.sigmoid(self.logit_p1)
        return expected_gates.sum(
            dim=tuple(range(1, len(expected_gates.shape)))).detach()

    def get_inference_nonzeros(self):
        u = self.floatTensor(self.logit_p1.size()).uniform_(0, 1)
        inference_gates = torch.sigmoid(self.logit_p1) > u
        return inference_gates.sum(
            dim=tuple(range(1, len(inference_gates.shape)))).detach()


class BinaryGatedConv2d(Module):
    """
    Convolutional layer with binary stochastic gates
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, learn_weight=True, bias=True,
                 droprate_init=0.5, l2_strength=1., l0_strength=1., **kwargs):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the gates will be initialized to
        :param l2_strength: Strength of the L2 penalty
        :param l0_strength: Strength of the L0 penalty
        """
        super(BinaryGatedConv2d, self).__init__()
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
        self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available()
                            else torch.cuda.FloatTensor)
        self.use_bias = False
        self.weight = self.floatTensor(out_channels, in_channels // groups,
                                       *self.kernel_size)
        if learn_weight:
            self.weight = Parameter(self.weight)
        self.logit_p1 = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                               *self.kernel_size))
        self.dim_z = out_channels
        self.input_shape = None

        if bias:
            self.bias = self.floatTensor(out_channels)
            if learn_weight:
                self.bias = Parameter(self.bias)
            self.use_bias = True

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode="fan_in")
        self.logit_p1.data.normal_(math.log(1 - self.droprate_init)
                                   - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        pass

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """
        p1 = torch.sigmoid(self.logit_p1)
        weight_decay_ungated = .5 * self.l2_strength * self.weight.pow(2)
        weight_l2_l0 = torch.sum((weight_decay_ungated + self.l0_strength) * p1)
        bias_l2 = (0 if not self.use_bias
                   else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
        return -weight_l2_l0 - bias_l2

    def sample_weight(self):
        u = self.floatTensor(self.logit_p1.size()).uniform_(0, 1)

        p1 = torch.sigmoid(self.logit_p1)
        mask = p1 > u

        def cc_to_p1(grad):
            ratio = p1 / (1 - p1)
            p1.backward(grad * torch.where(mask, 1 / ratio, ratio))
            return grad

        z = mask.float()
        z.requires_grad_()
        z.register_hook(cc_to_p1)

        return self.weight * z

    def forward(self, x):
        return F.conv2d(x, self.sample_weight(),
                        (self.bias if self.use_bias else None),
                        self.stride, self.padding, self.dilation,
                        self.groups)

    def get_expected_nonzeros(self):
        expected_gates = torch.sigmoid(self.logit_p1)
        return expected_gates.sum(
            dim=tuple(range(1, len(expected_gates.shape)))).detach()

    def get_inference_nonzeros(self):
        u = self.floatTensor(self.logit_p1.size()).uniform_(0, 1)
        inference_gates = torch.sigmoid(self.logit_p1) > u
        return inference_gates.sum(
            dim=tuple(range(1, len(inference_gates.shape)))).detach()
