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

The gate parameters p are learned using the gradient of the expected loss with
respect to those parameters. We use an estimator for this gradient that is based
on using the gradient of the loss wrt the gate's value z to compute a Taylor
approximation of the loss for different values of z, and using this
approximation we compute the expected loss. For Bernoulli random variables the
gradient of this loss simplifies, leading to estimator dE[L]/dp = dL/dz. When
gradient descent sets p to be outside range [0, 1], we allow it to continue
increasing or decreasing, making the gate's value more permanent, and we clamp
it to [0,1] when sampling the gate.
"""

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter


class HeavisideStep(Module):
    def __init__(self, neg1_activation, cancel_gradient, inplace):
        super(HeavisideStep, self).__init__()
        self.neg1_activation = neg1_activation
        self.cancel_gradient = cancel_gradient
        self.inplace = inplace

    def forward(self, x):
        if self.cancel_gradient:
            x = F.hardtanh(x, -1, 1)
        if not self.inplace:
            x = x.clone()
        mask = x.data >= 0
        x.data[mask] = 1
        x.data[~mask] = (-1 if self.neg1_activation else 0)
        return x


def sample_weight(p1, weight, deterministic=False, samples=1):
    if deterministic:
        u = 0.5
    else:
        t = (torch.FloatTensor if not torch.cuda.is_available()
             else torch.cuda.FloatTensor)
        if samples > 1:
            u = t(samples, *p1.size()).uniform_(0, 1)
        else:
            u = t(p1.size()).uniform_(0, 1)

    # Do this in a way that still propagates the gradient to p1.
    z = p1.clone()
    z.data = (z.data >= u).float()

    return weight * z


class BinaryGatedLinear(Module):
    """
    Linear layer with stochastic binary gates
    """
    def __init__(self, in_features, out_features, l0_strength=1.,
                 l2_strength=1., learn_weight=True, bias=True,
                 droprate_init=0.5, random_weight=True, deterministic=False,
                 use_baseline_bias=False, optimize_inference=False,
                 one_sample_per_item=False, decay_mean=False, **kwargs):
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
        self.deterministic = deterministic
        self.use_baseline_bias = use_baseline_bias
        self.optimize_inference = optimize_inference
        self.one_sample_per_item = one_sample_per_item
        self.decay_mean = decay_mean

        self.random_weight = random_weight
        if random_weight:
            exc_weight = torch.Tensor(out_features, in_features)
            inh_weight = torch.Tensor(out_features, in_features)
        else:
            exc_weight = torch.ones(out_features, in_features)
            inh_weight = torch.ones(out_features, in_features)

        if learn_weight:
            self.exc_weight = Parameter(exc_weight)
            self.inh_weight = Parameter(inh_weight)
        else:
            self.register_buffer("exc_weight", exc_weight)
            self.register_buffer("inh_weight", inh_weight)

        self.exc_p1 = Parameter(torch.Tensor(out_features, in_features))
        self.inh_p1 = Parameter(torch.Tensor(out_features, in_features))

        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.random_weight:
            init.kaiming_normal_(self.exc_weight, mode="fan_out")
            init.kaiming_normal_(self.inh_weight, mode="fan_out")
            self.exc_weight.data.abs_()
            self.inh_weight.data.abs_()
        self.exc_p1.data.normal_(1 - self.droprate_init, 1e-2)
        self.inh_p1.data.normal_(1 - self.droprate_init, 1e-2)
        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.exc_weight.data.clamp_(min=0.)
        self.inh_weight.data.clamp_(min=0.)

    def get_gate_probabilities(self):
        exc_p1 = torch.clamp(self.exc_p1.data, min=0., max=1.)
        inh_p1 = torch.clamp(self.inh_p1.data, min=0., max=1.)
        return exc_p1, inh_p1

    def weight_size(self):
        return self.exc_weight.size()

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """
        if self.l0_strength > 0 or self.l2_strength > 0:
            # Clamp these, but do it in a way that still always propagates the
            # gradient.
            exc_p1 = self.exc_p1.clone()
            torch.clamp(exc_p1.data, min=0, max=1, out=exc_p1.data)
            inh_p1 = self.inh_p1.clone()
            torch.clamp(inh_p1.data, min=0, max=1, out=inh_p1.data)

            if self.l2_strength == 0:
                if self.decay_mean:
                    connections_per_unit = (exc_p1 + inh_p1).sum(dim=1)
                    mean = torch.mean(connections_per_unit.data)

                    return self.l0_strength * torch.abs(connections_per_unit
                                                        - (mean / 2)).sum()
                else:
                    return self.l0_strength * (exc_p1 + inh_p1).sum()
            else:
                exc_weight_decay_ungated = (
                    .5 * self.l2_strength * self.exc_weight.pow(2))
                inh_weight_decay_ungated = (
                    .5 * self.l2_strength * self.inh_weight.pow(2))
                exc_weight_l2_l0 = torch.sum(
                    (exc_weight_decay_ungated + self.l0_strength) * exc_p1)
                inh_weight_l2_l0 = torch.sum(
                    (inh_weight_decay_ungated + self.l0_strength) * inh_p1)
                bias_l2 = (0 if not self.use_bias
                           else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
                return exc_weight_l2_l0 + inh_weight_l2_l0 + bias_l2
        else:
            return 0

    def get_inference_mask(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        if self.deterministic:
            exc_mask = (exc_p1 >= 0.5).float()
            inh_mask = (inh_p1 >= 0.5).float()
            return exc_mask, inh_mask
        else:
            exc_count1 = exc_p1.sum(dim=1).round().int()
            inh_count1 = inh_p1.sum(dim=1).round().int()

            # pytorch doesn't offer topk with varying k values.
            exc_mask = torch.zeros_like(exc_p1)
            inh_mask = torch.zeros_like(inh_p1)
            for i in range(exc_count1.size()[0]):
                _, exc_indices = torch.topk(exc_p1[i], exc_count1[i].item())
                _, inh_indices = torch.topk(inh_p1[i], inh_count1[i].item())
                exc_mask[i].scatter_(-1, exc_indices, 1)
                inh_mask[i].scatter_(-1, inh_indices, 1)

            return exc_mask, inh_mask

    def sample_weight_and_bias(self):
        if self.training or not self.optimize_inference:
            w = (sample_weight(self.exc_p1, self.exc_weight, self.deterministic)
                 - sample_weight(self.inh_p1, self.inh_weight, self.deterministic))
        else:
            exc_mask, inh_mask = self.get_inference_mask()
            w = exc_mask * self.exc_weight - inh_mask * self.inh_weight

        b = None
        if self.use_baseline_bias:
            b = -w.sum(dim=-1) / 2

        if self.use_bias:
            b = (b + self.bias
                 if b is not None
                 else self.bias)

        return w, b

    def forward(self, x):
        if self.one_sample_per_item and self.training and len(x.size()) > 1:
            results = []
            for i in range(x.size(0)):
                w, b = self.sample_weight_and_bias()
                results.append(F.linear(x[i:i + 1], w, b))
            return torch.cat(results)
        else:
            w, b = self.sample_weight_and_bias()
            return F.linear(x, w, b)

    def get_expected_nonzeros(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        # Flip two coins with probabilities pi_1 and pi_2. What is the
        # probability one of them is 1?
        #
        # 1 - (1 - pi_1)*(1 - pi_2)
        # = 1 - 1 + pi_1 + pi_2 - pi_1*pi_2
        # = pi_1 + pi_2 - pi_1*pi_2
        p1 = exc_p1 + inh_p1 - (exc_p1 * inh_p1)

        return p1.sum(dim=1).detach()

    def get_inference_nonzeros(self):
        exc_mask, inh_mask = self.get_inference_mask()

        return torch.sum(exc_mask.int() | inh_mask.int(), dim=1)

    def count_inference_flops(self):
        # For each unit, multiply with its n inputs then do n - 1 additions.
        # To capture the -1, subtract it, but only in cases where there is at
        # least one weight.
        nz_by_unit = self.get_inference_nonzeros()
        multiplies = torch.sum(nz_by_unit)
        adds = multiplies - torch.sum(nz_by_unit > 0)
        return multiplies.item(), adds.item()


class BinaryGatedConv2d(Module):
    """
    Convolutional layer with binary stochastic gates
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, learn_weight=True, bias=True,
                 droprate_init=0.5, l2_strength=1., l0_strength=1.,
                 random_weight=True, deterministic=False,
                 use_baseline_bias=False, optimize_inference=True,
                 one_sample_per_item=False, decay_mean=False, **kwargs):
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
        self.deterministic = deterministic
        self.use_baseline_bias = use_baseline_bias
        self.optimize_inference = optimize_inference
        self.one_sample_per_item = one_sample_per_item
        self.decay_mean = decay_mean

        self.random_weight = random_weight
        if random_weight:
            exc_weight = torch.Tensor(out_channels, in_channels // groups,
                                      *self.kernel_size)
            inh_weight = torch.Tensor(out_channels, in_channels // groups,
                                      *self.kernel_size)
        else:
            exc_weight = torch.ones(out_channels, in_channels // groups,
                                    *self.kernel_size)
            inh_weight = torch.ones(out_channels, in_channels // groups,
                                    *self.kernel_size)

        if learn_weight:
            self.exc_weight = Parameter(exc_weight)
            self.inh_weight = Parameter(inh_weight)
        else:
            self.register_buffer("exc_weight", exc_weight)
            self.register_buffer("inh_weight", inh_weight)
        self.exc_p1 = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                             *self.kernel_size))
        self.inh_p1 = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                             *self.kernel_size))
        self.input_shape = None

        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        if self.random_weight:
            init.kaiming_normal_(self.exc_weight, mode="fan_out")
            init.kaiming_normal_(self.inh_weight, mode="fan_out")
            self.exc_weight.data.abs_()
            self.inh_weight.data.abs_()
        self.exc_p1.data.normal_(1 - self.droprate_init, 1e-2)
        self.inh_p1.data.normal_(1 - self.droprate_init, 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.exc_weight.data.clamp_(min=0.)
        self.inh_weight.data.clamp_(min=0.)

    def weight_size(self):
        return self.exc_weight.size()

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """

        if self.l0_strength > 0 or self.l2_strength > 0:
            # Clamp these, but do it in a way that still always propagates the
            # gradient.
            exc_p1 = self.exc_p1.clone()
            torch.clamp(exc_p1.data, min=0, max=1, out=exc_p1.data)
            inh_p1 = self.inh_p1.clone()
            torch.clamp(inh_p1.data, min=0, max=1, out=inh_p1.data)

            if self.l2_strength == 0:
                if self.decay_mean:
                    connections_per_unit = (exc_p1 + inh_p1).sum(
                        dim=tuple(range(1, len(exc_p1.shape))))
                    mean = torch.mean(connections_per_unit.data)

                    return self.l0_strength * torch.abs(connections_per_unit
                                                        - (mean / 2)).sum()
                else:
                    return self.l0_strength * (exc_p1 + inh_p1).sum()
            else:
                exc_weight_decay_ungated = (
                    .5 * self.l2_strength * self.exc_weight.pow(2))
                inh_weight_decay_ungated = (
                    .5 * self.l2_strength * self.inh_weight.pow(2))
                exc_weight_l2_l0 = torch.sum(
                    (exc_weight_decay_ungated + self.l0_strength) * exc_p1)
                inh_weight_l2_l0 = torch.sum(
                    (inh_weight_decay_ungated + self.l0_strength) * inh_p1)
                bias_l2 = (0 if not self.use_bias
                           else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
                return exc_weight_l2_l0 + inh_weight_l2_l0 + bias_l2
        else:
            return 0

    def get_gate_probabilities(self):
        exc_p1 = torch.clamp(self.exc_p1.data, min=0., max=1.)
        inh_p1 = torch.clamp(self.inh_p1.data, min=0., max=1.)
        return exc_p1, inh_p1

    def get_inference_mask(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        if self.deterministic:
            exc_mask = (exc_p1 >= 0.5).float()
            inh_mask = (inh_p1 >= 0.5).float()
            return exc_mask, inh_mask
        else:
            exc_count1 = exc_p1.sum(
                dim=tuple(range(1, len(exc_p1.shape)))
            ).round().int()
            inh_count1 = inh_p1.sum(
                dim=tuple(range(1, len(inh_p1.shape)))
            ).round().int()

            # pytorch doesn't offer topk with varying k values.
            exc_mask = torch.zeros_like(exc_p1)
            inh_mask = torch.zeros_like(inh_p1)
            for i in range(exc_count1.size()[0]):
                _, exc_indices = torch.topk(exc_p1[i].flatten(),
                                            exc_count1[i].item())
                _, inh_indices = torch.topk(inh_p1[i].flatten(),
                                            inh_count1[i].item())
                exc_mask[i].flatten().scatter_(-1, exc_indices, 1)
                inh_mask[i].flatten().scatter_(-1, inh_indices, 1)

            return exc_mask, inh_mask

    def sample_weight_and_bias(self, samples=1):
        if self.training or not self.optimize_inference:
            w = (sample_weight(self.exc_p1, self.exc_weight,
                               self.deterministic, samples)
                 - sample_weight(self.inh_p1, self.inh_weight,
                                 self.deterministic, samples))
        else:
            exc_mask, inh_mask = self.get_inference_mask()
            w = exc_mask * self.exc_weight - inh_mask * self.inh_weight

        b = None
        if self.use_baseline_bias:
            b = -w.sum(dim=(-3, -2, -1)) / 2

        if self.use_bias:
            b = (b + self.bias
                 if b is not None
                 else self.bias)

        return w, b

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.size()

        if self.one_sample_per_item and self.training and len(x.size()) > 3:
            w, b = self.sample_weight_and_bias(x.size(0))

            if self.use_baseline_bias:
                b = b.view(x.size(0) * self.out_channels)
            else:
                b = b.repeat(x.size(0))

            x_ = x.view(1, x.size(0) * x.size(1), *x.size()[2:])
            w_ = w.view(w.size(0) * w.size(1), *w.size()[2:])
            result = F.conv2d(x_, w_, b,
                              self.stride, self.padding, self.dilation,
                              x.size(0) * self.groups)

            return result.view(x.size(0), self.out_channels, *result.size()[2:])
        else:
            w, b = self.sample_weight_and_bias()
            return F.conv2d(x, w, b,
                            self.stride, self.padding, self.dilation, self.groups)

    def get_expected_nonzeros(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        # Flip two coins with probabilities pi_1 and pi_2. What is the
        # probability one of them is 1?
        #
        # 1 - (1 - pi_1)*(1 - pi_2)
        # = 1 - 1 + pi_1 + pi_2 - pi_1*pi_2
        # = pi_1 + pi_2 - pi_1*pi_2
        p1 = exc_p1 + inh_p1 - (exc_p1 * inh_p1)

        return p1.sum(dim=tuple(range(1, len(p1.shape)))).detach()

    def get_inference_nonzeros(self):
        exc_mask, inh_mask = self.get_inference_mask()
        return torch.sum(exc_mask.int() | inh_mask.int(),
                         dim=tuple(range(1, len(exc_mask.shape))))

    def count_inference_flops(self):
        # For each unit, multiply with n inputs then do n - 1 additions.
        # Only subtract 1 in cases where is at least one weight.
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


#
# LOCAL REPARAMETERIZATION
#

class LocalReparamBinaryGatedLinear(BinaryGatedLinear):
    """
    Linear layer with stochastic binary gates
    """
    def forward(self, x):
        if self.training:
            # Allow gradient to keep pushing p1 outside (0,1).
            exc_p1 = self.exc_p1.clone()
            exc_p1.data.clamp_(0, 1)
            inh_p1 = self.inh_p1.clone()
            inh_p1.data.clamp_(0, 1)
            w_mu = (self.exc_weight * exc_p1) - (self.inh_weight * inh_p1)
            mu = F.linear(x, w_mu, (self.bias if self.use_bias else None))

            # Don't pass back gradients to p1 for the variance when p1 is
            # outside (0,1). They will be infinite (or very large with the
            # divide-by-zero handling).
            exc_p1 = torch.clamp(self.exc_p1, 0, 1)
            inh_p1 = torch.clamp(self.inh_p1, 0, 1)
            w_var = ((self.exc_weight.pow(2) * exc_p1 * (1 - exc_p1))
                     + (self.inh_weight.pow(2) * inh_p1 * (1 - inh_p1)))
            variance = F.linear(x.pow(2), w_var)
            # Don't backpropagate beyond this variance for units with variance
            # 0. It will divide by 0.
            variance = variance.clamp(0.000001)
            sigma = variance.sqrt()

            t = (torch.FloatTensor if not torch.cuda.is_available()
                 else torch.cuda.FloatTensor)
            u = t(mu.size()).normal_()
            return mu + (u * sigma)
        else:
            return super(LocalReparamBinaryGatedLinear, self).forward(x)


class LocalReparamBinaryGatedConv2d(BinaryGatedConv2d):
    """
    Convolutional layer with binary stochastic gates
    """
    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.size()

        if self.training:
            # Allow gradient to keep pushing p1 outside (0,1).
            exc_p1 = self.exc_p1.clone()
            exc_p1.data.clamp_(0, 1)
            inh_p1 = self.inh_p1.clone()
            inh_p1.data.clamp_(0, 1)
            w_mu = (self.exc_weight * exc_p1) - (self.inh_weight * inh_p1)
            mu = F.conv2d(x, w_mu, (self.bias if self.use_bias else None),
                          self.stride, self.padding, self.dilation, self.groups)

            w_var = ((self.exc_weight.pow(2) * exc_p1 * (1 - exc_p1))
                     + (self.inh_weight.pow(2) * inh_p1 * (1 - inh_p1)))
            variance = F.conv2d(x.pow(2), w_var, None,
                                self.stride, self.padding, self.dilation,
                                self.groups)
            # Don't backpropagate beyond this variance for units with variance
            # 0. It will divide by 0.
            variance = variance.clamp(0.000001)
            sigma = variance.sqrt()

            t = (torch.FloatTensor if not torch.cuda.is_available()
                 else torch.cuda.FloatTensor)
            u = t(mu.size()).normal_()
            return mu + (u * sigma)
        else:
            return super(LocalReparamBinaryGatedConv2d, self).forward(x)
