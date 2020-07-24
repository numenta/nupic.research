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

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter

from nupic.research.frameworks.pytorch.modules import MaskedConv2d


class VDropCentralData(nn.Module):
    """
    Stores data for a set of variational dropout (VDrop) modules in large
    central tensors. The VDrop modules access the data using views. This makes
    it possible to operate on all of the data at once, (rather than e.g. 53
    times with resnet50).

    Usage:
    1. Instantiate
    2. Pass into multiple constructed VDropLinear and VDropConv2d modules
    3. Call finalize

    Before calling forward on the model, call "compute_forward_data".
    After calling forward on the model, call "clear_forward_data".

    The parameters are stored in terms of z_mu and z_var rather than w_mu and
    w_var to support group variational dropout (e.g. to allow for pruning entire
    channels.)
    """
    def __init__(self, z_logvar_init=-10):
        super().__init__()
        self.z_chunk_sizes = []
        self.z_logvar_init = z_logvar_init
        self.z_logvar_min = min(z_logvar_init, -10)
        self.z_logvar_max = 10.
        self.epsilon = 1e-8
        self.data_views = {}
        self.modules = []

        # Populated during register(), deleted during finalize()
        self.all_z_mu = []
        self.all_z_logvar = []
        self.all_num_weights = []

        # Populated during finalize()
        self.z_mu = None
        self.z_logvar = None
        self.z_num_weights = None

        self.threshold = 3

    def extra_repr(self):
        s = f"z_logvar_init={self.z_logvar_init}"
        return s

    def __getitem__(self, key):
        return self.data_views[key]

    def register(self, module, z_mu, z_logvar, num_weights_per_z=1):
        self.all_z_mu.append(z_mu.flatten())
        self.all_z_logvar.append(z_logvar.flatten())
        self.all_num_weights.append(num_weights_per_z)

        self.modules.append(module)
        data_index = len(self.z_chunk_sizes)
        self.z_chunk_sizes.append(z_mu.numel())

        return data_index

    def finalize(self):
        self.z_mu = Parameter(torch.cat(self.all_z_mu))
        self.z_logvar = Parameter(torch.cat(self.all_z_logvar))
        self.z_num_weights = torch.tensor(
            self.all_num_weights, dtype=torch.float
        ).repeat_interleave(torch.tensor(self.z_chunk_sizes))
        del self.all_z_mu
        del self.all_z_logvar
        del self.all_num_weights

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        self.z_num_weights = self.z_num_weights.to(*args, **kwargs)
        return ret

    def compute_forward_data(self):
        if self.training:
            self.data_views["z_mu"] = self.z_mu.split(self.z_chunk_sizes)
            self.data_views["z_var"] = self.z_logvar.exp().split(
                self.z_chunk_sizes)
        else:
            self.data_views["z_mu"] = (
                self.z_mu
                * (self.compute_z_logalpha() < self.threshold).float()
            ).split(self.z_chunk_sizes)

    def clear_forward_data(self):
        self.data_views.clear()

    def compute_z_logalpha(self):
        return self.z_logvar - (self.z_mu.square() + self.epsilon).log()

    def regularization(self):
        return (vdrop_regularization(self.compute_z_logalpha())
                * self.z_num_weights).sum()

    def constrain_parameters(self):
        self.z_logvar.data.clamp_(min=self.z_logvar_min,
                                  max=self.z_logvar_max)


class MaskedVDropCentralData(VDropCentralData):
    def __init__(self, restore_precision_on_prune=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.restore_precision_on_prune = restore_precision_on_prune

        # Populated during register(), deleted during finalize()
        self.all_z_mask = []

        # Populated during finalize()
        self.register_buffer("z_mask", None)

        # Sentinel value to distinguish pruned weights from those that actually
        # reached z_logvar_max. (This has no effect on the algorithm, it's just
        # sometimes useful during analysis.)
        self.pruned_logvar_sentinel = self.z_logvar_max - 0.00058

    def register(self, module, z_mu, z_logvar, num_weights_per_z=1):
        data_index = super().register(module, z_mu, z_logvar, num_weights_per_z)
        self.all_z_mask.append(torch.ones(z_mu.numel(), dtype=torch.float16))
        return data_index

    def finalize(self):
        super().finalize()
        self.z_mask = torch.cat(self.all_z_mask)
        del self.all_z_mask

    def compute_forward_data(self):
        if self.training:
            self.data_views["z_mu"] = (self.z_mu
                                       * self.z_mask).split(self.z_chunk_sizes)
            self.data_views["z_var"] = (self.z_logvar.exp()
                                        * self.z_mask).split(self.z_chunk_sizes)
        else:
            z_mu = self.z_mu * self.z_mask * (
                (self.compute_z_logalpha() < self.threshold).float()
            )
            self.data_views["z_mu"] = z_mu.split(self.z_chunk_sizes)

    def regularization(self):
        return ((vdrop_regularization(self.compute_z_logalpha())
                 * self.z_mask)
                * self.z_num_weights).sum()

    def masked_parameters(self):
        """
        Get information needed to zero momentum in the optimizer.
        """
        yield self.z_mu, self.z_mask
        yield self.z_logvar, self.z_mask


class VDropLinear(nn.Module):
    def __init__(self, in_features, out_features, central_data, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store in a list to avoid having it registered as a module, otherwise
        # it will appear multiple times in the state dict.
        self.central_data = [central_data]

        w_mu = torch.Tensor(self.out_features, self.in_features)
        w_logvar = torch.Tensor(self.out_features, self.in_features)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.bias = None

        w_logvar.data.fill_(central_data.z_logvar_init)
        # Standard nn.Linear initialization.
        init.kaiming_uniform_(w_mu, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        self.data_index = central_data.register(self, w_mu, w_logvar)

        self.tensor_constructor = (torch.FloatTensor
                                   if not torch.cuda.is_available()
                                   else torch.cuda.FloatTensor)

    def extra_repr(self):
        s = f"{self.in_features}, {self.out_features}, "
        if self.bias is None:
            s += ", bias=False"
        return s

    def get_w_mu(self):
        return self.central_data[0]["z_mu"][self.data_index].view(
            self.out_features, self.in_features)

    def get_w_var(self):
        return self.central_data[0]["z_var"][self.data_index].view(
            self.out_features, self.in_features)

    def forward(self, x):
        if self.training:
            return vdrop_linear_forward(x, self.get_w_mu, self.get_w_var,
                                        self.bias, self.tensor_constructor)
        else:
            return F.linear(x, self.get_w_mu(), self.bias)


class VDropConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, central_data,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.groups = groups

        # Store in a list to avoid having it registered as a module, otherwise
        # it will appear multiple times in the state dict.
        self.central_data = [central_data]

        w_mu = torch.Tensor(out_channels,
                            in_channels // groups,
                            *self.kernel_size)
        w_logvar = torch.Tensor(out_channels,
                                in_channels // groups,
                                *self.kernel_size)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        w_logvar.data.fill_(central_data.z_logvar_init)

        # Standard nn.Conv2d initialization.
        init.kaiming_uniform_(w_mu, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(w_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        self.data_index = central_data.register(self, w_mu, w_logvar)

        self.tensor_constructor = (torch.FloatTensor
                                   if not torch.cuda.is_available()
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

    def get_w_mu(self):
        return self.central_data[0]["z_mu"][self.data_index].view(
            self.out_channels, self.in_channels, *self.kernel_size)

    def get_w_var(self):
        return self.central_data[0]["z_var"][self.data_index].view(
            self.out_channels, self.in_channels, *self.kernel_size)

    def forward(self, x):
        if self.training:
            return vdrop_conv_forward(x, self.get_w_mu, self.get_w_var,
                                      self.bias, self.stride, self.padding,
                                      self.dilation, self.groups,
                                      self.tensor_constructor)
        else:
            return F.conv2d(x, self.get_w_mu(), self.bias, self.stride,
                            self.padding, self.dilation, self.groups)


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
                lambda: self.alpha * (self.weight.square() * self.weight_mask),
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
    y = F.linear(x.square(), get_w_var())

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
        x.square(), get_w_var(), None, stride, padding, dilation, groups
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
    "VDropCentralData",
    "MaskedVDropCentralData",
    "VDropLinear",
    "VDropConv2d",
    "FixedVDropConv2d",
]
