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

import itertools

import numpy as np
import torch

# ------------------
# Utilities
# ------------------


def calc_sparsity(weight):
    return float(torch.Tensor([float((weight == 0).sum()) / np.prod(weight.shape)]))


def calc_onfrac(weight):
    return 1 - calc_sparsity(weight)


def topk_mask(tensor, k, exclusive=True):

    # Get value of top 'perc' percentile.
    tensor_flat = (tensor.cpu() if tensor.is_cuda else tensor).flatten()
    v = np.partition(tensor_flat, -k)[-k]

    # Return mask of values above v.
    topk = (tensor > v) if exclusive else (tensor >= v)
    return topk


def bottomk_mask(tensor, k, exclusive=True):
    k = np.prod(tensor.shape) - k
    bottomk = ~topk_mask(tensor, k, exclusive=~exclusive)
    return bottomk


def where(condition):
    """
    Similar to torch.where, but the result is packed into tuples and its
    shape is transposed via zip.
    Per the last statement, here's an example:
        zip((0, 1, 2), (0, 1, 2)) -> ((0, 0), (1, 1), (2, 2))

    :param condition: some condition in the form of a mask
    """
    idx = [tuple(i_.to("cpu").numpy()) for i_ in torch.where(condition)]
    idx = tuple(zip(*idx))
    return idx


def break_mask_ties(mask, num_remain=None, frac_remain=None):
    """
    Break ties in a mask - whether between zeros or ones.
    This function ensures exactly a num_remain number of
    non-zeros will be present in the output mask

    :param mask: mask of zeros and ones (or equivalents) - overwritten in place
    :param num_remain: number of desired non-zeros
    """

    assert num_remain is not None or frac_remain is not None

    if num_remain is None:
        num_remain = int(frac_remain * np.prod(mask.shape))

    idx_ones = where(mask == 1)
    num_ones = len(idx_ones)
    if num_ones > 0:
        idx_ones = tuple(
            idx_ones[i1_]
            for i1_ in np.random.choice(
                range(num_ones), min(num_remain, num_ones), replace=False
            )
        )

    if num_ones < num_remain:
        num_fill = num_remain - num_ones
        idx_zeros = where(mask == 0)
        num_zeros = len(idx_zeros)
        idx_remain = idx_ones + tuple(
            idx_zeros[i0_]
            for i0_ in np.random.choice(
                range(num_zeros), min(num_fill, num_zeros), replace=False
            )
        )

    else:
        idx_remain = idx_ones

    idx_remain = tuple(zip(*idx_remain))
    mask[...] = 0
    mask[idx_remain] = 1

    return mask


# ------------------
# Base
# ------------------


def init_coactivation_tracking(m):
    """
    Function used to start tracking coactivations.
    Call using :meth:`torch.nn.Module.apply` before training starts.
    For example: ``m.apply(init_coactivation_tracking)``

    :param m: torch.nn.Module
    """
    if isinstance(m, DynamicSparseBase):
        m.init_coactivation_tracking()


class DynamicSparseBase(torch.nn.Module):
    def _init_coactivations(self, weight, config=None):
        """
        This method
            1. registers a buffer the shape of weight to track coactivations
            2. adds a forward hook to the module to update the coactivations
               every 'update_interval'.

        :param weight: torch.tensor - corresponding weight of coactivations
        :param config: dict - configurable parameters for tracking coactivations
        """

        # Init defaults and override only when params are specified in the config.
        config = config or {}
        defaults = dict(
            moving_average_alpha=None,           # See `_update_coactivations`
            update_func=None,  # See `_update_coactivations`
            update_interval=1,  # See `forward_hook`
        )
        new_defaults = {k: (config.get(k, None) or v) for k, v in defaults.items()}
        self.__dict__.update(new_defaults)

        # Init buffer to keep track of coactivations.
        self._track_coactivations = False
        self.register_buffer("coactivations", torch.zeros_like(self.weight))

        # Init helper attrs to keep track of when to update coactivations.
        self.learning_iterations = 0

        # Register hook to update coactivations.
        assert hasattr(
            self, "calc_coactivations"
        ), "DynamicSparse modules must define a coactivation function."
        self.forward_hook_handle = self.register_forward_hook(self.forward_hook)

    def init_coactivation_tracking(self):
        self._track_coactivations = True

    def reset_coactivations(self):
        # Reset coactivations to zero.
        self.coactivations[:] = 0

    def _update_coactivations(self, input_tensor, output_tensor):

        new_coacts = self.calc_coactivations(input_tensor, output_tensor)
        if self.update_func:
            updated_coacts = self.update_func(self.coactivations, new_coacts)
        elif self.moving_average_alpha:
            alpha = self.moving_average_alpha
            updated_coacts = (1 - alpha) * self.coactivations + alpha * new_coacts
        else:
            updated_coacts = self.coactivations + new_coacts

        self.coactivations[:] = updated_coacts

    @staticmethod
    def forward_hook(module, input_tensor, output_tensor):
        if not module._track_coactivations:
            return

        # Update connections strengths.
        if isinstance(input_tensor, tuple):
            # TODO: This assumption of taking the first element may not
            #       work for all module.
            input_tensor = input_tensor[0]
        if module.training:
            if module.learning_iterations % module.update_interval == 0:
                module._update_coactivations(input_tensor, output_tensor)
            module.learning_iterations += 1


# ------------------
# Linear Layers
# ------------------


class DSLinear(torch.nn.Linear, DynamicSparseBase):
    def __init__(self, in_features, out_features, bias=False, config=None):

        super().__init__(in_features, out_features, bias=bias)

        # Initialize dynamic sparse attributes.
        config = config or {}
        self._init_coactivations(weight=self.weight, config=config)

    def _init_coactivations(self, weight, config=None):
        super()._init_coactivations(weight, config=config)

        # Init defaults and override only when params are specified in the config.
        config = config or {}
        defaults = dict(
            use_binary_coactivations=True,
        )
        new_defaults = {k: (config.get(k, None) or v) for k, v in defaults.items()}
        self.__dict__.update(new_defaults)

    def calc_coactivations(self, x, y):
        outer = 0
        n_samples = x.shape[0]
        with torch.no_grad():

            # Get active units.
            if self.use_binary_coactivations:
                prev_act = (x > 0).detach().float()
                curr_act = (y > 0).detach().float()
            else:
                prev_act = x.clone().detach().float()
                curr_act = y.clone().detach().float()

            # Cumulate outer product over all samples.
            # TODO: Vectorize this sum; for instance, using torch.einsum().
            for s in range(n_samples):
                outer += torch.ger(curr_act[s], prev_act[s])

        # Return coactivations.
        return outer

# ------------------
# Conv Layers
# ------------------


class _NullConv(torch.nn.Conv2d):
    """
    Exactly a regular conv, but without it's weights initialized. This is a helper class
    to DSConv2d. In some cases, like when initializing 'grouped_conv', we may want to
    manually set the weights and avoid the potentially computationally expensive
    procedure of random initializing them.
    """

    def reset_parameters(self):
        # Do nothing and don't initialize the weights.
        pass


class DSConv2d(torch.nn.Conv2d, DynamicSparseBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        config=None,
    ):
        """
        The primary params are the same for a regular Conv2d layer.
        Otherwise, they're described below.

        :param half_precision: whether to operate in half precision when calculating
                               calculating the coactivation - this only works when the
                               device is "cuda" and is mainly for memory saving during
                               training
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        # Initialize dynamic sparse attributes.
        config = config or {}
        self._init_coactivations(weight=self.weight, config=config)

        # -------------------------------------
        # 'calc_coactivation' related attr's
        # -------------------------------------

        # Init params for calculating coactivations.
        self.half_precision = half_precision
        self.coactivation_test = coactivation_test
        self.threshold_multiplier = threshold_multiplier

        # Specify number of groups for the helper convolutional layer.
        # This is equal to the number of connections in the last three dimensions:
        #      new_groups = in_channels x kernel_size[0] x kernel_size[1]
        self.new_groups = int(np.prod(self.weight.shape[1:]))

        # Compute indices that loop over all connections in the last three dimensions.
        # This will be used to help initialize the helper convolution.
        self.filter_indxs = list(
            itertools.product(*[range(d) for d in self.weight.shape[1:]])
        )

        # Compute indices that loop over all connections, including the out_channels.
        # This will be used to unpack the point-wise comparisons of the coactivations.
        self.connection_indxs = []
        for c_ in range(self.out_channels):
            for idx in self.filter_indxs:
                i_ = list(idx)
                self.connection_indxs.append([c_] + i_)
        self.connection_indxs = list(zip(*self.connection_indxs))

        # Compute indices that expand the output channels by the number of new groups.
        # This will be used to unpack the point-wise comparisons of the coactivations.
        self.perm_indices = []
        for c_i in range(self.out_channels):
            self.perm_indices.extend([c_i] * self.new_groups)

        # Create helper conv layer to aid in coactivation calculations.
        self.grouped_conv = _NullConv(
            in_channels=self.in_channels * self.new_groups,
            out_channels=self.new_groups,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
            dilation=self.dilation,
            groups=self.groups * self.new_groups,
            bias=False,
        )

        # Populate the weight matrix with stacked tensors having only one non-zero unit.
        single_unit_weights = [
            self._get_single_unit_weights(c, j, h) for c, j, h in self.filter_indxs
        ]
        stacked_weights = torch.cat(single_unit_weights, dim=0)
        self.grouped_conv.weight = torch.nn.Parameter(
            stacked_weights, requires_grad=False
        )

    def _init_coactivations(self, weight, config=None):
        super()._init_coactivations(weight, config=config)

        # Init defaults and override only when params are specified in the config.
        config = config or {}
        defaults = dict(
            padding_mode="zeros",
            update_interval=100,
            half_precision=False,
            coactivation_test="correlation_proxy",
            threshold_multiplier=1,
        )
        new_defaults = {k: (config.get(k, None) or v) for k, v in defaults.items()}
        self.__dict__.update(new_defaults)

    def _get_single_unit_weights(self, c, j, h):
        """
        Constructs and returns conv layer with training disabled and
        all zero weights except along the output channels for unit
        specified as (c, j, h).
        """

        # Construct weight.
        dtype = torch.float16 if self.half_precision else torch.float32
        weight = torch.zeros(1, *self.weight.shape[1:], dtype=dtype)

        # Set weights to zero except those specified.
        weight[0, c, j, h] = 1

        return weight

    def get_activity_threshold(self, input_tensor, output_tensor):
        """
        Returns tuple of input and output activity thresholds.
        """
        return (input_tensor.std(), output_tensor.std())

    def calc_coactivations(self, input_tensor, output_tensor):
        """
        This function updates self.coactivations.
        The computation is highly vectorized and unfortunately quite opaque.
        Generally, two units, say unit_in and unit_out, coactivate if
            1. unit_in is in the receptive field of unit_out
            2. (unit_in  - mean_input ) > input_activity_threshold
            3. (unit_out - mean_output) > output_activity_threshold
        """
        with torch.no_grad():

            # Switch to half-floating precision if needed.
            if self.half_precision:
                input_tensor = input_tensor.half()

            # Prep input to compute the coactivations.
            grouped_input = input_tensor.repeat((1, self.new_groups, 1, 1))
            grouped_input = self.grouped_conv(grouped_input).repeat(
                (1, self.out_channels, 1, 1)
            )

            mu_in = input_tensor.mean()
            mu_out = output_tensor.mean()
            std_in = input_tensor.std()
            std_out = output_tensor.std()

            if self.coactivation_test == "variance":

                a1, a2 = self.get_activity_threshold(input_tensor, output_tensor)
                a1 = a1 * self.threshold_multiplier
                a2 = a2 * self.threshold_multiplier
                s1 = torch.abs(grouped_input - mu_in)
                s1 = s1.gt_(a1)
                s2 = torch.abs(output_tensor - mu_out)
                s2 = s2.gt_(a2)[:, self.perm_indices, ...]

                # Save space on device
                del mu_in
                del mu_out
                del a1
                del a2
                del grouped_input

                h = torch.sum(s2.mul(s1), (0, 2, 3))

                del s1
                del s2

            elif self.coactivation_test == "correlation":

                s1 = grouped_input
                s2 = output_tensor[:, self.perm_indices, ...]

                mu_in = s1.mean(dim=0)
                mu_out = s2.mean(dim=0)

                std_in = s1.std(dim=0)
                std_out = s2.std(dim=0)

                corr = ((s1 - mu_in) * (s2 - mu_out)).mean(dim=0) / (std_in * std_out)
                corr[torch.where((std_in == 0) | (std_out == 0))] = 0
                corr = corr.abs()

                # Save space on device
                del s1
                del s2
                del grouped_input
                del mu_in
                del mu_out
                del std_in
                del std_out

                h = torch.sum(corr, (1, 2))
                h = h.type(self.coactivations.dtype)

                del corr

            elif self.coactivation_test == "correlation_proxy":

                del mu_in
                del mu_out

                s1 = grouped_input
                s2 = output_tensor[:, self.perm_indices, ...]

                corr_proxy = (s1 != 0) * (s2 != 0)
                h = torch.sum(corr_proxy, (0, 2, 3))
                h = h.type(self.coactivations.dtype)

                del corr_proxy

            new_coacts = torch.zeros_like(self.coactivations)
            new_coacts[self.connection_indxs] = h

            del h

            return new_coacts

    def __call__(self, input_tensor, *args, **kwargs):
        output_tensor = super().__call__(input_tensor, *args, **kwargs)
        return output_tensor


if __name__ == "__main__":
    pass
