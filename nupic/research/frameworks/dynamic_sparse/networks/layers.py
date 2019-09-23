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
                range(num_ones), min(num_remain, num_ones), replace=False)
        )

    if num_ones < num_remain:
        num_fill = num_remain - num_ones
        idx_zeros = where(mask == 0)
        num_zeros = len(idx_zeros)
        idx_remain = idx_ones + tuple(
            idx_zeros[i0_]
            for i0_ in np.random.choice(
                range(num_zeros), min(num_fill, num_zeros), replace=False)
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

    def _init_coactivations(self, weight, update_nsteps=1):
        """
        This method
            1. registers a buffer the shape of weight to track coactivations
            2. adds a forward hook to the module to update the coactivations
               every 'update_nsteps'.
        """

        # Init buffer to keep track of coactivations.
        self._track_coactivations = False
        self.register_buffer("coactivations", torch.zeros_like(self.weight))

        # Init helper attrs to keep track of when to update coactivations.
        self.learning_iterations = 0
        self.update_nsteps = update_nsteps

        # Register hook to update coactivations.
        assert hasattr(self, "calc_coactivations"), \
            "DynamicSparse modules must define a coactivation function."
        self.forward_hook_handle = self.register_forward_hook(self.forward_hook)

    def init_coactivation_tracking(self):
        self._track_coactivations = True

    def reset_coactivations(self):
        # Reset coactivations to zero.
        self.coactivations[:] = 0

    def _update_coactivations(self, input_tensor, output_tensor):

        new_coacts = self.calc_coactivations(input_tensor, output_tensor)
        self.coactivations[:] += new_coacts

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
            if module.learning_iterations % module.update_nsteps == 0:
                module._update_coactivations(input_tensor, output_tensor)
            module.learning_iterations += 1


# ------------------
# Linear Layers
# ------------------

class DSLinear(torch.nn.Linear, DynamicSparseBase):

    def __init__(self, in_features, out_features, bias=False):

        super().__init__(in_features, out_features, bias=bias)

        # Initialize dynamic sparse attributes.
        self._init_coactivations(weight=self.weight)

    def calc_coactivations(self, x, y):
        outer = 0
        n_samples = x.shape[0]
        with torch.no_grad():

            # Get active units.
            curr_act = (x > 0).detach().float()
            prev_act = (y > 0).detach().float()

            # Cumulate outer product over all samples.
            # TODO: Vectorize this sum; for instance, using torch.einsum().
            for s in range(n_samples):
                outer += torch.ger(prev_act[s], curr_act[s])

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
        padding_mode="zeros",
        hebbian_prune_frac=0.15,
        magnitude_prune_frac=0.00,
        sparsity=0.80,
        prune_dims=None,
        update_nsteps=100,
        half_precision=False,
        coactivation_test="correlation_proxy",
        threshold_multiplier=1,
    ):
        """
        The primary params are the same for a regular Conv2d layer.
        Otherwise, they're described below.

        :param hebbian_prune_frac: fraction of weights to keep by Hebbian based ranking
        :param magnitude_prune_frac: fraction of weights to keep by magn. based ranking
        :param sparsity: fraction of weights to maintain as zero
        :param prune_dims: take [0, 1] as an example, then pruning will occur
                           separately for each self.weight[i, j, :, :]
                           over all i, j combinations
        :param update_nsteps: period of training steps to wait before calculating the
                              coactivations needed for Hebbian pruning.
        :param half_precision: whether to operate in half precision when calculating
                               calculating the coactivation - this only works when the
                               device is "cuda" and is mainly for memory saving during
                               training
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
        )

        self._init_coactivations(self.weight, update_nsteps=update_nsteps)
        self.half_precision = half_precision

        if prune_dims is None:
            self.prune_dims = [0, 1]
        else:
            self.prune_dims = list(prune_dims).copy()

        # Track learning_iterations (number of forward passes during training)
        # and pruning_iterations (number of times connections are pruned).
        self.learning_iterations = 0
        self.pruning_iterations = 0

        # Set pruning params.
        #   num_connections - number of connections along each prune_dim
        #   k1_weight - the number of connections to keep by magnitude pruning
        #   k1_hebbian - the number of connections to keep by hebbian pruning
        #   k2 - the number of connections to keep non-zero
        self.num_connections = np.prod([
            d for i, d in enumerate(self.weight.shape) if i not in self.prune_dims
        ])
        self.total_connections = np.prod([self.weight.shape])
        self.nonzero_frac = 1 - sparsity
        self.magnitude_prune_frac = magnitude_prune_frac
        self.hebbian_prune_frac = hebbian_prune_frac
        self.k1_weight = max(int((1 - magnitude_prune_frac) * self.num_connections), 1)
        self.k1_hebbian = max(int((1 - hebbian_prune_frac) * self.num_connections), 1)
        self.k2 = max(int((1 - sparsity) * self.num_connections), 1)
        self.coactivation_test = coactivation_test
        self.threshold_multiplier = threshold_multiplier

        # Make the weight matrix sparse.
        self.nonzero_num = max(1, int(self.nonzero_frac * self.total_connections))
        # self.last_keep_mask = torch.rand(self.weight.shape) < self.nonzero_frac
        self.register_buffer("last_keep_mask",
                             torch.ones_like(self.weight, dtype=torch.bool))
        self.last_keep_mask[:] = break_mask_ties(self.last_keep_mask, self.nonzero_num)
        with torch.no_grad():
            self.weight.set_(self.weight.data * self.last_keep_mask.float())
            # Log sparsity
            self.weight_sparsity = calc_sparsity(self.weight)
        self.prune_grads_hook = self.weight.register_hook(
            lambda grad: grad * self.last_keep_mask.type(grad.dtype).to(grad.device))

        # Specify number of groups for the helper convolutional layer.
        # This is equal to the number of connections in the last three dimensions:
        #      new_groups = in_channels x kernel_size[0] x kernel_size[1]
        self.new_groups = int(np.prod(self.weight.shape[1:]))

        # Compute indices that loop over all connections in the last three dimensions.
        # This will be used to help initialize the helper convolution.
        self.filter_indxs = list(itertools.product(*[
            range(d) for d in self.weight.shape[1:]
        ]))

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
            self.perm_indices.extend(
                [c_i] * self.new_groups
            )

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
            self._get_single_unit_weights(
                c, j, h,
            )
            for c, j, h in self.filter_indxs
        ]
        stacked_weights = torch.cat(single_unit_weights, dim=0)
        self.grouped_conv.weight = torch.nn.Parameter(
            stacked_weights, requires_grad=False)

        self._init_logging_params()

    def _init_logging_params(self):

        # ------------------
        # For logging
        # ------------------

        # A note on notation:
        #
        #    `c*` => connection
        #    `0`  => dead
        #    `1`  => alive
        #    `x`  => either dead or alive
        #
        #    Example: (putting these all-together)
        #
        #    `c01` => a connection that was regrown - was dead but now alive
        #    `c01x => a connection that was regrown and anything
        #              can happen. Maybe it will stay alive, maybe it'll be pruned
        #    `keep_mask` => can also be thought of as `cx1`
        #

        # For sparsity...
        self.on2off_mask_sparsity = None
        self.off2on_mask_sparsity = None
        self.on2off_mask_num = None
        self.off2on_mask_num = None
        self.keep_mask_sparsity = None
        self.last_coactivations = None
        self.input_means = None
        self.output_means = None

        # For weights...
        self.last_c00_mask = None
        self.last_c01_mask = None
        self.last_c10_mask = None
        self.last_c11_mask = None
        self.weight_c01x_mean = None
        self.weight_c01x_std = None
        self.weight_c11x_mean = None
        self.weight_c11x_std = None
        self.weight_c01x_c11x_mean_diff = None
        self.c10_num = None
        self.c10_frac = None
        self.c10_frac_rel = None
        self.c01_num = None
        self.c01_frac = None
        self.c01_frac_rel = None
        self.c11_num = None
        self.c11_frac = None
        self.c11_frac_rel = None
        self.c00_num = None
        self.c00_frac = None
        self.c00_frac_rel = None
        self.c000_frac = None
        self.c001_frac = None
        self.c010_frac = None
        self.c011_frac = None
        self.c100_frac = None
        self.c101_frac = None
        self.c110_frac = None
        self.c111_frac = None
        self.c000_frac_rel = None
        self.c001_frac_rel = None
        self.c010_frac_rel = None
        self.c011_frac_rel = None
        self.c100_frac_rel = None
        self.c101_frac_rel = None
        self.c110_frac_rel = None
        self.c111_frac_rel = None
        self.c10_frac_rel = None
        self.c01_frac_rel = None
        self.survival_rate = None

        # For gradients...
        self.tot_grad_flow = None
        self.c00_grad_flow = None
        self.c00_grad_flow_centered = None
        self.c01_grad_flow = None
        self.c01_grad_flow_centered = None
        self.c10_grad_flow = None
        self.c10_grad_flow_centered = None
        self.c11_grad_flow = None
        self.c11_grad_flow_centered = None
        self.c01_c11_grad_flow_diff = None
        self.c01_c11_grad_flow_diff_centered = None

        self.running_c00_grad_flow = None
        self.running_c01_grad_flow = None
        self.running_c10_grad_flow = None
        self.running_c11_grad_flow = None
        self.running_tot_grad_flow = None
        self.log_grad_flows_hook = None
        self.log_grad_flows = False
        self.log_grad_flows_hook = self.weight.register_hook(
            lambda grad: self._log_grad_flows_hook(grad))

        self._reset_logging_params()

    def _log_grad_flows_hook(self, grad):

        if not self.log_grad_flows:
            return

        c00_mean = grad[self.last_c00_mask].mean()[None]
        c01_mean = grad[self.last_c01_mask].mean()[None]
        c10_mean = grad[self.last_c10_mask].mean()[None]
        c11_mean = grad[self.last_c11_mask].mean()[None]
        tot_mean = grad.mean()[None]

        dtype = tot_mean.dtype
        device = tot_mean.device

        if self.running_tot_grad_flow is None:
            self.running_c00_grad_flow = torch.tensor([]).type(dtype).to(device)
            self.running_c01_grad_flow = torch.tensor([]).type(dtype).to(device)
            self.running_c10_grad_flow = torch.tensor([]).type(dtype).to(device)
            self.running_c11_grad_flow = torch.tensor([]).type(dtype).to(device)
            self.running_tot_grad_flow = torch.tensor([]).type(dtype).to(device)

        self.running_c00_grad_flow = torch.cat((self.running_c00_grad_flow, c00_mean))
        self.running_c01_grad_flow = torch.cat((self.running_c01_grad_flow, c01_mean))
        self.running_c10_grad_flow = torch.cat((self.running_c10_grad_flow, c10_mean))
        self.running_c11_grad_flow = torch.cat((self.running_c11_grad_flow, c11_mean))
        self.running_tot_grad_flow = torch.cat((self.running_tot_grad_flow, tot_mean))

    def _reset_logging_params(self):
        self.input_means = np.array([])
        self.output_means = np.array([])

    def _get_single_unit_weights(self, c, j, h):
        """
        Constructs and returns conv layer with training disabled and
        all zero weights except along the output channels for unit
        specified as (c, j, h).
        """

        # Construct weight.
        dtype = torch.float16 if self.half_precision else torch.float32
        weight = torch.zeros(
            1, *self.weight.shape[1:],
            dtype=dtype
        )

        # Set weights to zero except those specified.
        weight[0, c, j, h] = 1

        return weight

    def get_activity_threshold(self, input_tensor, output_tensor):
        """
        Returns tuple of input and output activity thresholds.
        """
        return (
            input_tensor.std(),
            output_tensor.std()
        )

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
                (1, self.out_channels, 1, 1))

            mu_in = input_tensor.mean()
            mu_out = output_tensor.mean()
            std_in = input_tensor.std()
            std_out = output_tensor.std()

            self.input_means = np.append(self.input_means, mu_in.to("cpu").item())
            self.output_means = np.append(self.output_means, mu_out.to("cpu").item())

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

                h = torch.sum(s2.mul(s1), (0, 2, 3,))

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

    def progress_connections(self):
        """
        Prunes and add connections.
        """

        # Remove old hook to zero the gradients of pruned connections.
        self.prune_grads_hook.remove()
        if self.pruning_iterations > 0:
            self.last_coactivations = self.coactivations.clone().detach()

        # Decide which connections to prune and which to regrow.
        with torch.no_grad():

            # Get strengths of all connections.
            strengths_weight = self.weight.data.clone().detach().abs()
            strengths_hebbian = self.coactivations * strengths_weight

            # Determine all combinations of prune dimensions
            all_dims = range(len(self.weight.shape))
            prune_indxs = [
                range(self.weight.shape[d]) if d in self.prune_dims else [slice(None)]
                for d in all_dims
            ]
            prune_indxs = list(itertools.product(*prune_indxs))

            # Determine which indices to prune and which to add:
            #    - Keep strongest k1 connections
            #    - Prevent gradient flow weakest k2 connections.
            on_mask = self.last_keep_mask.clone().detach()
            off_mask = ~self.last_keep_mask.clone().detach()
            on2off_mask = self.last_keep_mask.clone().detach()
            off2on_mask = ~self.last_keep_mask.clone().detach()

            for prune_frac, strengths in [
                (self.magnitude_prune_frac, strengths_weight),
                (self.hebbian_prune_frac, strengths_hebbian),
            ]:

                if prune_frac == 0:
                    continue

                for idx in prune_indxs:

                    # Of the subset defined by 'idx', find out which
                    # connections are on and which are off.
                    on_submask = on_mask[idx]
                    off_submask = off_mask[idx]

                    # Tally on connections.
                    num_on = float(on_submask.sum())

                    # Case 1: Some connections are "on" and can be removed and replaced.
                    # Removing and adding is done in a 1-1 fashion.
                    if num_on != 0:

                        # Get strengths of on connections.
                        s = strengths[idx][on_submask]

                        # Find bottom on-connections.
                        prune_num = max(int(prune_frac * num_on), 1)
                        prune_submask = bottomk_mask(s, prune_num)
                        prune_submask = break_mask_ties(
                            prune_submask, num_remain=prune_num)

                        # Find top off-connections.
                        s = strengths[idx][off_submask]
                        new_submask = topk_mask(s, prune_num)
                        new_submask = break_mask_ties(
                            new_submask, num_remain=prune_num)

                        # Remove bottom-on connections, replace with top-off connections
                        on2off_mask[idx][on_submask] &= prune_submask
                        off2on_mask[idx][off_submask] &= new_submask

                    # Case 2: No connections are "on", so none may be replaced.
                    # Since none will be removed, ensure none are added.
                    else:
                        off2on_mask[idx][off_submask] = 0

            # Combine the result to form the new keep_mask.
            keep_mask = torch.zeros_like(self.weight, dtype=torch.bool)
            keep_mask[on_mask & ~on2off_mask] = 1
            keep_mask[off_mask & off2on_mask] = 1

            # ---------------------
            # For Logging
            # ---------------------

            # See __init__ for a note on notation.

            # Log stats of weight matrix.
            last_keep_mask = self.last_keep_mask
            if last_keep_mask is not None and self.last_c01_mask is not None:

                c01x_mask = self.last_c01_mask
                c11x_mask = self.last_c11_mask

                weights_mean = self.weight.data.std()
                weights_std = self.weight.data.std()

                self.weight_c01x_mean = float(
                    (self.weight.data[c01x_mask].mean() - weights_mean)
                    / weights_std)
                self.weight_c01x_std = float(self.weight.data[c01x_mask].std())

                self.weight_c11x_mean = float(
                    (self.weight.data[c11x_mask].mean() - weights_mean)
                    / weights_std)
                self.weight_c11x_std = float(self.weight.data[c11x_mask].std())

                self.weight_c01x_c11x_mean_diff = \
                    abs(self.weight_c01x_mean) - abs(self.weight_c11x_mean)

            # ----- END LOG BLOCK -----

            # Zero weights and prevent gradient flow.
            self.weight.data[~keep_mask] = 0
            self.weight_sparsity = calc_sparsity(self.weight.data)
            self.prune_grads_hook = self.weight.register_hook(
                lambda grad: grad * keep_mask.type(grad.dtype))

            # ---------------------
            # For Logging
            # ---------------------

            # See __init__ for a note on notation.

            self.log_grad_flows_hook.remove()

            self.on2off_mask_sparsity = calc_sparsity(on2off_mask)
            self.on2off_mask_num = float(on2off_mask.sum())
            self.off2on_mask_sparsity = calc_sparsity(off2on_mask)
            self.off2on_mask_num = float(off2on_mask.sum())
            self.keep_mask_sparsity = calc_sparsity(keep_mask)

            # ----- Log stats -----
            if last_keep_mask is not None:

                # Log stats of surviving connections.
                if self.last_c01_mask is not None:

                    c000_mask = self.last_c00_mask & ~keep_mask
                    c001_mask = self.last_c00_mask & keep_mask
                    c010_mask = self.last_c01_mask & ~keep_mask
                    c011_mask = self.last_c01_mask & keep_mask
                    c100_mask = self.last_c10_mask & ~keep_mask
                    c101_mask = self.last_c10_mask & keep_mask
                    c110_mask = self.last_c11_mask & ~keep_mask
                    c111_mask = self.last_c11_mask & keep_mask

                    self.c000_frac = calc_onfrac(c000_mask)
                    self.c000_frac_rel = calc_onfrac(c000_mask[self.last_c00_mask])
                    self.c001_frac = calc_onfrac(c001_mask)
                    self.c001_frac_rel = calc_onfrac(c001_mask[self.last_c00_mask])
                    self.c010_frac = calc_onfrac(c010_mask)
                    self.c010_frac_rel = calc_onfrac(c010_mask[self.last_c01_mask])
                    self.c011_frac = calc_onfrac(c011_mask)
                    self.c011_frac_rel = calc_onfrac(c011_mask[self.last_c01_mask])
                    self.c100_frac = calc_onfrac(c100_mask)
                    self.c100_frac_rel = calc_onfrac(c100_mask[self.last_c10_mask])
                    self.c101_frac = calc_onfrac(c101_mask)
                    self.c101_frac_rel = calc_onfrac(c101_mask[self.last_c10_mask])
                    self.c110_frac = calc_onfrac(c110_mask)
                    self.c110_frac_rel = calc_onfrac(c110_mask[self.last_c11_mask])
                    self.c111_frac = calc_onfrac(c111_mask)
                    self.c111_frac_rel = calc_onfrac(c111_mask[self.last_c11_mask])

                    self.survival_rate = float(
                        c011_mask.sum() / self.last_c01_mask.sum())

                # Log stats of grad flows.
                r_c00_grad_flow = self.running_c00_grad_flow
                r_c01_grad_flow = self.running_c01_grad_flow
                r_c10_grad_flow = self.running_c10_grad_flow
                r_c11_grad_flow = self.running_c11_grad_flow
                r_tot_grad_flow = self.running_tot_grad_flow

                if r_c00_grad_flow is not None:

                    mean_grad_flow = float(r_tot_grad_flow.mean())
                    std_grad_flow = float(r_tot_grad_flow.std())

                    self.c00_grad_flow = float(
                        r_c00_grad_flow.mean())
                    self.c00_grad_flow_centered = float(
                        (r_c00_grad_flow - mean_grad_flow).mean() / std_grad_flow)

                    self.c01_grad_flow = float(
                        r_c01_grad_flow.mean())
                    self.c01_grad_flow_centered = float(
                        (r_c01_grad_flow - mean_grad_flow).mean() / std_grad_flow)

                    self.c10_grad_flow = float(
                        r_c10_grad_flow.mean())
                    self.c10_grad_flow_centered = float(
                        (r_c10_grad_flow - mean_grad_flow).mean() / std_grad_flow)

                    self.c11_grad_flow = float(
                        r_c11_grad_flow.mean())
                    self.c11_grad_flow_centered = float(
                        (r_c11_grad_flow - mean_grad_flow).mean() / std_grad_flow)

                    self.c01_c11_grad_flow_diff = (
                        abs(self.c01_grad_flow)
                        - abs(self.c11_grad_flow))
                    self.c01_c11_grad_flow_diff_centered = (
                        abs(self.c01_grad_flow_centered)
                        - abs(self.c11_grad_flow_centered))

                # ----- Reset for next epoch -----

                # Reset connection masks...
                self.last_c00_mask = ~last_keep_mask & ~keep_mask
                self.last_c01_mask = ~last_keep_mask & keep_mask
                self.last_c10_mask = last_keep_mask & ~keep_mask
                self.last_c11_mask = last_keep_mask & keep_mask

                # Reset connection stats...
                self.c10_num = float(self.last_c10_mask.sum())
                self.c10_frac = calc_onfrac(self.last_c10_mask)
                self.c10_frac_rel = calc_onfrac(self.last_c10_mask[last_keep_mask])
                self.c01_num = float(self.last_c01_mask.sum())
                self.c01_frac = calc_onfrac(self.last_c01_mask)
                self.c01_frac_rel = calc_onfrac(self.last_c01_mask[~last_keep_mask])
                self.c11_num = float(self.last_c11_mask.sum())
                self.c11_frac = calc_onfrac(self.last_c11_mask)
                self.c11_frac_rel = calc_onfrac(self.last_c11_mask[last_keep_mask])
                self.c00_num = float(self.last_c00_mask.sum())
                self.c00_frac = calc_onfrac(self.last_c00_mask)
                self.c00_frac_rel = calc_onfrac(self.last_c00_mask[~last_keep_mask])

                # Reset grad stats...
                self.running_c00_grad_flow = None
                self.running_c01_grad_flow = None
                self.running_c10_grad_flow = None
                self.running_c11_grad_flow = None
                self.running_tot_grad_flow = None
                self.log_grad_flows = True

                self.log_grad_flows_hook = self.weight.register_hook(
                    lambda grad: self._log_grad_flows_hook(grad))

            # Reset keep mask...
            self.last_keep_mask[:] = keep_mask

            # Reset coactivations...
            self.reset_coactivations()
            self.pruning_iterations += 1
            self.learning_iterations = 0

            # ----- END LOG BLOCK -----

    def __call__(self, input_tensor, *args, **kwargs):
        output_tensor = super().__call__(input_tensor, *args, **kwargs)
        return output_tensor


class RandDSConv2d(DSConv2d):
    """
    Module like DSConv2d, but the dynamics of pruning and adding weights are entirely
    random.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coactivations.data[:] = torch.ones_like(self.weight)

    def calc_coactivations(self, input_tensor, output_tensor):

        with torch.no_grad():

            mu_in = input_tensor.mean()
            mu_out = output_tensor.mean()

            self.input_means = np.append(self.input_means, mu_in.to("cpu").item())
            self.output_means = np.append(self.output_means, mu_out.to("cpu").item())

    def progress_connections(self, *args, **kwargs):
        super().progress_connections(*args, **kwargs)
        self.coactivations.data[:] = torch.ones_like(self.weight)


class SparseConv2d(torch.nn.Conv2d, DynamicSparseBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_coactivations(self.weight)

    # def _init_logging_params(self):
    #     pass

    # def progress_connections(self, *args, **kwargs):
    #     pass

    # def calc_coactivations(self, *args, **kwargs):
    #     pass


if __name__ == "__main__":
    pass
