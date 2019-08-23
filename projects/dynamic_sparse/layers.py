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

import torch
import numpy as np

from nupic.torch.modules import SparseWeights2d


# ------------------
# Utilities
# ------------------


def calc_sparsity(weight):
    return float(torch.Tensor([float((weight == 0).sum()) / np.prod(weight.shape)]))


def where(condition):
    """
    Similar to torch.where, but the result is packed into tuples and its
    shape is transposed via zip.
    Per the last statement, here's an example:
        zip((0, 1, 2), (0, 1, 2)) -> ((0, 0), (1, 1), (2, 2))

    :param condition: some condition in the form of a mask
    """
    idx = [tuple(i_.to('cpu').numpy()) for i_ in torch.where(condition)]
    idx = tuple(zip(*idx))
    return idx


def break_ties(mask, num_remain):
    """
    Break ties in a mask - whether between zeros or ones.
    This function ensures exactly a num_remain number of
    non-zeros will be present in the output mask

    :param mask: mask of zeros and ones (or equivalents) - overwritten in place
    :param num_remain: number of desired non-zeros
    """

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

    def where(self, condition):
        """
        Similar to torch.where, but the result is packed into tuples and its
        shape is transposed via zip.
        Per the last statement, here's an example:
            zip((0, 1, 2), (0, 1, 2)) -> ((0, 0), (1, 1), (2, 2))

        :param condition: some condition in the form of a mask
        """
        idx = [tuple(i_.to('cpu').numpy()) for i_ in torch.where(condition)]
        idx = tuple(zip(*idx))
        return idx

    def break_ties(self, mask, num_remain):
        """
        Break ties in a mask - whether between zeros or ones.
        This function ensures exactly a num_remain number of
        non-zeros will be present in the output mask

        :param mask: mask of zeros and ones (or equivalents) - overwritten in place
        :param num_remain: number of desired non-zeros
        """

        idx_ones = self.where(mask == 1)
        num_ones = len(idx_ones)
        if num_ones > 0:
            idx_ones = tuple(
                idx_ones[i1_]
                for i1_ in np.random.choice(
                    range(num_ones), min(num_remain, num_ones), replace=False)
            )

        if num_ones < num_remain:
            num_fill = num_remain - num_ones
            idx_zeros = self.where(mask == 0)
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


class DSConv2d(torch.nn.Conv2d):

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
        padding_mode='zeros',
        hebbian_prune_frac=0.90,
        magnitude_prune_frac=0.00,
        sparsity=0.80,
        prune_dims=[0, 1],
        update_nsteps=100,
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
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
        )

        # Track learning_iterations (number of forward passes during training)
        # and pruning_iterations (number of times connections are pruned).
        self.learning_iterations = 0
        self.pruning_iterations = 0

        # Set pruning params.
        #   num_connections - number of connections along each prune_dim
        #   k1_weight - the number of connections to keep by magnitude pruning
        #   k1_hebbian - the number of connections to keep by hebbian pruning
        #   k2 - the number of connections to keep non-zero
        self.update_nsteps = update_nsteps
        self.num_connections = np.prod([
            d for i, d in enumerate(self.weight.shape) if i not in prune_dims
        ])
        self.nonzero_frac = 1 - sparsity
        self.magnitude_prune_frac = magnitude_prune_frac
        self.hebbian_prune_frac = hebbian_prune_frac
        self.k1_weight = max(int((1 - magnitude_prune_frac) * self.num_connections), 1)
        self.k1_hebbian = max(int((1 - hebbian_prune_frac) * self.num_connections), 1)
        self.k2 = max(int((1 - sparsity) * self.num_connections), 1)
        self.prune_dims = list(prune_dims).copy()

        # Make the weight matrix sparse.
        sparse_mask = torch.rand(self.weight.shape) < self.nonzero_frac
        with torch.no_grad():
            self.weight.set_(self.weight.data * sparse_mask.float())
            # Log sparsity
            self.weight_sparsity = calc_sparsity(self.weight)
        self.prune_grads_hook = self.weight.register_hook(
            lambda grad: grad * sparse_mask.type(grad.dtype).to(grad.device))

        # Set tensors to keep track of coactivations.
        self.register_buffer("coactivations", torch.zeros_like(self.weight))
        self.prune_mask = torch.ones_like(self.weight)

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

        # Attributes for logging purposes:
        self.last_keep_mask = None
        self.kept_frac = 0
        self.prune_mask_sparsity = -1
        self.keep_mask_sparsity = -1
        self.last_coactivations = torch.zeros_like(self.coactivations)

    def _get_single_unit_weights(self, c, j, h):
        """
        Constructs and returns conv layer with training disabled and
        all zero weights except along the output channels for unit
        specified as (c, j, h).
        """

        # Construct weight.
        weight = torch.zeros(
            1, *self.weight.shape[1:],
            dtype=torch.float32
        )

        # Set weights to zero except those specified.
        weight[0, c, j, h] = 1

        # TODO: The 'to' here may not be needed.
        return weight.to('cuda' if torch.cuda.device_count() > 0 else 'cpu')

    def get_activity_threshold(self, input_tensor, output_tensor):
        """
        Returns tuple of input and output activity thresholds.
        """
        return (
            input_tensor.std() / 2,
            output_tensor.std() / 2
        )

    def update_coactivations(self, input_tensor, output_tensor):
        """
        This function updates self.coactivations.
        The computation is highly vectorized and unfortunately quite opaque.
        Generally, two units, say unit_in and unit_out, coactivate if
            1. unit_in is in the receptive field of unit_out
            2. (unit_in  - mean_input ) > input_activity_threshold
            3. (unit_out - mean_output) > output_activity_threshold
        """

        with torch.no_grad():

            grouped_input = input_tensor.repeat((1, self.new_groups, 1, 1))
            grouped_input = self.grouped_conv(grouped_input).repeat(
                (1, self.out_channels, 1, 1))

            mu_in = input_tensor.mean()
            mu_out = output_tensor.mean()

            a1, a2 = self.get_activity_threshold(input_tensor, output_tensor)
            s1 = torch.abs(grouped_input - mu_in).gt_(a1)
            s2 = torch.abs(output_tensor - mu_out).gt_(a2)[:, self.perm_indices, ...]

            # Save space on device
            del mu_in
            del mu_out
            del a1
            del a2
            del grouped_input

            H_ = torch.sum(s2.mul(s1), (0, 2, 3,))

            self.H_ = H_

            del s1
            del s2

            self.coactivations[self.connection_indxs] += H_

            del H_

    def progress_connections(self):
        """
        Prunes and add connections.
        """

        # Remove old hook to zero the gradients of pruned connections.
        self.prune_grads_hook.remove()
        self.last_coactivations = self.coactivations.clone().detach()

        # Decide which connections to prune and which to regrow.
        with torch.no_grad():

            # Get strengths of all connections.
            strengths_hebbian = self.coactivations
            strengths_weight = self.weight.data.clone().detach().abs()

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
            prune_mask = torch.ones_like(self.weight, dtype=torch.bool)
            keep_mask = torch.ones_like(self.weight, dtype=torch.bool)
            for prune_frac, k1, k2, strengths in [
                (self.magnitude_prune_frac, self.k1_weight, self.k2, strengths_weight),
                (self.hebbian_prune_frac, self.k1_hebbian, self.k2, strengths_hebbian),
            ]:

                if prune_frac == 0:
                    continue
                for idx in prune_indxs:

                    # Get top k1'th coactivation.
                    s = strengths[idx]
                    s_flat = (s.cpu() if s.is_cuda else s).flatten()
                    v1 = np.partition(s_flat, -k1)[-k1]

                    # Set to keep top k1'th connection - prune those below.
                    prune_mask[idx] = prune_mask[idx] & (s < v1)

                    # Set to allow grad flow to top k2 connections.
                    v2 = np.partition(s_flat, -k2)[-k2]
                    keep_mask[idx] = keep_mask[idx] & (s >= v2)

            # Resolve ties.
            prune_mask = break_ties(
                prune_mask, num_remain=(self.num_connections - k1))
            keep_mask = break_ties(
                keep_mask, num_remain=k2)

            # Logging
            self.prune_mask_sparsity = calc_sparsity(prune_mask)
            self.keep_mask_sparsity = calc_sparsity(keep_mask)

            # Zero weights and prevent gradient flow.
            self.weight.data[prune_mask] = 0
            self.weight_sparsity = calc_sparsity(self.weight.data)
            self.prune_grads_hook = self.weight.register_hook(
                lambda grad: grad * keep_mask.type(grad.dtype))

            # Logging
            if self.last_keep_mask is not None:
                kept = (self.last_keep_mask == keep_mask)
                kept = kept[keep_mask == 1]
                self.kept_frac = float(kept.sum() / (self.last_keep_mask == 1).sum())
            self.last_keep_mask = keep_mask

            # Reset coactivations.
            self.coactivations.data[:] = torch.zeros_like(self.weight)
            self.pruning_iterations += 1

    def __call__(self, input_tensor, *args, **kwargs):
        output_tensor = super().__call__(input_tensor, *args, **kwargs)

        # Update connections strengths.
        if self.training and self.learning_iterations % self.update_nsteps == 0:
            self.update_coactivations(input_tensor, output_tensor)
        if self.training:
            self.learning_iterations += 1
        return output_tensor


class RandDSConv2d(DSConv2d):
    """
    Module like DSConv2d, but the dynamics of pruning and adding weights are entirely
    random.
    """

    def progress_connections(self, *args, **kwargs):

        if self.prune_grads_hook is not None:
            self.prune_grads_hook.remove()

        with torch.no_grad():

            keep_mask = torch.rand(self.weight.shape) < self.nonzero_frac
            self.weight[~keep_mask] = 0
            self.prune_grads_hook = self.weight.register_hook(
                lambda grad: grad * keep_mask.type(grad.dtype).to(grad.device))
            self.pruning_iterations += 1

            if self.last_keep_mask is not None:
                kept = (self.last_keep_mask == keep_mask)
                kept = kept[keep_mask == 1]
                self.kept_frac = 1 - calc_sparsity(kept)
            self.last_keep_mask = keep_mask

    def update_coactivations(self, *args, **kwargs):
        pass


class SparseConv2d(SparseWeights2d):
    """
    Conv layer with static sparsity.
    """

    def __init__(self, sparsity, *args, **kwargs):

        conv = torch.nn.Conv2d(*args, **kwargs)
        super(SparseConv2d, self).__init__(conv, 1 - sparsity)
        self.weight = self.module.weight

        # Zero out random weights.
        with torch.no_grad():
            zero_idx = (self.zero_weights[0], self.zero_weights[1])
            self.weight.view(self.module.out_channels, -1)[zero_idx] = 0.0

        # Block gradient flow to pruned connections.
        self.prune_grads_hook = self.weight.register_hook(self.zero_gradients)

    def zero_gradients(self, grad):
        zero_idx = (self.zero_weights[0], self.zero_weights[1])
        grad.view(self.module.out_channels, -1)[zero_idx] = 0.0
        return grad

    def forward(self, x):
        return self.module.forward(x)

    def rezero_weights(self):
        pass


if __name__ == '__main__':

    # --------------------------------
    # Exercise basic functionalities.
    # --------------------------------
    import torch.optim as optim

    torch.manual_seed(42)

    if True:

        conv1 = _NullConv(3, 3, 4)
        conv2 = DSConv2d(8, 8, 4, prune_dims=[])
        conv3 = RandDSConv2d(8, 8, 4)

        assert torch.tensor([calc_sparsity(conv2.weight)]).allclose(
            torch.Tensor([1 - conv2.nonzero_frac]), rtol=0, atol=0.1), \
            "Sparsity {}".format(conv2.calc_sparsity())

        torch.autograd.set_detect_anomaly(True)
        optimizer = optim.SGD(conv2.parameters(), lr=0.001, momentum=0.9)
        input_tensor = torch.randn(2, 8, 10, 10)
        output_tensor = super(DSConv2d, conv2).__call__(input_tensor)
        conv2.update_coactivations(input_tensor, output_tensor)
        assert calc_sparsity(conv2.coactivations) != 1

        conv2.progress_connections()

        w2 = conv2.weight.clone().detach()

        output_tensor.mean().backward()
        optimizer.step()

        grad_sparsity = torch.tensor([calc_sparsity(conv2.weight.grad)])
        assert grad_sparsity.allclose(
            torch.Tensor([1 - conv2.nonzero_frac]), rtol=0, atol=0.1), \
            "Sparsity = {} , Expected = {}".format(
                grad_sparsity, 1 - conv2.nonzero_frac)

        conv2.update_coactivations(input_tensor, output_tensor)
        assert calc_sparsity(conv2.coactivations) != 1

        output_tensor = super(DSConv2d, conv3).__call__(input_tensor)
        conv3.update_coactivations(input_tensor, output_tensor)
        conv3.progress_connections()

        conv4 = SparseConv2d(0.7, 3, 3, 4)
        optimizer = optim.SGD(conv4.parameters(), lr=0.001, momentum=0.9)

        input_tensor = torch.randn(4, 3, 10, 10)
        output_tensor = conv4(input_tensor)

        grad = output_tensor.mean().backward()
        optimizer.step()

        input_tensor = torch.randn(4, 3, 10, 10)
        output_tensor = conv4(input_tensor)

        grad = output_tensor.mean().backward()
        optimizer.step()

        sparsity = calc_sparsity(conv4.weight)
        assert np.isclose(sparsity, 0.7, rtol=0, atol=0.01), \
            "Expected sparsity {}, observed {}".format(0.7, sparsity)

    # ---------------------------------------------
    # Validate behavior against brute force method.
    # ---------------------------------------------

    subtract_mean_activations = True

    def coactivation(t1, t2, alpha, mean_activations):
        """
        :param t1: input unit
        :param t1: output unit
        :param alpha: activity threshold
        :param mean_activations: average activations of input and output
        """
        a1, a2 = alpha if hasattr(alpha, '__iter__') else (alpha, alpha)

        if subtract_mean_activations:
            t1, t2 = (t1 - mean_activations[0], t2 - mean_activations[1])

        s = torch.abs(t1).gt(a1) * torch.abs(t2).gt(a2)
        return s

    def get_indeces_of_input_and_filter(
            n, m, in_channels, kernel_size, padding, stride):
        """
        Assumes dilation=1 and grouping=1
        """

        k1, k2 = kernel_size
        p1, p2 = padding
        s1, s2 = stride

        i1, i2 = (0, 0)

        i1 -= p1
        i2 -= p2

        i1 += n * s1
        i2 += m * s2

        indxs = []
        for c_in in range(in_channels):
            for n_k1 in range(k1):
                for m_k2 in range(k2):
                    filter_indx = (c_in, n_k1, m_k2)
                    input_indx = (c_in, i1 + n_k1, i2 + m_k2)
                    indxs.append((input_indx, filter_indx))

        return indxs

    if True:

        in_channels = 16
        out_channels = 24
        kernel_size = (2, 2)
        stride = (1, 1)
        padding = 0
        conv = DSConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1
        )

        input_tensor = torch.randn(2, in_channels, 5, 3)
        output_tensor = super(DSConv2d, conv).__call__(input_tensor)
        conv.update_coactivations(input_tensor, output_tensor)
        mean_activations = (input_tensor.mean(), output_tensor.mean())

        B = output_tensor.shape[0]
        N_out = output_tensor.shape[2]
        M_out = output_tensor.shape[3]
        C_in = conv.weight.shape[1]
        C_out = conv.weight.shape[0]
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        alpha = conv.get_activity_threshold(input_tensor, output_tensor)

        def calc_coactivations():
            H = torch.zeros_like(conv.weight)
            for b in range(B):
                for c_out in range(C_out):
                    for n_out in range(N_out):
                        for m_out in range(M_out):
                            unit_1 = output_tensor[b, c_out, n_out, m_out]
                            indxs = get_indeces_of_input_and_filter(
                                n_out, m_out, in_channels, kernel_size, padding, stride)

                            for input_indx, filter_indx in indxs:
                                c_in, n_in, m_in = input_indx
                                c_fl, n_fl, m_fl = filter_indx
                                unit_2 = input_tensor[b, c_in, n_in, m_in]

                                if coactivation(unit_2,
                                                unit_1, alpha, mean_activations):
                                    H[c_out, c_fl, n_fl, m_fl] += 1
            return H

        H = calc_coactivations()
        assert conv.coactivations.allclose(H, atol=0, rtol=0)

        print('DONE.')
