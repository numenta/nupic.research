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


def calc_sparsity(weight):
    return float(torch.Tensor([float((weight == 0).sum()) / np.prod(weight.shape)]))


class _NullConv(torch.nn.Conv2d):
    """
    Exactly a regular conv, but without it's weights initialized. This is a helper class
    to DSConv2d. In some cases, like when initializing 'stacked_conv', we may want to
    manually set the weights and avoid the potentially computationally expensive
    procedure of random initializing them.
    """

    def reset_parameters(self):
        # Don nothing and don't initialize the weights.
        pass


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
        weight_prune_frac=0.00,
        sparsity=0.80,
        prune_dims=[0, 1],
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode,
        )

        # Track learning and pruning_iterations.
        self.learning_iterations = 0
        self.pruning_iterations = 0

        # Set pruning params.
        self.update_nsteps = 100
        self.num_prunable = np.prod([
            d for i, d in enumerate(self.weight.shape) if i not in prune_dims
        ])
        self.nonzero_frac = 1 - sparsity
        self.weight_prune_frac = weight_prune_frac
        self.hebbian_prune_frac = hebbian_prune_frac
        self.k1_weight = max(int((1 - weight_prune_frac) * self.num_prunable), 1)
        self.k1_hebbian = max(int((1 - hebbian_prune_frac) * self.num_prunable), 1)
        self.k2 = max(int((1 - sparsity) * self.num_prunable), 1)
        self.prune_dims = prune_dims.copy()

        # Make the weight matrix sparse.
        sparse_mask = torch.rand(self.weight.shape) < self.nonzero_frac
        with torch.no_grad():
            self.weight.set_(self.weight.data * sparse_mask.float())

        # Set tensors to keep track of coactivations.
        self.register_buffer("connections_tensor", torch.zeros_like(self.weight))
        self.prune_grads_hook = None
        self.prune_mask = torch.ones_like(self.weight)

        # Compute indices that loop over all connections of a channel.
        self.filter_indxs = list(itertools.product(*[
            range(d) for d in self.weight.shape[1:]
        ]))

        # Compute indices that loop over all channels and filters.
        # This will be used to unpack the point-wise comparisons of the coactivations.
        self.connection_indxs = []
        for idx in self.filter_indxs:
            i_ = list(idx)
            self.connection_indxs.extend([
                [c] + i_ for c in range(self.weight.shape[0])
            ])
        self.connection_indxs = list(zip(*self.connection_indxs))

        # Create new conv layer to aid in coactivation calculations.
        self.new_groups = len(self.filter_indxs)
        self.stacked_conv = _NullConv(
            in_channels=self.in_channels * self.new_groups,
            out_channels=self.out_channels * self.new_groups,
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
        self.stacked_conv.weight = torch.nn.Parameter(
            stacked_weights, requires_grad=False)

        # Log kept proportion of persistent connections when prunning.
        self.last_keep_mask = None
        self.kept_frac = 0

    def _get_single_unit_weights(self, c, j, h):
        """
        Constructs and returns conv layer with training disabled and
        all zero weights except along the output channels for unit
        specified as (c, j, h).
        """

        # Construct weight.
        weight = torch.zeros(
            self.weight.shape,
            dtype=torch.float32
        )

        # Set weights to zero except those specified.
        weight[:, c, j, h] = 1
        return weight.to('cuda' if torch.cuda.device_count() > 0 else 'cpu')

    def get_activity_threshold(self, input_tensor, output_tensor):
        return (
            input_tensor.mean() + input_tensor.std(),
            output_tensor.mean() + output_tensor.std()
        )

    def update_connections_tensor(self, input_tensor, output_tensor):

        with torch.no_grad():

            stacked_input = input_tensor.repeat((1, self.new_groups, 1, 1))
            stacked_output = self.stacked_conv(stacked_input)

            a1, a2 = self.get_activity_threshold(input_tensor, output_tensor)
            s1 = torch.abs(stacked_output).gt_(a1)
            s2 = torch.abs(output_tensor).gt_(a2).repeat((1, self.new_groups, 1, 1))

            del stacked_input
            del stacked_output

            H_ = torch.sum(s2.mul(s1), (0, 2, 3,))

            del s1
            del s2

            self.connections_tensor[self.connection_indxs] = H_

    def progress_connections(self):
        """
        Prunes and add connections.
        """

        # Remove old hook to zero the gradients of pruned connections.
        if self.prune_grads_hook is not None:
            self.prune_grads_hook.remove()

        # Decide which connections to prune and which to regrow.
        with torch.no_grad():

            # Get strengths of all connections.
            strengths_hebbian = self.connections_tensor
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
            #    - Reinitialize trailing k2 - k1 connections.
            prune_mask = torch.ones_like(self.weight, dtype=torch.uint8)
            # new_mask = torch.ones_like(self.weight, dtype=torch.uint8)
            keep_mask = torch.ones_like(self.weight, dtype=torch.uint8)
            for prune_frac, k1, k2, strengths in [
                (self.weight_prune_frac, self.k1_weight, self.k2, strengths_weight),
                (self.hebbian_prune_frac, self.k1_hebbian, self.k2, strengths_hebbian),
            ]:

                if prune_frac == 0:
                    continue
                for idx in prune_indxs:

                    # Get top k1'th strength.
                    s = strengths[idx]
                    s_flat = (s.cpu() if s.is_cuda else s).flatten()
                    v1 = np.partition(s_flat, -k1)[-k1]

                    # Set to keep top k1'th connection - prune those below
                    prune_mask[idx] = prune_mask[idx] & (s < v1).byte()

                    # Set to re-init trailing k2 - k1 connections.
                    v2 = np.partition(s_flat, -k2)[-k2]
                    keep_mask[idx] = keep_mask[idx] & (s >= v2).byte()
                    keep_mask[idx] = keep_mask[idx]

            self.weight.data[prune_mask] = 0
            self.prune_grads_hook = self.weight.register_hook(
                lambda grad: grad * keep_mask.type(grad.dtype))

            if self.last_keep_mask is not None:
                kept = (self.last_keep_mask == keep_mask)
                kept = kept[keep_mask == 1]
                self.kept_frac = float(kept.sum() / (keep_mask == 1).sum())
            self.last_keep_mask = keep_mask

            # Reset connection strengths.
            self.connections_tensor = torch.zeros_like(self.weight)
            self.pruning_iterations += 1

    def calc_sparsity(self, tensor=None):
        if tensor is None:
            tensor = self.weight
        return torch.Tensor([float((tensor == 0).sum()) / np.prod(tensor.shape)])

    def __call__(self, input_tensor, *args, **kwargs):
        output_tensor = super().__call__(input_tensor, *args, **kwargs)

        # Update connections strengths.
        if self.learning_iterations % self.update_nsteps == 0:
            self.pruning_iterations += 1
            self.update_connections_tensor(input_tensor, output_tensor)
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
            self.pruning_iterations = 0

            if self.last_keep_mask is not None:
                kept = (self.last_keep_mask == keep_mask)
                kept = kept[keep_mask == 1]
                self.kept_frac = 1 - calc_sparsity(kept)
            self.last_keep_mask = keep_mask

    def update_connections_tensor(self, *args, **kwargs):
        pass


class SparseConv2d(SparseWeights2d):

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

    conv1 = _NullConv(3, 3, 4)
    conv2 = DSConv2d(8, 8, 4)
    conv3 = RandDSConv2d(8, 8, 4)

    assert conv2.calc_sparsity().allclose(
        torch.Tensor([1 - conv2.nonzero_frac]), rtol=0, atol=0.1), \
        "Sparsity {}".format(conv2.calc_sparsity())

    w1 = conv2.weight.clone().detach()
    # print(w1)

    torch.autograd.set_detect_anomaly(True)
    input_tensor = torch.randn(2, 8, 10, 10)
    output_tensor = super(DSConv2d, conv2).__call__(input_tensor)
    conv2.update_connections_tensor(input_tensor, output_tensor)

    conv2.progress_connections()

    w2 = conv2.weight.clone().detach()

    output_tensor.mean().backward()

    grad_sparsity = conv2.calc_sparsity(conv2.weight.grad)
    assert grad_sparsity.allclose(
        torch.Tensor([1 - conv2.nonzero_frac]), rtol=0, atol=0.1), \
        "Sparsity = {} , Expected = {}".format(grad_sparsity, 1 - conv2.nonzero_frac)

    output_tensor = super(DSConv2d, conv3).__call__(input_tensor)
    conv3.update_connections_tensor(input_tensor, output_tensor)
    conv3.progress_connections()

    conv4 = SparseConv2d(0.7, 3, 3, 4)
    optimizer = optim.SGD(conv4.parameters(), lr=0.001, momentum=0.9)
    print(conv2.calc_sparsity(conv4.module.weight))
    print(conv4.module.in_channels)
    print(conv4.weight.shape)

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

    def coactivation(t1, t2, alpha):
        a1, a2 = alpha if hasattr(alpha, '__iter__') else (alpha, alpha)
        s = (torch.abs(t1) > a1) * (torch.abs(t2) > a2)
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
    conv.update_connections_tensor(input_tensor, output_tensor)

    B = output_tensor.shape[0]
    N_out = output_tensor.shape[2]
    M_out = output_tensor.shape[3]
    C_in = conv.weight.shape[1]
    C_out = conv.weight.shape[0]
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    alpha = conv.get_activity_threshold(input_tensor, output_tensor)

    def calc_connections_tensor():
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

                            if coactivation(unit_2, unit_1, alpha):
                                H[c_out, c_fl, n_fl, m_fl] += 1
        return H

    H = calc_connections_tensor()
    assert conv.connections_tensor.allclose(H, atol=0, rtol=0)

    # print(H)
