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


class DSConv2d(torch.nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_iterations = 0
        self.c = 0

        self.activity_threshold = 0.5
        self.k1 = max(int(0.1 * np.prod(self.weight.shape[2:])), 1)
        self.k2 = max(int(0.15 * np.prod(self.weight.shape[2:])), 1)
        self.prune_dims = [0, 1]

        self.connections_tensor = torch.zeros_like(self.weight)
        self.prune_mask = torch.ones_like(self.weight)

        # Compute inidices that loop over all connections of a channel.
        filter_indxs = list(itertools.product(*[
            range(d) for d in self.weight.shape[1:]
        ]))

        # Compute indeces that loop over all channels and filters.
        # This will be used to unpack the pointwise comparisons of the output.
        self.connection_indxs = []
        for idx in filter_indxs:
            i_ = list(idx)
            self.connection_indxs.extend([
                [c] + i_ for c in range(self.weight.shape[0])
            ])
        self.connection_indxs = list(zip(*self.connection_indxs))

        # Create new conv layer that groups it's input and output.
        self.new_groups = len(filter_indxs)
        self.stacked_conv = torch.nn.Conv2d(
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
            self.get_single_unit_weights(
                self.weight.shape,
                c, j, h,
            )
            for c, j, h in filter_indxs
        ]
        with torch.no_grad():
            self.stacked_conv.weight.set_(torch.cat(single_unit_weights, dim=0))

    def get_single_unit_weights(self, shape, c, j, h):
        """
        Constructs and returns conv layer with training disabled and
        all zero weights except along the output channels for unit
        specified as (c, j, h).
        """

        # Construct weight.
        weight = torch.zeros(self.weight.shape)

        # Set weights to zero except those specified.
        weight[:, c, j, h] = 1

        return weight

    def update_connections_tensor(self, input_tensor, output_tensor):

        with torch.no_grad():

            stacked_input = input_tensor.repeat((1, self.new_groups, 1, 1))
            stacked_output = self.stacked_conv(stacked_input)

            s1 = torch.abs(stacked_output).gt_(0.5)
            s2 = torch.abs(output_tensor).gt_(0.5).repeat((1, self.new_groups, 1, 1))
            H_ = torch.sum(s2.mul(s1), (0, 2, 3,))

            self.connections_tensor[self.connection_indxs] = H_

    def progress_connections(self):
        """
        Prunes and add connections.
        """

        with torch.no_grad():

            # Get strengths of all connections.
            strengths = self.connections_tensor.numpy()
            shape = strengths.shape

            # Determine all combinations of prune dimensions
            all_dims = range(len(shape))
            prune_indxs = [
                range(shape[d]) if d in self.prune_dims else [slice(None)]
                for d in all_dims
            ]
            prune_indxs = itertools.product(*prune_indxs)

            # Along all combinations of prune dimensions:
            #    - Keep strongest k1 connections
            #    - Reinitilize trailing k2 - k1 connections.
            k1 = self.k1
            k2 = self.k2
            for idx in prune_indxs:

                # Get top k1'th strength.
                s = strengths[idx].flatten()
                v1 = np.partition(s, -k1)[-k1]

                # Keep top k1'th connection - prune those below
                c = self.weight[idx].flatten()
                prune_mask = (s < v1).astype(np.uint8)
                c[prune_mask] = 0

                # Get trailing k2 - k1 connections.
                v2 = np.partition(s, -k2)[-k2]
                new_mask = (s > v2) & prune_mask

                # Reinitialized trailing k2 - k1 connections.
                # Note: [None, :] is added here as kaiming_uniform requires a 2d tensor
                if len(c[new_mask]) > 0:
                    torch.nn.init.kaiming_uniform_(c[new_mask][None, :])

                # Reshape connections and update the weight.
                self.weight[idx] = c.reshape(self.weight[idx].shape)
                self.prune_mask = prune_mask

            # Reset connection strengths.
            self.connections_tensor = torch.zeros_like(self.weight)

    def prune_randomly(self):

        with torch.no_grad():

            prune_mask = torch.rand(self.weight.shape) < 0.85
            self.weight[prune_mask] = 0

            # Reinitialize those that are zero.
            keep_mask = ~prune_mask
            new_mask = (self.weight == 0) & keep_mask
            new_weights = self.weight[new_mask]
            if len(new_weights) > 0:
                torch.nn.init.kaiming_uniform_(new_weights[None, :])
                self.weight[new_mask] = new_weights

    def __call__(self, input_tensor, *args, **kwargs):
        output_tensor = super().__call__(input_tensor, *args, **kwargs)
        if self.learning_iterations % 100 == 0:
            self.c += 1
            self.update_connections_tensor(input_tensor, output_tensor)
        self.learning_iterations += 1
