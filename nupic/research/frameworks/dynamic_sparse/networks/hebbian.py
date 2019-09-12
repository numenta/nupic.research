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

import torch
from torch import nn

from nupic.torch.modules import KWinners


class MLPHeb(nn.Module):
    """Simple 3 hidden layers + output MLP"""

    def __init__(self, config=None):
        super().__init__()

        defaults = dict(
            device="cpu",
            input_size=784,
            num_classes=10,
            hidden_sizes=[100, 100, 100],
            batch_norm=False,
            dropout=False,
            use_kwinners=False,
            hebbian_learning=False,
            bias=True,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        # decide which actiovation function to use
        if self.use_kwinners:
            self.activation_func = self._kwinners
        else:
            self.activation_func = lambda _: nn.ReLU()

        layers = []
        # add the first layer
        layers.extend(self._linear_block(self.input_size, self.hidden_sizes[0]))
        # all hidden layers
        for i in range(1, len(self.hidden_sizes)):
            layers.extend(
                self._linear_block(self.hidden_sizes[i - 1], self.hidden_sizes[i])
            )
        # last layer
        layers.append(
            nn.Linear(self.hidden_sizes[-1], self.num_classes, bias=self.bias)
        )

        # create the layers
        self.classifier = nn.Sequential(*layers)

    def _linear_block(self, a, b):
        """
        Clarifications on batch norm position at the linear block:
        - bn before relu at original paper
        - bn after relu in recent work
        (see fchollet @ https://github.com/keras-team/keras/issues/1802)
        - however, if applied after RELU or kWinners, breaks sparsity
        """
        block = [nn.Linear(a, b, bias=self.bias)]
        if self.batch_norm:
            block.append(nn.BatchNorm1d(b))
        block.append(self.activation_func(b))
        if self.dropout:
            block.append(nn.Dropout(p=self.dropout))
        return block

    def _kwinners(self, num_units):
        return KWinners(
            n=num_units, percent_on=0.25, boost_strength=1.4, boost_strength_factor=0.7
        )

    def forward(self, x):
        # need to flatten input before forward pass
        return self.classifier(x.view(-1, self.input_size))

    def init_hebbian(self):
        self.coactivations = []
        self.forward = self.forward_with_coactivations

    def _has_activation(self, idx, layer):
        return (
            idx == len(self.classifier) - 1
            or isinstance(layer, nn.ReLU)
            or isinstance(layer, KWinners)
        )

    def forward_with_coactivations(self, x):
        """A faster and approximate way to track correlations"""
        x = x.view(-1, self.input_size)  # resiaze if needed, eg mnist
        prev_act = (x > 0).detach().float()
        idx_activation = 0
        for idx_layer, layer in enumerate(self.classifier):
            # do the forward calculation normally
            x = layer(x)
            n_samples = x.shape[0]
            if self._has_activation(idx_layer, layer):
                with torch.no_grad():
                    curr_act = (x > 0).detach().float()
                    # add outer product to the coactivations, per sample
                    for s in range(n_samples):
                        outer = torch.ger(prev_act[s], curr_act[s])
                        if idx_activation + 1 > len(self.coactivations):
                            self.coactivations.append(outer)
                        else:
                            self.coactivations[idx_activation] += outer
                    # reassigning to the next
                    prev_act = curr_act
                    # move to next activation
                    idx_activation += 1

        return x
