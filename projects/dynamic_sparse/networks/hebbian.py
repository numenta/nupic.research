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


class MLP(nn.Module):
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
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        # decide which actiovation function to use
        if self.use_kwinners:
            self.activation_func = self._kwinners
        else:
            self.activation_func = lambda _: nn.ReLU()

        # create the layers
        layers = [
            *self._linear_block(self.input_size, self.hidden_sizes[0]),
            *self._linear_block(self.hidden_sizes[0], self.hidden_sizes[1]),
            *self._linear_block(self.hidden_sizes[1], self.hidden_sizes[2]),
            nn.Linear(self.hidden_sizes[2], self.num_classes),
        ]
        self.classifier = nn.Sequential(*layers)

    def _linear_block(self, a, b):
        block = [nn.Linear(a, b), self.activation_func(b)]
        if self.batch_norm:
            block.append(nn.BatchNorm1d(b))
        if self.dropout:
            block.append(nn.Dropout(p=self.dropout))
        return block

    def _kwinners(self, num_units):
        return KWinners(
            n=num_units, percent_on=0.3, boost_strength=1.4, boost_strength_factor=0.7
        )

    def forward(self, x):
        # need to flatten input before forward pass
        return self.classifier(x.view(-1, self.input_size))

    def alternative_forward(self, x):
        """Replace forward function by this to visualize activations"""
        # need to flatten before forward pass
        x = x.view(-1, self.input_size)
        for layer in self.classifier:
            # apply the transformation
            x = layer(x)
            # do something with the activation
            print(torch.mean(x).item())

        return x


class MLPHeb(MLP):
    """Replace forward layer to add hebbian learning"""

    def __init__(self, config=None):
        super().__init__(config)
        self.correlations = []

    def _has_activation(self, idx, layer):
        return (
            idx == len(self.classifier) - 1
            or isinstance(layer, nn.ReLU)
            or isinstance(layer, KWinners)
        )

    def forward(self, x):
        """A faster and approximate way to track correlations"""
        x = x.view(-1, self.input_size)  # resiaze if needed, eg mnist
        prev_act = (x > 0).detach().float()
        idx_activation = 0
        for idx_layer, layer in enumerate(self.classifier):
            # do the forward calculation normally
            x = layer(x)
            if self.hebbian_learning:
                n_samples = x.shape[0]
                if self._has_activation(idx_layer, layer):
                    with torch.no_grad():
                        curr_act = (x > 0).detach().float()
                        # add outer product to the correlations, per sample
                        for s in range(n_samples):
                            outer = torch.ger(prev_act[s], curr_act[s])
                            if idx_activation + 1 > len(self.correlations):
                                self.correlations.append(outer)
                            else:
                                self.correlations[idx_activation] += outer
                        # reassigning to the next
                        prev_act = curr_act
                        # move to next activation
                        idx_activation += 1

        return x
