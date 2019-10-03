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

from nupic.torch.modules import Flatten, KWinners, KWinners2d

from .layers import DSLinear, init_coactivation_tracking

# ------------------------------------------------------------------------------------
# DynamicSparse Linear Block
# ------------------------------------------------------------------------------------


class DSLinearBlock(nn.Sequential):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        batch_norm=None,
        batch_norm_affine=True,
        dropout=None,
        activation_func=None,
    ):

        # Clarifications on batch norm position at the linear block:
        # - bn before relu at original paper
        # - bn after relu in recent work
        # (see fchollet @ https://github.com/keras-team/keras/issues/1802)
        # - however, if applied after RELU or kWinners, breaks sparsity
        layers = [DSLinear(in_features, out_features, bias=bias)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features, affine=batch_norm_affine))
        if activation_func:
            layers.append(activation_func)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        super().__init__(*layers)

        # Transfer forward hook.
        dslayer = self[0]
        forward_hook = dslayer.forward_hook
        self.register_forward_hook(
            lambda module, in_, out_: forward_hook(dslayer, in_, out_)
        )
        dslayer.forward_hook_handle.remove()

    @property
    def weight(self):
        """
        Return weight of linear layer - needed for introspective networks.
        """
        return self[0].weight

    def forward(self, input_tensor):
        output_tensor = super().forward(input_tensor)
        return output_tensor


# ------------
# MLP Network
# ------------


class HebbianNetwork(nn.Module):
    @property
    def coactivations(self):
        if self._track_coactivations:
            return [m.coactivations.t() for m in self.dynamic_sparse_modules]
        else:
            return []

    def forward(self, x):
        if "features" in self._modules:
            x = self.features(x)
        x = self.classifier(x)
        return x

    def init_hebbian(self):
        self._track_coactivations = True
        self.apply(init_coactivation_tracking)


class MLPHeb(HebbianNetwork):
    """Simple 3 hidden layers + output MLP"""

    def __init__(self, config=None):
        super().__init__()

        defaults = dict(
            device="cpu",
            input_size=784,
            num_classes=10,
            hidden_sizes=[100, 100, 100],
            percent_on_k_winner=[1.0, 1.0, 1.0],
            boost_strength=[1.4, 1.4, 1.4],
            boost_strength_factor=[0.7, 0.7, 0.7],
            batch_norm=False,
            dropout=False,
            bias=True,
            k_inference_factor=1.0,
        )
        assert (
            config is None or "use_kwinners" not in config
        ), "use_kwinners is deprecated"

        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        # decide which actiovation function to use
        self.activation_funcs = []
        for layer, hidden_size in enumerate(self.hidden_sizes):
            if self.percent_on_k_winner[layer] < 0.5:
                self.activation_funcs.append(
                    KWinners(
                        hidden_size,
                        percent_on=self.percent_on_k_winner[layer],
                        boost_strength=self.boost_strength[layer],
                        boost_strength_factor=self.boost_strength_factor[layer],
                        k_inference_factor=self.k_inference_factor,
                    )
                )
            else:
                self.activation_funcs.append(nn.ReLU())

        # Construct layers.
        layers = []
        kwargs = dict(bias=self.bias, batch_norm=self.batch_norm, dropout=self.dropout)
        # Flatten image.
        layers = [nn.Flatten()]
        # Add the first layer
        layers.append(
            DSLinearBlock(
                self.input_size,
                self.hidden_sizes[0],
                activation_func=self.activation_funcs[0],
                **kwargs,
            )
        )
        # Add hidden layers.
        for i in range(1, len(self.hidden_sizes)):
            layers.append(
                DSLinearBlock(
                    self.hidden_sizes[i - 1],
                    self.hidden_sizes[i],
                    activation_func=self.activation_funcs[i],
                    **kwargs,
                )
            )
        # Add last layer.
        layers.append(
            DSLinearBlock(self.hidden_sizes[-1], self.num_classes, bias=self.bias)
        )

        # Create the classifier.
        self.dynamic_sparse_modules = [l[0] for l in layers[1:]]
        self.classifier = nn.Sequential(*layers)

        # Initialize attr to decide whether to update coactivations during learning.
        self._track_coactivations = False  # Off by default.


class GSCHeb(HebbianNetwork):
    """Simple 3 hidden layers + output MLP"""

    def __init__(self, config=None):
        super().__init__()

        defaults = dict(
            device="cpu",
            input_size=1024,
            num_classes=12,
            boost_strength=[1.5, 1.5, 1.5],
            boost_strength_factor=[0.9, 0.9, 0.9],
            duty_cycle_period=1000,
            k_inference_factor=1.5,
            percent_on_k_winner=[0.095, 0.125, 0.1],
            hidden_neurons_conv=[64, 64],
            hidden_neurons_fc=1000,
            batch_norm=True,
            dropout=False,
            bias=True,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        kwargs = dict(bias=self.bias, batch_norm=self.batch_norm, dropout=self.dropout)

        # decide which actiovation function to use for conv
        self.activation_funcs = []
        for layer, hidden_size in enumerate(self.hidden_neurons_conv):
            if self.percent_on_k_winner[layer] < 0.5:
                self.activation_funcs.append(
                    KWinners2d(
                        hidden_size,
                        percent_on=self.percent_on_k_winner[layer],
                        boost_strength=self.boost_strength[layer],
                        boost_strength_factor=self.boost_strength_factor[layer],
                        k_inference_factor=self.k_inference_factor,
                    )
                )
            else:
                self.activation_funcs.append(nn.ReLU())

        # decide which activvation to use for linear
        if self.percent_on_k_winner[-1] < 0.5:
            linear_activation = KWinners(
                self.hidden_neurons_fc,
                percent_on=self.percent_on_k_winner[-1],
                boost_strength=self.boost_strength[-1],
                boost_strength_factor=self.boost_strength_factor[-1],
                k_inference_factor=self.k_inference_factor,
            )
        else:
            linear_activation = nn.ReLU()

        # linear layers
        conv_layers = [
            # 28x28 -> 14x14
            *self._conv_block(1, self.hidden_neurons_conv[0], self.activation_funcs[0]),
            # 10x10 -> 5x5
            *self._conv_block(
                self.hidden_neurons_conv[0],
                self.hidden_neurons_conv[1],
                self.activation_funcs[1],
            ),
            Flatten(),
        ]
        linear_layers = [
            DSLinearBlock(
                self.hidden_neurons_conv[1] * 25,
                self.hidden_neurons_fc,
                activation_func=linear_activation,
                batch_norm_affine=False,
                **kwargs,
            ),
            DSLinearBlock(self.hidden_neurons_fc, self.num_classes),
            # nn.LogSoftmax(dim=1)
        ]

        self.features = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(*linear_layers)

    def _conv_block(self, fin, fout, activation_func):
        block = [
            nn.Conv2d(fin, fout, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(fout, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation_func,
        ]
        return block


def gsc_heb_small(config):
    config["hidden_neurons_conv"] = [12, 12]
    config["hidden_neurons_fc"] = 207
    return GSCHeb(config)
