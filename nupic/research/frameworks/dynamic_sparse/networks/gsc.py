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

from nupic.torch.models.sparse_cnn import GSCSparseCNN, MNISTSparseCNN
from nupic.torch.modules import Flatten, KWinners, KWinners2d, SparseWeights

from .layers import DSConv2d
from .main import VGG19
from .utils import get_dynamic_sparse_modules, make_dsnn, squash_layers, swap_layers

# --------------
# GSC Networks
# --------------


class GSCHeb(nn.Module):
    """LeNet like CNN used for GSC in how so dense paper."""

    def __init__(self, config=None):
        super(GSCHeb, self).__init__()

        defaults = dict(
            input_size=1024,
            num_classes=12,
            boost_strength=1.5,
            boost_strength_factor=0.9,
            k_inference_factor=1.5,
            duty_cycle_period=1000,
            use_kwinners=True,
            hidden_neurons_fc=1000,
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        if self.model == "DSNNMixedHeb":
            self.hebbian_learning = True
        else:
            self.hebbian_learning = False

        # hidden layers
        conv_layers = [
            *self._conv_block(1, 64, percent_on=0.095),  # 28x28 -> 14x14
            *self._conv_block(64, 64, percent_on=0.125),  # 10x10 -> 5x5
        ]
        linear_layers = [
            Flatten(),
            # *self._linear_block(1600, 1500, percent_on= 0.067),
            *self._linear_block(1600, self.hidden_neurons_fc, percent_on=0.1),
            nn.Linear(self.hidden_neurons_fc, self.num_classes),
        ]

        # classifier (*redundancy on layers to facilitate traversing)
        self.layers = conv_layers + linear_layers
        self.features = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(*linear_layers)

        # track correlations
        self.correlations = []

    def _conv_block(self, fin, fout, percent_on=0.1):
        block = [
            nn.Conv2d(fin, fout, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(fout, affine=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._activation_func(fout, percent_on),
        ]
        # if not self.use_kwinners:
        #     block.append(nn.Dropout(p=0.5))
        return block

    def _linear_block(self, fin, fout, percent_on=0.1):
        block = [
            nn.Linear(fin, fout),
            nn.BatchNorm1d(fout, affine=False),
            self._activation_func(fout, percent_on, twod=False),
        ]
        # if not self.use_kwinners:
        #     block.append(nn.Dropout(p=0.5))
        return block

    def _activation_func(self, fout, percent_on, twod=True):
        if self.use_kwinners:
            if twod:
                activation_func = KWinners2d
            else:
                activation_func = KWinners
            return activation_func(
                fout,
                percent_on=percent_on,
                boost_strength=self.boost_strength,
                boost_strength_factor=self.boost_strength_factor,
                k_inference_factor=self.k_inference_factor,
                duty_cycle_period=self.duty_cycle_period,
            )
        else:
            return nn.ReLU()

    def _has_activation(self, idx, layer):
        return idx == len(self.layers) - 1 or isinstance(layer, KWinners)

    def forward(self, x):
        """A faster and approximate way to track correlations"""
        # Forward pass through conv layers
        for layer in self.features:
            x = layer(x)

        # Forward pass through linear layers
        idx_activation = 0
        for layer in self.classifier:
            # do the forward calculation normally
            x = layer(x)
            if self.hebbian_learning:
                if isinstance(layer, Flatten):
                    prev_act = (x > 0).detach().float()
                if isinstance(layer, KWinners):
                    n_samples = x.shape[0]
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


# make a conv heb just by replacing the conv layers by special DSNN Conv layers
def gsc_conv_heb(config):

    net = make_dsnn(GSCHeb(config), config)
    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


def gsc_conv_only_heb(config):
    network = make_dsnn(GSCHeb(config), config)

    # replace the forward function to not apply regular convolution
    def forward(self, x):
        return self.classifier(self.features(x))

    network.forward = forward
    network.dynamic_sparse_modules = get_dynamic_sparse_modules(network)

    return network


def vgg19_dscnn(config):

    net = VGG19(config)
    net = make_dsnn(net)

    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


def mnist_sparse_cnn(config):

    net_params = config.get("net_params", {})
    net = MNISTSparseCNN(**net_params)
    return net


def mnist_sparse_dscnn(config, squash=True):

    net_params = config.get("net_params", {})
    net = MNISTSparseCNN(**net_params)
    net = make_dsnn(net, config)
    net = swap_layers(net, nn.MaxPool2d, KWinners2d)
    net = squash_layers(net, DSConv2d, KWinners2d)
    net = squash_layers(
        net, SparseWeights, nn.BatchNorm1d, KWinners, transfer_forward_hook=True
    )

    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


def gsc_sparse_cnn(config):

    net_params = config.get("net_params", {})
    net = GSCSparseCNN(**net_params)
    return net


def gsc_sparse_dscnn(config):

    net_params = config.get("net_params", {})
    net = GSCSparseCNN(**net_params)
    net = make_dsnn(net, config)
    net = swap_layers(net, nn.MaxPool2d, KWinners2d)
    net = squash_layers(net, DSConv2d, nn.BatchNorm2d, KWinners2d)
    net = squash_layers(
        net, SparseWeights, nn.BatchNorm1d, KWinners, transfer_forward_hook=True
    )

    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


class GSCSparseFullCNN(nn.Sequential):
    """Sparse CNN model used to classify `Google Speech Commands` dataset as
    described in `How Can We Be So Dense?`_ paper.

    .. _`How Can We Be So Dense?`: https://arxiv.org/abs/1903.11257

    :param cnn_out_channels: output channels for each CNN layer
    :param cnn_percent_on: Percent of units allowed to remain on each convolution
                           layer
    :param linear_units: Number of units in the linear layer
    :param linear_percent_on: Percent of units allowed to remain on the linear
                              layer
    :param linear_weight_sparsity: Percent of weights that are allowed to be
                                   non-zero in the linear layer
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    """

    def __init__(self,
                 cnn_out_channels=(32, 64, 32),
                 cnn_percent_on=(0.095, 0.125, 0.0925),
                 linear_units=1600,
                 linear_percent_on=0.1,
                 linear_weight_sparsity=0.4,
                 boost_strength=1.5,
                 boost_strength_factor=0.9,
                 k_inference_factor=1.5,
                 duty_cycle_period=1000
                 ):
        super(GSCSparseFullCNN, self).__init__()
        # input_shape = (1, 32, 32)
        # First Sparse CNN layer
        self.add_module("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5))
        self.add_module("cnn1_batchnorm", nn.BatchNorm2d(cnn_out_channels[0],
                                                         affine=False))
        self.add_module("cnn1_maxpool", nn.MaxPool2d(2))
        self.add_module("cnn1_kwinner", KWinners2d(
            channels=cnn_out_channels[0],
            percent_on=cnn_percent_on[0],
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period))

        # Second Sparse CNN layer
        self.add_module("cnn2", nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5))
        self.add_module("cnn2_batchnorm",
                        nn.BatchNorm2d(cnn_out_channels[1], affine=False))
        self.add_module("cnn2_maxpool", nn.MaxPool2d(2))
        self.add_module("cnn2_kwinner", KWinners2d(
            channels=cnn_out_channels[1],
            percent_on=cnn_percent_on[1],
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period))

        # # Third Sparse CNN layer
        # self.add_module("cnn3",
        #                 nn.Conv2d(cnn_out_channels[1], cnn_out_channels[2], 5))
        # self.add_module("cnn3_batchnorm",
        #                 nn.BatchNorm2d(cnn_out_channels[2], affine=False))
        # # self.add_module("cnn3_maxpool", nn.MaxPool2d(2))
        # self.add_module("cnn3_kwinner", KWinners2d(
        #     channels=cnn_out_channels[2],
        #     percent_on=cnn_percent_on[2],
        #     k_inference_factor=k_inference_factor,
        #     boost_strength=boost_strength,
        #     boost_strength_factor=boost_strength_factor,
        #     duty_cycle_period=duty_cycle_period))

        self.add_module("flatten", Flatten())

        # # Sparse Linear layer
        # self.add_module("linear", SparseWeights(
        #     nn.Linear(25 * cnn_out_channels[1], linear_units),
        #     weight_sparsity=linear_weight_sparsity))
        # self.add_module("linear_bn", nn.BatchNorm1d(linear_units, affine=False))
        # self.add_module("linear_kwinner", KWinners(
        #     n=linear_units,
        #     percent_on=linear_percent_on,
        #     k_inference_factor=k_inference_factor,
        #     boost_strength=boost_strength,
        #     boost_strength_factor=boost_strength_factor,
        #     duty_cycle_period=duty_cycle_period))

        # Classifier
        self.add_module("output", nn.Linear(1600, 12))
        self.add_module("softmax", nn.LogSoftmax(dim=1))


def gsc_sparse_dscnn_fullyconv(config):

    net_params = config.get("net_params", {})
    net = GSCSparseFullCNN(**net_params)
    net = make_dsnn(net, config)
    net = swap_layers(net, nn.MaxPool2d, KWinners2d)
    net = squash_layers(net, DSConv2d, nn.BatchNorm2d, KWinners2d)
    net = squash_layers(
        net, SparseWeights, nn.BatchNorm1d, KWinners, transfer_forward_hook=True
    )

    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net
