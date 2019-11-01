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

import numpy as np
import torch
from torch import nn

from nupic.torch.models.sparse_cnn import GSCSparseCNN, MNISTSparseCNN
from nupic.torch.modules import Flatten, KWinners, KWinners2d

from .layers import DSConv2d, DSLinear
from .utils import (
    get_dynamic_sparse_modules,
    make_dsnn,
    remove_layers,
    replace_sparse_weights,
    squash_layers,
    swap_layers,
)

__all__ = [
    "GSCHebDepreciated",
    "gsc_conv_heb_depreciated",
    "gsc_conv_only_heb_depreciated",
    "mnist_sparse_cnn",
    "mnist_sparse_dsnn",
]


# --------------
# GSC Networks
# --------------


class GSCHebDepreciated(nn.Module):
    """LeNet like CNN used for GSC in how so dense paper."""

    # NOTE: See `gsc_sparse_dsnn` for general method to construct
    #       dynamic-gsc network. For instance, one may use
    # ```
    # config = dict(
    #       prune_methods=[None, "dynamic-conv", "dynamic-linear", None]
    # )
    # net = gsc_sparse_dsnn(config)
    # ```
    #
    # This will yield a network with a dense-conv (implied by the None), then a
    # dynamic-conv, then a dynamic-linear and then a dense-linear
    # (also implied by the None). As well, all the other intermediate layers
    # expected from the GSC network will be included between them.
    #
    # This `GSCHeb` network would also work just fine; however, it would need some
    # adjustments to mimic the `GSCHeb` implementation first - that is, it would
    # need to use modules inherited from `DynamicSparseBase` which helps track
    # coactivations.
    #

    def __init__(self, config=None):
        super(GSCHebDepreciated, self).__init__()

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


def _make_dsnn(net, config):
    """
    This function performs "surgery" - so to speak on the
    `MNISTSparseCNN` and `GSCSparseCNN` networks to enable
    them to be dynamically sparse.
    """
    net = replace_sparse_weights(net)
    net = make_dsnn(net, config)
    net = swap_layers(net, nn.MaxPool2d, KWinners2d)
    net = squash_layers(net, DSConv2d, KWinners2d)
    net = squash_layers(net, DSLinear, nn.BatchNorm1d, KWinners)

    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


# make a conv heb just by replacing the conv layers by special DSNN Conv layers
def gsc_conv_heb_depreciated(config):

    net = make_dsnn(GSCHebDepreciated(config), config)
    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


def gsc_conv_only_heb_depreciated(config):
    network = make_dsnn(GSCHebDepreciated(config), config)

    # replace the forward function to not apply regular convolution
    def forward(self, x):
        return self.classifier(self.features(x))

    network.forward = forward
    network.dynamic_sparse_modules = get_dynamic_sparse_modules(network)

    return network


def mnist_sparse_cnn(config):

    net_params = config.get("net_params", {})
    net = MNISTSparseCNN(**net_params)
    return net


def mnist_sparse_dsnn(config, squash=True):

    net_params = config.get("net_params", {})
    net = MNISTSparseCNN(**net_params)
    net = replace_sparse_weights(net)
    net = make_dsnn(net, config)
    net = swap_layers(net, nn.MaxPool2d, KWinners2d)
    net = squash_layers(net, DSConv2d, KWinners2d)
    net = squash_layers(net, DSLinear, nn.BatchNorm1d, KWinners)

    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


def gsc_sparse_cnn(config):

    net_params = config.get("net_params", {})
    net = GSCSparseCNN(**net_params)
    return net


def gsc_sparse_dsnn(config):

    net_params = config.get("net_params", {})
    net = GSCSparseCNN(**net_params)
    net = replace_sparse_weights(net)
    net = make_dsnn(net, config)
    net = swap_layers(net, nn.MaxPool2d, KWinners2d)
    net = squash_layers(net, DSConv2d, nn.BatchNorm2d, KWinners2d)
    net = squash_layers(net, DSLinear, nn.BatchNorm1d, KWinners)
    net = remove_layers(net, nn.LogSoftmax)

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

    def __init__(
        self,
        cnn_out_channels=(32, 64, 32),
        cnn_percent_on=(0.095, 0.125, 0.0925),
        linear_units=1600,
        linear_percent_on=0.1,
        linear_weight_sparsity=0.4,
        boost_strength=1.5,
        boost_strength_factor=0.9,
        k_inference_factor=1.5,
        duty_cycle_period=1000,
    ):
        super(GSCSparseFullCNN, self).__init__()
        # input_shape = (1, 32, 32)
        # First Sparse CNN layer
        self.add_module("cnn1", nn.Conv2d(1, cnn_out_channels[0], 5))
        self.add_module(
            "cnn1_batchnorm", nn.BatchNorm2d(cnn_out_channels[0], affine=False)
        )
        self.add_module("cnn1_maxpool", nn.MaxPool2d(2))
        self.add_module(
            "cnn1_kwinner",
            KWinners2d(
                channels=cnn_out_channels[0],
                percent_on=cnn_percent_on[0],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            ),
        )

        # Second Sparse CNN layer
        self.add_module("cnn2", nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5))
        self.add_module(
            "cnn2_batchnorm", nn.BatchNorm2d(cnn_out_channels[1], affine=False)
        )
        self.add_module("cnn2_maxpool", nn.MaxPool2d(2))
        self.add_module(
            "cnn2_kwinner",
            KWinners2d(
                channels=cnn_out_channels[1],
                percent_on=cnn_percent_on[1],
                k_inference_factor=k_inference_factor,
                boost_strength=boost_strength,
                boost_strength_factor=boost_strength_factor,
                duty_cycle_period=duty_cycle_period,
            ),
        )

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


def gsc_sparse_dsnn_fullyconv(config):

    net_params = config.get("net_params", {})
    net = GSCSparseFullCNN(**net_params)
    net = make_dsnn(net, config)
    net = swap_layers(net, nn.MaxPool2d, KWinners2d)
    net = squash_layers(net, DSConv2d, nn.BatchNorm2d, KWinners2d)

    net.dynamic_sparse_modules = get_dynamic_sparse_modules(net)

    return net


def _get_gsc_small_dense_params(equivalent_on_perc, verbose=False):

    def vprint(*args):
        if verbose:
            print(*args)

    # Define number of params in dense GSC.
    # TODO: make configurable based off orignal `cnn_out_channels` and `linear_units`
    # default_config = dict(
    #     cnn_out_channels=(64, 64),
    #     linear_units=1000,
    # )
    large_dense_params = np.array([1600, 102400, 1600000, 12000])

    # Cacluate num params in large sparse GSC.
    large_sparse_params = large_dense_params * equivalent_on_perc

    # Init desired congfig.
    cnn_out_channels = np.array([0, 0])
    linear_units = None

    # Assume 1 channel input to first conv
    cnn_out_channels[0] = large_sparse_params[0] / 25
    cnn_out_channels[1] = large_sparse_params[1] / (cnn_out_channels[0] * 25)
    cnn_out_channels = np.round(cnn_out_channels).astype(np.int)
    linear_units = large_sparse_params[2] / (25 * cnn_out_channels[1])
    linear_units = int(np.round(linear_units))

    # Simulate foward pass for sanity check
    conv1 = torch.nn.Conv2d(1, cnn_out_channels[0], 5)
    maxp1 = torch.nn.MaxPool2d(2)
    conv2 = torch.nn.Conv2d(cnn_out_channels[0], cnn_out_channels[1], 5)
    maxp2 = torch.nn.MaxPool2d(2)
    flat = torch.nn.Flatten()
    lin1 = torch.nn.Linear(25 * cnn_out_channels[1], linear_units)
    lin2 = torch.nn.Linear(linear_units, 12)

    x = torch.rand(10, 1, 32, 32)
    x = conv1(x)
    x = maxp1(x)
    x = conv2(x)
    x = maxp2(x)
    x = flat(x)
    x = lin1(x)
    x = lin2(x)

    # Calculate number of params.
    small_dense_params = {
        "conv_1": np.prod(conv1.weight.shape),
        "conv_2": np.prod(conv2.weight.shape),
        "lin1": np.prod(lin1.weight.shape),
        "lin2": np.prod(lin2.weight.shape),
    }

    # Compare with desired.
    total_new = 0
    total_old = 0
    for p_old, (layer, p_new) in zip(large_sparse_params, small_dense_params.items()):
        abs_diff = p_new - p_old
        rel_diff = abs_diff / float(p_old)
        vprint("---- {} -----".format(layer))
        vprint("   new - ", p_new)
        vprint("   old - ", p_old)
        vprint("   abs diff:", abs_diff)
        vprint("   rel diff: {}% change".format(100 * rel_diff))
        vprint()
        total_new += p_new
        total_old += p_old

    total_abs_diff = total_new - total_old
    total_rel_diff = total_abs_diff / float(total_old)
    vprint("---- Summary ----")
    vprint("   total new - ", total_new)
    vprint("   total old - ", total_old)
    vprint("   total abs diff:", total_abs_diff)
    vprint("   total rel diff: {}% change".format(100 * total_rel_diff))

    # New config
    new_config = dict(
        cnn_out_channels=tuple(cnn_out_channels),
        linear_units=linear_units,
    )
    return new_config


def small_dense_gsc(config):

    equivalent_on_perc = config.get("equivalent_on_perc")
    verbose = config.get("debug_small_dense")
    small_gsc_config = _get_gsc_small_dense_params(equivalent_on_perc, verbose=verbose)

    net_params = config.get("net_params", {})
    net_params.update(small_gsc_config)

    return GSCSparseCNN(**net_params)


def small_dense_gsc_dsnn(config):
    net = small_dense_gsc(config)
    net = _make_dsnn(net, config)
    return net
