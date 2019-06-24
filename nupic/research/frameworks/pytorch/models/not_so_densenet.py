#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import math

import torch.nn as nn
from torchvision.models.densenet import _DenseBlock, _Transition

from nupic.torch.modules import Flatten, KWinners2d


def _sparsify(parent, relus, channels, percent_on, k_inference_factor,
              boost_strength, boost_strength_factor, duty_cycle_period):
    """
    Find ReLU and replace with k-winners where percent_on < 1.0
    :param parent: Parent Layer containing the ReLU to be replaced
    :param relus: List of ReLU module names to be replaced.
    :param channels: List of input channels for each k-winner.
    :param percent_on: List of 'percent_on' parameters for each ReLU
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    """
    for i, name in enumerate(relus):
        if percent_on[i] >= 1.0:
            continue
        parent.__setattr__(name, KWinners2d(
            channels=channels[i],
            percent_on=percent_on[i],
            k_inference_factor=k_inference_factor,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        ))


class DenseNetCIFAR(nn.Sequential):
    """
    DenseNet_ CIFAR model. Based on :mod:`torchvision.models.densenet` blocks.
    See original `densenet.lua`_ implementation for more details

    .. _DenseNet: https://arxiv.org/abs/1608.06993
    .. _`densenet.lua`: https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua # noqa
    :param block_config: how many layers in each pooling block. If None compute from `depth`
    :param depth: DenseNet network depth. If None then `block_config` must be given
    :param growth_rate: how many filters to add each layer (`k` in paper)
    :param reduction: Channel compress ratio at transition layer
    :param num_classes: number of classification classes
    :param bottleneck_size: multiplicative factor for number of bottle neck layers
    :param avg_pool_size: Average pool size for last transition layer
    """

    def __init__(self,
                 block_config=None,
                 depth=100,
                 growth_rate=12,
                 reduction=0.5,
                 num_classes=10,
                 bottleneck_size=4,
                 avg_pool_size=8):
        super(DenseNetCIFAR, self).__init__()

        # Compute blocks from depth
        if block_config is None:
            layers = (depth - 4) // 6
            block_config = (layers,) * 3

        # First convolution
        num_features = growth_rate * 2
        self.add_module("conv", nn.Conv2d(in_channels=3,
                                          out_channels=num_features,
                                          kernel_size=3,
                                          padding=1,
                                          bias=False))

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bottleneck_size,
                                growth_rate=growth_rate,
                                drop_rate=0)
            self.add_module("block{0}".format(i + 1), block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                out_features = math.floor(num_features * reduction)
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=out_features)
                self.add_module("transition{0}".format(i + 1), trans)
                num_features = out_features

        # Final batch norm
        self.add_module("norm", nn.BatchNorm2d(num_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("avg_pool", nn.AvgPool2d(kernel_size=avg_pool_size))

        # classifier layer
        outputs = int(num_features * 16 / (avg_pool_size * avg_pool_size))
        self.add_module("flatten", Flatten())
        self.add_module("classifier", nn.Linear(outputs, num_classes))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


class NoSoDenseNetCIFAR(DenseNetCIFAR):
    """
    Modified DenseNet_ architecture using sparse dense blocks and sparse
    transition layers. Inspired by the original `densenet.lua`_ implementation

    .. _DenseNet: https://arxiv.org/abs/1608.06993
    .. _`densenet.lua`: https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua # noqa

    :param block_config: how many layers in each pooling block.
                         If None compute from `depth`
    :param depth: DenseNet network depth. If None then `block_config` must be given
    :param growth_rate: how many filters to add each layer (`k` in paper)
    :param reduction: Channel compress ratio at transition layer
    :param num_classes: number of classification classes
    :param bottleneck_size: multiplicative factor for number of bottle neck layers
    :param avg_pool_size: Average pool size for last transition layer
    :param dense_percent_on: Percent of units allowed to remain before each
                             convolution layer of the dense layer.
    :param transition_percent_on: Percent of units allowed to remain the
                                  convolution layer of the transition layer
    :param classifier_percent_on: Percent of units allowed to remain after the
                                  last batch norm before the classifier
    :param k_inference_factor: During inference (training=False) we increase
                               `percent_on` in all sparse layers by this factor
    :param boost_strength: boost strength (0.0 implies no boosting)
    :param boost_strength_factor: Boost strength factor to use [0..1]
    :param duty_cycle_period: The period used to calculate duty cycles
    """

    def __init__(
        self,
        block_config=None,
        depth=100,
        growth_rate=12,
        reduction=0.5,
        num_classes=10,
        bottleneck_size=4,
        avg_pool_size=4,
        dense_percent_on=([1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]),
        transition_percent_on=(1.0, 1.0, 1.0),
        classifier_percent_on=1.0,
        k_inference_factor=1.0,
        boost_strength=1.5,
        boost_strength_factor=0.95,
        duty_cycle_period=1000,
    ):
        super(NoSoDenseNetCIFAR, self).__init__(block_config=block_config,
                                                depth=depth,
                                                growth_rate=growth_rate,
                                                reduction=reduction,
                                                num_classes=num_classes,
                                                bottleneck_size=bottleneck_size,
                                                avg_pool_size=avg_pool_size)

        # Sparsify relu after the last batch norm before the classifier
        _sparsify(parent=self,
                  relus=["relu"],
                  channels=[self.norm.num_features],
                  percent_on=[classifier_percent_on],
                  k_inference_factor=k_inference_factor,
                  boost_strength=boost_strength,
                  boost_strength_factor=boost_strength_factor,
                  duty_cycle_period=duty_cycle_period)

        # Sparsify dense blocks
        def _is_relu(x):
            return type(x[1]) == nn.ReLU

        def _is_norm(x):
            return type(x) == nn.BatchNorm2d

        def _is_denseblock(x):
            return type(x) == _DenseBlock

        for i, block in enumerate(filter(_is_denseblock, self.children())):
            for layer in block.children():
                channels = [bn.num_features for bn in
                            filter(_is_norm, layer.children())]
                relus = [x[0] for x in filter(_is_relu, layer.named_children())]
                _sparsify(parent=layer,
                          relus=relus,
                          channels=channels,
                          percent_on=dense_percent_on[i],
                          k_inference_factor=k_inference_factor,
                          boost_strength=boost_strength,
                          boost_strength_factor=boost_strength_factor,
                          duty_cycle_period=duty_cycle_period)

        # Sparsify transition block
        def _is_transition(x):
            return type(x) == _Transition

        for i, block in enumerate(filter(_is_transition, self.children())):
            channels = [bn.num_features for bn in filter(_is_norm, block.children())]
            relus = [x[0] for x in filter(_is_relu, block.named_children())]
            _sparsify(parent=block,
                      relus=relus,
                      channels=channels,
                      percent_on=(transition_percent_on[i],) * len(relus),
                      k_inference_factor=k_inference_factor,
                      boost_strength=boost_strength,
                      boost_strength_factor=boost_strength_factor,
                      duty_cycle_period=duty_cycle_period)


if __name__ == "__main__":
    nsdn = NoSoDenseNetCIFAR()
    print(nsdn)
