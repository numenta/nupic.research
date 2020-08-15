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
# summary
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# adapted from https://github.com/meliketoy/wide-resnet.pytorch/

from functools import partial

import torch.nn as nn

import nupic.torch.modules as nupic_modules
from nupic.research.frameworks.pytorch.models.resnets import BasicBlock, Bottleneck
from nupic.research.frameworks.pytorch.models.resnets import ResNet as ResNetCore
from nupic.research.frameworks.pytorch.models.resnets import cf_dict
from nupic.research.frameworks.pytorch.sparse_layer_params import (
    LayerParams,
    auto_sparse_activation_params,
    auto_sparse_conv_params,
)
from nupic.torch.modules import KWinners2d


def default_resnet_params(
    group_type,
    number_layers,
    layer_params_type=None,
    linear_params_func=None,
    conv_params_func=None,
    activation_params_func=None,
    layer_params_kwargs=None,
):
    """
    Creates dictionary with default parameters.

    :param group_type: defines whether group is BasicBlock or Bottleneck.
    :param number_layers: number of layers to be assigned to each group.

    :returns dictionary with default parameters
    """
    layer_params_type = layer_params_type or LayerParams
    layer_params_kwargs = layer_params_kwargs or {}

    # Set layer params w/ activation.
    layer_params = layer_params_type(
        linear_params_func=linear_params_func,
        conv_params_func=conv_params_func,
        activation_params_func=activation_params_func,
        **layer_params_kwargs
    )

    # Set layer params w/o activation.
    noact_layer_params = layer_params_type(
        linear_params_func=linear_params_func,
        conv_params_func=conv_params_func,
        **layer_params_kwargs
    )

    # Validate layer_params
    assert isinstance(layer_params, LayerParams), \
        "Expected {} to sub-classed from LayerParams".format(layer_params)

    # Set layers params by group type.
    if group_type == BasicBlock:
        params = dict(
            conv3x3_1=layer_params,
            conv3x3_2=noact_layer_params,
            shortcut=layer_params
        )
    elif group_type == Bottleneck:
        params = dict(
            conv1x1_1=layer_params,
            conv3x3_2=layer_params,
            conv1x1_3=layer_params,
            shortcut=layer_params,
        )

    return dict(
        stem=layer_params,
        filters64=[params] * number_layers[0],
        filters128=[params] * number_layers[1],
        filters256=[params] * number_layers[2],
        filters512=[params] * number_layers[3],
        linear=noact_layer_params,
    )


def as_kwarg(v):
    return dict(layer_params=v)


def format_as_conv_args(sparse_params, group_type, number_layers):
    """
    Take the dictionary and insert layer_params=value so that it works as a
    kwarg.

    By doing this here rather than in default_resnet_params, we preserve
    compatibility with any existing code that is providing an alternate to
    `default_resnet_params`.
    """
    result = {}
    for group_key in ResNetCore.group_keys:
        group_value = sparse_params[group_key]
        if isinstance(group_value, (list, tuple)):
            result[group_key] = [{k: as_kwarg(v)
                                  for k, v in params.items()}
                                 for params in group_value]
        else:
            result[group_key] = {k: as_kwarg(v)
                                 for k, v in group_value.items()}

    return dict(
        stem=as_kwarg(sparse_params["stem"]),
        linear=as_kwarg(sparse_params["linear"]),
        **result)


def format_as_activation_args(sparse_params, group_type, number_layers):
    """
    Take the dictionary that contains "conv1x1_1", etc, and convert it to one
    that contains "act1", etc., and inserting layer_params=value so that it
    works as a kwarg.

    By doing this here rather than in default_resnet_params, we preserve
    compatibility with any existing code that is providing an alternate to
    `default_resnet_params`.
    """
    result = {}
    if group_type == BasicBlock:
        for name in ResNetCore.group_keys:
            group_value = sparse_params[name]
            if isinstance(group_value, (list, tuple)):
                result[name] = [dict(act1=as_kwarg(params["conv3x3_1"]),
                                     act2=as_kwarg(params["shortcut"]))
                                for params in group_value]
            else:
                result[name] = dict(act1=as_kwarg(group_value["conv3x3_1"]),
                                    act2=as_kwarg(group_value["shortcut"]))
    elif group_type == Bottleneck:
        for name in ResNetCore.group_keys:
            group_value = sparse_params[name]
            if isinstance(group_value, (list, tuple)):
                result[name] = [dict(act1=as_kwarg(params["conv1x1_1"]),
                                     act2=as_kwarg(params["conv3x3_2"]),
                                     act3=as_kwarg(params["shortcut"]))
                                for params in group_value]
            else:
                result[name] = dict(act1=as_kwarg(group_value["conv1x1_1"]),
                                    act2=as_kwarg(group_value["conv3x3_2"]),
                                    act3=as_kwarg(group_value["shortcut"]))

    return dict(
        stem=as_kwarg(sparse_params["stem"]),
        linear=as_kwarg(sparse_params["linear"]),
        **result)


def linear_layer(input_size, output_size, layer_params, sparse_weights_type):
    """Basic linear layer, which accepts different sparse layer types."""
    layer = nn.Linear(input_size, output_size)

    # Compute params for sparse-weights module.
    if layer_params is not None:
        weight_params = layer_params.get_linear_params(
            input_size,
            output_size,
        )
    else:
        weight_params = None

    # Initialize sparse-weights module as specified.
    if weight_params is not None:
        sparse_weights_type = weight_params.pop(
            "sparse_weights_type", sparse_weights_type)

        if sparse_weights_type is not None:
            return sparse_weights_type(layer, **weight_params)

    # Default
    return layer


def conv_layer(
    in_planes,
    out_planes,
    kernel_size,
    layer_params,
    sparse_weights_type,
    stride=1,
    padding=0,
    bias=False,
):
    """Basic conv layer, which accepts different sparse layer types."""
    layer = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    # Compute params for sparse-weights module.
    if layer_params is not None:
        weight_params = layer_params.get_conv_params(
            in_planes,
            out_planes,
            kernel_size
        )
    else:
        weight_params = None

    # Initialize sparse-weights module as specified.
    if weight_params is not None:
        sparse_weights_type = weight_params.pop(
            "sparse_weights_type", sparse_weights_type)

        if sparse_weights_type is not None:
            return sparse_weights_type(layer, **weight_params)

    # Default
    return layer


def activation_layer(
    out,
    layer_params,
    kernel_size=1,
    base_activation=None,
):
    """Basic activation layer.
    Defaults to `base_activation` if `activation_params is None` from `layer_params`.

    :param out: number of output channels from the preceding (conv) layer
    :param layer_params: `LayerParams` object with `get_activation_params` function.
                         This gives `kwinner_class` and the kwargs to construct it.
                         Note: This entails a slightly different behavior from
                         `conv_layer` and `linear_layer` where the layer type
                         (e.g. kwinner, SparseWeights, ect.) are passed separately
                         and not through `layer_params`. This may be fixed for
                         consistency in the future.
    :param kernel_size: kernal size (e.g. 1, 3, ect) of preceding (conv) layer
    :param base_activation: this is the activation module applied irrespective
                            of the `kwinner_class`. If `kwinner_class` is present,
                            it's applied before. Otherwise, it's the only activation
                            that's applied.
    """

    # Determine default base_activation.
    if base_activation is None:
        base_activation = nn.ReLU(inplace=True)
    else:
        base_activation = base_activation()
    assert isinstance(base_activation, nn.Module), \
        "`base_activation` should be subclassed from torch.nn.Module"

    # Compute layer_params for kwinners activation module.
    if layer_params is not None:
        activation_params = layer_params.get_activation_params(0, out, kernel_size)
    else:
        activation_params = None

    # Initialize kwinners module as specified.
    if activation_params is not None:
        # Check if overriding default kwinner class
        kwinner_class = activation_params.pop("kwinner_class", KWinners2d)
        return nn.Sequential(
            base_activation,
            kwinner_class(
                out,
                **activation_params
            ),
        )
    else:
        return base_activation


class SparseResNet(ResNetCore):
    """
    A bridge from the previous SparseResNet API to the current ResNet API
    """

    def __init__(self, config=None):
        # update config
        defaults = dict(
            depth=50,
            num_classes=1000,
            base_activation=None,  # See `activation_layer` function above.
            linear_sparse_weights_type="SparseWeights",
            conv_sparse_weights_type="SparseWeights2d",
            resnet_params=default_resnet_params,
            defaults_sparse=False,
            layer_params_type=None,  # Sub-classed from `LayerParams`.
            # To be passed to layer_params_type:
            layer_params_kwargs=None,
            linear_params_func=None,
            conv_params_func=None,
            activation_params_func=None,
            batch_norm_args=None
        )
        defaults.update(config or {})
        self.__dict__.update(defaults)
        # sparse weights types ignored if layers_params are not defined
        if isinstance(self.linear_sparse_weights_type, str):
            self.linear_sparse_weights_type = getattr(
                nupic_modules, self.linear_sparse_weights_type)
        if isinstance(self.conv_sparse_weights_type, str):
            self.conv_sparse_weights_type = getattr(
                nupic_modules, self.conv_sparse_weights_type)

        if self.defaults_sparse:
            if self.conv_params_func is None:
                self.conv_params_func = auto_sparse_conv_params
            if self.activation_params_func is None:
                self.activation_params_func = auto_sparse_activation_params

        group_type, number_layers = cf_dict[str(self.depth)]
        if not hasattr(self, "sparse_params"):
            self.sparse_params = self.resnet_params(
                group_type,
                number_layers,
                layer_params_type=self.layer_params_type,
                layer_params_kwargs=self.layer_params_kwargs,
                linear_params_func=self.linear_params_func,
                conv_params_func=self.conv_params_func,
                activation_params_func=self.activation_params_func,
            )

        self.batch_norm_args = self.batch_norm_args or {}

        conv_args = format_as_conv_args(self.sparse_params, group_type,
                                        number_layers)
        act_args = format_as_activation_args(self.sparse_params, group_type,
                                             number_layers)

        super().__init__(
            depth=self.depth,
            num_classes=self.num_classes,
            conv_layer=partial(
                conv_layer,
                sparse_weights_type=self.conv_sparse_weights_type
            ),
            conv_args=conv_args,
            norm_args=self.batch_norm_args,
            act_layer=partial(
                activation_layer,
                base_activation=self.base_activation
            ),
            act_args=act_args,
            linear_layer=partial(
                linear_layer,
                sparse_weights_type=self.linear_sparse_weights_type
            ),
            linear_args=as_kwarg(self.sparse_params["linear"]),
            deprecated_compatibility_mode=True,
        )


# convenience classes
def build_resnet(depth, config=None):
    config = config or {}
    config["depth"] = depth
    return SparseResNet(config)


def resnet18(config=None):
    return build_resnet(18, config)


def resnet34(config=None):
    return build_resnet(34, config)


def resnet50(config=None):
    return build_resnet(50, config)


def resnet101(config=None):
    return build_resnet(101, config)


def resnet152(config=None):
    return build_resnet(152, config)
