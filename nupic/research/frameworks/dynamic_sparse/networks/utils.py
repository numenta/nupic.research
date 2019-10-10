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

from collections import OrderedDict

import torch

from nupic.torch.modules import SparseWeights, SparseWeights2d

from .layers import DSConv2d, DSLinear, DynamicSparseBase

# -------------------------------------------------
# General Utils - helper classes
# -------------------------------------------------


class NumScheduler(object):
    """
    Class like an lr_scheduler, but for any number of concern.
    In it's simplest form, this will just grab an element from
    a list of 'values' depending on the epic, but further development
    could allow the logic could be made more complex.
    """

    def __init__(self, values):
        self._values = values
        self._epoch = 0

    def step(self):
        self._epoch += 1

    def get_value(self):
        idx = min(self._epoch, len(self._values) - 1)
        return self._values[idx]


# -------------------------------------------------
# General Utils - network mutators
# -------------------------------------------------

def remove_layers(sequential, *types):
    """
    Removes all nn.Modules of any type in types.
    Returns new sequential object - does not overwrite the one passed
    to this function.

    :param sequential: torch.nn.Sequential
    :param types: types of layers
    """

    old_seq = dict(sequential.named_children())
    names = list(old_seq.keys())
    modules = list(old_seq.values())

    # Make copy of sequence.
    new_seq = OrderedDict()

    # Edit copy in place.
    for name, m in sequential.named_children():

        if not isinstance(m, types):
            new_seq[name] = m

    # Return new sequential.
    new_seq = torch.nn.Sequential(new_seq)
    return new_seq


def replace_sparse_weights(sequence):

    new_seq = OrderedDict()
    for name, m in sequence.named_children():

        # Case 1: Linear
        if isinstance(m, SparseWeights):

            linear = m.module
            new_module = torch.nn.Linear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=(linear.bias is not None),
            )

        # Case 2: Conv2d
        elif isinstance(m, SparseWeights2d):

            conv = m.module
            new_module = torch.nn.Conv2d(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                padding_mode=conv.padding_mode,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=(conv.bias is not None),
            )

        # Case 3: Default, add back-in as is.
        else:
            new_module = m

        # Add new_module.
        new_seq[name] = new_module

    # Return new sequential.
    new_seq = torch.nn.Sequential(new_seq)
    return new_seq


def swap_layers(sequential, layer_type_a, layer_type_b):
    """
    If 'layer_type_a' appears immediately before 'layer_type_2',
    this function will swap their position in a new sequential.

    :param sequential: torch.nn.Sequential
    :param layer_type_a: type of first layer
    :param layer_type_a: type of second layer
    """

    old_seq = dict(sequential.named_children())
    names = list(old_seq.keys())
    modules = list(old_seq.values())

    # Make copy of sequence.
    new_seq = list(old_seq.items())

    # Edit copy in place.
    layer_a = modules[0]
    name_a = names[0]
    for i, (name_b, layer_b) in enumerate(list(old_seq.items())[1:], 1):

        if isinstance(layer_a, layer_type_a) and isinstance(layer_b, layer_type_b):
            new_seq[i - 1] = (name_b, layer_b)
            new_seq[i] = (name_a, layer_a)

        layer_a = layer_b
        name_a = name_b

    # Turn sequence into nn.Sequential.
    new_seq = OrderedDict(new_seq)
    new_seq = torch.nn.Sequential(new_seq)
    return new_seq


def squash_layers(sequential, *types, transfer_forward_hook=True):
    """
    This function squashes layers matching the sequence of 'types'.
    For instance, if 'types' is [Conv2d, BatchNorm, KWinners] and
    "sequential" has layers [..., Conv2d, BatchNorm, KWinners, ...],
    then a new "sequential" will be returned of the form
    [..., SubSequence, ...] where SubSequence contains the 'types'
    specified.

    More importantly, if 'transfer_foward_hook=True'
    the SubSequence will use the same hook (if any)
    as the original Conv2d, although with the output from KWinners
    - at least in this example case.

    :param sequential: torch.nn.Sequential
    :param types: types of layers

    :returns: a new torch.nn.Sequential
    """
    assert len(types) <= len(sequential), "More types than layers passed."
    assert len(types) > 1, "Expected more than one type to squash."

    named_children = dict(sequential.named_children())
    names = list(named_children.keys())
    modules = list(named_children.values())

    i0 = 0
    new_seq = []
    while i0 < len(modules):
        i1 = i0 + len(types)
        if i1 <= len(modules) + 1:

            sublayers = modules[i0:i1]
            subnames = names[i0:i1]
            matches = [
                isinstance(layer, ltype) for layer, ltype in zip(sublayers, types)
            ]

        else:
            matches = [False]

        if all(matches):

            # Get base dynamic sparse layers.
            base_layer = modules[i0]

            # Save forward hook of base layer.
            if transfer_forward_hook and hasattr(base_layer, "forward_hook"):
                forward_hook = base_layer.forward_hook
                if hasattr(base_layer, "forward_hook_handle"):
                    base_layer.forward_hook_handle.remove()
            else:
                forward_hook = None

            # Squash layers.
            squashed = OrderedDict(zip(subnames, sublayers))
            squashed = torch.nn.Sequential(squashed)

            assert squashed[0] == base_layer

            # Maintain same forward hook.
            if forward_hook:
                forward_hook_handle = squashed.register_forward_hook(
                    lambda module, in_, out_: forward_hook(module[0], in_, out_)
                )
                squashed.forward_hook = forward_hook
                squashed.forward_hook_handle = forward_hook_handle

            # Append squashed sequence
            name = "squashed" + str(i0)
            new_seq.append((name, squashed))

            # Iterate i0.
            i0 = i1

        else:

            # Append layer as is.
            name = names[i0]
            module = modules[i0]
            new_seq.append((name, module))

            # Iterate i0.
            i0 += 1

    # Turn sequence into nn.Sequential.
    new_seq = OrderedDict(new_seq)
    new_seq = torch.nn.Sequential(new_seq)
    return new_seq


def set_module(net, name, new_module):
    """
    Mimics "setattr" in purpose and argument types.
    Sets module "name" of "net" to "new_module".
    This is done recursively as "name" may be
    of the form '0.subname-1.subname-2.3 ...'
    where 0 and 3 indicate indices of a
    torch.nn.Sequential.
    """

    subnames = name.split(".")
    subname0, subnames_remaining = subnames[0], subnames[1:]

    if subnames_remaining:

        if subname0.isdigit():
            subnet = net[int(subname0)]
        else:
            subnet = getattr(net, subname0)

        set_module(subnet, ".".join(subnames_remaining), new_module)

    else:

        if subname0.isdigit():
            net[int(subname0)] = new_module
        else:
            setattr(net, subname0, new_module)


# ------------------------------------------------------------
# Dynamic Utils - dynamic-network builders and introspectors.
# ------------------------------------------------------------


def get_dynamic_sparse_modules(net):
    """
    Inspects all children recursively to collect the
    Dynamic-Sparse modules.
    """
    sparse_modules = []
    for module in net.modules():

        if isinstance(module, DynamicSparseBase):
            sparse_modules.append(module)

    return sparse_modules


def make_dsnn(net, config=None):
    """
    Edits net in place to replace Conv2d layers with those
    specified in config.
    """

    config = config or {}

    named_layers = [
        (name, layer)
        for name, layer in net.named_modules()
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear)
    ]
    num_convs = len(named_layers)

    def tolist(param):
        if isinstance(param, list):
            return param
        else:
            return [param] * num_convs

    def get_layer_type(prune_method):
        if prune_method == "dynamic-conv":
            return DSConv2d
        elif prune_method == "dynamic-linear":
            return DSLinear

    # Get DSConv2d params from config.
    prune_methods = tolist(config.get("prune_methods", None))
    assert (
        len(prune_methods) == num_convs
    ), "Not enough prune_methods specified in config. Expected {}, got {}".format(
        num_convs, prune_methods
    )

    # # Populate kwargs for new layers.
    # possible_args = {
    #     "dynamic-conv": [
    #         "update_nsteps",
    #         "half_precision",
    #         "coactivation_test",
    #         "threshold_multiplier",
    #     ],
    #     "dynamic-linear": [
    #     ],
    #     None: [],
    # }
    # kwargs_s = []
    # for c_i in range(num_convs):
    #     layer_args = {}
    #     prune_method = prune_methods[c_i]
    #     for arg in possible_args[prune_method]:
    #         if arg in config:
    #             layer_args[arg] = tolist(config.get(arg))[c_i]
    #     kwargs_s.append(layer_args)

    # assert (
    #     len((kwargs_s)) == len(named_layers) == len(prune_methods)
    # ), "Sizes do not match"

    # Replace conv layers.
    for method, (name, layer) in zip(prune_methods, named_layers):

        layer_type = get_layer_type(method)
        if layer_type is None:
            continue

        if isinstance(layer, torch.nn.Conv2d):
            set_module(
                net,
                name,
                layer_type(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    padding_mode=layer.padding_mode,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=(layer.bias is not None),
                    config=config,
                ),
            )

        elif isinstance(layer, torch.nn.Linear):
            set_module(
                net,
                name,
                layer_type(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    bias=(layer.bias is not None),
                    config=config,
                ),
            )

    return net
