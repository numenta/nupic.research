#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

from copy import deepcopy

from nupic.research.frameworks.pytorch.model_utils import get_parent_module


def clone_model(model, keep_params=None, keep_hooks=True):
    """
    Clones a model by creating a deepcopy and then for each param either
        1) cloning it from the original to the copied model
        2) passing a reference of it from the original to the copied model

    This implementation is largely based on
    https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py

    :param keep_params: a list params to keep by reference, instead of cloning
    :type keep_params: list of names
    :param keep_hooks: whether to keep the models original module hooks; this is
                       useful when the hooks continuously track some updated state.
    :type keep_params: list of names
    """

    try:
        new_model = deepcopy(model)
    except RuntimeError as err:
        raise Exception(
            f"Received RuntimeError: {err}\n"
            "Make sure no operations (such as clone) have been taken on the params. "
            "You can't deepcopy non-leaf tensors."
        )

    keep_params = keep_params or []
    for (name, old_param) in model.named_parameters():
        if name in keep_params:
            # Keep param as reference.
            new_param = old_param
        else:
            # Clone param.
            new_param = old_param.clone()

        parent_module = get_parent_module(new_model, name)
        base_name = name.split(".")[-1]
        parent_module._parameters[base_name] = new_param

    if keep_hooks:
        _copy_over_hooks(model, new_model)

    return new_model


def _copy_over_hooks(old_model, new_model):
    """
    Since the parameters are cloned, in `clone_module`, there hooks are maintained. Only
    the module hooks need to be copied over.
    """
    for old_module, new_module in zip(old_model.modules(), new_model.modules()):
        new_module._backward_hooks = old_module._backward_hooks
        new_module._forward_hooks = old_module._forward_hooks
        new_module._forward_pre_hooks = old_module._forward_pre_hooks
