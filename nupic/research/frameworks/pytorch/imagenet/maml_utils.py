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

# These are extracted from:
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json

from copy import deepcopy

import torch

from nupic.research.frameworks.pytorch.model_utils import get_parent_module


def clone_module(module, keep_as_reference=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.
    * **keep_as_reference** (list) - list of parameters to keep as a reference instead
                                     of cloning

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    keep_as_reference = keep_as_reference or []

    # Second, re-write all parameters
    if hasattr(clone, "_parameters"):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                if param_key not in keep_as_reference:
                    # Clone param
                    param = param.clone()
                clone._parameters[param_key] = param  # reference or clone

    # Third, handle the buffers if necessary
    if hasattr(clone, "_buffers"):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                clone._buffers[buffer_key] = module._buffers[buffer_key].clone()

    # Then, recurse for each submodule
    if hasattr(clone, "_modules"):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(module._modules[module_key])

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    clone = clone._apply(lambda x: x)
    return clone


def clone_model(model, keep_as_reference=None):
    """
    Clones a model by creating a deepcopy and then for each param either
        1) cloning it from the original to the copied model
        2) passing a reference of it from the original to the copied model

    :param keep_as_reference: which params to pass as a reference
    :type keep_as_reference: list of names
    """

    try:
        new_model = deepcopy(model)
    except RuntimeError as err:
        raise Exception(
            f"Received RuntimeError: {err}\n"
            "Make sure no operations (such as clone) have been taken on the params. "
            "You can't deepcopy non-leaf tensors."
        )

    keep_as_reference = keep_as_reference or []
    for (name, old_param) in model.named_parameters():
        if name in keep_as_reference:
            # Keep param as reference.
            new_param = old_param
        else:
            # Clone param.
            new_param = old_param.clone()

        parent_module = get_parent_module(new_model, name)
        base_name = name.split(".")[-1]
        parent_module._parameters[base_name] = new_param

    return new_model
