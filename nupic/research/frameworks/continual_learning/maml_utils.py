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


def clone_model(model, keep_as_reference=None):
    """
    Clones a model by creating a deepcopy and then for each param either
        1) cloning it from the original to the copied model
        2) passing a reference of it from the original to the copied model

    This implementation is largely based on
    https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py

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
