# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
import io
import logging
import os
import pickle

import torch

from nupic.research.frameworks.pytorch.model_utils import deserialize_state_dict


def get_state_dict(checkpoint_path):

    checkpoint_path = os.path.expanduser(checkpoint_path)
    with open(checkpoint_path, "rb") as loaded_state:
        checkpoint_dict = pickle.load(loaded_state)

    if "model" in checkpoint_dict:
        with io.BytesIO(checkpoint_dict["model"]) as buffer:
            state_dict = deserialize_state_dict(buffer)
        return state_dict
    else:
        return None


def get_linear_param_names(model):

    linear_params = []
    for name_m, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            for name_p, _ in m.named_parameters():
                full_name = name_m + ("." if name_m else "") + name_p
                linear_params.append(full_name)
    return linear_params


def load_multi_state(
    model,
    restore_full_model=None,
    restore_linear=None,
    restore_nonlinear=None,
):
    """
    Example 1:
    ```
    checkpoint_linear = "~/.../checkpoint_20/checkpoint"
    checkpoint_nonlinear = "~/.../checkpoint_1/checkpoint""

    kwargs = {
        "restore_linear": checkpoint_linear,
        "restore_nonlinear": checkpoint_nonlinear,
    }

    model = ResNet()
    model = load_multi_state(model, **kwargs)
    ```

    Example 2:
    ```
    checkpoint_model = "~/.../checkpoint_1/checkpoint""

    kwargs = {
        "restore_full_model": checkpoint_model,
    }

    model = ResNet()
    model = load_multi_state(model, **kwargs)
    ```
    """

    # Case 1: Full model state specified.
    if restore_full_model:
        state_dict = get_state_dict(restore_full_model)

        if state_dict:
            model.load_state_dict(state_dict, strict=False)

        return model

    # Case 2: Use separate sources for Linear and Non-Linear states.
    linear_params = get_linear_param_names(model)

    # Case 2a:  Linear param states
    linear_state = dict()
    if restore_linear:
        state_dict = get_state_dict(restore_linear)

        if state_dict:

            linear_state = {
                param_name: state_dict[param_name]
                for param_name, param in state_dict.items()
                if param_name in linear_params
            }
            model.load_state_dict(linear_state, strict=False)

    # Case 2b:  Non-Linear param states
    nonlinear_state = dict()
    if restore_nonlinear:
        state_dict = get_state_dict(restore_nonlinear)

        if state_dict:

            nonlinear_state = {
                param_name: state_dict[param_name]
                for param_name, param in state_dict.items()
                if param_name not in linear_params
            }
            model.load_state_dict(nonlinear_state, strict=False)

    # Validate results / quick sanity-check.
    assert set(linear_state.keys()).isdisjoint(nonlinear_state.keys())
    if linear_params and restore_linear:
        if not set(linear_state.keys()) == set(linear_params):
            logging.warning(
                "Warning: Unable to load all linear params [{}] from {}".format(
                    linear_params, restore_linear)
            )

    return model


def freeze_all_params(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_linear_params(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
