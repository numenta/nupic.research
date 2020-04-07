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
import os
import pickle

import torch

from nupic.research.frameworks.pytorch.model_utils import deserialize_state_dict
from nupic.torch.modules.sparse_weights import SparseWeightsBase


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


def get_linear_param_names(
    model,
    include_buffers=True,
    include_sparse_weights_params=True,
):

    linear_params = []
    for name_m, m in model.named_modules():

        # Identify whether to treat `SparseWeightsBase` as have linear params.
        if include_sparse_weights_params and isinstance(m, SparseWeightsBase):
            module = m.module  # -> may contain a linear layer
        else:
            module = m

        # Check if the 'module' is linear then iterate over params/buffers of 'm'.
        # Note: 'm' will either be a `torch.nn.Linear` or a `SparseWeightsBase`.
        if isinstance(module, torch.nn.Linear):

            # Iterate over all params of m.
            for name_p, _ in m.named_parameters():
                full_name = name_m + ("." if name_m else "") + name_p
                linear_params.append(full_name)

            # Iterate over all buffers of m.
            if include_buffers:

                for name_b, _ in m.named_buffers():
                    full_name = name_m + ("." if name_m else "") + name_b
                    linear_params.append(full_name)

    return list(set(linear_params))


def get_nonlinear_param_names(
    model,
    include_buffers=True,
    include_sparse_weights_params=True,
):
    linear_param_names = get_linear_param_names(
        model, include_buffers, include_sparse_weights_params
    )
    nonlinear_params = []
    for name_p, _ in model.named_parameters():
        if name_p not in linear_param_names:
            nonlinear_params.append(name_p)

    if include_buffers:
        for name_b, _ in model.named_buffers():
            if name_b not in linear_param_names:
                nonlinear_params.append(name_b)

    return list(set(nonlinear_params))


def remap_state_dict(state_dict, param_map):
    """
    Remaps the names of the params according to 'param_map'.
    """

    new_state_dict = {}
    for param, state in state_dict.items():

        if param in param_map:
            new_param = param_map[param]
            new_state_dict[new_param] = state
        else:
            new_state_dict[param] = state

    return new_state_dict


def load_multi_state(
    model,
    restore_full_model=None,
    restore_linear=None,
    restore_nonlinear=None,
    strict=True,
    param_map=None,
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

    # Validate paths.
    if restore_full_model:
        assert os.path.isfile(restore_full_model)
    if restore_linear:
        assert os.path.isfile(restore_linear)
    if restore_nonlinear:
        assert os.path.isfile(restore_nonlinear)

    # Case 1: Full model state specified.
    if restore_full_model:
        state_dict = get_state_dict(restore_full_model)

        if state_dict:

            # Remap param names in state_dict.
            if param_map:
                state_dict = remap_state_dict(state_dict, param_map)

            # Load state.
            model.load_state_dict(state_dict, strict=True)

        return model

    # Case 2: Use separate sources for Linear and Non-Linear states.
    linear_params = get_linear_param_names(model)
    nonlinear_params = get_nonlinear_param_names(model)

    # Case 2a:  Linear param states
    linear_state = dict()
    if restore_linear:
        state_dict = get_state_dict(restore_linear)

        if state_dict:

            # Remap param names in state_dict.
            if param_map:
                state_dict = remap_state_dict(state_dict, param_map)

            # Check all desired linear params are present.
            if strict:
                assert set(linear_params) <= set(state_dict.keys()), "".join([
                    "Found linear params in the model ",
                    "which are not present in the checkpoint '{}'.\n".format(
                        restore_linear),
                    "Params not present include:\n {}".format(
                        set(linear_params) - set(state_dict.keys()))
                ])

            # Get and load desired linear params.
            linear_state = {
                param_name: state_dict[param_name]
                for param_name in linear_params
            }

            # Load state.
            model.load_state_dict(linear_state, strict=False)

    # Case 2b:  Non-Linear param states
    nonlinear_state = dict()
    if restore_nonlinear:
        state_dict = get_state_dict(restore_nonlinear)

        if state_dict:

            # Remap param names in state_dict.
            if param_map:
                state_dict = remap_state_dict(state_dict, param_map)

            # Check all desired nonlinear params are present.
            if strict:
                assert set(nonlinear_params) <= set(state_dict.keys()), "".join([
                    "Found nonlinear params in the model ",
                    "which are not present in the checkpoint '{}'.\n".format(
                        restore_nonlinear),
                    "Params not present include:\n {}".format(
                        set(nonlinear_params) - set(state_dict.keys()))
                ])

            # Get and load desired nonlinear params.
            nonlinear_state = {
                param_name: state_dict[param_name]
                for param_name in nonlinear_params
            }

            # Load state.
            model.load_state_dict(nonlinear_state, strict=False)

    # Validate results / quick sanity-check.
    assert set(linear_state.keys()).isdisjoint(nonlinear_state.keys())

    return model


def freeze_all_params(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_linear_params(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
