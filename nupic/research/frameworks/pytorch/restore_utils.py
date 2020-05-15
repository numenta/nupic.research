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

# ----------------
# Main utils
# ----------------


def load_multi_state(
    model,
    restore_full_model=None,
    restore_linear=None,
    restore_nonlinear=None,
    strict=True,
    include_buffers=True,
    resize_buffers=False,
    param_map=None,
):
    """
    A function for flexible loading of torch.nn.Module's.

    :param model: model to load state; instance of torch.nn.Module
    :param restore_full_model: path to checkpoint; loads full model, supersedes
                               restore_linear and restore_nonlinear
    :param restore_linear: path to checkpoint; loads linear state (including
                           SparseWeights params), may be used in conjunction with
                           restore_nonlinear.
    :param restore_nonlinear: path to checkpoint; loads non-linear state (the difference
                              between all-state and linear-state), may be used in
                              conjunction with restore_linear.
    :param strict: similar to `strict` of pytorch's `load_state_dict`; If True, the
                   loadable state of models params must be a subset (<=) of the
                   state stored within the checkpoint. If False, the overlap between
                   the model's loadable state and checkpoint's save state will be loaded
    :param include_buffers: whether to load the models buffers as well
    :param resize_buffers: whether to resize the models buffers before loading the
                           state; this ensure the model has the same buffer sizes as
                           those saved within the checkpoint prior to loading.
                           Otherwise, pytorch may throw an error.
    :param param_map: a dict mapping names of state within the checkpoint to new desired
                      names; this helps when the names of the model's state don't quite
                      match that of the checkpoints. Ex:
                      `param_map = {"features.weight": "features.new_weight"}`
                      where `model.weight` exists within the checkpoint and the model
                      has the attribute `model.features.new_weight`.

    Example:
    ```
    model = ResNet()
    model = load_multi_state(
        model,
        restore_linear=path_to_checkpoint_1     # include linear state
        restore_nonlinear=path_to_checkpoint_2  # include nonlinear state
    )
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
    linear_params = get_linear_param_names(model, include_buffers=include_buffers)
    nonlinear_params = get_nonlinear_param_names(model, include_buffers=include_buffers)

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

            # Resize the linear buffers.
            if resize_buffers:
                resize_model_buffers(model, linear_state)  # done in place

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

            # Resize the linear buffers.
            if resize_buffers:
                resize_model_buffers(model, nonlinear_state)  # done in place

            # Load state.
            model.load_state_dict(nonlinear_state, strict=False)

    # Validate results / quick sanity-check.
    assert set(linear_state.keys()).isdisjoint(nonlinear_state.keys())

    return model


# -------------------
# Supplemental utils
# -------------------


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
    Ex:
    ```
    param_map["foo"] == "bar"
    state_dict["foo"] == new_state_dict["bar"]
    """

    new_state_dict = {}
    assert set(param_map.keys()) <= set(state_dict.keys()), \
        "The given map should be from keys that are subset of the loadable params."
    for param, state in state_dict.items():

        if param in param_map:
            new_param = param_map[param]
            new_state_dict[new_param] = state
        else:
            new_state_dict[param] = state

    return new_state_dict


def _get_sub_module(module, name):
    """
    Gets a submodule either by name or index - pytorch either uses names for module
    attributes (e.g. "module.classifier") or indices for sequential models
    (e.g. `module[0]`).
    ```
    """
    if name.isdigit():
        return module[int(name)]
    else:
        return getattr(module, name)


def get_module_attr(module, name):
    """
    Get model attribute of torch.nn.Module given its name.
    Ex:
    ```
    name = "features.stem.weight"
    weight = get_module_attr(net, name)
    ```
    """

    # Split off the last sub-name.
    # Ex. "features.stem.weight" -> ("features.stem", "weight")
    parent_name, sub_name = (name.split(".", 1) + [None])[0:2]

    if sub_name is not None:
        parent_module = _get_sub_module(module, parent_name)
        sub_module = get_module_attr(parent_module, sub_name)
        return sub_module
    else:
        sub_module = _get_sub_module(module, parent_name)
        return sub_module


def set_module_attr(module, name, value):
    """
    Set model attribute of torch.nn.Module given its name.
    Ex:
    ```
    name = "features.stem.weight"
    weight = Parameter(...)
    set_module_attr(net, name, weight)
    ```
    """

    # Split name: pytorch convention uses "." for each child module.
    all_names = name.split(".")

    # Get all names except the last.
    # Ex. "features.stem.weight" -> "features.stem"
    parents_names = all_names[:-1]

    if not parents_names:
        setattr(module, name, value)

    else:
        # Get the parent module of the last child module.
        parent_name = ".".join(parents_names)
        parent_module = get_module_attr(module, parent_name)

        # Set the new value of the last child module.
        child_name = all_names[-1]
        setattr(parent_module, child_name, value)


def resize_model_buffers(model, state_dict):
    """
    Resizes the models buffers by initializing a zero tensor
    matching the same size as that within the state_dict.
    """

    for name, init_buffer in list(model.named_buffers()):

        if name not in state_dict:
            continue

        saved_buffer = state_dict[name]
        new_buffer = torch.zeros(
            saved_buffer.shape,
            dtype=init_buffer.dtype,
            layout=init_buffer.layout,
            device=init_buffer.device,
        )

        set_module_attr(model, name, new_buffer)
