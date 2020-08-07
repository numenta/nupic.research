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


def load_state_from_checkpoint(
    model,
    chekpoint_path,
    strict=True,
    subset=None,
    resize_buffers=False,
    param_map=None,
    state_dict_transform=None,
):
    """
    A function for flexible loading of torch.nn.Module's.

    :param model: model to load state; instance of torch.nn.Module
    :param chekpoint_path: path to checkpoint
    :param strict: similar to `strict` of pytorch's `load_state_dict`
    :param subset: List of param names to accompany `strict=True`. This enables a user
                   define a set of params that will only be loaded and must be present
                   in both the model and the checkpoint
    :param include_buffers: whether to load the models buffers as well; doesn't work
                            with restore_full_model.
    :param resize_buffers: whether to resize the models buffers before loading the
                           state; this ensures the model has the same buffer sizes as
                           those saved within the checkpoint prior to loading.
                           Otherwise, pytorch may throw an error.
    :param param_map: a dict mapping names of state within the checkpoint to new desired
                      names; this helps when the names of the model's state don't quite
                      match that of the checkpoints. Ex:
                      `param_map = {"features.weight": "features.new_weight"}`
                      where "features.weight" exists within the checkpoint and the model
                      has the attribute `model.features.new_weight`.
    :param state_dict_transform: this is a callable that takes the state_dict and
                                 transforms it in some way; useful for a custom
                                 re-mapping such as parameters with new naming
                                 schemes or formats. The output should be a new
                                 state_dict.
    """

    assert os.path.isfile(chekpoint_path), (
        "Double check the checkpoint exists and is a file."
    )

    # Load the state dict from the checkpoint.
    state_dict = get_state_dict(chekpoint_path)

    assert state_dict is not None, (
        "Couldn't load the state_dict. Maybe check it's in the right format."
    )

    # Remap param names in state_dict.
    if param_map:
        state_dict = remap_state_dict(state_dict, param_map)

    # Check all desired params are present.
    if strict and subset:

        # Ensure subset is present in the checkpoint's state.
        assert set(subset) <= set(state_dict.keys()), "".join([
            "Found params in the subset which are not present in the checkpoint: ",
            f"'{chekpoint_path}'"
            "Params not present include:"
            f"\n {set(subset) - set(state_dict.keys())}"
        ])

        # Ensure subset is present in the model's state.
        model_params = model.state_dict()
        assert set(subset) <= set(model_params.keys()), "".join([
            "Found params in the subset which are not present in the model: ",
            f"'{chekpoint_path}'"
            "Params not present include:"
            f"\n {set(subset) - set(model_params.keys())}"
        ])

        # Retrieve the subset of params.
        new_state_dict = {
            param_name: state_dict[param_name]
            for param_name in subset
        }
        state_dict = new_state_dict
        strict = False  # we now only care about the subset

    # Resize the linear buffers.
    if resize_buffers:
        resize_model_buffers(model, state_dict)  # done in place

    # Apply custom transform.
    if state_dict_transform:
        state_dict = state_dict_transform(state_dict, model)

    # Load state.
    model.load_state_dict(state_dict, strict=strict)

    return model


def load_multi_state(  # noqa: C901
    model,
    restore_full_model=None,
    restore_linear=None,
    restore_nonlinear=None,
    strict=True,
    include_buffers=True,
    resize_buffers=False,
    param_map=None,
    state_dict_transform=None,
):
    """
    A function for flexible loading of torch.nn.Module's.

    :param restore_full_model: path to checkpoint; loads full model, supersedes
                               restore_linear and restore_nonlinear
    :param restore_linear: path to checkpoint; loads linear state (including
                           SparseWeights params), may be used in conjunction with
                           restore_nonlinear.
    :param restore_nonlinear: path to checkpoint; loads non-linear state (the difference
                              between all-state and linear-state), may be used in
                              conjunction with restore_linear.

    Example:
    ```
    model = ResNet()
    model = load_multi_state(
        model,
        restore_linear=path_to_checkpoint_1     # includes linear state
        restore_nonlinear=path_to_checkpoint_2  # includes nonlinear state
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
        load_state_from_checkpoint(
            model,
            restore_full_model,
            strict=strict,
            resize_buffers=resize_buffers,
            param_map=param_map,
            state_dict_transform=state_dict_transform,
        )

        return model

    # Case 2: Use separate sources for Linear and Non-Linear states.
    linear_params = get_linear_param_names(model, include_buffers=include_buffers)
    nonlinear_params = get_nonlinear_param_names(model, include_buffers=include_buffers)

    # Case 2a:  Linear param states
    if restore_linear:

        load_state_from_checkpoint(
            model,
            restore_linear,
            strict=True,  # the subset must be present
            subset=linear_params,
            resize_buffers=resize_buffers,
            param_map=param_map,
            state_dict_transform=state_dict_transform,
        )

    # Case 2b:  Non-Linear param states
    if restore_nonlinear:
        load_state_from_checkpoint(
            model,
            restore_nonlinear,
            strict=True,  # the subset must be present
            subset=nonlinear_params,
            resize_buffers=resize_buffers,
            param_map=param_map,
            state_dict_transform=state_dict_transform,
        )

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


def freeze_all_params(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_linear_params(net):
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            for p in m.parameters():
                p.requires_grad = True
