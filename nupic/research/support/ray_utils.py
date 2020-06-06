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
import glob
import json
import os

import ray
import torch
from ray.tune.suggest.variant_generator import (
    generate_variants, format_vars
)
from ray.tune.trial_runner import _TuneFunctionDecoder


def load_ray_tune_experiments(
    experiment_path, load_results=False
):
    """Load multiple ray tune experiment states. This is useful if you want
    to collect the results from multiple runs into one collection

    :param experiment_path: ray tune experiment directory
    :type experiment_path: str
    :param load_results: Whether or not to load experiment results
    :type load_results: bool

    :return: list of dictionaries with ray tune experiment state results
    :rtype: list(dict)
    """
    experiment_state_paths = glob.glob(
        os.path.join(experiment_path, "experiment_state*.json")
    )
    if not experiment_state_paths:
        raise RuntimeError("No experiment state found: " + experiment_path)

    experiment_states = [
        load_ray_tune_experiment(experiment_path, filename, load_results)
        for filename in experiment_state_paths
    ]

    return experiment_states


def load_ray_tune_experiment(
    experiment_path, experiment_filename=None, load_results=False
):
    """Load ray tune experiment state.

    :param experiment_path: ray tune experiment directory
    :type experiment_path: str
    :param experiment_filename: experiment state to load. None for latest
    :type experiment_filename: str
    :param load_results: Whether or not to load experiment results
    :type load_results: bool

    :return: dictionary with ray tune experiment state results
    :rtype: dict
    """
    if experiment_filename is None:
        experiment_state_paths = glob.glob(
            os.path.join(experiment_path, "experiment_state*.json")
        )
        if not experiment_state_paths:
            raise RuntimeError("No experiment state found: " + experiment_path)

        # Get latest experiment only
        experiment_filename = max(experiment_state_paths)

    # Load experiment checkpoints
    with open(experiment_filename) as f:
        experiment_state = json.load(f)

    if "checkpoints" not in experiment_state:
        raise RuntimeError("Experiment state is invalid; no checkpoints found!")

    all_experiments = experiment_state["checkpoints"]
    for experiment in all_experiments:
        # Make logs relative to experiment path
        logdir = experiment["logdir"]
        logpath = os.path.join(experiment_path, os.path.basename(logdir))
        experiment["results"] = None

        if load_results:
            # Load results
            result_file = os.path.join(logpath, "result.json")
            if not result_file:
                print("No results for experiment:", experiment["experiment_tag"])
                continue

            with open(result_file) as f:
                rows = f.readlines()
                if not rows:
                    print("No data for experiment:", experiment["experiment_tag"])
                    continue
                experiment["results"] = [json.loads(s) for s in rows]

    return experiment_state


def get_last_checkpoint(results_dir):
    """
    Find the last checkpoint given the ray tune results directory

    :param results_dir:  ray tune results directory
    :return: checkpoint file or None when no checkpoints are found
    """
    # Get experiment states sorted by date
    experiment_state_json = glob.glob(f"{results_dir}/experiment_state-*.json")
    experiment_state_json.sort(key=os.path.getmtime, reverse=True)

    for file_path in experiment_state_json:
        with open(file_path, mode="r") as f:
            experiment_state = json.load(f, cls=_TuneFunctionDecoder)

        # Get newest checkpoint from last experiment
        checkpoint = next(reversed(experiment_state["checkpoints"]))
        newest_checkpoint = checkpoint["checkpoint_manager"].newest_checkpoint

        # Update checkpoint location
        checkpoint_file = newest_checkpoint.value
        if checkpoint_file is not None:
            checkpoint_file = checkpoint_file.replace(
                checkpoint["local_dir"], str(results_dir))
            return checkpoint_file

    # No checkpoint available
    return None


def register_torch_serializers():
    """
    Registers ray custom serializer and deserializer for torch.tensor types. According
    to the ray documentation:
        "The serializer and deserializer are used when transferring objects of cls
        across processes and nodes."

    In particular, these are found handy when array-like logs (from a
    tune.Trainable) are transfered across nodes.

    Example:
    ```
    ray.init()
    register_torch_serializers()
    ```
    """

    # Register serializer and deserializer - needed when logging arrays and tensors.
    def serializer(obj):
        if obj.requires_grad:
            obj = obj.detach()
        if obj.is_cuda:
            return obj.cpu().numpy()
        else:
            return obj.numpy()

    for tensor_type in [
        torch.FloatTensor,
        torch.DoubleTensor,
        torch.HalfTensor,
        torch.ByteTensor,
        torch.CharTensor,
        torch.ShortTensor,
        torch.IntTensor,
        torch.LongTensor,
        torch.Tensor,
    ]:
        def deserializer(serialized_obj):
            return tensor_type(serialized_obj)  # cast to tensor_type

        ray.register_custom_serializer(
            tensor_type, serializer=serializer, deserializer=deserializer
        )


def generate_trial_variants(config):
    """
    Generate configuration for each trial variant evaluating 'ray.tune'
    functions (grid_search, sample_from, ...) into its final values.

    :param config: Ray tune configuration with 'ray.tune' functions
    :return: list of dict for each trial configuration variant
    """
    trials = []
    num_samples = config["num_samples"]
    for i in range(num_samples):
        for variables, variant in generate_variants(config):
            # Update experiment tag with variant vars
            if len(variables) > 0:
                variant["experiment_tag"] = f"{i}_{format_vars(variables)}"
            else:
                variant["experiment_tag"] = str(i)

            trials.append(variant)

    return trials
