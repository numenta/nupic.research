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


def load_ray_tune_experiments(
    experiment_path, load_results=False
):
    """Load multiple ray tune experiment state. This is useful if you want
    to collect the results from multiple runs into one collection

    :param experiment_path: ray tune experiment directory
    :type experiment_path: str
    :param load_results: Whether or not to load experiment results
    :type load_results: bool

    :return: list of dictionaries with ray tune experiment state results
    :rtype: dict
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
