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

import json
import os

import ray
from ax import (
    Experiment,
    Metric,
    OptimizationConfig,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.core.objective import MultiObjective
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ray import tune

from nupic.research.frameworks.ax import AxService, CoreAxClient, NonblockingAxSearch

NUM_TRIALS = 20
NUM_RANDOM = 10
SERIALIZED_FILEPATH = None  # Set to enable resuming experiments.


# The two loss functions
def example_f17(x):
    # Distance from a multiple of 17
    mod17 = x - (x // 17) * 17
    if mod17 > 17 // 2:
        return 17 - mod17
    else:
        return mod17


def example_f33(x):
    # Distance from a multiple of 33
    mod33 = x - (x // 33) * 33
    if mod33 > 33 // 2:
        return 33 - mod33
    else:
        return mod33


class MyTuneExperiment(tune.Trainable):
    def _setup(self, config):
        self.x = config["x"]

    def _train(self):
        return {
            "distance17": example_f17(self.x),
            "distance33": example_f33(self.x),
            "done": True,
        }


class MyAxClient(CoreAxClient):
    def __init__(self, serialized_filepath=None):
        # Give ourselves the ability to resume this experiment later.
        self.serialized_filepath = serialized_filepath
        if serialized_filepath is not None and os.path.exists(
                serialized_filepath):
            with open(serialized_filepath, "r") as f:
                serialized = json.load(f)
            self.initialize_from_json_snapshot(serialized)
        else:
            # Create a CoreAxClient.
            search_space = SearchSpace(parameters=[
                RangeParameter("x", ParameterType.FLOAT, lower=12.2,
                               upper=602.2),
            ])

            optimization_config = OptimizationConfig(
                objective=MultiObjective(
                    metrics=[
                        # Currently MultiObjective doesn't work with
                        # lower_is_better=True.
                        # https://github.com/facebook/Ax/issues/289
                        Metric(name="neg_distance17",
                               lower_is_better=False),
                        Metric(name="neg_distance33",
                               lower_is_better=False)],
                    minimize=False,
                ),
            )

            generation_strategy = choose_generation_strategy(
                search_space,
                enforce_sequential_optimization=False,
                no_max_parallelism=True,
                num_trials=NUM_TRIALS,
                num_initialization_trials=NUM_RANDOM)

            super().__init__(
                experiment=Experiment(search_space=search_space,
                                      optimization_config=optimization_config),
                generation_strategy=generation_strategy,
                verbose=True)

    def save(self):
        if self.serialized_filepath is not None:
            with open(self.serialized_filepath, "w") as f:
                json.dump(self.to_json_snapshot(), f)


def metric_function(result):
    return {
        "neg_distance17": (-result["distance17"], None),
        "neg_distance33": (-result["distance33"], None),
    }


ray.init()
ax_service = AxService(MyAxClient, SERIALIZED_FILEPATH)
tune.run(
    MyTuneExperiment,
    num_samples=20,
    search_alg=NonblockingAxSearch(
        ax_service.frontend, metric_function, max_concurrent=20,

        # Instruct it not to poll Ax too many times before any results are
        # returned. (See docstring for more details.)
        m_suggestions_allowed_before_nth_completion=(
            (NUM_RANDOM, 1)
        )
    ),
    resources_per_trial={
        "cpu": 1,
    },
    verbose=1,
)
