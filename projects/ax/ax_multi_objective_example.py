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

import pprint
from collections import defaultdict

import numpy as np
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

from nupic.research.frameworks.ax import CoreAxClient

#
# PART 1: RUN THE EXPERIMENT
#


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


NUM_TRIALS = 20
NUM_RANDOM = 10

search_space = SearchSpace(parameters=[
    RangeParameter("x", ParameterType.FLOAT, lower=12.2, upper=602.2),
])

optimization_config = OptimizationConfig(
    objective=MultiObjective(
        metrics=[
            # Currently MultiObjective doesn't work with lower_is_better=True.
            # https://github.com/facebook/Ax/issues/289
            Metric(name="neg_distance17",
                   lower_is_better=False),
            Metric(name="neg_distance33",
                   lower_is_better=False)],
        minimize=False,
    ),
)

generation_strategy = choose_generation_strategy(
    search_space, num_trials=NUM_TRIALS, num_initialization_trials=NUM_RANDOM
)

ax_client = CoreAxClient(
    experiment=Experiment(search_space=search_space,
                          optimization_config=optimization_config),
    generation_strategy=generation_strategy)


for _ in range(NUM_TRIALS):
    parameters, trial_index = ax_client.get_next_trial(
        model_gen_options={
            "acquisition_function_kwargs": {
                "random_scalarization": True,
            },
        }
    )
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={
            "neg_distance17": (-example_f17(parameters["x"]), None),
            "neg_distance33": (-example_f33(parameters["x"]), None)
        })


#
# PART 2: PRINT THE RESULTS
#


def get_parameters_and_results(ax_client):
    trial_by_arm_name = defaultdict(dict)
    for _, row in ax_client.experiment.fetch_data().df.iterrows():
        trial_by_arm_name[row["arm_name"]][row["metric_name"]] = row["mean"]

    trials = []
    arms_by_name = ax_client.experiment.arms_by_name
    for arm_name, metrics in trial_by_arm_name.items():
        trial = dict(arms_by_name[arm_name].parameters)
        trial.update(metrics)
        trials.append(trial)
    return trials


def filter_to_pareto(trials, metrics):
    scores = np.empty((len(trials), len(metrics)), dtype=np.float)
    for i, trial in enumerate(trials):
        for j, metric in enumerate(metrics):
            scores[i, j] = trial[metric]

    filtered = []
    for i in range(len(trials)):
        s = scores[i]
        on_frontier = True
        for i2 in range(len(trials)):
            if i2 != i:
                s2 = scores[i2]
                if (s2 > s).all():
                    on_frontier = False
        if on_frontier:
            filtered.append(trials[i])

    return filtered


print("Pareto frontier:")
pprint.pprint(sorted(
    filter_to_pareto(
        get_parameters_and_results(ax_client),
        ["neg_distance17", "neg_distance33"]),
    key=lambda x: x["neg_distance17"],
    reverse=True))
