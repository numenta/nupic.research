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

import math
from pprint import pprint

from sigopt.examples import franke_function

from nupic.research.frameworks.sigopt import SigOptExperiment

"""
Standalone example of SigOpt that uses SigOptExperiment. The example experiment uses
linear constraints, checkpointing, and multitask experiments. Unfortunately
checkpointing and multitask are incompatible, so set the flag appropriately before
running.
"""
USE_CHECKPOINTS = False


def evaluate_model(assignments):
    """
    Evaluate your model with the suggested parameter assignments
    Franke function - http://www.sfu.ca/~ssurjano/franke2d.html
    """
    return franke_function(assignments["x"], assignments["y"])


def get_multitask_config():
    # Docs here: https://app.sigopt.com/docs/overview/multitask
    return dict(
        name="SigOptExperiment multitask example",
        parameters=[
            dict(name="x", type="double", bounds=dict(min=0.0, max=1.0)),
            dict(name="y", type="double", bounds=dict(min=0.0, max=1.0)),
            dict(name="cycle_momentum", type="categorical",
                 categorical_values=["True", "False"]),
        ],
        metrics=[dict(name="function_value", objective="maximize")],
        observation_budget=30,
        project="sigopt-examples",

        # Add a constraint, just for grins: x - y >= 0.1
        linear_constraints=[
            dict(
                type="greater_than",
                threshold=0.1,
                terms=[
                    dict(name="x", weight=1),
                    dict(name="y", weight=-1),
                ],
            ),
        ],

        # Multi task. Costs must be between 0 and 1 and there must be at one task with
        # cost 1.0. We use a trick here to help convert to something meaningful, such as
        # the number of epochs. We store max_epochs in the metadata, and then multiply
        # the task cost by max_epochs to get the actual number of epochs to run.
        tasks=[
            dict(name="epochs30", cost=30.0 / 100.0),
            dict(name="epochs60", cost=60.0 / 100.0),
            dict(name="epochs100", cost=1.0)
        ],
        metadata=dict(max_epochs=100.0)
    )


def get_checkpointing_config():
    return dict(
        name="SigOptExperiment checkpointing example",
        parameters=[
            dict(name="x", type="double", bounds=dict(min=0.0, max=1.0)),
            dict(name="y", type="double", bounds=dict(min=0.0, max=1.0)),
            dict(name="cycle_momentum", type="categorical",
                 categorical_values=["True", "False"]),
        ],
        metrics=[dict(name="function_value", objective="maximize")],
        observation_budget=30,
        project="sigopt-examples",

        # Add a constraint, just for grins: x - y >= 0.1
        linear_constraints=[
            dict(
                type="greater_than",
                threshold=0.1,
                terms=[
                    dict(name="x", weight=1),
                    dict(name="y", weight=-1),
                ],
            ),
        ],

        # Required for checkpointing
        training_monitor={"max_checkpoints": 30},
    )


if __name__ == "__main__":

    if USE_CHECKPOINTS:
        sigopt_config = get_checkpointing_config()
    else:
        sigopt_config = get_multitask_config()

    # Set to None to create a new experiment, or use an existing id to keep
    # populating an existing one.
    experiment_id = 163935

    # Create the SigOptExperiment object and SigOpt experiment if there is no
    # experiment_id.
    s = SigOptExperiment(experiment_id, sigopt_config)
    if experiment_id is None:
        s.create_experiment(sigopt_config)
    print("Created: https://app.sigopt.com/experiment/" + str(s.experiment_id))
    print("Current observation count:", s.get_observation_count())
    pprint(sigopt_config)

    # Test out checkpointing by "running" a suggestion with fake values without
    # completing the observation
    if USE_CHECKPOINTS:
        print("Creating checkpoints...")
        suggestion = s.get_next_suggestion()
        s.create_training_run(suggestion)
        for v in range(1, 20):
            s.create_checkpoint(1.0 / math.exp(1.0 / float(v)))

    # Run a bunch of suggestions with complete observations.
    for _ in range(5):
        suggestion = s.get_next_suggestion()
        value = evaluate_model(suggestion.assignments)
        print()
        print("Observation number:", s.get_observation_count())
        print("   Suggestion: ", suggestion)
        print("   Value:", value)

        # For multitask experiments only, here's an example of how to convert to
        # number of epochs.
        if suggestion.task is not None:
            max_epochs = sigopt_config["metadata"]["max_epochs"]
            print("Suggested task/cost/epochs for this multitask experiment: ",
                  suggestion.task.name, suggestion.task.cost,
                  int(max_epochs * suggestion.task.cost))
        s.update_observation(suggestion, value)

    # Fetch the best configuration and explore your experiment
    best_assignments = s.get_best_assignments()
    if best_assignments is not None:
        best_assignment = best_assignments.assignments
        best_value = evaluate_model(best_assignment)
        print()
        print("Best Assignments: " + str(best_assignment))
        print("Best x value: " + str(best_assignment["x"]))
        print("Best y value: " + str(best_assignment["y"]))
        print("Best model value:", best_value)
    else:
        print("No best assignment yet!")

    print("Explore your experiment: https://app.sigopt.com/experiment/"
          + str(s.experiment_id) + "/analysis")
