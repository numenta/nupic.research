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

from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptExperiment


# Evaluate your model with the suggested parameter assignments
# Franke function - http://www.sfu.ca/~ssurjano/franke2d.html
def evaluate_model(assignments):
    return franke_function(assignments["x"], assignments["y"])


if __name__ == "__main__":

    experiment_id = 163908
    sigopt_config = dict(
        name="SigOptExperiment test",
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

    # Create the SigOptExperiment object and SigOpt experiment if there is no
    # experiment_id.
    s = SigOptExperiment(experiment_id, sigopt_config)
    if experiment_id is None:
        s.create_experiment(sigopt_config)
    print("Created: https://app.sigopt.com/experiment/" + str(s.experiment_id))
    pprint(sigopt_config)
    print("Current observation count:", s.get_observation_count())

    # Test out checkpointing by "running" a suggestion with fake values without
    # completing the observation
    print("Creating checkpoints...")
    suggestion = s.get_next_suggestion()
    s.create_training_run(suggestion)
    for v in range(1, 20):
        s.create_checkpoint(1.0 / math.exp(1.0 / float(v)))

    # Run a bunch more suggestions with complete observations.
    for _ in range(5):
        suggestion = s.get_next_suggestion()
        value = evaluate_model(suggestion.assignments)
        print()
        print("Observation number:", s.get_observation_count())
        print("   Suggestion: ", suggestion.assignments)
        print("   Value:", value)
        s.update_observation(suggestion, value)

    # Fetch the best configuration and explore your experiment
    best_assignments = s.get_best_assignments().assignments
    best_value = evaluate_model(best_assignments)
    print()
    print("Best Assignments: " + str(best_assignments))
    print("Best x value: " + str(best_assignments["x"]))
    print("Best y value: " + str(best_assignments["y"]))
    print("Best model value:", best_value)
    print("Explore your experiment: https://app.sigopt.com/experiment/"
          + str(s.experiment_id) + "/analysis")
