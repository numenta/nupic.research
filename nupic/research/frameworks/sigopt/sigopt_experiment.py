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
import logging
import os

from sigopt import Connection

from nupic.research.frameworks.pytorch.imagenet.imagenet_experiment import (
    ImagenetExperiment
)

__all__ = ["SigOptExperiment"]


class SigOptExperiment:
    """
    SigOpt class used to sit between an experiment runner (such as Ray) and the
    ImagenetExperiment class.
    """

    def __init__(self):
        self.logger = None
        self.experiment_id = None
        self.imagenet_experiment = ImagenetExperiment()

        self.api_key = os.environ.get("SIGOPT_KEY", None)
        if self.api_key is None:
            self.api_key = os.environ.get("SIGOPT_DEV_KEY", None)
        assert self.api_key is not None, "No SigOpt API key!"

        try:
            self.conn = Connection(client_token=self.api_key)
        except:
            print("Could not connect to SigOpt!")
            raise


    def setup_experiment(self, config, sigopt_config, experiment_id=None):

        """
        Configure the sigopt experiment for training

        :param config: Dictionary containing ImagenetExperiment config parameters.
                       This dict will be updated using the SigOpt suggestions and
                       then passed onto ImagenetExperiment.setup()

        :param sigopt_config: Dictionary containing the SigOpt experiment parameters

        :param experiment_id: If None, create a new experiment ID. If not None,
                              reuse an existing experiment.
        """
        # Configure logging related stuff
        log_format = config.get("log_format", logging.BASIC_FORMAT)
        log_level = getattr(logging, config.get("log_level", "INFO").upper())
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger(config.get("name", type(self).__name__))
        self.logger.setLevel(log_level)
        self.logger.addHandler(console)

        # Create SigOpt experiment if needed
        if experiment_id is None:
            experiment = s.conn.experiments().create(
                name=sigopt_config["name"],
                # Define which parameters you would like to tune
                parameters=sigopt_config["parameters"],
                metrics=[dict(name='function_value', objective='maximize')],
                parallel_bandwidth=1,
                # Define an Observation Budget for your experiment
                observation_budget=sigopt_config["observation_budget"],
                project=sigopt_config["project"],
            )
            self.experiment_id = experiment.id
            self.logger.info(
                "Created experiment: https://app.sigopt.com/experiment/%s",
                experiment.id)

        # Get the next suggestion
        experiment = s.conn.experiments(self.experiment_id).fetch()
        suggestion = s.conn.experiments(experiment.id).suggestions().create()
        self.logger.debug("   suggestion: %s", suggestion.assignments)

        # Configure model config
        config.update(suggestion)

        # Call ImagenetExperiment setup()
        self.imagenet_experiment.setup_experiment(config)

        # Somewhere we need to return the end value. We also need to be able to
        # send in validation results after every validation call so that we can
        # do early stopping.

    def run_epoch(self, epoch):
        return self.imagenet_experiment.run_epoch(epoch)

    def get_state(self):
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        return self.imagenet_experiment.get_state()

    def set_state(self, state):
        """
        Restore the experiment from the state returned by `get_state`
        :param state: dictionary with "model", "optimizer", "lr_scheduler", and "amp"
                      states
        """
        self.imagenet_experiment.set_state(state)

    def stop_experiment(self):
        self.imagenet_experiment.stop_experiment()

    def get_node_ip(self):
        """Returns the IP address of the current ray node."""
        return self.imagenet_experiment.get_node_ip()


from sigopt.examples import franke_function

def evaluate_model(assignments):
    return franke_function(assignments['x'], assignments['y'])


if __name__ == "__main__":
    s = SigOptExperiment()
    experiment = s.conn.experiments(160659).fetch()
    # experiment = s.conn.experiments().create(
    #     name='Franke Optimization (Python)',
    #     # Define which parameters you would like to tune
    #     parameters=[
    #         dict(name='x', type='double', bounds=dict(min=0.0, max=1.0)),
    #         dict(name='y', type='double', bounds=dict(min=0.0, max=1.0)),
    #     ],
    #     metrics=[dict(name='function_value', objective='maximize')],
    #     parallel_bandwidth=1,
    #     # Define an Observation Budget for your experiment
    #     observation_budget=30,
    #     project="sigopt-examples",
    # )
    print("Created experiment: https://app.sigopt.com/experiment/" + experiment.id)

    # Run the Optimization Loop until the Observation Budget is exhausted
    for _ in range(5):
        print("observation count:", experiment.progress.observation_count)
        suggestion = s.conn.experiments(experiment.id).suggestions().create()
        print("   suggestion: ", suggestion.assignments)
        value = evaluate_model(suggestion.assignments)
        s.conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        # Update the experiment object
        experiment = s.conn.experiments(experiment.id).fetch()

    # Fetch the best configuration and explore your experiment
    all_best_assignments = s.conn.experiments(experiment.id).best_assignments().fetch()
    # Returns a list of dict-like Observation objects
    best_assignments = all_best_assignments.data[0].assignments
    best_value = evaluate_model(best_assignments)
    print("Best Assignments: " + str(best_assignments))
    print("Best x value: " + str(best_assignments['x']))
    print("Best y value: " + str(best_assignments['y']))
    print("Best model value:", best_value)
    print("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")
