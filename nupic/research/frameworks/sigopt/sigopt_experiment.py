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
import os

from sigopt import Connection


class SigOptExperiment:
    """
    Class used to wrap around the SigOpt API and designed to be used in any experiment
    runner. A particular experiment runner, such as ImagenetTrainable, will want to
    subclass and redefine update_config_with_suggestion() to be specific to their
    config.
    """

    def __init__(self, experiment_id=None, sigopt_config=None):
        """
        Initiate a connection to the SigOpt API and optionally store the id
        and config for an existing experiment.  The SigOpt API key should be
        defined in the environment variable 'SIGOPT_KEY'.

        :param experiment_id: (int) An existing experiment id.
        :param sigopt_config: (dict) The config used to create experiment id.
        """
        self.experiment_id = experiment_id
        self.sigopt_config = sigopt_config
        self.conn = None
        self.training_run = None

        self.api_key = os.environ.get("SIGOPT_KEY", None)
        if self.api_key is None:
            self.api_key = os.environ.get("SIGOPT_DEV_KEY", None)
        assert self.api_key is not None, "No SigOpt API key!"

        try:
            self.conn = Connection(client_token=self.api_key)
        except Exception:
            print("Could not connect to SigOpt!")
            raise

    def create_experiment(self, sigopt_config=None):
        """
        Create a new sigopt experiment using the config.

        :param sigopt_config: dictionary containing the SigOpt experiment parameters. If
        this is None, this method does nothing and acts as a pass through.

        If sigopt_config contains the key experiment_id we reuse the corresponding
        existing experiment. If None, or this key doesn't exist, we create a brand new
        experiment using sigopt_config, and update sigopt_config with the new
        experiment_id.
        """
        if sigopt_config is None:
            return
        self.sigopt_config = sigopt_config

        # Create SigOpt experiment if requested
        experiment = self.conn.experiments().create(**sigopt_config)
        self.experiment_id = experiment.id
        self.sigopt_config = sigopt_config
        sigopt_config["experiment_id"] = experiment.id
        print("Created experiment: https://app.sigopt.com/experiment/"
              + str(experiment.id))

        return self.experiment_id

    def get_next_suggestion(self):
        experiment = self.conn.experiments(self.experiment_id).fetch()
        suggestion = self.conn.experiments(experiment.id).suggestions().create()
        return suggestion

    def update_observation(self, suggestion, values):
        self.conn.experiments(self.experiment_id).observations().create(
            suggestion=suggestion.id,
            values=values,
        )

    def get_observation_count(self):
        experiment = self.conn.experiments(self.experiment_id).fetch()
        return experiment.progress.observation_count

    def observations(self):
        observations = self.conn.experiments(self.experiment_id).observations().fetch()
        return observations.data

    def create_observation(self, assignments, value, task=None):
        """
        Create an observation with custom assignments.
        """
        if task is None:
            self.conn.experiments(self.experiment_id).observations().create(
                assignments=assignments,
                value=value
            )
        else:
            self.conn.experiments(self.experiment_id).observations().create(
                assignments=assignments,
                value=value,
                task=task
            )

    def open_suggestions(self):
        suggestions = self.conn.experiments(self.experiment_id).suggestions().fetch(
            state="open")
        return suggestions.data

    def delete_suggestion(self, suggestion):
        self.conn.experiments(self.experiment_id).suggestions(
            suggestion.id).delete()

    def delete_open_suggestions(self):
        """
        Delete all open suggestions.
        """
        self.conn.experiments(self.experiment_id).suggestions().delete(state="open")

    def get_best_assignments(self):
        a = self.conn.experiments(self.experiment_id).best_assignments().fetch().data

        # If you have not completed any observations, or you are early on a multitask
        # experiment, you could have no best assignments.
        if len(a) == 0:
            return None
        else:
            return a[0]

    def create_training_run(self, suggestion):
        """
        Create training run using this suggestion. The training run is cached
        for later creating checkpoints.
        """
        self.training_run = self.conn.experiments(
            self.experiment_id).training_runs().create(suggestion=suggestion.id)

    def create_checkpoint(self, metric_value):
        """
        Create a checkpoint for the (single) metric that is being optimized. In order to
        use this you must have specified training_monitor when creating the experiment,
        and must have called create_training_run() for this training run.
        """
        assert self.training_run is not None
        self.conn.experiments(self.experiment_id).training_runs(
            self.training_run.id).checkpoints().create(
            values=[dict(name=self.sigopt_config["metrics"][0]["name"],
                         value=metric_value)],
        )

    def get_experiment_details(self):
        return self.conn.experiments(self.experiment_id).fetch()

    @staticmethod
    def update_config_with_suggestion(config, suggestion):
        """
        Given a SigOpt suggestion, update this config dict.
        """
        config.update(suggestion.assignments)


class SigOptImagenetExperiment(SigOptExperiment):
    """
    A subclass of SigOptExperiment used to sit between an experiment runner (such as
    Ray) and the ImagenetExperiment class. update_config_with_suggestion() is specific
    to our ImagenetExperiment config.
    """

    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update our various config dicts properly so that we
        can pass it onto ImagenetExperiment.
        """
        assignments = suggestion.assignments

        assert "optimizer_args" in config
        assert "lr_scheduler_args" in config

        # For multi-task experiments where epoch is the task. Must have a metadata
        # field called max_epochs.
        if suggestion.task is not None and "epoch" in suggestion.task.name:
            max_epochs = self.sigopt_config["metadata"]["max_epochs"]
            epochs = int(max_epochs * suggestion.task.cost)
            print("Suggested task/cost/epochs for this multitask experiment: ",
                  suggestion.task.name, suggestion.task.cost, epochs)
            config["epochs"] = epochs
            config["lr_scheduler_args"]["epochs"] = epochs

        # Optimizer args
        if "log_lr" in assignments:
            config["optimizer_args"]["lr"] = math.exp(assignments["log_lr"])
            assignments.pop("log_lr")

        if "momentum" in assignments:
            config["optimizer_args"]["momentum"] = assignments["momentum"]
            config["lr_scheduler_args"]["max_momentum"] = assignments["momentum"]
            assignments.pop("momentum")

        if "weight_decay" in assignments:
            config["optimizer_args"]["weight_decay"] = assignments["weight_decay"]
            assignments.pop("weight_decay")

        # lr_scheduler args
        if "gamma" in assignments:
            config["lr_scheduler_args"]["gamma"] = assignments["gamma"]
            assignments.pop("gamma")

        if "step_size" in assignments:
            config["lr_scheduler_args"]["step_size"] = assignments["step_size"]
            assignments.pop("step_size")

        # Parameters for OneCycleLR
        if "pct_start" in assignments:
            # Ensure integer number of epochs in pct start to avoid crash.
            start_epochs = int(assignments["pct_start"] * config["epochs"] + 0.5)
            pct_start = float(start_epochs) / config["epochs"]
            config["lr_scheduler_args"]["pct_start"] = pct_start
            assignments.pop("pct_start")

        if "cycle_momentum" in assignments:
            config["lr_scheduler_args"]["cycle_momentum"] = \
                eval(assignments["cycle_momentum"])
            assignments.pop("cycle_momentum")

        if "max_lr" in assignments or "init_lr" in assignments:
            # From the OneCycleLR docs:
            #   initial_lr = max_lr/div_factor
            #   min_lr = initial_lr/final_div_factor
            max_lr = assignments.get("max_lr", 6.0)
            init_lr = assignments.get("init_lr", 1.0)

            config["lr_scheduler_args"]["max_lr"] = max_lr
            config["lr_scheduler_args"]["div_factor"] = max_lr / init_lr
            config["lr_scheduler_args"]["final_div_factor"] = init_lr / 0.00025
            assignments.pop("init_lr", None)
            assignments.pop("max_lr", None)

        config.update(assignments)
