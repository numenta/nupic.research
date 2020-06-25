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

import math
import os

import ray
import sigopt
import torch
import torch.optim
from ray import tune

import nupic.research.frameworks.backprop_structure.dataset_managers as datasets
import nupic.research.frameworks.backprop_structure.networks as networks
from nupic.research.frameworks.backprop_structure import experiments
from nupic.research.frameworks.dynamic_sparse.common.ray_custom_loggers import (
    DEFAULT_LOGGERS,
)

# CLIENT_TOKEN = "INSERT KEY HERE"  # Test
CLIENT_TOKEN = "INSERT KEY HERE"  # Prod

BUDGET = 80


class TuneExperiment(tune.Trainable):
    def _setup(self, config):
        super()._setup(config)

        self.sigopt_exp_id = config["sigopt_experiment_id"]
        self.conn = sigopt.Connection(client_token=CLIENT_TOKEN)
        self.suggestion = self.conn.experiments(
            self.sigopt_exp_id).suggestions().create()

        assignments = self.suggestion.assignments
        print(f"Testing assignments {assignments}")

        batch_size = 1 << assignments["log_2_batch_size"]

        params = dict(
            network_class=networks.resnet18_mnist,
            network_args=dict(),

            dataset_class=datasets.MNIST,
            dataset_args={},

            optim_class=getattr(torch.optim, assignments["optimizer"]),
            optim_args=dict(
                lr=math.exp(assignments["log_lr"]),
                weight_decay=math.exp(assignments["log_weight_decay"])
            ),

            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args=dict(
                step_size=assignments["step_size"],
                gamma=assignments["gamma"],
            ),

            training_iterations=30,

            use_tqdm=False,
            batch_size_train=(batch_size, batch_size),
            batch_size_test=1000,
        )

        self.exp = experiments.Supervised(**params)

        self.mean_accuracy = None

    def _train(self):
        result = self.exp.run_epoch(self.iteration)
        self.mean_accuracy = result["mean_accuracy"]
        return result

    def _stop(self):
        print(f"Recording result {self.mean_accuracy} "
              f"for assignments {self.suggestion.assignments}")
        self.conn.experiments(self.sigopt_exp_id).observations().create(
            suggestion=self.suggestion.id,
            value=self.mean_accuracy,
        )


if __name__ == "__main__":
    conn = sigopt.Connection(client_token=CLIENT_TOKEN)
    experiment = conn.experiments().create(
        name="Hello ResNet MNIST",
        parameters=[
            dict(name="log_2_batch_size",
                 type="int",
                 bounds=dict(min=5, max=8)),
            dict(name="log_lr",
                 type="double",
                 bounds=dict(min=math.log(1e-6), max=math.log(3e-1))),
            dict(name="gamma",
                 type="double",
                 bounds=dict(min=0.1, max=0.97)),
            dict(name="step_size",
                 type="int",
                 bounds=dict(min=1, max=10)),
            dict(name="log_weight_decay",
                 type="double",
                 bounds=dict(min=math.log(1e-8), max=math.log(1e-1))),
            dict(name="optimizer",
                 type="categorical",
                 categorical_values=["Adam", "AdamW"]),
        ],
        metrics=[dict(name="function_value", objective="maximize")],
        parallel_bandwidth=(torch.cuda.device_count()
                            if torch.cuda.is_available()
                            else 1),
        observation_budget=BUDGET,
        project="sigopt-examples",
    )

    # experiment = conn.experiments(161691).fetch()

    ray.init(webui_host="0.0.0.0")
    tune.run(
        TuneExperiment,
        name=os.path.basename(__file__).replace(".py", ""),
        config={"sigopt_experiment_id": experiment.id},
        num_samples=(experiment.observation_budget
                     - experiment.progress.observation_count),
        resources_per_trial={
            "cpu": 1,
            "gpu": (1 if torch.cuda.is_available() else 0)},
        loggers=DEFAULT_LOGGERS,
        verbose=1,
    )

    # Update the experiment object
    experiment = conn.experiments(experiment.id).fetch()

    # Fetch the best configuration and explore your experiment
    all_best_assignments = conn.experiments(
        experiment.id).best_assignments().fetch()
    # Returns a list of dict-like Observation objects
    best_assignments = all_best_assignments.data[0].assignments
    print("Best Assignments: " + str(best_assignments))
    print("Best x value: " + str(best_assignments["x"]))
    print("Best y value: " + str(best_assignments["y"]))
    print("Explore your experiment: "
          f"https://app.sigopt.com/experiment/{experiment.id}/analysis")
