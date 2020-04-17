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
import time

import numpy as np
import ray
import sigopt
import torch
import torch.optim
from ray import tune

import nupic.research.frameworks.backprop_structure.dataset_managers as datasets
import nupic.research.frameworks.backprop_structure.networks as networks
from nupic.research.frameworks.backprop_structure import experiments
from nupic.research.frameworks.backprop_structure.experiments import mixins

# CLIENT_TOKEN = "[Insert here]"  # Test
CLIENT_TOKEN = "[Insert here]"  # Prod

BUDGET = 1000


class VDropExperiment(mixins.ConstrainParameters,
                      mixins.LogStructure,
                      mixins.Regularize,
                      experiments.Supervised):
    pass


class TuneExperiment(tune.Trainable):
    def _setup(self, config):
        super()._setup(config)

        self.time_start = time.time()

        self.sigopt_exp_id = config["sigopt_experiment_id"]
        self.conn = sigopt.Connection(client_token=CLIENT_TOKEN)
        self.suggestion = self.conn.experiments(
            self.sigopt_exp_id).suggestions().create()

        assignments = self.suggestion.assignments
        print(f"Testing assignments {assignments}")

        batch_size = 0x1 << int(assignments["log2_batch_size"])
        warmup_start_iteration = round(assignments["reg_warmup_start_time"]
                                       * (assignments["num_epochs"] - 1))
        warmup_end_iteration = round(assignments["reg_warmup_end_time"]
                                     * (assignments["num_epochs"] - 1))
        reg_factor_start = math.exp(assignments["log_reg_factor_start"])
        reg_factor_end = math.exp(assignments["log_reg_factor_end"])

        self.warmup_start_iteration = warmup_start_iteration
        self.warmup_end_iteration = warmup_end_iteration

        params = dict(
            logdir=self.logdir,

            network_class=networks.gsc_lenet_vdrop,
            network_args=dict(),

            dataset_class=datasets.PreprocessedGSC,
            dataset_args=dict(),

            optim_class=torch.optim.Adam,
            optim_args=dict(
                lr=math.exp(assignments["log_lr"]),
            ),

            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args=dict(
                step_size=1,
                gamma=assignments["gamma_prewarmup"],
            ),

            training_iterations=assignments["num_epochs"],

            use_tqdm=False,
            batch_size_test=128,
            batch_size_train=(batch_size, batch_size),

            log_verbose_structure=False,

            reg_schedule=dict(
                ([(0, reg_factor_start)]
                 if warmup_start_iteration > 0
                 else [])
                + list(
                    zip(range(warmup_start_iteration, warmup_end_iteration),
                        np.linspace(reg_factor_start, reg_factor_end,
                                    warmup_end_iteration - warmup_start_iteration,
                                    endpoint=False)))
                + [(warmup_end_iteration, reg_factor_end)]
            ),
            downscale_reg_with_training_set=True,
        )

        self.exp = VDropExperiment(**params)

        self.mean_accuracy = None

    def _train(self):
        assignments = self.suggestion.assignments
        if self.iteration == self.warmup_end_iteration:
            self.exp.lr_scheduler.gamma = assignments["gamma_postwarmup"]
        elif self.iteration == self.warmup_start_iteration:
            self.exp.lr_scheduler.gamma = assignments["gamma_warmup"]

        result = self.exp.run_epoch(self.iteration)
        self.mean_accuracy = result["mean_accuracy"]
        self.inference_nz = result["inference_nz"]

        return result

    def _stop(self):
        time_stop = time.time()
        duration = time_stop - self.time_start

        print(f"Recording result {self.mean_accuracy}, {self.inference_nz} "
              f"for assignments {self.suggestion.assignments}")

        self.conn.experiments(self.sigopt_exp_id).observations().create(
            suggestion=self.suggestion.id,
            values=[{"name": "log_error",
                     "value": math.log(1 - self.mean_accuracy)},
                    {"name": "log_num_nonzero_weights",
                     "value": math.log(self.inference_nz)},
                    {"name": "duration_seconds", "value": duration}],
        )


if __name__ == "__main__":
    conn = sigopt.Connection(client_token=CLIENT_TOKEN)
    experiment = conn.experiments().create(
        name="Variational Dropout GSC",
        parameters=[
            dict(name="num_epochs",
                 type="int",
                 bounds=dict(min=30, max=200)),
            dict(name="log2_batch_size",
                 type="int",
                 bounds=dict(min=5, max=8)),
            dict(name="log_lr",
                 type="double",
                 bounds=dict(min=math.log(1e-5), max=math.log(3e-1))),
            dict(name="gamma_prewarmup",
                 type="double",
                 bounds=dict(min=0.5, max=1.0)),
            dict(name="gamma_warmup",
                 type="double",
                 bounds=dict(min=0.5, max=1.0)),
            dict(name="gamma_postwarmup",
                 type="double",
                 bounds=dict(min=0.5, max=0.97)),

            # SigOpt doesn't support constraints with integers, so we need to
            # use doubles rather than the more obvious choice of using epoch
            # integers.
            dict(name="reg_warmup_start_time",
                 type="double",
                 bounds=dict(min=0.0, max=1.0)),
            dict(name="reg_warmup_end_time",
                 type="double",
                 bounds=dict(min=0.0, max=1.0)),

            dict(name="log_reg_factor_start",
                 type="double",
                 bounds=dict(min=math.log(1e-4),
                             max=math.log(1.0))),
            dict(name="log_reg_factor_end",
                 type="double",
                 bounds=dict(min=math.log(0.1),
                             max=math.log(10.0))),
        ],

        linear_constraints=[
            dict(
                type="greater_than",
                threshold=0,
                terms=[
                    dict(name="reg_warmup_end_time", weight=1),
                    dict(name="reg_warmup_start_time", weight=-1),
                ],
            ),
            dict(
                type="greater_than",
                threshold=0,
                terms=[
                    dict(name="log_reg_factor_end", weight=1),
                    dict(name="log_reg_factor_start", weight=-1),
                ],
            ),
        ],

        metrics=[
            dict(name="log_error", objective="minimize",
                 strategy="optimize"),
            dict(name="log_num_nonzero_weights", objective="minimize",
                 strategy="optimize"),
            dict(name="duration_seconds", objective="minimize",
                 strategy="store"),
        ],
        parallel_bandwidth=32,
        observation_budget=BUDGET,
        project="gsc-vdrop",
    )

    # experiment = conn.experiments(185010).fetch()

    ray.init(address="auto", redis_password="5241590000000000")
    tune.run(
        TuneExperiment,
        name=os.path.basename(__file__).replace(".py", ""),
        config={"sigopt_experiment_id": experiment.id},
        num_samples=(experiment.observation_budget
                     - experiment.progress.observation_count),
        local_dir=os.path.expanduser("~/nta/results/ray_results"),
        checkpoint_freq=0,
        checkpoint_at_end=False,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1,
        },
        verbose=1,
        queue_trials=True,
        sync_to_driver=False,
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
