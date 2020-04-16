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
import math
import os
import time
import uuid

import numpy as np
import ray
import torch.optim
from ax import (
    Experiment,
    Metric,
    OptimizationConfig,
    OrderConstraint,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.core.objective import MultiObjective
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ray import tune

import nupic.research.frameworks.backprop_structure.dataset_managers as datasets
import nupic.research.frameworks.backprop_structure.experiments as experiments
import nupic.research.frameworks.backprop_structure.experiments.mixins as mixins
import nupic.research.frameworks.backprop_structure.networks as networks
from nupic.research.frameworks.ax import AxService, CoreAxClient, NonblockingAxSearch


class VDropExperiment(mixins.ConstrainParameters,
                      mixins.LogStructure,
                      mixins.Regularize,
                      experiments.Supervised):
    def __init__(self, logdir, num_epochs, log2_batch_size, lr,
                 gamma_prewarmup, gamma_warmup, gamma_postwarmup,
                 reg_warmup_start_epoch, reg_warmup_end_epoch,
                 log_reg_factor_start, log_reg_factor_end):
        batch_size = 0x1 << log2_batch_size
        warmup_start_iteration = reg_warmup_start_epoch - 1
        warmup_end_iteration = reg_warmup_end_epoch - 1
        reg_factor_start = math.exp(log_reg_factor_start)
        reg_factor_end = math.exp(log_reg_factor_end)
        self.warmup_start_iteration = warmup_start_iteration
        self.warmup_end_iteration = warmup_end_iteration
        self.gamma_warmup = gamma_warmup
        self.gamma_postwarmup = gamma_postwarmup

        super().__init__(
            logdir=logdir,
            network_class=networks.gsc_lenet_vdrop,
            network_args=dict(),

            dataset_class=datasets.PreprocessedGSC,
            dataset_args=dict(),

            optim_class=torch.optim.Adam,
            optim_args=dict(
                lr=lr,
            ),

            lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_args=dict(
                step_size=1,
                gamma=gamma_prewarmup,
            ),

            training_iterations=num_epochs,

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

    def run_epoch(self, iteration):
        if iteration == self.warmup_end_iteration:
            self.lr_scheduler.gamma = self.gamma_postwarmup
        elif iteration == self.warmup_start_iteration:
            self.lr_scheduler.gamma = self.gamma_warmup

        self.result = super().run_epoch(iteration)
        return self.result


class MyAxClient(CoreAxClient):
    def __init__(self, serialized_filepath=None):
        self.serialized_filepath = serialized_filepath

        if serialized_filepath is not None and os.path.exists(
                serialized_filepath):
            with open(serialized_filepath, "r") as f:
                serialized = json.load(f)
            other = CoreAxClient.from_json_snapshot(serialized)
            self.__dict__.update(other.__dict__)
        else:
            parameters = [
                RangeParameter("num_epochs", ParameterType.INT,
                               lower=30, upper=200),
                RangeParameter("log2_batch_size", ParameterType.INT,
                               lower=5, upper=8),
                RangeParameter("lr", ParameterType.FLOAT,
                               lower=1e-5, upper=0.3, log_scale=True),
                RangeParameter("gamma_prewarmup", ParameterType.FLOAT,
                               lower=0.5, upper=1.0),
                RangeParameter("gamma_warmup", ParameterType.FLOAT,
                               lower=0.5, upper=1.0),
                RangeParameter("gamma_postwarmup", ParameterType.FLOAT,
                               lower=0.5, upper=0.985),
                RangeParameter("reg_warmup_start_epoch", ParameterType.INT,
                               lower=1, upper=200),
                RangeParameter("reg_warmup_end_epoch", ParameterType.INT,
                               lower=1, upper=200),

                # Parameter constraints not allowed on log scale
                # parameters. So implement the log ourselves.
                RangeParameter("log_reg_factor_start", ParameterType.FLOAT,
                               lower=math.log(1e-4), upper=math.log(1.0)),
                RangeParameter("log_reg_factor_end", ParameterType.FLOAT,
                               lower=math.log(0.1), upper=math.log(10.0)),
            ]

            pm = {p.name: p for p in parameters}
            search_space = SearchSpace(
                parameters=parameters,
                parameter_constraints=[
                    # reg_warmup_start_epoch <= reg_warmup_end_epoch
                    OrderConstraint(pm["reg_warmup_start_epoch"],
                                    pm["reg_warmup_end_epoch"]),
                    # reg_warmup_end_epoch <= num_epochs
                    OrderConstraint(pm["reg_warmup_end_epoch"],
                                    pm["num_epochs"]),
                    # log_reg_factor_start <= log_reg_factor_end
                    OrderConstraint(pm["log_reg_factor_start"],
                                    pm["log_reg_factor_end"]),
                ]
            )

            optimization_config = OptimizationConfig(
                objective=MultiObjective(
                    metrics=[
                        Metric(name="neg_log_error",
                               lower_is_better=False),
                        Metric(name="neg_log_num_nonzero_weights",
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
                generation_strategy=generation_strategy
            )

    def get_next_trial(self):
        return super().get_next_trial(
            model_gen_options={
                "acquisition_function_kwargs": {
                    "random_scalarization": True,
                },
            }
        )

    def save(self):
        if self.serialized_filepath is not None:
            with open(self.serialized_filepath, "w") as f:
                json.dump(self.to_json_snapshot(), f)


def metric_function(result):
    return {
        "neg_log_error": (
            -math.log(1 - result["mean_accuracy"]), None),
        "neg_log_num_nonzero_weights": (
            -math.log(max(result["inference_nz"], 0.5)), None),
    }


if __name__ == "__main__":
    NUM_TRIALS = 1000
    NUM_RANDOM = 300

    ray.init(
        address="auto", redis_password="5241590000000000"
    )

    experiment_name = os.path.basename(__file__).replace(".py", "")
    ax_dir = os.path.expanduser(
        f"~/nta/results/ax_experiments/{experiment_name}/")
    os.makedirs(ax_dir, exist_ok=True)

    restore = False
    if restore:
        serialized_filepath = os.path.join(
            ax_dir, "[INSERT PREVIOUS SERIALIZED_FILEPATH]")
    else:
        serialized_filepath = os.path.join(
            ax_dir, "{}_{}.json".format(time.strftime("%Y%m%d-%H%M%S"),
                                        uuid.uuid4().hex))

        print(f"Ax model will be saved to {serialized_filepath}")

    ax_service = AxService(MyAxClient,
                           serialized_filepath=serialized_filepath,
                           actor_resources=dict(OnDemandInstances=0.001))
    ax_service.backend.queue_incomplete_trials.remote()

    local_dir = os.path.expanduser("~/nta/results/ray_results")
    num_completed = ray.get(
        ax_service.backend.get_num_completed_trials.remote()
    )

    tune.run(
        experiments.as_ray_trainable(VDropExperiment),
        name=experiment_name,
        num_samples=NUM_TRIALS - num_completed,
        search_alg=NonblockingAxSearch(
            ax_service.frontend, metric_function, max_concurrent=20,
            m_suggestions_allowed_before_nth_completion=(
                (NUM_RANDOM, 1) if num_completed == 0 else None
            )
        ),
        local_dir=local_dir,
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
