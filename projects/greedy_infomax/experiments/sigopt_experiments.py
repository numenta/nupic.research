# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

import os
from copy import deepcopy

from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptExperiment

from .default_base import CONFIGS as DEFAULT_BASE_CONFIGS
from .sparse_resnets import CONFIGS as SPARSE_CONFIGS


def make_reg_schedule(
    epochs, pct_ramp_start, pct_ramp_end, peak_value, pct_drop, final_value
):
    def reg_schedule(epoch, batch_idx, steps_per_epoch):
        pct = (epoch + batch_idx / steps_per_epoch) / epochs

        if pct < pct_ramp_start:
            return 0.0
        elif pct < pct_ramp_end:
            progress = (pct - pct_ramp_start) / (pct_ramp_end - pct_ramp_start)
            return progress * peak_value
        elif pct < pct_drop:
            return peak_value
        else:
            return final_value

    return reg_schedule


class DenseGIMSigOpt(SigOptExperiment):
    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the optimizer_args with SGD optimizer params.

        :param config:
            - model_args
            - optimizer_args
        :param suggestion:
            - assignments (all optional)
                - negative_samples
                - lr
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments
        assert "model_args" in config
        assert "optimizer_args" in config

        # Optimizer args
        negative_samples = assignments.get("negative_samples", 16)
        lr = assignments.get("lr", 2e-4)
        config["optimizer_args"] = dict(lr=lr)
        model_args = config["model_args"]
        model_args.update(dict(negative_samples=negative_samples))
        config["model_args"] = model_args

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["update_config_with_suggestion"].append(
            "DenseGIMSigOpt.update_config_with_suggestion"
        )
        return eo


class SparseGIMSigOpt(SigOptExperiment):
    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the optimizer_args with SGD optimizer params.

        :param config:
            - model_args
            - optimizer_args
        :param suggestion:
            - assignments (all optional)
                - negative_samples
                - weight_density
                - percent_on
                - lr
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments
        assert "model_args" in config

        # Optimizer args
        weight_density = assignments.get("weight_density", 0.2)
        percent_on = assignments.get("percent_on", 0.2)
        model_args = config["model_args"]
        model_args.update(
            dict(sparsity=[1 - weight_density] * 3, percent_on=[percent_on] * 3)
        )
        config["model_args"] = model_args

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["update_config_with_suggestion"].append(
            "SparseGIMSigOpt.update_config_with_suggestion"
        )
        return eo


DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]
SPARSE_BASE = SPARSE_CONFIGS["sparse_base"]


NUM_EPOCHS = 6
SIGOPT_DENSE_BASE = deepcopy(DEFAULT_BASE)
SIGOPT_DENSE_BASE.update(
    dict(
        # wandb
        wandb_args=dict(
            project="greedy_infomax-sparsity", name="dense_resnet_base_val_acc"
        ),
        # sigopt
        sigopt_experiment_class=DenseGIMSigOpt,
        sigopt_config=dict(
            name="sigopt_GIM_dense_val_acc",
            parameters=[
                dict(name="negative_samples", type="int", bounds=dict(min=8, max=32)),
                dict(
                    name="lr",
                    type="double",
                    bounds=dict(min=1e-4, max=2e-3),
                    transformation="log",
                ),
            ],
            metrics=[dict(name="mean_accuracy", objective="maximize")],
            parallel_bandwidth=1,
            observation_budget=40,
            project="greedy_infomax",
        ),
        sigopt_experiment_id=397541,
        api_key=os.environ.get("SIGOPT_KEY", None),
        # experiment args
        epochs=NUM_EPOCHS,
        # batches_in_epoch=2,
        epochs_to_validate=[NUM_EPOCHS - 1],
        supervised_training_epochs_per_validation=20,
    )
)


SIGOPT_SPARSE_BASE = deepcopy(SPARSE_BASE)
SIGOPT_SPARSE_BASE.update(
    dict(
        # wandb
        wandb_args=dict(project="greedy_infomax-sparsity", name="sparse_resnet_base"),
        # sigopt
        sigopt_experiment_class=SparseGIMSigOpt,
        sigopt_config=dict(
            name="sigopt_GIM_sparse_base",
            parameters=[
                # TODO: dynamic sparsity?
                dict(
                    name="weight_density",
                    type="double",
                    bounds=dict(min=0.05, max=0.8),
                    transformation="log",
                ),
                dict(
                    name="percent_on",
                    type="double",
                    bounds=dict(min=0.025, max=0.51),
                    transformation="log",
                ),
            ],
            metrics=[dict(name="mean_accuracy", objective="maximize")],
            parallel_bandwidth=5,
            observation_budget=40,
            project="greedy_infomax",
        ),
        sigopt_experiment_id=398138,
        api_key=os.environ.get("SIGOPT_KEY", None),
        # experiment args
        epochs=NUM_EPOCHS,
        epochs_to_validate=[NUM_EPOCHS - 1],
        supervised_training_epochs_per_validation=20,
    )
)

CONFIGS = dict(
    sigopt_dense_base=SIGOPT_DENSE_BASE, sigopt_sparse_base=SIGOPT_SPARSE_BASE
)
