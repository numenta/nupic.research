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
                - k_predictions
                - lr
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments
        assert "model_args" in config
        assert "optimizer_args" in config

        # Optimizer args
        negative_samples = assignments.get("negative_samples", 16)
        k_predictions = assignments.get("k_predictions", 5)
        lr = assignments.get("lr", 2e-4)
        config["optimizer_args"] = dict(lr=lr)
        model_args = config["model_args"]
        model_args.update(
            dict(negative_samples=negative_samples, k_predictions=k_predictions)
        )
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
                - k_predictions
                - weight_density
                - percent_on
                - lr
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments
        assert "model_args" in config
        assert "optimizer_args" in config

        # Optimizer args
        negative_samples = assignments.get("negative_samples", 16)
        k_predictions = assignments.get("k_predictions", 5)
        lr = assignments.get("lr", 2e-4)
        weight_density = assignments.get("weight_density", 0.2)
        percent_on = assignments.get("percent_on", 0.2)
        config["optimizer_args"] = dict(lr=lr)
        model_args = config["model_args"]
        model_args.update(
            dict(
                negative_samples=negative_samples,
                k_predictions=k_predictions,
                sparsity=[1 - weight_density] * 3,
                percent_on=[percent_on] * 3,
            )
        )
        config["model_args"] = model_args

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["update_config_with_suggestion"].append(
            "SparseGIMSigOpt.update_config_with_suggestion"
        )
        return eo


BATCH_SIZE = 32
NUM_EPOCHS = 10
DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]
SPARSE_BASE = SPARSE_CONFIGS["sparse_base"]
SIGOPT_DENSE_BASE = deepcopy(DEFAULT_BASE)
SIGOPT_DENSE_BASE.update(
    dict(
        sigopt_experiment_class=DenseGIMSigOpt,
        sigopt_config=dict(
            name="sigopt_GIM_dense_base",
            parameters=[
                dict(name="negative_samples", type="int", bounds=dict(min=8, max=32)),
                dict(name="k_predictions", type="int", bounds=dict(min=2, max=8)),
                dict(
                    name="lr",
                    type="double",
                    bounds=dict(min=2e-3, max=1e-4),
                    transformation="log",
                ),
            ],
            metrics=[dict(name="train_loss", objective="minimize")],
            parallel_bandwidth=1,
            observation_budget=10,
            project="greedy_infomax",
        ),
        sigopt_experiment_id=None,
        api_key=os.environ.get("SIGOPT_KEY", None),
    )
)


SIGOPT_SPARSE_BASE = deepcopy(SPARSE_BASE)
SIGOPT_SPARSE_BASE.update(
    dict(
        sigopt_experiment_class=SparseGIMSigOpt,
        sigopt_config=dict(
            name="sigopt_GIM_sparse_base",
            parameters=[
                dict(name="negative_samples", type="int", bounds=dict(min=8, max=32)),
                dict(name="k_predictions", type="int", bounds=dict(min=2, max=8)),
                dict(
                    name="lr",
                    type="double",
                    bounds=dict(min=1e-4, max=2e-3),
                    transformation="log",
                ),
                # TODO: dynamic sparsity?
                dict(
                    name="weight_density",
                    type="double",
                    bounds=dict(min=0.02, max=0.20),
                    transformation="log",
                ),
                dict(
                    name="percent_on",
                    type="double",
                    bounds=dict(min=0.02, max=0.20),
                    transformation="log",
                ),
            ],
            metrics=[dict(name="train_loss", objective="minimize")],
            parallel_bandwidth=1,
            observation_budget=10,
            project="greedy_infomax",
        ),
        sigopt_experiment_id=None,
        api_key=os.environ.get("SIGOPT_KEY", None),
    )
)

CONFIGS = dict(
    sigopt_dense_base=SIGOPT_DENSE_BASE, sigopt_sparse_base=SIGOPT_SPARSE_BASE
)
