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

import numpy as np

from nupic.research.frameworks.vernon.mixins.step_based_logging import StepBasedLogging

__all__ = [
    "ExtraValidationsPerEpoch",
]


class ExtraValidationsPerEpoch(StepBasedLogging):
    """
    Perform validations mid-epoch.
    """
    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - extra_validations_per_epoch: number of additional validations to
                                           perform mid-epoch. Additional
                                           validations are distributed evenly
                                           across training batches.
        """
        super().setup_experiment(config)

        # A list of [(timestep, result), ...] for the current epoch.
        self.extra_val_results = []

        extra_validations = config.get("extra_validations_per_epoch", 0)
        batches_to_validate = np.linspace(
            min(self.total_batches, self.batches_in_epoch),
            0,
            1 + extra_validations,
            endpoint=False
        )[::-1].round().astype("int").tolist()
        self.additional_batches_to_validate = batches_to_validate[:-1]
        if extra_validations > 0:
            self.logger.info(
                f"Extra validations per epoch: {extra_validations}, "
                f"batch indices: {self.additional_batches_to_validate}")

    def run_epoch(self):
        ret = super().run_epoch()
        ret["extra_val_results"] = self.extra_val_results
        self.extra_val_results = []
        return ret

    def post_batch(self, batch_idx, **kwargs):
        super().post_batch(batch_idx=batch_idx, **kwargs)
        validate = (batch_idx in self.additional_batches_to_validate
                    and self.current_epoch in self.epochs_to_validate)
        if validate:
            result = self.validate()
            self.extra_val_results.append(
                (self.current_timestep, result)
            )
            self.model.train()

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        # Get the end-of-epoch validation results.
        result_by_timestep = super().expand_result_to_time_series(result, config)

        # Add additional validations to the time series.
        k_mapping = {
            "mean_loss": "validation_loss",
            "mean_accuracy": "validation_accuracy",
            "learning_rate": "learning_rate",
            "complexity_loss": "complexity_loss",
        }
        for timestep, val_result in result["extra_val_results"]:
            result_by_timestep[timestep].update({
                k2: val_result[k1]
                for k1, k2 in k_mapping.items()
                if k1 in val_result
            })

        return result_by_timestep

    @classmethod
    def insert_pre_experiment_result(cls, result, pre_experiment_result):
        if pre_experiment_result is not None:
            result["extra_val_results"].insert(0, (0, pre_experiment_result))

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "ExtraValidationsPerEpoch"

        # Extended methods
        eo["run_epoch"].append(
            name + ": Put extra validations into result")
        eo["post_batch"].append(name + ": Maybe run validation")
        eo["expand_result_to_time_series"].append(
            name + ": Insert extra validations")
        eo["insert_pre_experiment_result"].append(
            name + ": Insert pre experiment validation")

        return eo
