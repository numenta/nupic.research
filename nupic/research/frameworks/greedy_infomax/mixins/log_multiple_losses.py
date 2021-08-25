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
from nupic.research.frameworks.vernon.mixins import StepBasedLogging
from nupic.research.frameworks.greedy_infomax.utils.train_utils import train_block_model
import torch

__all__ = [
    "LogMultipleLoss",
]


class LogMultipleLoss(StepBasedLogging):
    """
    Include the training loss for each BilinearInfo module in a BlockModule for
    every batch in the result dict.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.train_model = train_block_model
        self.multiple_module_loss_history = []

    def post_batch(self, error_loss, complexity_loss, batch_idx, **kwargs):
        super().post_batch(error_loss=error_loss,
                           complexity_loss=complexity_loss, batch_idx=batch_idx,
                           **kwargs)
        if self.should_log_batch(batch_idx):
            self.multiple_module_loss_history.append(error_loss.clone())
            if complexity_loss is not None:
                self.complexity_loss_history.append(complexity_loss.clone())

    def run_epoch(self):
        result = super().run_epoch()

        if len(self.multiple_module_loss_history) > 0:
            log = torch.stack(self.multiple_module_loss_history)
            error_loss_history = log.cpu().numpy()
            result["error_loss_history"] = error_loss_history.tolist()
            result["train_loss"] = error_loss_history[-25:].mean()
            self.error_loss_history = []

        if len(self.complexity_loss_history) > 0:
            log = torch.stack(self.complexity_loss_history)
            result["complexity_loss_history"] = log.cpu().numpy().tolist()
            self.complexity_loss_history = []

        return result

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        result_by_timestep = super().expand_result_to_time_series(result,
                                                                  config)

        recorded_timesteps = cls.get_recorded_timesteps(result, config)
        for t, loss in zip(recorded_timesteps, result["error_loss_history"]):
            result_by_timestep[t].update(
                train_loss=loss,
            )

        if "complexity_loss_history" in result:
            for t, loss in zip(recorded_timesteps,
                               result["complexity_loss_history"]):
                result_by_timestep[t].update(
                    complexity_loss=loss,
                )

        return result_by_timestep

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("LogEveryLoss: initialize")
        eo["post_batch"].append("LogEveryLoss: record losses")
        eo["run_epoch"].append("LogEveryLoss: to result dict")
        eo["expand_result_to_time_series"].append(
            "LogEveryLoss: error_loss, complexity_loss"
        )
        return eo
