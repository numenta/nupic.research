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

import torch


class LogEveryLoss:
    """
    Include the training loss for every batch in the result dict.

    This class must be placed earlier in the mixin order than other mixins that
    modify the loss.

    class MyExperiment(mixins.LogEveryLoss,
                       ...
                       mixins.RegularizeLoss,
                       ImagenetExperiment):
        pass

    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.error_loss_history = []
        self.complexity_loss_history = []

    def error_loss(self, *args, **kwargs):
        loss = super().error_loss(*args, **kwargs)
        if self.model.training:
            self.error_loss_history.append(loss.detach().clone())
        return loss

    def complexity_loss(self, *args, **kwargs):
        loss = super().complexity_loss(*args, **kwargs)
        if loss is not None:
            if self.model.training:
                self.complexity_loss_history.append(loss.detach().clone())
        return loss

    def run_epoch(self):
        result = super().run_epoch()

        log = torch.stack(self.error_loss_history)
        result["error_loss_history"] = log.cpu().numpy().tolist()
        self.error_loss_history = []

        if len(self.complexity_loss_history) > 0:
            log = torch.stack(self.complexity_loss_history)
            result["complexity_loss_history"] = log.cpu().numpy().tolist()
            self.complexity_loss_history = []

        return result

    @classmethod
    def aggregate_results(cls, results):
        aggregated = super().aggregate_results(results)

        k = "error_loss_history"
        loss_by_process_and_batch = torch.Tensor(len(results),
                                                 len(results[0][k]))
        for rank, result in enumerate(results):
            loss_by_process_and_batch[rank, :] = torch.tensor(result[k])
        aggregated[k] = loss_by_process_and_batch.mean(dim=0).tolist()

        # "complexity_loss_history" doesn't need to be aggregated, since it's
        # the same on every process.

        return aggregated

    @classmethod
    def expand_result_to_time_series(cls, result):
        result_by_timestep = super().expand_result_to_time_series(result)

        end = result["timestep"]
        start = end - len(result["error_loss_history"])
        for t, loss in zip(range(start, end),
                           result["error_loss_history"]):
            result_by_timestep[t].update(
                train_loss=loss,
            )

        if "complexity_loss_history" in result:
            for t, loss in zip(range(start, end),
                               result["complexity_loss_history"]):
                result_by_timestep[t].update(
                    complexity_loss=loss,
                )

        return result_by_timestep

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("LogEveryLoss: initialize")
        eo["error_loss"].append("LogEveryLoss: copy loss")
        eo["complexity_loss"].append("LogEveryLoss: copy loss")
        eo["run_epoch"].append("LogEveryLoss: to result dict")
        eo["aggregate_results"].append("LogEveryLoss: Aggregate")
        eo["expand_result_to_time_series"].append(
            "LogEveryLoss: error_loss, complexity_loss"
        )
        return eo
