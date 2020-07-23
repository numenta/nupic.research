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

    These numbers are stored on the GPU until the end of the epoch. If every
    kilobyte of GPU memory is being used by the experiment, this could result in
    running out of memory. Adjust config["log_timestep_freq"] to reduce the
    logging frequency.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.error_loss_history = []
        self.complexity_loss_history = []
        self.learning_rate_history = []

    def post_batch(self, model, error_loss, complexity_loss, batch_idx,
                   *args, **kwargs):
        super().post_batch(model, error_loss, complexity_loss, batch_idx,
                           *args, **kwargs)
        if self.should_log_batch(batch_idx):
            self.learning_rate_history.append(self.get_lr()[0])
            self.error_loss_history.append(error_loss.clone())
            if complexity_loss is not None:
                self.complexity_loss_history.append(complexity_loss.clone())

    def run_epoch(self):
        result = super().run_epoch()

        if len(self.error_loss_history) > 0:
            log = torch.stack(self.error_loss_history)
            result["error_loss_history"] = log.cpu().numpy().tolist()
            self.error_loss_history = []

        if len(self.complexity_loss_history) > 0:
            log = torch.stack(self.complexity_loss_history)
            result["complexity_loss_history"] = log.cpu().numpy().tolist()
            self.complexity_loss_history = []

        if len(self.learning_rate_history) > 0:
            result["learning_rate_history"] = self.learning_rate_history
            self.learning_rate_history = []

        return result

    @classmethod
    def aggregate_results(cls, results):
        aggregated = super().aggregate_results(results)

        k = "error_loss_history"
        if k in aggregated:
            loss_by_process_and_batch = torch.Tensor(len(results),
                                                     len(results[0][k]))
            for rank, result in enumerate(results):
                loss_by_process_and_batch[rank, :] = torch.tensor(result[k])
            aggregated[k] = loss_by_process_and_batch.mean(dim=0).tolist()

        # "complexity_loss_history" and "learning_rate_history" doesn't need to be
        # aggregated, since it's the same on every process.

        return aggregated

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        result_by_timestep = super().expand_result_to_time_series(result, config)

        recorded_timesteps = cls.get_recorded_timesteps(result, config)
        for t, loss in zip(recorded_timesteps, result["error_loss_history"]):
            result_by_timestep[t].update(
                train_loss=loss,
            )

        for log_name in ["complexity_loss_history", "learning_rate_history"]:
            if log_name in result:
                for t, log_hist in zip(recorded_timesteps, result[log_name]):
                    result_by_timestep[t].update(
                        {log_name.replace("_history", ""): log_hist},
                    )

        return result_by_timestep

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("LogEveryLoss: initialize")
        eo["post_batch"].append("LogEveryLoss: record losses")
        eo["run_epoch"].append("LogEveryLoss: to result dict")
        eo["aggregate_results"].append("LogEveryLoss: Aggregate")
        eo["expand_result_to_time_series"].append(
            "LogEveryLoss: error_loss, complexity_loss"
        )
        return eo
