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


class RegularizeLoss(object):
    """
    Implement the complexity_loss as the sum all module.regularization()
    functions, times some specified scalar.
    """
    def setup_experiment(self, config):
        """
        :param config:
            Dictionary containing the following parameters that set the schedule
            for the regularization strength:
            - initial_reg_scalar, final_reg_scalar
            - pct_begin_reg_ramp, pct_end_reg_ramp
            - num_reg_cycles: How many times to repeat the reg schedule
            - downscale_reg_with_training_set: If True, multiply the
                 regularization term by (1 / size_of_training_set)
        """
        super().setup_experiment(config)

        self.num_reg_cycles = config.get("num_reg_cycles", 1)
        self.initial_reg_scalar = config.get("initial_reg_scalar", 1.0)
        self.final_reg_scalar = config.get("final_reg_scalar", 1.0)
        self.pct_begin_ramp = config.get("pct_begin_reg_ramp", 0.0)
        self.pct_end_ramp = config.get("pct_end_reg_ramp", 0.0)
        self.reg_coefficient = (
            1 / len(self.train_loader.dataset)
            if config.get("downscale_reg_with_training_set", False)
            else 1
        )

        self.reg_scalar_history = []
        self.model_complexity_history = []

        # Cache these floats
        self.reg_ramp_window_length = self.pct_end_ramp - self.pct_begin_ramp
        self.reg_ramp_amount = self.final_reg_scalar - self.initial_reg_scalar

        self.reg_scalar = self._compute_reg_scalar(0.0)

        self._regularized_modules = [module
                                     for module in self.model.modules()
                                     if hasattr(module, "regularization")]
        assert len(self._regularized_modules) > 0

    def _compute_reg_scalar(self, pct):
        if pct < self.pct_begin_ramp:
            return self.initial_reg_scalar
        elif pct < self.pct_end_ramp:
            ramp_pct = (pct - self.pct_begin_ramp) / self.reg_ramp_window_length
            return (self.initial_reg_scalar
                    + ramp_pct * self.reg_ramp_amount)
        else:
            return self.final_reg_scalar

    def pre_batch(self, model, batch_idx):
        super().pre_batch(model, batch_idx)

        epochs_per_cycle = self.epochs / self.num_reg_cycles
        epoch = self.current_epoch % epochs_per_cycle

        pct = ((epoch * self.total_batches + batch_idx)
               / (epochs_per_cycle * self.total_batches))
        self.reg_scalar = self._compute_reg_scalar(pct)
        self.reg_scalar_history.append(self.reg_scalar)

    def complexity_loss(self, model):
        c_loss = super().complexity_loss(model)
        assert model is self.model
        reg = torch.stack([module.regularization()
                           for module in self._regularized_modules]).sum()
        if self.model.training:
            self.model_complexity_history.append(reg.detach().clone())
        reg *= self.reg_scalar * self.reg_coefficient
        c_loss = (c_loss + reg
                  if c_loss is not None
                  else reg)
        return c_loss

    def run_epoch(self):
        result = super().run_epoch()
        result["reg_scalar_history"] = self.reg_scalar_history
        self.reg_scalar_history = []

        result["model_complexity_history"] = torch.stack(
            self.model_complexity_history).cpu().tolist()
        self.model_complexity_history = []

        return result

    @classmethod
    def expand_result_to_time_series(cls, result):
        result_by_timestep = super().expand_result_to_time_series(result)

        end = result["timestep"]
        start = end - len(result["reg_scalar_history"])
        for t, rs, c in zip(range(start, end),
                            result["reg_scalar_history"],
                            result["model_complexity_history"]):
            result_by_timestep[t].update(
                complexity_loss_scalar=rs,
                model_complexity=c,
            )

        return result_by_timestep

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("RegularizeLoss: initialization")
        eo["complexity_loss"].append("RegularizeLoss: Compute")
        eo["pre_batch"].append("RegularizeLoss: update and log "
                               "regularization scalar")
        eo["run_epoch"].append("RegularizeLoss: add regularization weight log "
                               "to result dict")
        eo["expand_result_to_time_series"].append(
            "RegularizeLoss: regularization scalar and model complexity"
        )
        return eo
