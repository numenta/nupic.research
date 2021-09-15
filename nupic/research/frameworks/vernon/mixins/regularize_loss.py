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

from nupic.research.frameworks.vernon.mixins.step_based_logging import StepBasedLogging


class RegularizeLoss(StepBasedLogging):
    """
    Implement the complexity_loss as the sum all module.regularization()
    functions, times some specified scalar.

    This mixin also records the model complexity (the regularization() sum) for
    each batch. These numbers are stored on the GPU until the end of the epoch.
    If every kilobyte of GPU memory is being used by the experiment, this could
    result in running out of memory. Adjust config["log_timestep_freq"] to
    reduce the logging frequency.
    """
    def setup_experiment(self, config):
        """
        :param config:
            Dictionary containing the following parameters that set the schedule
            for the regularization strength:
            - reg_scalar: (float or callable) Returns the reg_scalar, given the
                 epoch number, batch index, and total batches per epoch
            - downscale_reg_with_training_set: If True, multiply the
                 regularization term by (1 / size_of_training_set)
        """
        super().setup_experiment(config)

        self.reg_scalar = config.get("reg_scalar", 0)
        self._update_reg_scalar(0)

        self.reg_scalar_downscale = (
            1 / len(self.train_loader.dataset)
            if config.get("downscale_reg_with_training_set", False)
            else 1
        )

        self.reg_scalar_history = []
        self.model_complexity_history = []
        self.prev_model_complexity = None

        # Cache these floats
        self._regularized_modules = [module
                                     for module in self.model.modules()
                                     if hasattr(module, "regularization")]
        assert len(self._regularized_modules) > 0

    def _update_reg_scalar(self, batch_idx):
        reg_scalar = self.reg_scalar
        if callable(reg_scalar):
            reg_scalar = reg_scalar(self.current_epoch, batch_idx,
                                    self.total_batches)

        self.reg_scalar_value = reg_scalar

    def pre_batch(self, batch_idx, **kwargs):
        super().pre_batch(batch_idx=batch_idx, **kwargs)
        self._update_reg_scalar(batch_idx)

    def complexity_loss(self, model):
        c_loss = super().complexity_loss(model)
        reg = torch.stack([module.regularization()
                           for module in self._regularized_modules]).sum()
        if self.model.training:
            # Save this now, decide whether to log it in post_batch when we know
            # the batch index.
            self.prev_model_complexity = reg.detach().clone()
        reg *= self.reg_scalar_value * self.reg_scalar_downscale
        c_loss = (c_loss + reg
                  if c_loss is not None
                  else reg)
        return c_loss

    def post_batch(self, batch_idx, **kwargs):
        super().post_batch(batch_idx=batch_idx, **kwargs)

        if self.should_log_batch(batch_idx):
            self.model_complexity_history.append(self.prev_model_complexity)
            self.reg_scalar_history.append(self.reg_scalar_value)
        self.prev_model_complexity = None

    def run_epoch(self):
        result = super().run_epoch()
        result["reg_scalar_history"] = self.reg_scalar_history
        self.reg_scalar_history = []

        if len(self.model_complexity_history) > 0:
            result["model_complexity_history"] = torch.stack(
                self.model_complexity_history).cpu().tolist()
            self.model_complexity_history = []

        return result

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        result_by_timestep = super().expand_result_to_time_series(result,
                                                                  config)

        for t, rs, c in zip(cls.get_recorded_timesteps(result, config),
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
        eo["pre_batch"].append("RegularizeLoss: update regularization scalar")
        eo["run_epoch"].append("RegularizeLoss: add regularization weight log "
                               "to result dict")
        eo["post_batch"].append("RegularizeLoss: Log regularization scalar and "
                                "model complexity")
        eo["expand_result_to_time_series"].append(
            "RegularizeLoss: regularization scalar and model complexity"
        )
        return eo
