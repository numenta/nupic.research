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
from collections import defaultdict

__all__ = [
    "StepBasedLogging",
]


class StepBasedLogging:
    """
    Adds logic and extensibility points to keep a step-based log (rather than an
    epoch-based log).

    Conceptually, the "current_timestep" is a version number for the model's
    parameters. By default, this is the elapsed number of batches that the model
    has been trained on. Experiments may also increment this on other events
    like model prunings. When validation is performed after a training batch,
    the validation results are assigned to the next timestep after that training
    batch, since it was performed on the subsequent version of the parameters.
    """
    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - log_timestep_freq: Configures mixins and subclasses that log every
                                 timestep to only log every nth timestep (in
                                 addition to the final timestep of each epoch).
                                 Set to 0 to log only at the end of each epoch.
        """
        super().setup_experiment(config)
        self.current_timestep = 0
        self.log_timestep_freq = config.get("log_timestep_freq", 1)

    def run_epoch(self):
        timestep_begin = self.current_timestep
        ret = super().run_epoch()
        ret.update(
            timestep_begin=timestep_begin,
            timestep_end=self.current_timestep,
        )
        return ret

    def post_batch(self, **kwargs):
        super().post_batch(**kwargs)
        self.current_timestep += 1

    def should_log_batch(self, train_batch_idx):
        """
        Returns true if the current timestep should be logged, either because it's a
        logged timestep or the final training batch of an epoch.

        This is a utility method, not intended for extensibility.
        """
        return (train_batch_idx == self.total_batches - 1) or (
            self.log_timestep_freq > 0
            and (self.current_timestep % self.log_timestep_freq) == 0)

    def get_state(self):
        state = super().get_state()
        state["current_timestep"] = self.current_timestep
        return state

    def set_state(self, state):
        super().set_state(state)
        if "current_timestep" in state:
            self.current_timestep = state["current_timestep"]

    @classmethod
    def get_recorded_timesteps(cls, result, config):
        """
        Given an epoch result dict and config, returns a list of timestep numbers
        that are supposed to be logged for that epoch.

        This is a utility method, not intended for extensibility.
        """
        log_timestep_freq = config.get("log_timestep_freq", 1)
        timestep_end = result["timestep_end"]
        if log_timestep_freq == 0:
            ret = [timestep_end - 1]
        else:
            # Find first logged timestep in range
            logged_begin = int(math.ceil(result["timestep_begin"]
                                         / log_timestep_freq)
                               * log_timestep_freq)

            ret = list(range(logged_begin, timestep_end, log_timestep_freq))

            last_batch_timestep = timestep_end - 1
            if last_batch_timestep % log_timestep_freq != 0:
                ret.append(last_batch_timestep)

        return ret

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        """
        Given a result dict containing data for multiple batches, returns a mapping
        from timesteps to results. The mapping is stored as a dict so that
        subclasses and mixins can easily add data to it.

        Result keys are converted from Ray Tune requirements to better names,
        and the keys are filtered to those that make useful charts.

        :return: defaultdict mapping timesteps to result dicts
        """
        result_by_timestep = defaultdict(dict)

        # Assign the epoch result to the appropriate timestep.
        k_mapping = {
            "mean_loss": "validation_loss",
            "mean_accuracy": "validation_accuracy",
            "learning_rate": "learning_rate",
            "complexity_loss": "complexity_loss",
        }
        result_by_timestep[result["timestep_end"]].update({
            k2: result[k1]
            for k1, k2 in k_mapping.items()
            if k1 in result
        })

        return result_by_timestep

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "StepBasedLogging"
        eo.update(
            # New methods
            expand_result_to_time_series=[exp + ": common result dict keys"],
            post_batch_wrapper=[exp + ".post_batch_wrapper"],
        )
        return eo
