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

from nupic.research.frameworks.vernon import interfaces

__all__ = [
    "StepBasedLogging",
]


class StepBasedLogging(
    interfaces.Experiment,  # Requires
    interfaces.StepBasedLogging,  # Implements
):
    @staticmethod
    def step_based_logging_interface_implemented():
        return True

    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - log_timestep_freq: Configures mixins and subclasses that log every
                                 timestep to only log every nth timestep (in
                                 addition to the final timestep of each epoch).
                                 Set to 0 to log only at the end of each epoch.
        """
        super().setup_experiment(config)
        self._current_timestep = 0
        self.log_timestep_freq = config.get("log_timestep_freq", 1)

    @property
    def current_timestep(self):
        return self._current_timestep

    @current_timestep.setter
    def current_timestep(self, value):
        self._current_timestep = value

    def run_iteration(self):
        timestep_begin = self.current_timestep
        ret = super().run_iteration()
        ret.update(
            timestep_begin=timestep_begin,
            timestep_end=self.current_timestep,
        )
        return ret

    def post_batch(self, **kwargs):
        super().post_batch(**kwargs)
        # FIXME: move to post_optimizer_step
        self.current_timestep += 1

    def should_log_batch(self, train_batch_idx):
        return (train_batch_idx == len(self.train_loader) - 1) or (
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
        result_by_timestep = defaultdict(dict)

        # Assign the epoch result to the appropriate timestep.
        result_by_timestep[result["timestep_end"]].update(
            cls.get_readable_result(result)
        )

        return result_by_timestep

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "StepBasedLoggingCore"

        eo["run_iteration"].append(exp + ": Add timestep info")
        eo["post_batch"].append(exp + ": Increment timestep")
        eo["get_state"].append(exp + ": Get current timestep")
        eo["set_state"].append(exp + ": Set current timestep")

        eo.update(
            # StepBasedLogging
            expand_result_to_time_series=[exp + ": common result dict keys"],
        )
        return eo
