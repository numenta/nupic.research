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

import abc
import collections

__all__ = [
    "StepBasedLogging",
]


class StepBasedLogging(abc.ABC):
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

    @property
    @abc.abstractmethod
    def current_timestep(self) -> int:
        """
        Get the current StepBasedLogging timestep.
        """

    @current_timestep.setter
    @abc.abstractmethod
    def current_timestep(self, value: int):
        """
        Set the current StepBasedLogging timestep.
        """

    @abc.abstractmethod
    def should_log_batch(self, train_batch_idx: int) -> bool:
        """
        Returns true if the current timestep should be logged, either because it's a
        logged timestep or the final training batch of an epoch.

        This is a utility method, not intended for extensibility.
        """

    @classmethod
    @abc.abstractmethod
    def get_recorded_timesteps(cls, result: dict, config: dict) -> list:
        """
        Given an epoch result dict and config, returns a list of timestep numbers
        that are supposed to be logged for that epoch.

        This is a utility method, not intended for extensibility.
        """

    @classmethod
    @abc.abstractmethod
    def expand_result_to_time_series(cls,
                                     result: dict,
                                     config: dict) -> collections.defaultdict:
        """
        Given a result dict containing data for multiple batches, returns a mapping
        from timesteps to results. The mapping is stored as a dict so that
        subclasses and mixins can easily add data to it.

        Result keys are converted from Ray Tune requirements to better names,
        and the keys are filtered to those that make useful charts.

        :return: defaultdict mapping timesteps to result dicts
        """

    @staticmethod
    @abc.abstractmethod
    def step_based_logging_interface_implemented() -> bool:
        """
        Implement this method to signal that this interface is implemented,
        enabling the class to be instantiated.

        This method ensures invalid classes can't be constructed. The other
        abstract methods are often sufficient for ensuring this, but they can be
        fooled by classes extending those methods (and calling `super`). This
        method prevents this type of error because there is no reason for a
        class to extend this method.
        """
