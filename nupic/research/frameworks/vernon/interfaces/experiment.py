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
import logging
from typing import Optional

__all__ = [
    "Experiment",
]


class Experiment(abc.ABC):
    """
    This class provides an interface for machine learning experiments.
    Subclasses define what should happen during setup, what should happen on
    each iteration of the experiment, and when the experiment should stop. The
    API is designed for extensibility via object-oriented-programming and/or
    mixins.

    Callers will implement the loop:

        exp = ExperimentClass()
        exp.setup_experiment(config)

        first_run = True
        while not exp.should_stop():
            if first_run:
                pre_result = exp.run_pre_experiment()
                print(pre_result)
            result = exp.run_iteration()
            if first_run:
                ExperimentClass.insert_pre_experiment_result(result, pre_result)
                first_run = False
            print(ExperimentClass.get_printable_result(result))
            save_iteration_somehow(result)  # Example-specific logging
    """
    @abc.abstractmethod
    def setup_experiment(self, config) -> None:
        """
        Configure the experiment for training

        :param config: Dictionary containing the configuration parameters
        """

    @abc.abstractmethod
    def run_iteration(self) -> dict:
        """
        Run one training iteration of the experiment and return some sort of
        result.
        """

    @abc.abstractmethod
    def should_stop(self) -> bool:
        """
        Whether or not the experiment should stop. Usually determined by the
        number of epochs but customizable to any other stopping criteria
        """

    @abc.abstractmethod
    def stop_experiment(self) -> None:
        """
        Perform any needed cleanup.
        """

    @abc.abstractmethod
    def run_pre_experiment(self) -> Optional[dict]:
        """
        Perform some sort of initial analysis (for example, run validation) and
        return some sort of result. If the experiment is configured to skip the
        pre_experiment, return None.
        """

    @classmethod
    @abc.abstractmethod
    def insert_pre_experiment_result(cls, result, pre_experiment_result) -> None:
        """
        Modify result to incorporate the pre_experiment_result. By default, this
        method performs no update. (This may be sufficient, for example if the
        pre_experiment_result has already been printed to the console.)

        :param pre_experiment_results: The return value of pre_experiment
        """

    @property
    @abc.abstractmethod
    def logger(self) -> logging.Logger:
        """
        Get the experiment's logger.
        """

    @classmethod
    @abc.abstractmethod
    def create_logger(cls, config: dict) -> logging.Logger:
        """
        Create the experiment's logger.
        """

    @abc.abstractmethod
    def get_state(self) -> dict:
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """

    @abc.abstractmethod
    def set_state(self, state: dict) -> None:
        """
        Restore the experiment from the state returned by `get_state`
        :param state: dictionary with "model", "optimizer", "lr_scheduler"
        states
        """

    @classmethod
    @abc.abstractmethod
    def get_printable_result(cls, result: dict):
        """
        Return a stripped down version of result that has its large data structures
        removed so that the result can be printed to the console.
        """

    @classmethod
    @abc.abstractmethod
    def get_execution_order(cls) -> dict:
        """
        Gets a dict that can be printed to show the order of events that occur
        for each method. Subclasses and mixins should extend this method,
        modifying each list of events according to when/whether they call
        super().
        """

    @staticmethod
    @abc.abstractmethod
    def experiment_interface_implemented() -> bool:
        """
        Implement this method to signal that this interface is implemented,
        enabling the class to be instantiated.

        This method ensures invalid classes can't be constructed. The other
        abstract methods are often sufficient for ensuring this, but they can be
        fooled by classes extending those methods (and calling `super`). This
        method prevents this type of error because there is no reason for a
        class to extend this method.
        """
