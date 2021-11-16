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

__all__ = [
    "DistributedAggregation",
]


class DistributedAggregation(abc.ABC):
    """
    Methods that take distributed experiment results and combine them.
    """

    @classmethod
    @abc.abstractmethod
    def aggregate_results(cls, results):
        """
        Aggregate multiple processes' "run_iteration" results into a single
        result.

        :param results:
            A list of return values from run_iteration from different processes.
        :type results: list

        :return:
            A single result dict with results aggregated.
        :rtype: dict
        """

    @classmethod
    @abc.abstractmethod
    def aggregate_pre_experiment_results(cls, results):
        """
        Aggregate multiple processes' "run_pre_experiment" results into a single
        result.

        :param results:
            A list of return values from run_iteration from different processes.
        :type results: list

        :return:
            A single result dict with results aggregated.
        :rtype: dict
        """

    @staticmethod
    @abc.abstractmethod
    def distributed_aggregation_interface_implemented() -> bool:
        """
        Implement this method to signal that this interface is implemented,
        enabling the class to be instantiated.

        This method ensures invalid classes can't be constructed. The other
        abstract methods are often sufficient for ensuring this, but they can be
        fooled by classes extending those methods (and calling `super`). This
        method prevents this type of error because there is no reason for a
        class to extend this method.
        """
