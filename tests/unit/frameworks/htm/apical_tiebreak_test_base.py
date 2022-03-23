# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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

"""
Test the Temporal Memory with explicit basal and apical input. Test that it
correctly uses the "apical tiebreak" approach to basal/apical input.
"""

import random
import unittest
from abc import ABCMeta, abstractmethod

import torch

from nupic.research.frameworks.htm import PairMemoryApicalTiebreak

real_type = torch.float32
int_type = torch.int64


class ApicalTiebreakTestBase(object, metaclass=ABCMeta):
    """
    Test that a Temporal Memory successfully uses the following approach to basal
    and apical connections:
    - Basal input provides the "context". For a cell to be predicted, it must have
        an active basal segment.
    - When multiple cells in a single minicolumn have basal support, they are all
        predicted *unless* one of them also has an active apical segment. In that
        case, only the cells with basal and apical support are predicted.
    The apical dendrites resolve ambiguity when there are multiple cells in a
    minicolumn with basal support. In other words, they handle the situation where
    the basal input is a union.
    """

    apical_input_size = 1000
    basal_input_size = 1000
    num_minicolumns = 2048
    w = 40

    def setUp(self):

        self.num_cells_per_minicolumn = None

        print(
            (
                "\n"
                "======================================================\n"
                "Test: {0} \n"
                "{1}\n"
                "======================================================\n"
            ).format(self.id(), self.shortDescription())
        )

    def testbasal_inputRequiredForPredictions(self):
        """
        Learn A for basal_input1, apical_input1.
        Now observe A with apical_input1 but no basal input. It should burst.
        """

        self.init()

        active_minicolumns = self.randomColumnPattern()
        basal_input = self.randomBasalPattern()
        apical_input = self.randomApicalPattern()

        for _ in range(3):
            self.compute(active_minicolumns, basal_input, apical_input, learn=True)

        self.compute(
            active_minicolumns, basal_input=(), apical_input=apical_input, learn=False
        )

        self.assertEqual(set(active_minicolumns), self.getBurstingColumns())

    def testBasalPredictionsWithoutApical(self):
        """
        Learn A for two contexts:
        - basal_input1, apical_input1
        - basal_input2, apical_input2
        Now observe A with a union of basal_input1 and basal_input2, and no apical
        input. It should predict both contexts.
        """

        self.init()

        active_minicolumns = self.randomColumnPattern()

        basal_input1 = self.randomBasalPattern()
        basal_input2 = self.randomBasalPattern()

        apical_input1 = self.randomApicalPattern()
        apical_input2 = self.randomApicalPattern()

        for _ in range(3):
            self.compute(active_minicolumns, basal_input1, apical_input1, learn=True)
            activeCells1 = set(self.get_active_cells())
            self.compute(active_minicolumns, basal_input2, apical_input2, learn=True)
            activeCells2 = set(self.get_active_cells())

        self.compute(
            active_minicolumns,
            basal_input1 | basal_input2,
            apical_input=(),
            learn=False,
        )

        self.assertEqual(activeCells1 | activeCells2, set(self.get_active_cells()))

    def testApicalNarrowsThePredictions(self):
        """
        Learn A for two contexts:
        - basal_input1, apical_input1
        - basal_input2, apical_input2
        Now observe A with a union of basal_input1 and basal_input2, and apical_input1.
        It should only predict one context.
        """

        self.init()

        active_minicolumns = self.randomColumnPattern()
        basal_input1 = self.randomBasalPattern()
        basal_input2 = self.randomBasalPattern()
        apical_input1 = self.randomApicalPattern()
        apical_input2 = self.randomApicalPattern()

        for _ in range(3):
            self.compute(active_minicolumns, basal_input1, apical_input1, learn=True)
            activeCells1 = set(self.get_active_cells())
            self.compute(active_minicolumns, basal_input2, apical_input2, learn=True)

        self.compute(
            active_minicolumns, basal_input1 | basal_input2, apical_input1, learn=False
        )

        self.assertEqual(activeCells1, set(self.get_active_cells()))

    def testUnionOfFeedback(self):
        """
        Learn A for three contexts:
        - basal_input1, apical_input1
        - basal_input2, apical_input2
        - basal_input3, apical_input3
        Now observe A with a union of all 3 basal inputs, and a union of
        apical_input1 and apical_input2. It should predict 2 of the 3 contexts.
        """

        self.init()

        active_minicolumns = self.randomColumnPattern()
        basal_input1 = self.randomBasalPattern()
        basal_input2 = self.randomBasalPattern()
        basal_input3 = self.randomBasalPattern()
        apical_input1 = self.randomApicalPattern()
        apical_input2 = self.randomApicalPattern()
        apical_input3 = self.randomApicalPattern()

        for _ in range(3):
            self.compute(active_minicolumns, basal_input1, apical_input1, learn=True)
            activeCells1 = set(self.get_active_cells())
            self.compute(active_minicolumns, basal_input2, apical_input2, learn=True)
            activeCells2 = set(self.get_active_cells())
            self.compute(active_minicolumns, basal_input3, apical_input3, learn=True)

        self.compute(
            active_minicolumns,
            basal_input1 | basal_input2 | basal_input3,
            apical_input1 | apical_input2,
            learn=False,
        )

        self.assertEqual(activeCells1 | activeCells2, set(self.get_active_cells()))

    # ==============================
    # Helper functions
    # ==============================

    def init(self, overrides=None):
        """
        Initialize Temporal Memory, and other member variables.
        @param overrides (dict)
        Overrides for default Temporal Memory parameters
        """

        params = {
            "num_minicolumns": self.num_minicolumns,
            "basal_input_size": self.basal_input_size,
            "apical_input_size": self.apical_input_size,
            "num_cells_per_minicolumn": 32,
            "initial_permanence": 0.5,
            "connected_permanence": 0.6,
            "matching_threshold": 25,
            "sample_size": 30,
            "permanence_increment": 0.1,
            "permanence_decrement": 0.02,
            "basal_segment_incorrect_decrement": 0.0,
            "activation_threshold": 25,
            "seed": 42,
        }

        params.update(overrides or {})

        self.num_cells_per_minicolumn = params["num_cells_per_minicolumn"]

        self.constructTM(**params)

    def getBurstingColumns(self):
        predicted = set(
            cell // self.num_cells_per_minicolumn for cell in self.get_predicted_cells()
        )
        active = set(
            cell // self.num_cells_per_minicolumn for cell in self.get_active_cells()
        )

        return active - predicted

    def randomColumnPattern(self):
        return set(random.sample(range(self.num_minicolumns), self.w))

    def randomApicalPattern(self):
        return set(random.sample(range(self.apical_input_size), self.w))

    def randomBasalPattern(self):
        return set(random.sample(range(self.basal_input_size), self.w))

    # ==============================
    # Extension points
    # ==============================

    @abstractmethod
    def constructTM(
        self,
        num_minicolumns,
        basal_input_size,
        apical_input_size,
        num_cells_per_minicolumn,
        initial_permanence,
        connected_permanence,
        matching_threshold,
        sample_size,
        permanence_increment,
        permanence_decrement,
        predictedSegmentDecrement,
        activation_threshold,
        seed,
    ):
        """
        Construct a new TemporalMemory from these parameters.
        """
        pass

    @abstractmethod
    def compute(self, active_minicolumns, basal_input, apical_input, learn):
        """
        Run one timestep of the TemporalMemory.
        """
        pass

    @abstractmethod
    def get_active_cells(self):
        """
        Get the currently active cells.
        """
        pass

    @abstractmethod
    def get_predicted_cells(self):
        """
        Get the cells that were predicted for the current timestep.
        In other words, the set of "correctly predicted cells" is the intersection
        of these cells and the active cells.
        """
        pass


class ApicalTiebreakTM_ApicalTiebreakTests(ApicalTiebreakTestBase, unittest.TestCase):
    """
    Run the "apical tiebreak" tests on the ApicalTiebreakTemporalMemory.
    """

    def constructTM(
        self,
        num_minicolumns,
        basal_input_size,
        apical_input_size,
        num_cells_per_minicolumn,
        initial_permanence,
        connected_permanence,
        matching_threshold,
        sample_size,
        permanence_increment,
        permanence_decrement,
        basal_segment_incorrect_decrement,
        activation_threshold,
        seed,
    ):

        params = {
            "num_minicolumns": num_minicolumns,
            "num_cells_per_minicolumn": num_cells_per_minicolumn,
            "initial_permanence": initial_permanence,
            "connected_permanence": connected_permanence,
            "matching_threshold": matching_threshold,
            "sample_size": sample_size,
            "permanence_increment": permanence_increment,
            "permanence_decrement": permanence_decrement,
            "basal_segment_incorrect_decrement": basal_segment_incorrect_decrement,
            "apical_segment_incorrect_decrement": 0.0,
            "activation_threshold": activation_threshold,
            "seed": seed,
            "basal_input_size": basal_input_size,
            "apical_input_size": apical_input_size,
        }

        self.tm = PairMemoryApicalTiebreak(**params)

    def compute(self, active_minicolumns, basal_input, apical_input, learn):
        active_minicolumns = torch.Tensor(list(sorted(active_minicolumns))).to(int_type)
        basal_input = torch.Tensor(list(sorted(basal_input))).to(int_type)
        apical_input = torch.Tensor(list(sorted(apical_input))).to(int_type)

        self.tm.compute(
            active_minicolumns=active_minicolumns,
            basal_input=basal_input,
            basal_growth_candidates=basal_input,
            apical_input=apical_input,
            apical_growth_candidates=apical_input,
            learn=learn,
        )

    def get_active_cells(self):
        return self.tm.get_active_cells().tolist()

    def get_predicted_cells(self):
        return self.tm.get_predicted_cells().tolist()


if __name__ == "__main__":
    unittest.main()
