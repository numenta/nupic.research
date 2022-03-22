# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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
Sequence memory tests that focus on the effects of feedback.
"""

import unittest
import random
from abc import ABCMeta, abstractmethod
import torch

from nupic.research.frameworks.htm import SequenceMemoryApicalTiebreak

real_type = torch.float32
int_type = torch.int64

class ApicalTiebreakSequencesTestBase(object, metaclass=ABCMeta):
    """
    Test that a Temporal Memory uses apical dendrites as part of sequence
    inference.
    The expected basal / apical algorithm is:
    - Basal input provides the "context". For a cell to be predicted, it must have
        an active basal segment.
    - When multiple cells in a single minicolumn have basal support, they are all
        predicted *unless* one of them also has an active apical segment. In that
        case, only the cells with basal and apical support are predicted.
    The apical dendrites resolve ambiguity when there are multiple cells in a
    minicolumn with basal support. In other words, they handle the situation where
    the previous input is bursting.
    """

    num_minicolumns = 2048
    w = 40
    apical_input_size = 1000

    def setUp(self):
        self.num_cells_per_minicolumn = None

        print(("\n"
               "======================================================\n"
               "Test: {0} \n"
               "{1}\n"
               "======================================================\n"
               ).format(self.id(), self.shortDescription()))

    def testSequenceMemory_BasalInputRequiredForPredictions(self):
        """
        Learn ABCDE with F1.
        Reset, then observe B with F1.
        It should burst, despite the fact that the B cells have apical support.
        """

        self.init()

        abcde = [self.randomColumnPattern() for _ in range(5)]
        feedback = self.randomApicalPattern()

        for _ in range(4):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=feedback, learn=True)

        self.reset()
        self.compute(abcde[1], apical_input=feedback, learn=False)

        self.assertEqual([], self.get_predicted_cells())
        self.assertEqual(set(abcde[1]), self.get_bursting_minicolumns())

    def testSequenceMemory_BasalPredictionsWithoutFeedback(self):
        """
        Train on ABCDE with F1, XBCDY with F2.
        Test with BCDE. Without feedback, two patterns are predicted.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(10):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=feedback1, learn=True)

            eCells = set(self.get_active_cells())

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=feedback2, learn=True)

            yCells = set(self.get_active_cells())

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apical_input=(), learn=False)

        # The E cells should be active, and so should any Y cells that happen to be
        # in a minicolumn shared between E and Y.
        expectedActive = eCells | set(self.filterCellsByColumn(yCells, abcde[4]))

        self.assertEqual(expectedActive, set(self.get_active_cells()))
        self.assertEqual(eCells | yCells, set(self.get_predicted_cells()))

    def testSequenceMemory_FeedbackNarrowsThePredictions(self):
        """
        Train on ABCDE with F1, XBCDY with F2.
        Test with BCDE with F1. One pattern is predicted.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(10):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=feedback1, learn=True)

            eCells = set(self.get_active_cells())

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=feedback2, learn=True)

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apical_input=feedback1, learn=False)

        self.assertEqual(eCells, set(self.get_active_cells()))
        self.assertEqual(eCells, set(self.get_predicted_cells()))

    def testSequenceMemory_IncorrectFeedbackLeadsToBursting(self):
        """
        Train on ABCDE with F1, XBCDY with F2.
        Test with BCDE with F2. E should burst.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(10):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=feedback1, learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=feedback2, learn=True)

            yCells = set(self.get_active_cells())

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apical_input=feedback2, learn=False)

        self.assertEqual(yCells, set(self.get_predicted_cells()))

        # E should burst, except for columns that happen to be shared with Y.
        self.assertEqual(set(abcde[4]) - set(xbcdy[4]),
                         set(self.get_bursting_minicolumns()))

    def testSequenceMemory_UnionOfFeedback(self):
        """
        Train on ABCDE with F1, XBCDY with F2, MBCDN with F3.
        Test with BCDE with F1 | F2. The last step should predict E and Y.
        """

        self.init()

        bcd = [self.randomColumnPattern() for _ in range(3)]
        abcde = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        xbcdy = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        mbcdn = [self.randomColumnPattern()] + bcd + [self.randomColumnPattern()]
        feedback1 = self.randomApicalPattern()
        feedback2 = self.randomApicalPattern()
        feedback3 = self.randomApicalPattern()

        # First learn the sequences without feedback. We need to let it work through
        # the common subsequence, choosing new cell SDRs for elements later in the
        # sequence, before allowing it to grow apical segments.
        for _ in range(20):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=(), learn=True)

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=(), learn=True)

            self.reset()
            for pattern in mbcdn:
                self.compute(pattern, apical_input=(), learn=True)

        # Learn the apical connections
        for _ in range(2):
            self.reset()
            for pattern in abcde:
                self.compute(pattern, apical_input=feedback1, learn=True)

            eCells = set(self.get_active_cells())

            self.reset()
            for pattern in xbcdy:
                self.compute(pattern, apical_input=feedback2, learn=True)

            yCells = set(self.get_active_cells())

            self.reset()
            for pattern in mbcdn:
                self.compute(pattern, apical_input=feedback3, learn=True)

        # Test
        self.reset()
        for pattern in abcde[1:]:
            self.compute(pattern, apical_input=feedback1 | feedback2, learn=False)

        # The E cells should be active, and so should any Y cells that happen to be
        # in a minicolumn shared between E and Y.
        expectedActive = eCells | set(self.filterCellsByColumn(yCells, abcde[4]))

        self.assertEqual(expectedActive, set(self.get_active_cells()))
        self.assertEqual(eCells | yCells, set(self.get_predicted_cells()))

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
            "apical_input_size": self.apical_input_size,
            "num_cells_per_minicolumn": 32,
            "initial_permanence": 0.5,
            "connected_permanence": 0.6,
            "matching_threshold": 25,
            "sample_size": 30,
            "permanence_increment": 0.1,
            "permanence_decrement": 0.02,
            "basal_segment_incorrect_decrement": 0.08,
            "activation_threshold": 25,
            "seed": 42,
        }

        params.update(overrides or {})

        self.num_cells_per_minicolumn = params["num_cells_per_minicolumn"]

        self.constructTM(**params)

    def get_bursting_minicolumns(self):
        predicted = set(cell // self.num_cells_per_minicolumn
                        for cell in self.get_predicted_cells())
        active = set(cell // self.num_cells_per_minicolumn
                     for cell in self.get_active_cells())

        return active - predicted

    def randomColumnPattern(self):
        return set(random.sample(range(self.num_minicolumns), self.w))

    def randomApicalPattern(self):
        return set(random.sample(range(self.apical_input_size), self.w))

    def filterCellsByColumn(self, cells, columns):
        return [cell
                for cell in cells
                if (cell // self.num_cells_per_minicolumn) in columns]

    # ==============================
    # Extension points
    # ==============================

    @abstractmethod
    def constructTM(self, num_minicolumns, apical_input_size, num_cells_per_minicolumn,
                    initial_permanence, connected_permanence, matching_threshold,
                    sample_size, permanence_increment, permanence_decrement,
                    basal_segment_incorrect_decrement, activation_threshold, seed):
        """
        Construct a new TemporalMemory from these parameters.
        """
        pass

    @abstractmethod
    def compute(self, active_minicolumns, apical_input, learn):
        """
        Run one timestep of the TemporalMemory.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the TemporalMemory.
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

class ApicalTiebreakTM_ApicalTiebreakSequencesTests(ApicalTiebreakSequencesTestBase,
                                                    unittest.TestCase):
    """
    Runs the "apical tiebreak sequences" tests on the ApicalTiebreakTemporalMemory
    """

    def constructTM(self, num_minicolumns, apical_input_size, num_cells_per_minicolumn,
                    initial_permanence, connected_permanence, matching_threshold,
                    sample_size, permanence_increment, permanence_decrement,
                    basal_segment_incorrect_decrement, activation_threshold, seed):

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
            "apical_input_size": apical_input_size,
        }

        self.tm = SequenceMemoryApicalTiebreak(**params)

    def compute(self, active_minicolumns, apical_input, learn):
        import time
        # start = time.time()

        active_minicolumns = torch.Tensor(list(sorted(active_minicolumns))).to(int_type)
        apical_input = torch.Tensor(list(sorted(apical_input))).to(int_type)

        self.tm.compute(active_minicolumns,
                        apical_input=apical_input,
                        apical_growth_candidates=apical_input,
                        learn=learn)
        # print(time.time() - start)

    def reset(self):
        self.tm.reset()

    def get_active_cells(self):
        return self.tm.get_active_cells().tolist()

    def get_predicted_cells(self):
        return self.tm.get_predicted_cells().tolist()

if __name__ == "__main__":
    unittest.main()