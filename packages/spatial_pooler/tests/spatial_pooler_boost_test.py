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

import numpy as np
import time
import unittest

from nupic.research.frameworks.spatial_pooler import SpatialPooler

real_type = np.float32
uint_type = np.uint32

SEED = int((time.time() % 10000) * 10)


def compute_overlap(x, y):
    """
    compute overlap between 2 binary arrays
    """

    return ((x + y) == 2).sum()


def sdrs_unique(sdrs_dict):
    """
    return True iff all SDRs in dict are unique
    """

    for k1, v1 in sdrs_dict.items():
        for k2, v2 in sdrs_dict.items():
            if (k1 != k2) and ((v1 == v2).sum() == v1.size):
                return False
    return True


class SpatialPoolerBoostTest(unittest.TestCase):
    """
    Test boosting.

    The test is constructed as follows: we construct a set of 5 known inputs. Two
    of the input patterns have 50% overlap while all other combinations have 0%
    overlap. Each input pattern has 20 bits on to ensure reasonable overlap with
    almost all minicolumns.

    SP parameters:  The SP is set to have 600 minicolumns with 10% output sparsity.
    This ensures that the 5 inputs cannot use up all the minicolumns. Yet we still can
    have a reasonable number of winning minicolumns at each step in order to test
    overlap properties. boostStrength is set to 10 so that some boosted minicolumns are
    guaranteed to win eventually but not necessarily quickly. potentialPct is set
    to 0.9 to ensure all minicolumns have at least some overlap with at least one
    input bit. Thus, when sufficiently boosted, every minicolumn should become a
    winner at some point. We set permanence increment and decrement to 0 so that
    winning minicolumns don't change unless they have been boosted.

    Learning is OFF for Phase 1 & 4 and ON for Phase 2 & 3

    Phase 1: Run spatial pooler on the dataset with learning off to get a baseline
    The boosting factors should be all ones in this phase. A significant fraction
    of the minicolumns will not be used at all. There will be significant overlap
    between the first two inputs.

    Phase 2: Learning is on over the next 10 iterations. During this phase,
    minicolumns that are active frequently will have low boost factors, and minicolumns
    that are not active enough will have high boost factors. All minicolumns should
    be active at some point in phase 2.

    Phase 3: Run one more batch on with learning on. Because of the artificially
    induced thrashing behavior in this test due to boosting, all the inputs should
    now have pretty distinct patterns.

    Phase 4: Run spatial pooler with learning off. Make sure boosting factors
    do not change when learning is off.
    """

    def set_up(self):
        """
        set up various constraints. create input patterns and spatial pooler.
        """

        self.input_size = 90
        self.minicolumn_dims = 600

        self.x = np.zeros((5, self.input_size), dtype=uint_type)
        self.x[0, 0:20] = 1  # pattern A
        self.x[1, 10:30] = 1  # pattern A' (half of bits overlap with A)
        self.x[2, 30:50] = 1  # pattern B (no overlap with others)
        self.x[3, 50:70] = 1  # pattern C (no overlap with others)
        self.x[4, 70:90] = 1  # pattern D (no overlap with others)

        # for each minicolmn, this will contain the last iteration number where that column was a winner
        self.winning_iteration = np.zeros(self.minicolumn_dims)

        # for each input vector i, last_sdr[i] contains the most recent SDR output by the SP
        self.last_sdr = {}

        self.sp = SpatialPooler(
            input_dims=[self.input_size],
            minicolumn_dims=[self.minicolumn_dims],
            num_active_minicolumns_per_inh_area=60,
            local_density=-1,
            potential_radius=self.input_size,
            potential_percent=0.9,
            global_inhibition=True,
            stimulus_threshold=0.0,
            synapse_perm_inc=0.0,
            synapse_perm_dec=0.0,
            synapse_perm_connected=0.1,
            min_percent_overlap_duty_cycles=0.001,
            duty_cycle_period=10,
            boost_strength=10.0,
            seed=SEED,
        )

    def verify_sdr_properties(self):
        """
        verify that all SDRs have properties desired for this test.

        the bounds fo checking overlap are loosely set since there is some variance due to randomness and the artificial parameters used in this test.
        """

        # verify that all SDRs are unique
        self.assertTrue(sdrs_unique(self.last_sdr), "all SDRs are unique")

        # verify that first two SDRs have some overlap
        self.assertGreater(
            compute_overlap(self.last_sdr[0], self.last_sdr[1]),
            9,
            "first two SDRs do not overlap much",
        )

        # verify that last three SDRs have low overlap with everyone else
        for i in [2, 3, 4]:
            for j in range(5):
                if i != j:
                    self.assertLess(
                        compute_overlap(self.last_sdr[i], self.last_sdr[j]),
                        18,
                        "one of the last three SDRs has high overlap",
                    )

    def boost_test_phase1(self):
        y = np.zeros(self.minicolumn_dims, dtype=uint_type)

        # one batch through input patterns while learning is off
        for idx, v in enumerate(self.x):
            y.fill(0)
            self.sp.compute(v, False, y)
            self.winning_iteration[y.nonzero()[0]] = self.sp.get_iteration_learn_num()
            self.last_sdr[idx] = y.copy()

        # boost factor for all minicolumns should be at 1
        boost = self.sp.get_boost_factors()
        self.assertEqual(
            (boost == 1).sum(), self.minicolumn_dims, "boost factors are not all 1"
        )

        # at least half of minicolumns should have never been active
        self.assertGreaterEqual(
            (self.winning_iteration == 0).sum(),
            self.minicolumn_dims // 2,
            "more than half of all minicolumns have been active",
        )

        self.verify_sdr_properties()

    def boost_test_phase2(self):
        y = np.zeros(self.minicolumn_dims, dtype=uint_type)

        # 10 training batch through input patterns
        for _ in range(10):
            for idx, v in enumerate(self.x):
                y.fill(0)
                self.sp.compute(v, True, y)
                self.winning_iteration[
                    y.nonzero()[0]
                ] = self.sp.get_iteration_learn_num()
                self.last_sdr[idx] = y.copy()

        # all never-active minicolumns should have duty cycle of 0
        duty_cycles = self.sp.get_active_duty_cycles()
        self.assertEqual(
            duty_cycles[self.winning_iteration == 0].sum(),
            0,
            "inactive minicolumns have positive duty cycle",
        )

        boost = self.sp.get_boost_factors()
        self.assertLessEqual(
            np.max(boost[np.where(duty_cycles > 0.1)]),
            1.0,
            "strongly active minicolumns have high boost factors",
        )
        self.assertGreaterEqual(
            np.min(boost[np.where(duty_cycles < 0.1)]),
            1.0,
            "weakly active columns have low boost factors",
        )

        # every minicolumn should have been sufficiently boosted to win at least once. number of minicolumns that have never won should be 0
        num_losers_after = (self.winning_iteration == 0).sum()
        self.assertEqual(num_losers_after, 0)

        # artificially induced thrashing --> even the first two patterns should have low overlap. verify this.
        self.assertLess(
            compute_overlap(self.last_sdr[0], self.last_sdr[1]),
            7,
            "first two SDRs overlap significantly when they should not",
        )

    def boost_test_phase3(self):
        y = np.zeros(self.minicolumn_dims, dtype=uint_type)

        # one more training batch through input patterns
        for idx, v in enumerate(self.x):
            y.fill(0)
            self.sp.compute(v, True, y)
            self.winning_iteration[y.nonzero()[0]] = self.sp.get_iteration_learn_num()
            self.last_sdr[idx] = y.copy()

        # by now, every minicolumn should have been sufficiently boosted to win at least once. number of minicolumns that have never won should be 0.
        num_losers_after = (self.winning_iteration == 0).sum()
        self.assertEqual(num_losers_after, 0)

        # artificially induced thrashing --> even the first two patterns should have low overlap. verify this.
        self.assertLess(
            compute_overlap(self.last_sdr[0], self.last_sdr[1]),
            7,
            "first two SDRs overlap significantly when they should not",
        )

    def boost_test_phase4(self):
        y = np.zeros(self.minicolumn_dims, dtype=uint_type)

        boost_at_beginning = self.sp.get_boost_factors()

        # one more training batch through input parameters with **learning OFF**
        for idx, v in enumerate(self.x):
            y.fill(0)
            self.sp.compute(v, False, y)

            boost = self.sp.get_boost_factors()
            self.assertEqual(
                boost.sum(),
                boost_at_beginning.sum(),
                "boost factors changed when learning was off",
            )

    def test_boost(self):
        self.set_up()

        self.boost_test_phase1()
        self.boost_test_phase2()
        self.boost_test_phase3()
        self.boost_test_phase4()


if __name__ == "__main__":
    unittest.main()
