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

from nupic.research.frameworks.columns.apical_tiebreak_temporal_memory import (
    ApicalTiebreakPairMemory,
)


class ApicalTiebreakPairMemoryWrapper(ApicalTiebreakPairMemory):
    def __init__(
        self,
        proximal_n,
        proximal_w,
        basal_n,
        basal_w,
        apical_n,
        apical_w,
        cells_per_column,
        activation_threshold,
        reduced_basal_threshold,
        initial_permanence,
        connected_permanence,
        matching_threshold,
        sample_size,
        permanence_increment,
        permanence_decrement,
        seed
    ):
        """
        wrapper class around ApicalTiebreakPairMemory that uses Pythonic
        variables instead of camelCase.
        FIXME: need to change variable structure in ApicalTiebreakTemporalMemory
        """
        super().__init__(
            columnCount=proximal_n,
            basalInputSize=basal_n,
            apicalInputSize=apical_n,
            cellsPerColumn=cells_per_column,
            activationThreshold=activation_threshold,
            reducedBasalThreshold=reduced_basal_threshold,
            initialPermanence=initial_permanence,
            connectedPermanence=connected_permanence,
            minThreshold=matching_threshold,
            sampleSize=sample_size,
            permanenceIncrement=permanence_increment,
            permanenceDecrement=permanence_decrement,
            seed=seed
        )

        self.proximal_n = proximal_n
        self.proximal_w = proximal_w
        self.basal_n = basal_n
        self.basal_w = basal_w
        self.apical_n = apical_n
        self.apical_w = apical_w

    def compute(
        self,
        active_columns,
        basal_input,
        apical_input=(),
        basal_growth_candidates=None,
        apical_growth_candidates=None,
        learn=True
    ):
        super().compute(
            activeColumns=active_columns,
            basalInput=basal_input,
            apicalInput=apical_input,
            basalGrowthCandidates=basal_growth_candidates,
            apicalGrowthCandidates=apical_growth_candidates,
            learn=learn
        )

    def get_winner_cells(self):
        return super().getWinnerCells()

    def get_predicted_active_cells(self):
        return super().getPredictedActiveCells()
