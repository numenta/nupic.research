# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from nupic.research.frameworks.dendrites import BoostedDendritesAbsMaxGate1d

from .dendritic_networks import ReplicateANMLDendriticNetwork

__all__ = [
    "BoostedANMLDendriticNetwork",
]


class BoostedANMLDendriticNetwork(ReplicateANMLDendriticNetwork):
    """
    Similar to the parent network, but with boosting applied to the dendrites.

    Input shape expected is 3 x 28 x 28 (C x W x H) but an adaptive average pooling
    layer helps accept other heights and width.

    With default parameters and `num_classes-963`, it uses 5,118,291 weights-on
    out of a total of 8,345,095 weights.
    """

    def __init__(
        self,
        num_classes,
        num_segments=10,
        dendrite_sparsity=0.856,
        dendrite_bias=True,
        boost_strength=1.0,
        boost_strength_factor=0.995,
        duty_cycle_period=1000,
    ):

        super().__init__(
            num_classes=num_classes,
            num_segments=num_segments,
            dendrite_sparsity=dendrite_sparsity,
            dendrite_bias=dendrite_bias,
        )

        num_units = self.segments.num_units
        num_segments = self.segments.num_segments
        self.apply_boosted_dendrites = BoostedDendritesAbsMaxGate1d(
            num_units,
            num_segments,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

    def apply_dendrites(self, y, dendrite_activations):
        """
        Apply dendrites as a gating mechanism and update boost strength.
        """
        return self.apply_boosted_dendrites(y, dendrite_activations).values
