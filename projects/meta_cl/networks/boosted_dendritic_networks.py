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

# %%
import torch

from nupic.research.frameworks.dendrites import DendriticAbsoluteMaxGate1d
from nupic.research.frameworks.pytorch.mask_utils import indices_to_mask

from .dendritic_networks import (
    CloserToANMLDendriticNetwork,
    WiderVisualANMLDendriticNetwork,
)

__all__ = [
    "BoostedANMLDendriticNetwork",
    "DendriteSegementsBooster",
    "WiderVisualBoostedANMLDendriticNetwork",
]


class DendriteSegementsBooster(torch.nn.Module):
    """
    Example usage:
    ```
    # Iterate over data in an epoch.
    for x, target in data_loader:

        # Compute output.
        y = net(x)

        # Modulate output (or intermediate values) of network via dendrites.
        dendrite_activations = dendrite_segments(context)
        boosted_activations = segments_booster.boost_activations(dendrite_activations)
        y_new, winning_indices = apply_dendrites(boosted_activations)

        # Update the duty_cyle to keep track of which segments won.
        segments_booster.update_duty_cylces(winning_indices)

    # Update the boost_strength at the end of the epoch.
    net.segments_booster.update_boost_strength()
    ```
    """

    def __init__(
        self,
        num_units,
        num_segments,
        boost_strength=1.0,
        boost_strength_factor=1.0,
        duty_cycle_period=1000,
    ):
        super().__init__()
        assert boost_strength >= 0.0
        assert 0.0 <= boost_strength_factor <= 1.0

        self.num_units = num_units
        self.num_segments = num_segments
        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period

        boost_strength = torch.tensor(boost_strength, dtype=torch.float)
        duty_cycles = torch.zeros((num_units, num_segments))

        self.register_buffer("duty_cycles", duty_cycles)
        self.register_buffer("boost_strength", boost_strength)
        self.register_buffer("learning_iterations", torch.tensor(0))

    def extra_repr(self):
        return (
            f"num_units={self.num_units}, "
            f"num_segments={self.num_segments}, "
            f"boost_strength={self.boost_strength}, "
            f"boost_strength_factor={self.boost_strength_factor}, "
            f"duty_cycle_period={self.duty_cycle_period}"
        )

    def boost_activations(self, dendrite_activations):
        boosted = dendrite_activations.detach()
        boosted *= torch.exp(-self.boost_strength * self.duty_cycles)
        return boosted

    def update_duty_cylces(self, indices):

        # Figure out which segments won for each unit for each batch.
        batch_size = indices.shape[0]
        shape = (batch_size, self.num_units, self.num_segments)
        winning_mask = indices_to_mask(indices, shape, dim=2)

        # Sum over the batches.
        winning_mask = winning_mask.sum(dim=0)

        # Update the duty cycle.
        self.learning_iterations += batch_size
        period = min(self.duty_cycle_period, self.learning_iterations)
        self.duty_cycles.mul_(period - batch_size)
        self.duty_cycles.add_(winning_mask)
        self.duty_cycles.div_(period)

    def update_boost_strength(self):
        """
        Update boost strength by multiplying by the boost strength factor.
        This is typically done during training at the beginning of each epoch.
        """
        self.boost_strength.mul_(self.boost_strength_factor)


class BoostedANMLDendriticNetwork(CloserToANMLDendriticNetwork):
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
        self.segments_booster = DendriteSegementsBooster(
            num_units,
            num_segments,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.dendritic_gate = DendriticAbsoluteMaxGate1d()

    def apply_dendrites(self, y, dendrite_activations):
        """
        Apply dendrites as a gating mechanism and update boost strength.
        """

        boosted = self.segments_booster.boost_activations(dendrite_activations)
        output, indices = self.dendritic_gate(y, boosted)

        if self.training:
            self.segments_booster.update_duty_cylces(indices)

        return output


class WiderVisualBoostedANMLDendriticNetwork(WiderVisualANMLDendriticNetwork):
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
        self.segments_booster = DendriteSegementsBooster(
            num_units,
            num_segments,
            boost_strength=boost_strength,
            boost_strength_factor=boost_strength_factor,
            duty_cycle_period=duty_cycle_period,
        )

        self.dendritic_gate = DendriticAbsoluteMaxGate1d()

    def apply_dendrites(self, y, dendrite_activations):
        """
        Apply dendrites as a gating mechanism and update boost strength.
        """

        boosted = self.segments_booster.boost_activations(dendrite_activations)
        output, indices = self.dendritic_gate(y, boosted)

        if self.training:
            self.segments_booster.update_duty_cylces(indices)

        return output
