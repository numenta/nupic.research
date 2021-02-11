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

import torch

from nupic.research.frameworks.pytorch.mask_utils import indices_to_mask

__all__ = [
    "DendriteSegementsBooster"
]


class DendriteSegementsBooster(torch.nn.Module):
    """
    This class boosts the dendritic activations yielded by a `DendriteSegments` class.
    This is similar to the `KWinners`_ boosting functionality; however, the
    implementation differs in that the boosting mechanism is kept separate from the
    associated module. Thus, this class is a stand alone `torch.nn.Module` which tracks
    the duty-cycles and boosts the activations of an associated `DedriteSegments` module
    accordingly.

    At the end of each batch, be sure to call `update_duty_cylces` with the indices of
    the winning segments. At the end of each epoch, be sure to call
    `update_boost_strength` to multiply the boost strength by the
    `boost_strength_factor`.

    Example usage:
    ```
    # Instantiate segments and booster separately.
    dendrite_segments = DendriteSegments(num_units, num_segments, ...)
    dendrite_booster = DendriteSegementsBooster(num_units, num_segments, ...)

    # Run forward pass.
    y = net(x)
    dendrite_activations = dendrite_segments(context)

    # Boost activations.
    boosted_activations = dendrite_booster.boost_activations(dendrite_activations)

    # Modulate output with boosted activations.
    y_new, winning_indices = apply_dendrites(boosted_activations)

    # Update duty_cylces after each batch.
    dendrite_segments.update_duty_cylces(winning_indices)

    # Update boost_strength after each epoch.
    segments_booster.update_boost_strength()
    ```

    :param num_units: number of units i.e. neurons;
                      each unit has it's own set of dendrite segments
    :param num_segments: number of dendrite segments per unit
    :param boost_strength: initial boost strength; the higher the value, the more
                           scarcely winning segments will be promoted
    :param boost_strength_factor: the boost_strength decay factor; the user calls
                                  `update_boost_strength` to multiply this factor by the
                                  current boost strength.
    :param duty_cycle_period: the size of the rolling window to calculate the frequency
                              of times each segments wins for a given unit.

    .. _KWinners:
        https://github.com/numenta/nupic.torch/blob/master/nupic/torch/modules/k_winners.py
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
        """
        Boost dendrite activations according to the duty cycles of the winning segments.

        :param dendrite_activations: output from a DendriteSegments module;
                                     shape batch_size x num_units x num_segments
        """
        boosted = dendrite_activations.detach()
        boosted *= torch.exp(-self.boost_strength * self.duty_cycles)
        return boosted

    def update_duty_cylces(self, indices):
        """
        Update the moving average of winning segments.

        :param indices: indices of winning segments; shape batch_size x num_units
        """

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
