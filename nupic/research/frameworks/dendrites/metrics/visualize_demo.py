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

"""
This module provides an example of how to visualize dendrite activations

NOTE: Wandb will create and store files in ./wandb/ which will not be removed after
      this script has finished executing, and the new files must be manually deleted
"""

import torch
import wandb

from nupic.research.frameworks.dendrites import plot_dendrite_activations


def run_wandb_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dendrite activations for a dendritic layer which has 1 unit, and 7 segments per
    # unit; the input consists of a batch of 5 examples
    dendrite_activations = torch.tensor(
        [
            [[-0.2114, 0.1766, 0.6050, 0.6600, -0.9530, 0.4609, -0.1015]],
            [[-0.2442, -0.7882, 0.3758, -0.0182, -0.5847, 0.0360, -0.2504]],
            [[0.8437, 0.2504, 0.2197, 0.4834, 0.2731, -0.3862, -0.1709]],
            [[-0.3383, 0.7264, 0.1994, -0.6035, 0.3236, 0.0790, 0.2049]],
            [[0.5494, -0.4031, 0.6324, 0.1556, -0.2197, 0.2124, 0.0208]]
        ]
    )
    dendrite_activations = dendrite_activations.to(device)

    # Binary mask of winning activations
    winning_mask = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        ]
    )
    winning_mask = winning_mask.to(device)

    # The routing function's mask value for a particular output unit, across all masks
    mask_values = [0, 1, 1, 0, 0]

    # Call `plot_dendrite_activations` to create a heatmap of just the first output
    # unit's dendrite activations; note that in a non-routing scenario, `mask_values`
    # can be None
    visual = plot_dendrite_activations(
        dendrite_activations=dendrite_activations,
        winning_mask=winning_mask,
        mask_values=mask_values,
        unit_to_plot=0
    )

    # Plot heatmap of dendrite activations using wandb
    wandb.init(name="Output unit 0", project="Dendrites demo")
    wandb.log({"Initial activations": visual})


if __name__ == "__main__":
    run_wandb_demo()
