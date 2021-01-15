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

from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    plot_dendrite_activations,
)
from nupic.research.frameworks.dendrites.routing import generate_context_vectors


def run_wandb_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dendrite network with 10 input units, 10 output units, and 10
    # dendrite weights per output unit
    dendrite_network = AbsoluteMaxGatingDendriticLayer(
        module=torch.nn.Linear(in_features=10, out_features=10),
        num_segments=10,
        dim_context=100,
        module_sparsity=0.7,
        dendrite_sparsity=0.0
    )

    # Initialize 10 different context vectors that each have 100 dimensions
    context_vectors = generate_context_vectors(
        num_contexts=10,
        n_dim=100,
        percent_on=0.2
    )

    dendrite_network = dendrite_network.to(device)
    context_vectors = context_vectors.to(device)

    # The routing function's mask value for a particular output unit, across all masks
    mask_values = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

    # Call `plot_dendrite_activations` to create a heatmap of just the first output
    # unit's dendrite activations; note that in a non-routing scenario, `mask_values`
    # can be None
    visual = plot_dendrite_activations(
        dendrite_segments=dendrite_network.segments,
        context_vectors=context_vectors,
        selection_criterion="absolute",
        mask_values=mask_values
    )

    # Plot heatmap of dendrite activations using wandb
    wandb.init(name="Output unit 0", project="Dendrites demo")
    wandb.log({"Initial activations": visual})


if __name__ == "__main__":
    run_wandb_demo()
