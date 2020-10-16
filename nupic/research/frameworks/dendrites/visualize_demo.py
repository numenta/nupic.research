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
This module provides an example of how to visualize dendritic weight activations
"""

import torch

from nupic.research.frameworks.dendrites import (
    AbsoluteMaxGatingDendriticLayer,
    visualize_dendritic_activations,
)
from nupic.research.frameworks.dendrites.routing import generate_context_vectors


def run_wandb_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dendritic network with 10 input units, 10 output units, and 10
    # dendritic weights per output unit
    dendritic_network = AbsoluteMaxGatingDendriticLayer(
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

    dendritic_network = dendritic_network.to(device)
    context_vectors = context_vectors.to(device)

    dendritic_weight_tensor = dendritic_network.segments.weights.data

    # Visualize the absolute activations from just the first output unit's dendritic
    # weights using Weights & Biases
    visualize_dendritic_activations(
        dendritic_weights=dendritic_weight_tensor[0, :, :],
        context_vectors=context_vectors,
        project_title="Dendrites demo",
        run_name="Output unit 0",
        plot_name="Epoch 0",
        use_absolute_activations=True
    )


if __name__ == "__main__":
    run_wandb_demo()
