#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

from copy import deepcopy

import torch

from experiment_classes import OMLExperiment
from nupic.research.frameworks.vernon import mixins

from .oml_replicate import metacl_oml_replicate


class TrackStatsMetaCLExperiment(mixins.TrackRepresentationSparsityMetaCL,
                                 OMLExperiment):
    pass


metacl_with_sparse_stats = deepcopy(metacl_oml_replicate)
metacl_with_sparse_stats.update(
    experiment_class=TrackStatsMetaCLExperiment,

    # Track sparsity statistics.
    track_input_sparsity_args=dict(
        include_modules=[torch.nn.Linear]
    ),
    track_output_sparsity_args=dict(
        include_modules=[torch.nn.ReLU]
    ),

    # Log results to wandb.
    wandb_args=dict(
        name="metacl_with_sparse_stats",
        project="test_metacl",
    ),
)


# Export configurations in this file
CONFIGS = dict(
    metacl_with_sparse_stats=metacl_with_sparse_stats,
)
