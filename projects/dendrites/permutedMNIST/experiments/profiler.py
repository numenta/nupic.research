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

"""
Experiments profiling dendrites experiments
"""
from copy import deepcopy

import torch

from nupic.research.frameworks.vernon import mixins

from .prototype import PROTOTYPE_10, PROTOTYPE_50

__all__ = ["CONFIGS"]

PROFILER_ARGS = {
    "with_stack": True,
    "record_shapes": False,
    "schedule": torch.profiler.schedule(wait=1, warmup=1, active=5),
}
WANDB_ARGS = {
    "project": "dendrite_baselines",
    "group": "profiler",
    "notes": "Profiler for dendrite network",
}
PROTOTYPE_10_PROFILER = deepcopy(PROTOTYPE_10)
experiment_class = PROTOTYPE_10_PROFILER["experiment_class"]
PROTOTYPE_10_PROFILER.update(
    experiment_class=mixins.inject_torch_profiler_mixin(experiment_class),
    epochs=1,
    num_samples=1,
    profiler=PROFILER_ARGS,
    wandb_args=WANDB_ARGS,
)

PROTOTYPE_10_ONE_SEGMENT_PROFILER = deepcopy(PROTOTYPE_10_PROFILER)
PROTOTYPE_10_ONE_SEGMENT_PROFILER["model_args"].update(num_segments=1)

PROTOTYPE_50_PROFILER = deepcopy(PROTOTYPE_50)
experiment_class = PROTOTYPE_50_PROFILER["experiment_class"]
PROTOTYPE_50_PROFILER.update(
    experiment_class=mixins.inject_torch_profiler_mixin(experiment_class),
    epochs=1,
    num_samples=1,
    profiler=PROFILER_ARGS,
    wandb_args=WANDB_ARGS,
)

PROTOTYPE_10_TWO_SEGMENT_PROFILER = deepcopy(PROTOTYPE_10_PROFILER)
PROTOTYPE_10_TWO_SEGMENT_PROFILER["model_args"].update(num_segments=2)

# Export configurations in this file
CONFIGS = dict(
    prototype_10_profiler=PROTOTYPE_10_PROFILER,
    prototype_10_one_segment_profiler=PROTOTYPE_10_ONE_SEGMENT_PROFILER,
    prototype_10_two_segment_profiler=PROTOTYPE_10_TWO_SEGMENT_PROFILER,
    prototype_50_profiler=PROTOTYPE_50_PROFILER,
)
