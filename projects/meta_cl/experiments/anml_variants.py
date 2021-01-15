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

from copy import deepcopy

from experiment_classes import IIDTrainingOMLExperiment

from .anml_replicate import metacl_anml_replicate

"""
These experiments alter the ANML-replicated experiment in some way
for ablation studies.
"""


anml_first_order = deepcopy(metacl_anml_replicate)
anml_first_order.update(
    use_2nd_order_grads=False,

    # Log results to wandb.
    wandb_args=dict(
        name="anml_first_order",
        project="metacl",
    ),
)


anml_iid = deepcopy(metacl_anml_replicate)
anml_iid.update(
    experiment_class=IIDTrainingOMLExperiment,

    # Take 20 sequential steps in inner loop, each over one image sampled iid.
    tasks_per_epoch=20,

    # Batch size for each inner and outer step, respectively.
    batch_size=1,       # use 1 images per step w/in the inner loop

    # The slow batch is intentionally not sampled by the IIDTrainingOMLExperiment
    slow_batch_size=1,  # this is effectively equal to 0

    # Number of steps per task within the inner loop.
    num_fast_steps=1,  # => we take 1 * 'tasks_per_epoch' sequential grad steps
    # num_slow_steps=1,  # not actually defined, but technically this is always one.

    # Sampled for the remember set; 64 as per usual plus another 20 i.i.d.
    # The extra 20 corresponds to the 20 images usually sampled for the given task.
    replay_batch_size=84,

    # Log results to wandb.
    wandb_args=dict(
        name="anml_iid",
        project="metacl",
    ),
)

# ------------
# All configs.
# ------------

CONFIGS = dict(
    anml_first_order=anml_first_order,
    anml_iid=anml_iid,
)
