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

from experiment_classes import IIDTrainingOMLExperiment, OMLExperiment
from networks import KWinnerOMLNetwork
from nupic.research.frameworks.meta_continual_learning import mixins as metacl_mixins
from nupic.research.frameworks.meta_continual_learning.experiments import (
    MetaContinualLearningExperiment,
)
from nupic.research.frameworks.meta_continual_learning.models import OMLNetwork
from nupic.research.frameworks.pytorch.datasets import omniglot
from nupic.research.frameworks.vernon import mixins as vernon_mixins

from .anml_replicate import metacl_anml_replicate
from .oml_replicate import metacl_oml_replicate, oml_datasplit_without_norm

"""
These experiments alter the OML-replicated experiment in some way
for ablation studies.
"""


# OML but in the i.i.d setting for meta-training; still taking
# second order gradients.
oml_fully_iid = deepcopy(metacl_oml_replicate)
oml_fully_iid.update(
    experiment_class=IIDTrainingOMLExperiment,

    # Number of tasks (i.e. distinct classes) for the inner loops and outer loop
    tasks_per_epoch=15,

    # Batch size for each inner and outer step, respectively.
    batch_size=1,       # use 1 images per step w/in the inner loop
    slow_batch_size=1,  # this should not make any difference

    # Number of steps per task within the inner loop.
    num_fast_steps=1,  # => we take 1 * 'tasks_per_epoch' sequential grad steps
    # num_slow_steps=1,  # not actually defined, but technically this is always one.

    # Sampled for the remember set; 15 as per usual plus another 15 i.i.d.
    replay_batch_size=30,

    # Log results to wandb.
    wandb_args=dict(
        name="oml_fully_iid",
        project="metacl",
    ),
)


oml_fully_iid_fixed_batch = deepcopy(oml_fully_iid)
oml_fully_iid_fixed_batch.update(

    # Instead of 20000 epochs with 15 images per inner loop and 30 per outer loop,
    # we'll do 60000 epochs of 0 images per inner loop and 15 per outer loop.
    epochs=60000,

    # Number of tasks (i.e. distinct classes) for the inner loops and outer loop
    tasks_per_epoch=0,

    # Batch size for each inner and outer step, respectively.
    batch_size=1,       # this is effectively 0
    slow_batch_size=1,  # this is effectively 0

    # Number of steps per task within the inner loop.
    num_fast_steps=0,  # => we take 0 sequential grad steps
    # num_slow_steps=1,  # not actually defined, but technically this is always one.

    # Sampled for the remember set; 15 as per usual plus another 15 i.i.d.
    replay_batch_size=15,

    # Log results to wandb.
    wandb_args=dict(
        name="oml_fully_iid_fixed_batch",
        project="metacl",
    ),
)


# OML in the i.i.d. setting. This time, we'll exclusively use 1st order gradients.
oml_fully_iid_first_order = deepcopy(oml_fully_iid)
oml_fully_iid_first_order.update(
    use_2nd_order_grads=False,

    # Log results to wandb.
    wandb_args=dict(
        name="oml_fully_iid_first_order",
        project="metacl",
    ),
)


oml_fully_iid_with_datasplit = deepcopy(oml_fully_iid)
oml_fully_iid_with_datasplit.update(
    # Split classes among fastslow and replay sets.
    replay_classes=list(range(0, 481)),
    fast_and_slow_classes=list(range(481, 963)),

    # Log results to wandb.
    wandb_args=dict(
        name="oml_fully_iid_with_datasplit",
        project="metacl",
    ),
)


# Use OML's model, but ANML training paradigm.
# |---------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |     LR |
# |--------------:|:-----------------|:------------------|-------:|
# |            10 | 0.94 ± 0.03      | 0.98 ± 0.02       | 0.001  |
# |            50 | 0.86 ± 0.02      | 0.97 ± 0.01       | 0.0006 |
# |           100 | 0.82 ± 0.02      | 0.96 ± 0.01       | 0.0006 |
# |           200 | 0.74 ± 0.01      | 0.95 ± 0.00       | 0.0004 |
# |           600 | 0.62 ± 0.01      | 0.90 ± 0.00       | 0.0006 |
# |---------------------------------------------------------------|
#
oml_trained_like_anml = deepcopy(metacl_anml_replicate)
oml_trained_like_anml.update(
    experiment_class=OMLExperiment,
    dataset_class=omniglot,
    dataset_args=dict(root="~/nta/datasets"),
    model_class=OMLNetwork,

    fast_params=["adaptation.*"],
    test_train_params=["adaptation.*"],

    # Identify the params of the output layer.
    output_layer_params=["adaptation.0.weight", "adaptation.0.bias"],

    # Log results to wandb.
    wandb_args=dict(
        name="oml_trained_like_anml",
        project="metacl",
    ),

    # Running on -r 5: without oml's lr 0.1/0.001
    # Running on -r 4: with omls's lr 0.03/0.0001
    adaptation_lr=0.03,
    optimizer_args=dict(lr=1e-4),
)


# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            50 | 0.70 ± 0.03      | 0.95 ± 0.01       | 0.001 |
# |--------------------------------------------------------------|
#
oml_trained_like_anml_2000 = deepcopy(oml_trained_like_anml)
oml_trained_like_anml_2000.update(
    epochs=2000,
    num_meta_test_classes=[50, 200],

    # Log results to wandb.
    wandb_args=dict(
        name="oml_trained_like_anml_2000",
        project="metacl",
    ),
)


oml_datasplit_without_norm_2000 = deepcopy(oml_datasplit_without_norm)
oml_datasplit_without_norm_2000.update(
    epochs=2000,
    num_meta_test_classes=[50],
)


class KWinnerOMLExperiment(metacl_mixins.OnlineMetaLearning,
                           vernon_mixins.UpdateBoostStrength,
                           MetaContinualLearningExperiment):
    pass


# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.57 ± 0.14      | 0.75 ± 0.13       | 0.01  |
# |            50 | 0.26 ± 0.06      | 0.42 ± 0.10       | 0.001 |
# |           100 | 0.52 ± 0.06      | 0.89 ± 0.10       | 0.003 |
# |           200 | 0.46 ± 0.05      | 0.89 ± 0.08       | 0.001 |
# |           600 | 0.53 ± 0.00      | 1.00 ± 0.00       | 0.001 |
# |--------------------------------------------------------------|
#
oml_with_kwinners = deepcopy(oml_datasplit_without_norm)
oml_with_kwinners.update(
    experiment_class=KWinnerOMLExperiment,

    model_class=KWinnerOMLNetwork,
    model_args=dict(
        num_classes=963,
        boost_strength_factor=0.9995,  # almost fully decayed half-way through
    ),

    # Log results to wandb.
    wandb_args=dict(
        name="oml_with_kwinners",
        project="metacl",
    ),

    # Identify the params of the output layer.
    output_layer_params=["adaptation.1.weight", "adaptation.1.bias"],
)


oml_with_kwinners_percent_on_20 = deepcopy(oml_with_kwinners)
oml_with_kwinners_percent_on_20.update(
    model_args=dict(
        num_classes=963,
        boost_strength_factor=0.9995,  # almost fully decayed half-way through
        percent_on=0.20
    ),

    # Log results to wandb.
    wandb_args=dict(
        name="oml_with_kwinners_percent_on_20",
        project="metacl",
    ),
)


oml_with_kwinners_percent_on_15 = deepcopy(oml_with_kwinners)
oml_with_kwinners_percent_on_15.update(
    model_args=dict(
        num_classes=963,
        boost_strength_factor=0.9995,  # almost fully decayed half-way through
        percent_on=0.15
    ),

    # Log results to wandb.
    wandb_args=dict(
        name="oml_with_kwinners_percent_on_15",
        project="metacl",
    ),
)


# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            50 | 0.48 ± 0.10      | 0.75 ± 0.15       | 0.003 |
# |--------------------------------------------------------------|
#
oml_with_kwinners_2000 = deepcopy(oml_with_kwinners)
oml_with_kwinners_2000.update(
    epochs=2000,
    num_meta_test_classes=[50],
    model_args=dict(
        num_classes=963,
        boost_strength_factor=0.995,  # almost fully decayed half-way through
    ),

    # Log results to wandb.
    wandb_args=dict(
        name="oml_with_kwinners_2000",
        project="metacl",
    ),
)


# ------------
# All configs.
# ------------

CONFIGS = dict(
    oml_fully_iid=oml_fully_iid,
    oml_fully_iid_fixed_batch=oml_fully_iid_fixed_batch,
    oml_fully_iid_first_order=oml_fully_iid_first_order,
    oml_fully_iid_with_datasplit=oml_fully_iid_with_datasplit,
    oml_trained_like_anml=oml_trained_like_anml,
    oml_trained_like_anml_2000=oml_trained_like_anml_2000,
    oml_datasplit_without_norm_2000=oml_datasplit_without_norm_2000,
    oml_with_kwinners=oml_with_kwinners,
    oml_with_kwinners_2000=oml_with_kwinners_2000,
    oml_with_kwinners_percent_on_20=oml_with_kwinners_percent_on_20,
    oml_with_kwinners_percent_on_15=oml_with_kwinners_percent_on_15,
)
