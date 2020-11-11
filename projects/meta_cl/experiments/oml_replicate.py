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

from torchvision import transforms

from .oml import metacl_oml

# """
# These experiments serves to further narrow the gap between our implementation
# of the OML algorithm and that of the original repos.
# """


transform_without_normalization = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
])


# OML-repo results:
# |-----------------------------------------------------------|
# | Num Classes | Meta test-test   | Meta test-train  |  LR   |
# |:----------- |:----------------:|:-----------------|------:|
# |          10 | 0.91 ± 0.04      | 0.98 ± 0.02      | 0.003 |
# |          50 | 0.82 ± 0.03      | 0.96 ± 0.01      | 0.001 |
# |         100 | 0.76 ± 0.03      | 0.96 ± 0.01      | 0.001 |
# |         200 | 0.69 ± 0.02      | 0.94 ± 0.01      | 0.001 |
# |         600 | 0.55 ± 0.01      | 0.91 ± 0.00      | 0.001 |
# |-----------------------------------------------------------|
#


# --------------------
# Vernon OML Variants
# --------------------


meta_test_test_kwargs = dict(

    # Setup the meta-testing phase and allow it to run.
    run_meta_test=True,

    # This resets the fast params (in this case the output layer of the OMLNetwork)
    reset_output_params=True,

    # Results reported over 15 sampled.
    meta_test_sample_size=15,

    # The best lr was chosen among the following; done separately for each number of
    # classes trained on.
    lr_sweep_range=[0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],

    # Run through meta-test testing 5 images at a time. No training occurs here.
    test_test_batch_size=5,
)


# This attempts to replicate the results from the original OML paper.
# It's missing two additional variations incorporated in the configs below
#    1) Reserving classes 1 to 480 for the replay set and classes 481 to 962
#       for the fast/slow sets.
#    2) Keeping the data unnormalized.
#
# Results:
#
# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |  LR   |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.87 ± 0.07      | 0.95 ± 0.03       | 0.001 |
# |            50 | 0.87 ± 0.02      | 0.96 ± 0.02       | 0.001 |
# |           100 | 0.81 ± 0.02      | 0.94 ± 0.01       | 0.001 |
# |           200 | 0.73 ± 0.02      | 0.91 ± 0.02       | 0.001 |
# |           600 | 0.59 ± 0.01      | 0.85 ± 0.01       | 0.001 |
# |--------------------------------------------------------------|
#
metacl_oml_replicate = deepcopy(metacl_oml)
metacl_oml_replicate.update(

    # How often to checkpoint (in epochs)
    checkpoint_freq=1000,
    keep_checkpoints_num=1,
    checkpoint_at_end=True,
    checkpoint_score_attr="training_iteration",

    # Number of tasks (i.e. distinct classes) for the inner loops and outer loop
    tasks_per_epoch=3,

    # Batch size for each inner and outer step, respectively.
    batch_size=1,       # use 1 images per step w/in the inner loop
    slow_batch_size=5,  # use 5 images per class w/in the outer loop.

    # Number of steps per task within the inner loop.
    num_fast_steps=5,  # => we take 5 * 'tasks_per_epoch' sequential grad steps
    # num_slow_steps=1,  # not actually defined, but technically this is always one.

    # Number of images sampled for training (should be b/w 1 and 20)
    train_train_sample_size=20,  # This applies to both the slow and fast iterations.

    # Sampled for the remember set; taken from the whole dataset.
    # Total images used for one step in the outer loop equals:
    #      replay_batch_size + (slow_batch_size * tasks_per_epoch) = 30
    replay_batch_size=15,

    # Since all samples are being used in the slow and fast loops (all 20 of them),
    # there are none being held out for a separate validation phase.
    val_batch_size=1,  # for the validation set; not being used.

    # The number of outer (i.e. slow) steps. The OML-repo README recommends 700000,
    # but we use less here. Note, the OML-repo results above report 20000 steps as well
    # for a consistent comparison.
    epochs=20000,

    # Identify the params of the output layer.
    output_layer_params=["adaptation.0.weight", "adaptation.0.bias"],

    # Reset task params in the output layer prior to meta-train training on that task.
    reset_task_params=True,

    # Whether to run the meta-testing phase at the end of the experiment.
    run_meta_test=False,  # we won't run this for now

    # Log results to wandb.
    wandb_args=dict(
        name="metacl_oml_replicate",
        project="metacl",
    ),

    # Meta-testing specific arguments.
    **deepcopy(meta_test_test_kwargs),
)


# Split the dataset using classes 481 to 962 for the replay set
# and classes 0 to 480 for the fast and slow datasets.
# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |  LR   |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.84 ± 0.05      | 0.93 ± 0.04       | 0.001 |
# |            50 | 0.82 ± 0.03      | 0.96 ± 0.01       | 0.001 |
# |           100 | 0.74 ± 0.03      | 0.94 ± 0.01       | 0.001 |
# |           200 | 0.67 ± 0.02      | 0.92 ± 0.01       | 0.001 |
# |           600 | 0.53 ± 0.01      | 0.87 ± 0.01       | 0.001 |
# |--------------------------------------------------------------|
#
oml_fastslow_replay_split = deepcopy(metacl_oml_replicate)
oml_fastslow_replay_split.update(

    # Log results to wandb.
    wandb_args=dict(
        name="oml_fastslow_replay_split",
        project="metacl",
    ),

    # Split classes among fastslow and replay sets.
    replay_classes=list(range(0, 481)),
    slowfast_classes=list(range(481, 963)),

    # Meta-testing specific arguments.
    **deepcopy(meta_test_test_kwargs),
)


# Train as normal, but without a normalized dataset.
# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |   LR  |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.92 ± 0.05      | 0.97 ± 0.03       | 0.003 |
# |            50 | 0.87 ± 0.02      | 0.96 ± 0.02       | 0.001 |
# |           100 | 0.79 ± 0.03      | 0.93 ± 0.01       | 0.001 |
# |           200 | 0.71 ± 0.02      | 0.89 ± 0.02       | 0.001 |
# |           600 | 0.57 ± 0.01      | 0.82 ± 0.01       | 0.001 |
# |--------------------------------------------------------------|
#
oml_without_normalization = deepcopy(metacl_oml_replicate)
oml_without_normalization["dataset_args"].update(
    transform=transform_without_normalization,
)
oml_without_normalization.update(
    # Log results to wandb.
    wandb_args=dict(
        name="oml_without_normalization",
        project="metacl",
    ),

    # Meta-testing specific arguments.
    **deepcopy(meta_test_test_kwargs),
)

# This is a combination of the last two.
# |---------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |   LR   |
# |--------------:|:-----------------|:------------------|-------:|
# |            10 | 0.86 ± 0.07      | 0.96 ± 0.04       | 0.003  |
# |            50 | 0.81 ± 0.02      | 0.96 ± 0.02       | 0.001  |
# |           100 | 0.73 ± 0.03      | 0.93 ± 0.01       | 0.001  |
# |           200 | 0.65 ± 0.02      | 0.90 ± 0.01       | 0.001  |
# |           600 | 0.52 ± 0.01      | 0.86 ± 0.00       | 0.001  |
# |---------------------------------------------------------------|
#
oml_datasplit_without_norm = deepcopy(oml_without_normalization)
oml_datasplit_without_norm.update(

    # Log results to wandb.
    wandb_args=dict(
        name="oml_datasplit_without_norm",
        project="metacl",
    ),

    # Split classes among fastslow and replay sets.
    replay_classes=list(range(0, 481)),
    slowfast_classes=list(range(481, 963)),

    # Meta-testing specific arguments.
    **deepcopy(meta_test_test_kwargs),
)


# ------------
# All configs.
# ------------

CONFIGS = dict(
    metacl_oml_replicate=metacl_oml_replicate,
    oml_fastslow_replay_split=oml_fastslow_replay_split,
    oml_without_normalization=oml_without_normalization,
    oml_datasplit_without_norm=oml_datasplit_without_norm,
)
