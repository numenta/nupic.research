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

import os
from copy import deepcopy

import numpy as np
from PIL import Image
from torch.optim import Adam
from torchvision.transforms import Normalize, ToTensor

from networks import ANMLNetwork

from .oml import metacl_oml


"""
These experiments serve to narrow the gap between our implementation
of ANML and the original repos.
"""


class ANMLTransform:
    """
    Transform images to RGB of size 28 x 28 and normalize with mean=0.92206 * 256
    and std=0.08426 * 256.
    """
    def __init__(self):
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.92206 * 256] * 3,
                                   std=[0.08426 * 256 * 256] * 3)

    def __call__(self, pil_image):
        img = pil_image.convert("L").convert("RGB")
        img = img.resize((28, 28), resample=Image.LANCZOS)
        img = np.array(img, dtype=np.float32)
        img = self.to_tensor(img)
        return self.normalize(img)


# ANML-repo results:
# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.84 ± 0.05      | 0.92 ± 0.02       | 0.001 |
# |            50 | 0.84 ± 0.02      | 0.97 ± 0.01       | 0.001 |
# |           100 | 0.81 ± 0.01      | 0.98 ± 0.01       | 0.001 |
# |           200 | 0.77 ± 0.01      | 0.97 ± 0.00       | 0.001 |
# |           600 | 0.64 ± 0.00      | 0.92 ± 0.00       | 0.001 |
# |--------------------------------------------------------------|
#
# Training command:
# ```
# python mrcl_classification.py --rln 7 --meta_lr 0.001 --update_lr 0.1
#                               --name mrcl_omniglot --steps 20000 --seed 9
#                               --model_name ANML_Model --treatment Neuromodulation
# ```
#
# Evaluation command:
# ```
# python evaluate_classification.py --rln 13 --model ANML_Model.net
#                                   --name ANML_meta_tesing --runs 10
#                                   --schedule 10 50 --neuromodulation
# ```
#


# --------------------
# Vernon ANML Variants
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
    lr_sweep_range=[0.001, 0.0006, 0.0004, 0.00035, 0.0003, 0.00025, 0.0002, 0.00015,
                    0.0001, 0.00009, 0.00008, 0.00006, 0.00003, 0.00001],

    # Perform this many runs over each lr to find the best.
    num_lr_search_runs=10,

    # Perform this many meta-testing runs to average over.
    num_meta_testing_runs=10,

    # Run through meta-test testing 5 images at a time. No training occurs here.
    test_test_batch_size=1,
)


# This attempts to replicate the results from the original ANML paper.
# |----------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |      LR |
# |--------------:|:-----------------|:------------------|--------:|
# |            10 | 0.71 ± 0.08      | 0.79 ± 0.05       | 0.001   |
# |            50 | 0.76 ± 0.03      | 0.88 ± 0.02       | 0.0006  |
# |           100 | 0.64 ± 0.03      | 0.79 ± 0.02       | 0.00035 |
# |           200 | 0.49 ± 0.02      | 0.65 ± 0.02       | 0.0003  |
# |           600 | 0.26 ± 0.01      | 0.39 ± 0.01       | 0.00015 |
# |----------------------------------------------------------------|
#
metacl_anml_replicate = deepcopy(metacl_oml)
metacl_anml_replicate.update(

    local_dir=os.path.expanduser("~/nta/results/experiments/meta_cl"),

    # ANML Network
    model_class=ANMLNetwork,
    model_args=dict(num_classes=1000),  # output layer with 1000 units, more than needed

    # What's trained and when.
    fast_params=["prediction.*", "classifier.*"],  # trained during meta-train training
    test_train_params=["classifier.*"],  # trained during meta-test training

    # Transform the data as done in the ANML repo.
    dataset_args=dict(
        root="~/nta/datasets",
        transform=ANMLTransform(),
    ),

    # Num classes used in meta-training and meta-testing.
    num_classes=963,
    num_classes_eval=660,
    # replay_classes
    # fast_and_slow_classes

    # Learning rate & optimizer.
    adaptation_lr=0.1,  # applied in inner loop via SGD like update rule
    optimizer_class=Adam,
    optimizer_args=dict(lr=1e-3),

    # How often to checkpoint (in epochs)
    checkpoint_freq=1000,
    keep_checkpoints_num=1,
    checkpoint_at_end=True,
    checkpoint_score_attr="training_iteration",

    # Number of tasks (i.e. distinct classes) for the inner loops and outer loop
    tasks_per_epoch=1,

    # Batch size for each inner and outer step, respectively.
    batch_size=1,       # use 1 images per step w/in the inner loop
    slow_batch_size=20,  # use 20 images per class w/in the outer loop.

    # Number of steps per task within the inner loop.
    num_fast_steps=20,  # => we take 20 * 'tasks_per_epoch' sequential grad steps
    # num_slow_steps=1,  # not actually defined, but technically this is always one.

    # Number of images sampled for training (should be b/w 1 and 20)
    train_train_sample_size=20,  # This applies to both the slow and fast iterations.

    # Sampled for the remember set; taken from the whole dataset.
    # Total images used for one step in the outer loop equals:
    #      replay_batch_size + (slow_batch_size * tasks_per_epoch) = 84
    replay_batch_size=64,

    # Since all samples are being used in the slow and fast loops (all 20 of them),
    # there are none being held out for a separate validation phase.
    val_batch_size=1,  # for the validation set; not being used.

    # The number of outer (i.e. slow) steps. The OML-repo README recommends 700000,
    # but we use less here. Note, the OML-repo results above report 20000 steps as well
    # for a consistent comparison.
    epochs=20000,

    # Identify the params of the output layer.
    output_layer_params=["classifier.weight", "classifier.bias"],

    # Reset task params in the output layer prior to meta-train training on that task.
    reset_task_params=True,

    # Log results to wandb.
    wandb_args=dict(
        name="metacl_anml_replicate",
        project="metacl",
    ),

    # Meta-testing specific arguments.
    **deepcopy(meta_test_test_kwargs),
)


# ------------
# All configs.
# ------------

CONFIGS = dict(
    metacl_anml_replicate=metacl_anml_replicate,
)
