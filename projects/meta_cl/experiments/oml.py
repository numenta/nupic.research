#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import copy
import os

import torch

from nupic.research.frameworks.pytorch.datasets import omniglot
from nupic.research.frameworks.pytorch.models import OMLNetwork
from nupic.research.frameworks.vernon import MetaContinualLearningExperiment, mixins


class OMLExperiment(mixins.OnlineMetaLearning,
                    mixins.ResetOMLTaskParams,
                    MetaContinualLearningExperiment):
    pass


# Alternative to run on a single GPU
def run_experiment(config):
    exp = config.get("experiment_class")()
    exp.setup_experiment(config)
    print(f"Training started....")
    while not exp.should_stop():
        result = exp.run_epoch()
        print(f"Accuracy: {result['mean_accuracy']:.4f}")
    print(f"....Training finished")


# Simple Omniglot Experiment
metacl_base = dict(
    # training infrastructure
    distributed=False,
    # problem specific
    experiment_class=MetaContinualLearningExperiment,
    dataset_class=omniglot,
    model_class=OMLNetwork,
    model_args=dict(num_classes=963),
    # metacl variables
    num_classes=963,
    num_classes_eval=660,
    batch_size=5,
    val_batch_size=15,
    slow_batch_size=64,
    replay_batch_size=64,
    epochs=1000,
    tasks_per_epoch=10,
    adaptation_lr=0.03,
    fast_params=["adaptation.*"],
    test_train_params=["adaptation.*"],
    output_layer_params=["adaptation.0.weight", "adaptation.0.bias"],
    # generic
    optimizer_class=torch.optim.Adam,
    optimizer_args=dict(lr=1e-4),
    dataset_args=dict(root="~/nta/datasets"),
    local_dir=os.path.expanduser("~/nta/results/experiments/meta_cl"),
)

metacl_oml = copy.deepcopy(metacl_base)
metacl_oml.update(
    experiment_class=OMLExperiment,
    run_meta_test=True,
)

# This attempts to replicate the results from the original OML paper.
# See results below.
metacl_oml_replicate = copy.deepcopy(metacl_oml)
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

    # Whether to run the meta-testing phase at the end of the experiment.
    run_meta_test=False,  # we won't run this for now

    # Log results to wandb.
    wandb_args=dict(
        name="metacl_oml_replicate",
        project="metacl",
    )
)


# This config is explicitly for running the meta-testing phase.
# Here are the best results so far and a comparison with the OML repo:
#
# |----------------------------------------------------------------------------|
# | Num Classes | Meta test-test               | Meta test-train               |
# |:----------- |:--------------:|:-----------:|:---------------:|:-----------:|
# |             |     Vernon     |     OML     |     Vernon      |     OML     |
# | 10          |  0.85 ± 0.06   | 0.91 ± 0.04 |   0.95 ± 0.03   | 0.98 ± 0.02 |
# | 50          |  0.72 ± 0.02   | 0.82 ± 0.03 |   0.92 ± 0.01   | 0.96 ± 0.01 |
# | 100         |  0.69 ± 0.03   | 0.76 ± 0.03 |   0.94 ± 0.01   | 0.96 ± 0.01 |
# | 200         |  0.62 ± 0.01   | 0.69 ± 0.02 |   0.94 ± 0.01   | 0.94 ± 0.01 |
# | 200         |  0.43 ± 0.01   | 0.55 ± 0.01 |   0.84 ± 0.00   | 0.91 ± 0.00 |
# |----------------------------------------------------------------------------|
#
metacl_oml_replicate_metatest = copy.deepcopy(metacl_oml_replicate)
metacl_oml_replicate_metatest.update(

    # Setup the meta-testing phase and allow it to run.
    run_meta_test=True,

    # This resets the fast params (in this case the output layer of the OMLNetwork)
    reset_fast_params=True,

    # Results reported over 15 sampled.
    test_train_sample_size=15,

    # The best lr was chosen among the following; done separately for each number of
    # classes trained on.
    lr_sweep_range=[0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],
)

# Export configurations in this file
CONFIGS = dict(
    metacl_base=metacl_base,
    metacl_oml=metacl_oml,
    metacl_oml_replicate=metacl_oml_replicate,
    metacl_oml_replicate_metatest=metacl_oml_replicate_metatest,
)
