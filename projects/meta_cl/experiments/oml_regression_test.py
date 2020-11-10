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

from .oml import metacl_oml_replicate

meta_test_test_kwargs = dict(

    # Setup the meta-testing phase and allow it to run.
    run_meta_test=True,

    # This resets the fast params (in this case the output layer of the OMLNetwork)
    reset_fast_params=True,

    # Results reported over 15 sampled.
    meta_test_sample_size=15,

    # Run meta-testing over 10 and 50 classes.
    num_meta_test_classes=[10, 50],

    # The best lr was chosen among the following; done separately for each number of
    # classes trained on.
    lr_sweep_range=[0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],

    # Run through meta-test testing 5 images at a time. No training occurs here.
    test_test_batch_size=5,
)


# Run OML for 2000 steps to ensure meta=testing accuracy hasn't regressed.
# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.84 ± 0.06      | 0.94 ± 0.05       | 0.003 |
# |            50 | 0.75 ± 0.03      | 0.95 ± 0.01       | 0.001 |
# |--------------------------------------------------------------|
oml_regression_test = deepcopy(metacl_oml_replicate)
oml_regression_test.update(

    # The number of outer (i.e. slow) steps.
    epochs=2000,

    # Log results to wandb.
    wandb_args=dict(
        name="oml_regression_test",
        project="metacl",
    ),

    # Meta-testing specific arguments.
    **deepcopy(meta_test_test_kwargs),
)

# ------------
# All configs.
# ------------

CONFIGS = dict(
    oml_regression_test=oml_regression_test,
)
