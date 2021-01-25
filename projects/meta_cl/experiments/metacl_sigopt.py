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

from sigopt_utils import MetaCLSigOptExperiment, MetaCLSigOptRemoteProcessTrainable

from .oml_variants import oml_trained_like_anml

oml_trained_like_anml_lr_search = deepcopy(oml_trained_like_anml)
oml_trained_like_anml_lr_search.update(
    ray_trainable=MetaCLSigOptRemoteProcessTrainable,
    sigopt_experiment=MetaCLSigOptExperiment,
    epochs=20000,
    num_meta_test_classes=[100],
    stop=dict(early_stop=1.0),
    sigopt_config=dict(
        name="oml_trained_like_anml_lr_search",
        parameters=[
            dict(name="log10_inner_lr", type="double", bounds=dict(min=-5, max=-1)),
            dict(name="log10_outer_lr", type="double", bounds=dict(min=-5, max=-1)),
        ],
        metrics=[dict(name="mean_test_test_acc", objective="maximize")],
        parallel_bandwidth=1,
        observation_budget=61,
        project="mcaporale-metacl",
    ),
    sigopt_experiment_id=368294,
)

# |--------------------------------------------------------------|
# |   Num Classes | Meta-test test   | Meta-test train   |    LR |
# |--------------:|:-----------------|:------------------|------:|
# |            10 | 0.91 ± 0.05      | 0.97 ± 0.02       | 0.001 |
# |            50 | 0.88 ± 0.01      | 0.98 ± 0.01       | 0.001 |
# |           100 | 0.83 ± 0.02      | 0.98 ± 0.01       | 0.001 |
# |           200 | 0.77 ± 0.01      | 0.96 ± 0.01       | 0.001 |
# |           600 | 0.63 ± 0.01      | 0.93 ± 0.00       | 0.001 |
# |--------------------------------------------------------------|
#
oml_trained_like_anml_best_lr = deepcopy(oml_trained_like_anml)
oml_trained_like_anml_best_lr.update(
    # Using best lr's found in sigopt search.
    adaptation_lr=0.07041400155996859,  # OML originally used 0.03
    optimizer_args=dict(lr=0.0002724584690422078),  # OML originally used 0.0001

    # Log results to wandb.
    wandb_args=dict(
        name="oml_trained_like_anml_best_lr",
        project="metacl",
    ),
)


CONFIGS = dict(
    oml_trained_like_anml_lr_search=oml_trained_like_anml_lr_search,
    oml_trained_like_anml_best_lr=oml_trained_like_anml_best_lr,
)
