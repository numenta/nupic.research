# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

import ray.tune as tune
import torch

from nupic.research.frameworks.greedy_infomax.models.classification_model import (
    Classifier,
)
from nupic.research.frameworks.greedy_infomax.models.full_model import (
    SparseSmallVisionModel,
    VDropSparseSmallVisionModel,
    WrappedSparseSmallVisionModel,
    WrappedSuperGreedySmallSparseVisionModel,
)
from nupic.research.frameworks.greedy_infomax.utils.model_utils import (
    evaluate_gim_model,
    train_gim_model,
)
from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptExperiment
from nupic.research.frameworks.vernon.distributed import experiments, mixins
from nupic.torch.modules import SparseWeights2d

from .default_base import CONFIGS as DEFAULT_BASE_CONFIGS


class GreedyInfoMaxExperimentSparse(
    mixins.LogEveryLoss,
    mixins.RezeroWeights,
    mixins.LogBackpropStructure,
    mixins.NoiseRobustnessTest,
    experiments.SelfSupervisedExperiment,
):
    # avoid changing key names for sigopt
    @classmethod
    def get_readable_result(cls, result):
        return result


DEFAULT_BASE = DEFAULT_BASE_CONFIGS["default_base"]

BATCH_SIZE = 32
NUM_EPOCHS = 10
model_args = DEFAULT_BASE["model_args"]
static_sparse_weights = dict(
    # weight sparsity
    sparsity=dict(
        conv1=0.01,  # dense
        encoder1=dict(
            block1=dict(conv1=0.7, conv2=0.7),
            block2=dict(conv1=0.7, conv2=0.7),
            block3=dict(conv1=0.7, conv2=0.7),
            bilinear_info=0.1,  # dense weights
        ),
    )
)
static_sparse_activations = dict(
    percent_on=dict(
        encoder1=dict(
            block1=dict(nonlinearity1=0.2, nonlinearity2=0.2),
            block2=dict(nonlinearity1=0.2, nonlinearity2=0.2),
            block3=dict(nonlinearity1=0.2, nonlinearity2=0.2),
        )
    )
)

model_args.update(dict(sparse_weights_class=SparseWeights2d))
sparse_weights_only_args = deepcopy(model_args)
sparse_weights_only_args.update(static_sparse_weights)
sparse_activations_only_args = deepcopy(model_args)
sparse_activations_only_args.update(static_sparse_activations)
sparse_weights_and_activations_args = deepcopy(sparse_weights_only_args)
sparse_weights_and_activations_args.update(static_sparse_activations)

SMALL_SPARSE_BASE = deepcopy(DEFAULT_BASE)
SMALL_SPARSE_BASE.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity", name="sparse_resnet_base"
        ),
        experiment_class=GreedyInfoMaxExperimentSparse,
        epochs=NUM_EPOCHS,
        epochs_to_validate=range(NUM_EPOCHS),
        supervised_training_epochs_per_validation=10,
        batch_size=BATCH_SIZE,
        model_class=SparseSmallVisionModel,
        noise_levels=[0.1, 0.5, 0.9],
        model_args=model_args,
        classifier_config=dict(
            model_class=Classifier,
            model_args=dict(in_channels=64, num_classes=10),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)

STATIC_SPARSE_WEIGHTS_SMALL = deepcopy(SMALL_SPARSE_BASE)
STATIC_SPARSE_WEIGHTS_SMALL.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity", name="small_static_sparse_weights"
        ),
        model_args=sparse_weights_only_args,
    )
)


STATIC_SPARSE_ACTIVATIONS_SMALL = deepcopy(SMALL_SPARSE_BASE)
STATIC_SPARSE_ACTIVATIONS_SMALL.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity",
            name="small_static_sparse_activations",
        ),
        model_args=sparse_activations_only_args,
    )
)

SPARSE_WEIGHTS_ACTIVATIONS_SMALL = deepcopy(STATIC_SPARSE_ACTIVATIONS_SMALL)
SPARSE_WEIGHTS_ACTIVATIONS_SMALL.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-static-sparsity",
            name="small_static_sparse_weights_activations",
        ),
        model_args=sparse_weights_and_activations_args,
    )
)

# only used to determine largest layer size that can still fit on a p2.8xlarge
dimension_search_model_args = deepcopy(model_args)
dimension_search_model_args.update(num_channels=350)
SMALL_SPARSE_LARGEST_DIMENSION = deepcopy(SMALL_SPARSE_BASE)
SMALL_SPARSE_LARGEST_DIMENSION.update(
    wandb_args=dict(
        project="greedy_infomax-static-sparsity",
        name="small_static_sparse_largest_dimension",
    ),
    model_args=dimension_search_model_args,
    classifier_config=dict(
        model_class=Classifier,
        model_args=dict(num_classes=dimension_search_model_args["num_channels"]),
        loss_function=torch.nn.functional.cross_entropy,
        # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
        optimizer_class=torch.optim.Adam,
        # Optimizer class class arguments passed to the constructor
        optimizer_args=dict(lr=2e-4),
    ),
)
# LARGEST CHANNEL WIDTH = 350
grid_search_weight_sparsity = tune.grid_search([0, 0.5, 0.85])
grid_search_percent_on = tune.grid_search([1.0, 0.2, 0.1])
grid_search_dimensionality = tune.grid_search([64, 128, 256])

grid_search_model_args = deepcopy(model_args)
grid_search_model_args.update(sparsity=grid_search_weight_sparsity)
grid_search_model_args.update(percent_on=grid_search_percent_on)
grid_search_model_args.update(num_channels=grid_search_dimensionality)

# only used to split up work amongst different machines
grid_search_model_args_1 = deepcopy(grid_search_model_args)
grid_search_model_args_1.update(sparsity=0)
grid_search_model_args_2 = deepcopy(grid_search_model_args)
grid_search_model_args_2.update(sparsity=0.5)
grid_search_model_args_3 = deepcopy(grid_search_model_args)
grid_search_model_args_3.update(sparsity=0.85)


class GreedyInfoMaxExperimentSparse2(
    mixins.LogEveryLoss,
    mixins.RezeroWeights,
    mixins.LogBackpropStructure,
    mixins.NoiseRobustnessTest,
    experiments.SelfSupervisedExperiment,
):
    # avoid changing key names for sigopt
    @classmethod
    def get_readable_result(cls, result):
        return result

    def setup_experiment(self, config):
        num_channels = config["model_args"]["num_channels"]
        config["classifier_config"]["model_args"].update(in_channels=num_channels)
        super().setup_experiment(config)


class MultiHeadedGreedyInfoMaxExperiment(
    mixins.LogEveryLoss,
    mixins.RezeroWeights,
    mixins.LogBackpropStructure,
    mixins.NoiseRobustnessTest,
    experiments.SelfSupervisedExperiment,
):
    # avoid changing key names for sigopt
    @classmethod
    def get_readable_result(cls, result):
        return result

    def post_batch(self, error_loss, complexity_loss, batch_idx, **kwargs):
        module_wise_losses = error_loss
        error_loss = module_wise_losses.sum()
        super().post_batch(error_loss, complexity_loss, batch_idx, **kwargs)

    def setup_experiment(self, config):
        num_channels = config["model_args"]["num_channels"]
        config["classifier_config"]["model_args"].update(in_channels=num_channels)
        config["train_model_func"] = train_gim_model
        config["evaluate_model_func"] = evaluate_gim_model
        super().setup_experiment(config)


SPARSE_SMALL_GRID_SEARCH = deepcopy(STATIC_SPARSE_WEIGHTS_SMALL)
SPARSE_SMALL_GRID_SEARCH.update(
    dict(
        experiment_class=GreedyInfoMaxExperimentSparse2,
        wandb_args=dict(
            project="greedy_infomax-static-sparsity",
            name="static_sparse_small_dimensionality_grid_search",
        ),
        model_class=WrappedSparseSmallVisionModel,
        model_args=grid_search_model_args,
        classifier_config=dict(
            model_class=Classifier,
            model_args=dict(num_classes=10),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)


def make_reg_schedule(
    epochs, pct_ramp_start, pct_ramp_end, peak_value, pct_drop, final_value
):
    def reg_schedule(epoch, batch_idx, steps_per_epoch):
        pct = (epoch + batch_idx / steps_per_epoch) / epochs

        if pct < pct_ramp_start:
            return 0.0
        elif pct < pct_ramp_end:
            progress = (pct - pct_ramp_start) / (pct_ramp_end - pct_ramp_start)
            return progress * peak_value
        elif pct < pct_drop:
            return peak_value
        else:
            return final_value

    return reg_schedule


class GreedyInfoMaxExperimentSparsePruning(
    mixins.LogEveryLoss,
    mixins.RegularizeLoss,
    mixins.ConstrainParameters,
    mixins.LogBackpropStructure,
    mixins.PruneLowSNRGlobal,
    mixins.NoiseRobustnessTest,
    experiments.SelfSupervisedExperiment,
):
    pass


SPARSE_VDROP_SMALL = deepcopy(SMALL_SPARSE_BASE)
SPARSE_VDROP_SMALL.update(
    dict(
        wandb_args=dict(
            project="greedy_infomax-sparsity-tests", name="sparse_resnet_vdrop_small"
        ),
        experiment_class=GreedyInfoMaxExperimentSparsePruning,
        epochs=30,
        epochs_to_validate=[0, 3, 6, 9, 12, 15, 19, 23, 27, 29],
        model_class=VDropSparseSmallVisionModel,
        model_args=dict(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            block_dims=None,
            num_channels=None,
            grayscale=True,
            patch_size=16,
            overlap=2,
            # percent_on=None,
        ),
        prune_schedule=[(8, 0.8), (14, 0.6), (20, 0.4), (25, 0.3)],
        log_module_sparsities=True,
        reg_scalar=make_reg_schedule(
            epochs=30,
            pct_ramp_start=4 / 30,
            pct_ramp_end=6 / 30,
            peak_value=0.0003,
            pct_drop=26 / 30,
            final_value=0.00001,
        ),
        optimizer_class=torch.optim.Adam,
        # Optimizer class class arguments passed to the constructor
        optimizer_args=dict(lr=2e-4),
        # batches_in_epoch=1,
        # batches_in_epoch_supervised=1,
        # batches_in_epoch_val=1,
        # supervised_training_epochs_per_validation=1,
        # batch_size=16,
    )
)


# grid search using dense activations, sparse weights, nonzero params constant,
# varying channels, using OneCycleLR
# (num_channels, required weight density)
# channels_density = [
# (64, 1.0),
# (90, 0.5069810763533928),
# (117, 0.30042822545578884),
# (143, 0.20129180763773824),
# (203, 0.10000452594328949),
# (287, 0.050073336993638744),
# ]

channels_density = [(32, 1.0), (45, 0.503), (58, 0.301), (72, 0.194), (100, 0.0993)]


experiment_idx = 2
exp_num_channels, exp_required_density = channels_density[experiment_idx]
exp_required_sparsity = 1.0 - exp_required_density
dimensionality_study_args = deepcopy(model_args)
dimensionality_study_args.update(percent_on=1.0)
dimensionality_study_args.update(num_channels=exp_num_channels)
dimensionality_study_args.update(sparsity=exp_required_sparsity)
STATIC_SPARSE_SMALL_DIMENSIONALITY_STUDY = deepcopy(STATIC_SPARSE_WEIGHTS_SMALL)
STATIC_SPARSE_SMALL_DIMENSIONALITY_STUDY.update(
    dict(
        experiment_class=GreedyInfoMaxExperimentSparse2,
        wandb_args=dict(
            project="greedy_infomax-static-sparsity-hyperparameters",
            name=f"constant_nonzero_params_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse",
            group=f"constant_nonzero_params_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse",
        ),
        batch_size=32,
        batch_size_supervised=32,
        val_batch_size=32,
        # batch_size=16,
        # batch_size_supervised=16,
        # val_batch_size=16,
        model_class=WrappedSparseSmallVisionModel,
        model_args=dimensionality_study_args,
        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=2e-4),
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_args=dict(
            max_lr=0.185,  # change based on sparsity/dimensionality
            div_factor=5,  # initial_lr = 0.01
            final_div_factor=1000,  # min_lr = 0.0000025
            pct_start=1.0 / 10.0,
            epochs=10,
            anneal_strategy="linear",
            max_momentum=1e-4,
            cycle_momentum=False,
        ),
        classifier_config=dict(
            model_class=Classifier,
            model_args=dict(num_classes=10),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)

# Sigopt optimize the onecycleLR


class OneCycleLRGreedyInfoMaxSparseSigOpt(SigOptExperiment):
    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the optimizer_args with SGD optimizer params.

        :param config:
            - lr_scheduler_args
        :param suggestion:
            - assignments (all optional)
                - max_lr
                - initial_lr
                - min_lr
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments
        assert "lr_scheduler_args" in config

        # Optimizer args
        config["lr_scheduler_args"].update(
            dict(
                max_lr=assignments["max_lr"],
                div_factor=assignments["div_factor"],
                final_div_factor=assignments["final_div_factor"],
            )
        )


sigopt_experiment_idx = 2
exp_num_channels, exp_required_density = channels_density[sigopt_experiment_idx]
exp_required_sparsity = 1.0 - exp_required_density
sigopt_dimensionality_study_args = deepcopy(model_args)
sigopt_dimensionality_study_args.update(percent_on=1.0)
sigopt_dimensionality_study_args.update(num_channels=exp_num_channels)
sigopt_dimensionality_study_args.update(sparsity=exp_required_sparsity)
ONE_CYCLE_LR_DIMENSIONALITY_SIGOPT = deepcopy(STATIC_SPARSE_SMALL_DIMENSIONALITY_STUDY)
ONE_CYCLE_LR_DIMENSIONALITY_SIGOPT.update(
    dict(
        sigopt_experiment_class=OneCycleLRGreedyInfoMaxSparseSigOpt,
        wandb_args=dict(
            project="greedy_infomax-static-sparsity",
            name=f"constant_nonzero_params_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse_sigopt",
            group=f"constant_nonzero_params_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse_sigopt",
        ),
        model_args=sigopt_dimensionality_study_args,
        sigopt_config=dict(
            name=f"sigopt_GIM_sparse_onecycle_{exp_num_channels}_"
            f"{str(exp_required_sparsity)[:4]}",
            parameters=[
                dict(
                    name="max_lr",
                    type="double",
                    bounds=dict(min=0.01, max=1.0),
                    transformation="log",
                ),
                dict(
                    name="div_factor",
                    type="double",
                    bounds=dict(min=5, max=200),
                    transformation="log",
                ),
                dict(
                    name="final_div_factor",
                    type="double",
                    bounds=dict(min=200, max=2000),
                    transformation="log",
                ),
            ],
            metrics=[dict(name="mean_accuracy", objective="maximize")],
            parallel_bandwidth=3,
            observation_budget=40,
            project="greedy_infomax",
        ),
        sigopt_experiment_id=409217,
        api_key=os.environ.get("SIGOPT_KEY", None),
        batch_size=32,
        batch_size_supervised=32,
        val_batch_size=32,
        # batch_size=16,
        # batch_size_supervised=16,
        # val_batch_size=16,
        lr_scheduler_args=dict(
            max_lr=1e-1,  # change based on sparsity/dimensionality
            div_factor=10,  # initial_lr = 0.01
            final_div_factor=300,  # min_lr = 0.0000025
            pct_start=2.0 / 10.0,
            epochs=10,
            anneal_strategy="linear",
            max_momentum=1e-4,
            cycle_momentum=False,
        ),
    )
)


# using LR range test to optimize LR
class GreedyInfoMaxExperimentSparseLRRangeTest(
    mixins.LRRangeTest,
    mixins.LogEveryLoss,
    mixins.LogEveryLearningRate,
    mixins.RezeroWeights,
    mixins.LogBackpropStructure,
    mixins.NoiseRobustnessTest,
    experiments.SelfSupervisedExperiment,
):
    # avoid changing key names for sigopt
    @classmethod
    def get_readable_result(cls, result):
        return result

    def setup_experiment(self, config):
        num_channels = config["model_args"]["num_channels"]
        config["classifier_config"]["model_args"].update(in_channels=num_channels)
        super().setup_experiment(config)


#
# channels_density = [
# (64, 1.0),
# (45, 0.503),
# (58, 0.301),
# (72, 0.194),
# (100, 0.0993),
# ]

range_test_experiment_idx = 0
exp_num_channels, exp_required_density = channels_density[range_test_experiment_idx]
exp_required_sparsity = 1.0 - exp_required_density
lr_range_test_args = deepcopy(model_args)
lr_range_test_args.update(percent_on=1.0)
lr_range_test_args.update(num_channels=exp_num_channels)
lr_range_test_args.update(sparsity=exp_required_sparsity)
LR_RANGE_TEST = deepcopy(STATIC_SPARSE_WEIGHTS_SMALL)
LR_RANGE_TEST.update(
    dict(
        experiment_class=GreedyInfoMaxExperimentSparseLRRangeTest,
        wandb_args=dict(
            project="greedy_infomax-static-sparsity-hyperparameters",
            name=f"lr_range_test_constant_nonzero_params"
            f"_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse_range_test",
            group=f"lr_range_test_constant_nonzero_params"
            f"_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse_range_test",
        ),
        batch_size=32,
        batch_size_supervised=32,
        val_batch_size=32,
        epochs=3,
        # batch_size=16,
        # batch_size_supervised=16,
        # val_batch_size=16,
        model_class=WrappedSparseSmallVisionModel,
        model_args=lr_range_test_args,
        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=2e-4),
        lr_scheduler_class=dict(),
        lr_scheduler_args=dict(min_lr=2e-4, max_lr=2),
        classifier_config=dict(
            model_class=Classifier,
            model_args=dict(num_classes=10),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
            lr_scheduler_class=dict(),
            lr_scheduler_args=dict(min_lr=2e-4, max_lr=2e-4),
            epochs=3,
        ),
    )
)

#
# channels_density = [
# (32, 1.0),
# (45, 0.503),
# (58, 0.301),
# (72, 0.194),
# (100, 0.0993),
# ]
max_lr_grid_search_experiment_idx = 2
exp_num_channels, exp_required_density = channels_density[
    max_lr_grid_search_experiment_idx
]
exp_required_sparsity = 1.0 - exp_required_density
max_lr_grid_search_args = deepcopy(model_args)
max_lr_grid_search_args.update(percent_on=1.0)
max_lr_grid_search_args.update(num_channels=exp_num_channels)
max_lr_grid_search_args.update(sparsity=exp_required_sparsity)
MAX_LR_GRID_SEARCH = deepcopy(STATIC_SPARSE_WEIGHTS_SMALL)
MAX_LR_GRID_SEARCH.update(
    dict(
        experiment_class=GreedyInfoMaxExperimentSparse2,
        wandb_args=dict(
            project="greedy_infomax-static-sparsity-hyperparameters",
            name=f"max_lr_grid_search_constant_nonzero_params"
            f"_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse",
            group=f"constant_nonzero_params_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse",
        ),
        batch_size=32,
        batch_size_supervised=32,
        val_batch_size=32,
        # batch_size=16,
        # batch_size_supervised=16,
        # val_batch_size=16,
        model_class=WrappedSparseSmallVisionModel,
        model_args=max_lr_grid_search_args,
        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=2e-4),
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_args=dict(
            # max_lr=tune.grid_search([0.15, 0.18, 0.21, 0.24, 0.27]),
            max_lr=tune.grid_search([0.19, 0.2, 0.21, 0.22, 0.213]),
            div_factor=100,  # initial_lr = 0.01
            final_div_factor=1000,  # min_lr = 0.0000025
            pct_start=1.0 / 10.0,
            epochs=10,
            anneal_strategy="linear",
            max_momentum=1e-4,
            cycle_momentum=False,
        ),
        classifier_config=dict(
            model_class=Classifier,
            model_args=dict(num_classes=10),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)


# Super Greedy Implementation (all blocks trained concurrently)

experiment_idx = 0
exp_num_channels, exp_required_density = channels_density[experiment_idx]
exp_required_sparsity = 1.0 - exp_required_density
super_greedy_model_args = deepcopy(model_args)
super_greedy_model_args.update(percent_on=1.0)
super_greedy_model_args.update(num_channels=exp_num_channels)
super_greedy_model_args.update(sparsity=exp_required_sparsity)
SUPER_GREEDY_BASE = deepcopy(STATIC_SPARSE_WEIGHTS_SMALL)
SUPER_GREEDY_BASE.update(
    dict(
        experiment_class=GreedyInfoMaxExperimentSparse2,
        wandb_args=dict(
            project="greedy_infomax-super-greedy",
            name=f"constant_nonzero_params_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse",
            group=f"constant_nonzero_params_{str(exp_num_channels)}_channels_"
            f"{str(exp_required_sparsity)[:4]}_sparse",
        ),
        batch_size=32,
        batch_size_supervised=32,
        val_batch_size=32,
        # batch_size=16,
        # batch_size_supervised=16,
        # val_batch_size=16,
        model_class=WrappedSuperGreedySmallSparseVisionModel,
        model_args=super_greedy_model_args,
        optimizer_class=torch.optim.SGD,
        optimizer_args=dict(lr=2e-4),
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_args=dict(
            max_lr=0.24,  # change based on sparsity/dimensionality
            div_factor=100,  # initial_lr = 0.01
            final_div_factor=1000,  # min_lr = 0.0000025
            pct_start=1.0 / 10.0,
            epochs=10,
            anneal_strategy="linear",
            max_momentum=1e-4,
            cycle_momentum=False,
        ),
        classifier_config=dict(
            model_class=Classifier,
            model_args=dict(num_classes=10),
            loss_function=torch.nn.functional.cross_entropy,
            # Classifier Optimizer class. Must inherit from "torch.optim.Optimizer"
            optimizer_class=torch.optim.Adam,
            # Optimizer class class arguments passed to the constructor
            optimizer_args=dict(lr=2e-4),
        ),
    )
)


CONFIGS = dict(
    sparse_vdrop_small=SPARSE_VDROP_SMALL,
    static_sparse_weights_small=STATIC_SPARSE_WEIGHTS_SMALL,
    static_sparse_activations_small=STATIC_SPARSE_ACTIVATIONS_SMALL,
    static_sparse_weights_activations_small=SPARSE_WEIGHTS_ACTIVATIONS_SMALL,
    static_sparse_small_dimensionality_grid_search=SPARSE_SMALL_GRID_SEARCH,
    small_sparse_largest_dimension=SMALL_SPARSE_LARGEST_DIMENSION,
    static_sparse_small_dimensionality_study=STATIC_SPARSE_SMALL_DIMENSIONALITY_STUDY,
    one_cycle_lr_dimensionality_sigopt=ONE_CYCLE_LR_DIMENSIONALITY_SIGOPT,
    lr_range_test=LR_RANGE_TEST,
    max_lr_grid_search=MAX_LR_GRID_SEARCH,
    super_greedy_base=SUPER_GREEDY_BASE,
)
