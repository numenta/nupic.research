#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

"""
Run a simple GSC experiment using variational dropout modules.
"""

from copy import deepcopy

import torch
import numpy as np

from .base import DEFAULT_BASE
from nupic.research.frameworks.backprop_structure.networks import VDropLeNet, \
    gsc_lenet_vdrop_sparse, gsc_lenet_vdrop_super_sparse
from nupic.research.frameworks.vernon.distributed import experiments, mixins
from nupic.research.frameworks.sigopt.sigopt_experiment import SigOptExperiment
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params


class GSCVDropExperiment(mixins.RegularizeLoss,
                        mixins.ConstrainParameters,
                        mixins.LogBackpropStructure,
                        mixins.LogEveryLoss,
                        mixins.LogEveryLearningRate,
                        experiments.SupervisedExperiment):

    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.warmup_start_iteration = config.get('warmup_start_iteration', 50)
        self.warmup_end_iteration = config.get('warmup_end_iteration', 80)
        self.gamma_warmup = config.get('gamma_warmup', 0.5)
        self.gamma_postwarmup = config.get('gamma_postwarmup', 0.9)

    def pre_epoch(self):
        super().pre_epoch()
        if self.current_epoch == self.warmup_start_iteration:
            self.lr_scheduler.gamma = self.gamma_warmup
        elif self.current_epoch == self.warmup_end_iteration:
            self.lr_scheduler.gamma = self.gamma_postwarmup


REG_FACTOR_START = np.exp(-3.8539378170655567)
REG_FACTOR_START_EPOCH = 50
REG_FACTOR_END = np.exp(-2.2155542642611037)
REG_FACTOR_END_EPOCH = 80
LR = 0.0007980119944707181




def get_reg_scalar(epoch, batch_idx, total_batches):
    # Set to reg_factor_start before reg_factor_start_epoch, then scale linearly
    # until reaching reg_factor_end at reg_factor_end_epoch, then hold constant
    if epoch < REG_FACTOR_START_EPOCH:
        return REG_FACTOR_START
    elif epoch > REG_FACTOR_END_EPOCH:
        return REG_FACTOR_END
    return (epoch - REG_FACTOR_START_EPOCH) / (REG_FACTOR_END_EPOCH -
                                               REG_FACTOR_START_EPOCH)  * (
        REG_FACTOR_END - REG_FACTOR_START) + REG_FACTOR_START



GSC_VDROP = deepcopy(DEFAULT_BASE)
GSC_VDROP.update(
    experiment_class = GSCVDropExperiment,
    # Training batch size
    batch_size=32,
    # Validation batch size
    val_batch_size=32,
    reg_scalar = get_reg_scalar,
    gamma_warmup = 0.72,
    gamma_postwarmup = 0.568,
    warmup_start_iteration = 50,
    warmup_end_iteration = 60,
    wandb_args = dict(
        name="gsc_vdrop_2",
        project="gsc_vdrop_experiments",
        notes="This is a cool experiment",
    ),

    lr_scheduler_class=torch.optim.lr_scheduler.StepLR,
    lr_scheduler_args=dict(
        step_size=1,
        gamma=1.0,
    ),
    optimizer_args=dict(
        lr=LR,
    ),
    num_epochs= 100,
    epochs_to_validate = range(100),
    model_class = VDropLeNet,
    downscale_reg_with_training_set = True,
    model_args = dict(),
)


class SNRPruningGSCVDrop(

        mixins.LogEveryLoss,
        mixins.LogBackpropStructure,
        mixins.LogEveryLearningRate,
        mixins.ExtraValidationsPerEpoch,

        mixins.RegularizeLoss,
        mixins.ConstrainParameters,
        mixins.MultiCycleLR,
        mixins.PruneLowSNR,

        experiments.SupervisedExperiment,):
    pass

def make_reg_schedule(epochs, pct_ramp_start, pct_ramp_end, peak_value,
                      pct_drop, final_value):
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

SUBSEQUENT_LR_SCHED_ARGS = dict(
    max_lr=6.0,
    pct_start=0.0625,
    anneal_strategy="linear",
    base_momentum=0.6,
    max_momentum=0.75,
    cycle_momentum=True,
    final_div_factor=1000.0
)

NUM_EPOCHS = 120


#94% accuracy, ~170k params = 10% sparsity
GSC_VDROP_SNR_PRUNING = deepcopy(GSC_VDROP)
GSC_VDROP_SNR_PRUNING.update(dict(
    name="GSC_VDROP_INITIAL_MULTICYCLE",
    experiment_class = SNRPruningGSCVDrop,
    epochs_to_validate=range(1, NUM_EPOCHS),
    epochs=NUM_EPOCHS,
    wandb_args=dict(
        project="gsc-snr-pruning",
        name="3rd-run-log-sparsity",
        notes="just a demo to see if the multi-cycle LR w/ pruning is working"
    ),
    log_timestep_freq=5,
    validate_on_prune=True,

    model_class = gsc_lenet_vdrop_sparse,

    multi_cycle_lr_args=(
        (0, dict(max_lr=6.0,
                 pct_start=0.2,
                 anneal_strategy="linear",
                 base_momentum=0.6,
                 max_momentum=0.75,
                 cycle_momentum=True,
                 div_factor=6.0,
                 final_div_factor=1000.0)),
        (30, SUBSEQUENT_LR_SCHED_ARGS),
        (33, SUBSEQUENT_LR_SCHED_ARGS),
        (36, SUBSEQUENT_LR_SCHED_ARGS),
        (39, SUBSEQUENT_LR_SCHED_ARGS),
        (42, SUBSEQUENT_LR_SCHED_ARGS),
        (45, SUBSEQUENT_LR_SCHED_ARGS),
        (80, SUBSEQUENT_LR_SCHED_ARGS),
    ),

    reg_scalar=make_reg_schedule(
        epochs=NUM_EPOCHS,
        pct_ramp_start=10 / 120,
        pct_ramp_end=30 / 60,
        peak_value=0.01,
        pct_drop=45 / 60,
        final_value=0.0005,
    ),

    prune_schedule=[
        (30, 1 / 6),
        (33, 2 / 6),
        (36, 3 / 6),
        (39, 4 / 6),
        (42, 5 / 6),
        (45, 6 / 6),
    ],
))



GSC_VDROP_SNR_PRUNING_SUPER_SPARSE = deepcopy(GSC_VDROP_SNR_PRUNING)
GSC_VDROP_SNR_PRUNING_SUPER_SPARSE.update(dict(
    name="GSC_VDROP_SNR_PRUNING_SUPER_SPARSE",
    wandb_args=dict(
        project="gsc-snr-pruning-sparse",
        name="very_sparse",
        notes="how sparse can it get?"
    ),
    model_class = gsc_lenet_vdrop_super_sparse,

    multi_cycle_lr_args=(
        (0, dict(max_lr=6.0,
                 pct_start=0.2,
                 anneal_strategy="linear",
                 base_momentum=0.6,
                 max_momentum=0.75,
                 cycle_momentum=True,
                 div_factor=6.0,
                 final_div_factor=1000.0)),
        (30, SUBSEQUENT_LR_SCHED_ARGS),
        (33, SUBSEQUENT_LR_SCHED_ARGS),
        (36, SUBSEQUENT_LR_SCHED_ARGS),
        (39, SUBSEQUENT_LR_SCHED_ARGS),
        (42, SUBSEQUENT_LR_SCHED_ARGS),
        (46, SUBSEQUENT_LR_SCHED_ARGS),
        (50, SUBSEQUENT_LR_SCHED_ARGS),
        (54, SUBSEQUENT_LR_SCHED_ARGS),
        (80, SUBSEQUENT_LR_SCHED_ARGS),
    ),
    reg_scalar=make_reg_schedule(
        epochs=NUM_EPOCHS,
        pct_ramp_start=10 / 120,
        pct_ramp_end=30 / 60,
        peak_value=0.01,
        pct_drop=54/120,
        final_value=0.0005,
    ),
    prune_schedule=[
        (30, 1 / 8),
        (33, 2 / 8),
        (36, 3 / 8),
        (39, 4 / 8),
        (42, 5 / 8),
        (46, 6 / 8),
        (50, 7 / 8),
        (54, 8 / 8),
    ],
))





class SNRPruningGSCVDropSIGOPT(
        mixins.LogEveryLoss,
        mixins.LogBackpropStructure,
        mixins.LogEveryLearningRate,
        mixins.ExtraValidationsPerEpoch,

        mixins.RegularizeLoss,
        mixins.ConstrainParameters,
        mixins.MultiCycleLR,
        mixins.PruneLowSNR,
        SigOptExperiment):

    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the optimizer_args with SGD optimizer params.

        :param config:
            - multi_cycle_lr_args
            - reg_scalar
            - prune_schedule
            - num_epochs
        :param suggestion:
            - assignments (all optional)
                - num_pruning_iterations
                - epochs_per_pruning_cycle
                - max_lr
                - reg_scalar_max_value
                - reg_scalar_min_value
                - max_momentum
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments

        assert "multi_cycle_lr_args" in config
        assert "reg_scalar" in config

        multi_cycle_lr_args = config["multi_cycle_lr_args"]
        prune_schedule = config["prune_schedule"]
        reg_scalar = config["reg_scalar"]
        num_epochs = config["num_epochs"]

        # Optimizer args
        num_pruning_iterations = assignments.get("num_pruning_iterations", 5)
        epochs_per_pruning_cycle = assignments.get("epochs_per_pruning_cycle", 4)
        max_lr = assignments.get("max_lr", 1.5)
        reg_scalar_max_value = assignments.get("reg_scalar_max_value", 0.1)
        reg_scalar_final_value = assignments.get("reg_scalar_final_value", 0.001)
        momentum_max_value = assignments.get("momentum_max_value", 0.75)

        CYCLE_LR_ARGS = dict(
            max_lr=max_lr,
            pct_start=0.0625,
            anneal_strategy="linear",
            base_momentum=0.6,
            max_momentum=momentum_max_value,
            cycle_momentum=True,
            final_div_factor=1000.0
        )
        multi_cycle_lr_args = ((0, dict(max_lr=6.0,
                                        pct_start=0.2,
                                        anneal_strategy="linear",
                                        base_momentum=0.6,
                                        max_momentum=0.75,
                                        cycle_momentum=True,
                                        div_factor=6.0,
                                        final_div_factor=1000.0)),
                               ) + tuple([(30 + prune_iteration *
                                           epochs_per_pruning_cycle, CYCLE_LR_ARGS)
                                          for prune_iteration in range(
                num_pruning_iterations)]) + tuple((30 + num_pruning_iterations *
                                                   epochs_per_pruning_cycle,
                                                   CYCLE_LR_ARGS))
        reg_scalar = make_reg_schedule(
            epochs=NUM_EPOCHS,
            pct_ramp_start=10 / num_epochs,
            pct_ramp_end=30 / num_epochs,
            peak_value=reg_scalar_max_value,
            pct_drop=(
                                 30 + num_pruning_iterations * epochs_per_pruning_cycle) / num_epochs,
            final_value=reg_scalar_final_value,
        )

        prune_schedule = tuple([(30 + prune_iteration * epochs_per_pruning_cycle,
                                 prune_iteration / num_pruning_iterations) for
                                prune_iteration in range(num_pruning_iterations)])

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["update_config_with_suggestion"].append(
            "SNRPruningParams.update_config_with_suggestion"
        )
        return eo


# Initial ~30 epochs one large cycle, followed by pruning iterations
# how many pruning iterations, and how long is each iteration (num epochs/pruning cycle)
    # how should the learning rate and momentum change? (how high LR should go)
# how high should the regularization warmup go?

SIGOPT_GSC_VDROP_SNR_PRUNING = deepcopy(GSC_VDROP_SNR_PRUNING)
SIGOPT_GSC_VDROP_SNR_PRUNING.update(
    sigopt_experiment_class=SNRPruningGSCVDropSIGOPT,
    sigopt_config=dict(
        name="sigopt_vdrop_gsc_snr_pruning",
        parameters=[
            dict(name="num_pruning_iterations", type="int", bounds=dict(min=3,
                                                                     max=10)),
            dict(name="epochs_per_pruning_cycle", type="int", bounds=dict(min=2,
                                                                        max=8)),
            dict(name="max_lr", type="double", bounds=dict(min=0.001, max=1.5),
                    transformation="log"),
            dict(name="reg_scalar_max_value", type="double", bounds=dict(min=0.01,
                                                                         max=0.3),
                 transformation="log"),
            dict(name="reg_scalar_final_value", type="double", dounds=dict(min=0.0001,
                                                                         max=0.005)),
            dict(name="momentum_max_value", type="double", dounds=dict(min=0.6,
                                                                         max=0.9))
        ],
        metrics=[dict(name="mean_accuracy", objective="maximize")],
        parallel_bandwidth=1,
        observation_budget=10,
        project="gsc_snr_pruning",
    ),

    sigopt_experiment_id=None,

)










CONFIGS = dict(
    gsc_vdrop=GSC_VDROP,
    gsc_vdrop_snr_pruning = GSC_VDROP_SNR_PRUNING,
    gsc_vdrop_snr_pruning_super_sparse = GSC_VDROP_SNR_PRUNING_SUPER_SPARSE,
    sigopt_gsc_vdrop_snr_pruning = SIGOPT_GSC_VDROP_SNR_PRUNING
)

