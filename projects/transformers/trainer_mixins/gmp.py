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

import numpy as np
import wandb
from transformers.modeling_utils import unwrap_model

from nupic.research.frameworks.dynamic_sparse import (
    ThreeStageGMPLR,
    global_prune_by_abs_weight,
)
from nupic.research.frameworks.pytorch.model_utils import (
    calc_model_sparsity,
    count_nonzero_params,
    filter_modules,
)
from nupic.torch.modules.sparse_weights import SparseWeightsBase


class GradualMagnitudePruningMixin:
    """
    This mixin implements Gradual Magnitude Pruning (GMP) as described in the paper
        https://arxiv.org/pdf/1710.01878.pdf
    More accurately, it implements the overview of the method described in the blog post
        https://neuralmagic.com/blog/pruning-gmp/

    To summarize, GMP consists of three stages, typically performed on a pre-trained
    network. These include:
        1) stabilization - allows the network to stabilize before pruning occurs;
                           in this mixin it's called the warmup phase
        2) pruning - when pruning occurs; will take the network from being fully dense
                     to the desired end sparsity; pruning occurs according to a cubic
                     function where more weights are pruned earlier on and less so later
        2) fine-tuning - no pruning occurs; at this point the sparse network is
                         fine-tuned to recover or solidify it's evaluation accuracy;
                         in this mixin it's called the cooldown stage.


    This mixin can be used to prune a pre-trained fully dense model or to perform GMP
    throughout pre-training to start dense and end sparse. For the former, use the
    ThreeStageLRMixin or, for the latter, use whatever LR scheduler makes sense, such as
    the one for OneCycleLR.

    Params to add to 'trainer_mixin_args':
    :param warmup_steps: how many training iterations in the warm-up phase
    :param cooldown_steps: how many training iterations in the cool-down phase
    :param prune_period: how often pruning occurs during the pruning phase
    :param start_sparsity: this can be given just for the user's clarity of
                           documentation in their config, but only a starting sparsity
                           of 0 is allowed; that is, the model must start fully dense
    :param end_sparsity: the desired ending sparsity of the model; given between 0 and 1
    :param verbose_gmp_logging: this will additional pruning info to wandb

    Note: The sparsities (start_sparsity and end_sparsity) indicate the overall sparsity
          of the BERT model. This is in contrast to the sparsity measured among just the
          sparse modules as done with other experiment configs. Specifically, this mixin
          is concerned with the sparsity measured via `calc_model_sparsity(model.bert)`.

    Note: This mixin can or cannot be used with ThreeStageLRMixin. If pruning a fully
          dense, already pre-trained, network, then it makes sense to use the
          three-stage lr mixin. Otherwise, if GMP pruning during pre-training, it may
          make more sense to use OneCycleLR.

    Note: Notice how this mixin requires the same params for warmup_steps and
          cooldown_steps as does ThreeStageLRMixin. This is intentional as the phases
          are meant to align.

    Note: If using `gradient_accumulation_steps` be careful to define warmup_steps and
          cooldown_steps as the overall number of training iterations will be increased.
          For instance, if `gradient_accumulation_steps=2`, the training steps will be
          doubled and so you'll need to double the number of warmup_steps and
          cooldown_steps as well.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        mixin_args = self.args.trainer_mixin_args
        assert "warmup_steps" in mixin_args
        assert "cooldown_steps" in mixin_args
        assert "prune_period" in mixin_args

        # The start sparsity is inferred from the model.
        self.end_sparsity = mixin_args.get("end_sparsity", 0.2)
        self.start_sparsity = mixin_args.get("start_sparsity", 0.2)
        assert self.start_sparsity == 0, "For now, only starting dense is supported."

        # These dictate when during training pruning will occur and how often.
        self.warmup_steps = mixin_args["warmup_steps"]
        self.cooldown_steps = mixin_args["cooldown_steps"]
        self.prune_period = mixin_args["prune_period"]

        # For verbose logging.
        self.verbose_gmp_logging = mixin_args.get("verbose_gmp_logging", False)

        # These get initialized in `setup_pruning`
        self.start_on_params = None
        self.end_on_params = None
        self.total_prune_iterations = None

        self._setup_done = False
        self.max_steps = self.args.max_steps
        self.prune_iteration = 1

    def setup_pruning(self, args, model):
        """
        This infers things about the pruning setup such how many many steps to prune
        and how many params to start and end with. Note that we're concerned with
        pruning the parameters within the sparse modules. Thus, any SparseWeightsBase
        will be involved in global pruning. However, the start and end sparsity will
        be measured over all parameters within `model.bert`.
        """

        # Calculate the number of steps in the pruning phase.
        pruning_steps = self.max_steps - self.warmup_steps - self.cooldown_steps
        assert pruning_steps > 0
        assert pruning_steps % self.prune_period == 0

        # Calculate how many times we'll prune.
        self.total_prune_iterations = pruning_steps / self.prune_period + 1

        # Calculate the params that belong to BERT and how many to start and end with.
        total_params, _ = count_nonzero_params(model.bert)
        model_start_params = total_params * (1 - self.start_sparsity)
        model_end_params = total_params * (1 - self.end_sparsity)

        # Get all the sparse modules in BERT.
        sparse_modules = filter_modules(model.bert, include_modules=[SparseWeightsBase])
        self.sparse_modules = list(sparse_modules.values())

        # Calculate the number of params that belong to the non-sparse modules.
        sparse_module_params = 0
        for m in self.sparse_modules:
            sparse_module_params += m.weight.numel()
        non_sparse_module_params = total_params - sparse_module_params

        # Solve for the number of params that will be on. Specifically for the
        # sparse module parameters.
        self.start_on_params = model_start_params - non_sparse_module_params
        self.end_on_params = model_end_params - non_sparse_module_params

    def training_step(self, model, inputs):
        """
        Prune every `prune_period` steps during the pruning phase.
        """

        model = unwrap_model(model)  # extract from DistributedDataParallel
        train_loss = super().training_step(model, inputs)

        if not self._setup_done:
            self.setup_pruning(self.args, model)
            self._setup_done = True

        # Track the steps by starting at `step=1`
        train_step = self.state.global_step + 1

        # Return if still in warm-up phase.
        if train_step < self.warmup_steps:
            return train_loss

        # Return if in cool-down phase.
        if train_step > self.max_steps - self.cooldown_steps:
            return train_loss

        # Return if step within pruning phase isn't divisible by pruning period.
        if not (train_step - self.warmup_steps) % self.prune_period == 0:
            return train_loss

        # Fraction of the way through the pruning phase.
        fraction_through_pruning = self.prune_iteration / self.total_prune_iterations

        # Target number of on-params at time t.
        lambda_t = np.power(1 - fraction_through_pruning, 3)  # varies from 0 to 1
        target_on_params = self.end_on_params + (
            self.start_on_params - self.end_on_params) * lambda_t

        # Actual number of on-params at time t.
        current_on_params = calc_on_params(self.sparse_modules)
        remove_params = current_on_params - int(target_on_params)

        # Prune the specified number of params.
        actual_removed = global_prune_by_abs_weight(
            self.sparse_modules,
            num_remove=remove_params
        )

        # Log how much was pruned and resulting sparsity.
        if self.verbose_gmp_logging:

            bert_sparsity = calc_model_sparsity(model.bert)
            logs = dict({
                "gmp/target_pruned_params": remove_params,
                "gmp/actual_pruned_params": actual_removed,
                "gmp/post_prune_bert_sparsity": bert_sparsity,
            })

        if wandb.run is not None:
            wandb.log(logs, commit=False)
            self.control.should_log = True

        self.prune_iteration += 1
        return train_loss


class ThreeStageLRMixin:
    """
    This mixin overrides the lr scheduler to use the ThreeStageGMPLR which consists of
    three stages:
        1) warmup - linearly ramp up from 0 to max_lr
        2) pruning - maintain a constant max_lr
        3) cooldown - decay the learning rate twice (like a StepLR)

    This LR is used for GMP pruning on already pre-trained fully dense network. In
    contrast, one would most likely not want to use this mixin for GMP style pruning
    during pre-training. Instead, it may be better to use OneCycle LR to achieve a
    better eval-loss.

    Params to add to 'trainer_mixin_args':
    :param warmup_steps: number of warmup steps (i.e. the the stabilization phase)
    :param cooldown_steps: number of cooldown steps
    :param max_lr: the lr held constant throughout the pruning phase
    :param cooldown_gamma: (optional) how much to decay the lr during the cooldown phase
                           e.g. lr <- lr * cooldown_gamma; used to decay the lr twice;
                           defaults to 0.1

    Note: This mixin can't be used with the OneCycleLRMixin as they both override
          `create_scheduler`.

    Note: Notice how this mixin requires the same params for warmup_steps and
          cooldown_steps as does GradualMagnitudePruningMixin. This is intentional as
          the phases are meant to align
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        mixin_args = self.args.trainer_mixin_args
        assert "warmup_steps" in mixin_args
        assert "cooldown_steps" in mixin_args
        assert "max_lr" in mixin_args

        self.max_steps = self.args.max_steps
        self.warmup_steps = mixin_args["warmup_steps"]
        self.cooldown_steps = mixin_args["cooldown_steps"]
        self.max_lr = mixin_args["max_lr"]
        self.cooldown_gamma = mixin_args.get("cooldown_gamma", 0.1)

        # The LambdaLR (parent class of ThreeStageGMPLR) will multiply this base lr of
        # 1 times the one at the given step.
        kwargs["args"].learning_rate = 1

    def create_scheduler(self, num_training_steps: int):

        self.lr_scheduler = ThreeStageGMPLR(
            self.optimizer,
            max_lr=self.max_lr,
            total_steps=self.max_steps,
            warmup_steps=self.warmup_steps,
            cooldown_steps=self.cooldown_steps,
            cooldown_gamma=self.cooldown_gamma,
        )


def calc_on_params(sparse_modules):
    """
    Calculate the total number of on-params throughout the sparse modules.
    """
    on_params = 0
    for m in sparse_modules:
        on_mask = ~m.zero_mask.bool()
        on_params += on_mask.sum().item()
    return on_params
