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

import logging

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from nupic.research.frameworks.pytorch.model_utils import (
    filter_modules,
    set_module_attr,
)

try:
    import deepspeed
except ImportError:
    # Fails later if using deepspeed configuration without installing deepspeed
    pass


class DistillationTrainerMixin:
    """
    Mixin to HF Trainer.
    Initialize required arguments for Knowledge Distillation.
    Replace compute_loss with modified function.
    Replaces label_smoother with KDLoss (will override regular label_smoother).

    To deactivate distillation, call trainer.deactivate_distillation() and will
    fall back to regular loss computation.
    """

    def __init__(self, *args, **kwargs):
        """
        Add following variables under 'trainer_mixin_args' through the training
        arguments.

        :param teacher_model_names_or_paths: List of pretrained model names or paths to
                                             use as teachers in knowledge distillation.
        :param teacher_models_cache_dir: (optional) directory to load and save
                                         pre-trained teacher models
        :param kd_ensemble_weights: List of weights to apply to each teacher model
                                during distillation.
                                If the total is > 1 the loss will be scaled out
                                of proportion, acting in practice as a scaling factor
                                to the learning rate (the equivalence is true
                                in the composite loss model, and only approximate
                                for the regular distillation model. Scaling the
                                softmax out of proportion creates a target that
                                is impossible to reach, since the output distribution
                                can only sum to 1)
        :param kd_factor_init: Determines the percentage of the target that comes
                            from the teacher model. Value should be float
                            between 0 and 1. Defaults to 1.
        :param kd_factor_end: KD factor at last epoch. Will calculate linear decay
                            based on initial kd_factor_init and kd_factor_end.
                            Value should be float between 0 and 1.
                            If None, no decay is applied. Defaults to None.
        :param kd_temperature_init: Determines the temperature T applied to softmax.
                                If T > 1, it smoothes the softmax distribution.
                                If T < 1, it sharpens the distribution (more mass to
                                few points). If kd_temperature_end is also defined,
                                this variable equals the temperature at the beginning
                                of training. Defaults to 1.0
        :param kd_temperature_end: Determines the temperature applied to softmax.
                                Will calculate linear decay based on
                                kd_temperature_init and kd_temperature_end.
                                If None, no decay is applied. Defaults to None.
        """

        super().__init__(*args, **kwargs)

        mixin_args = self.args.trainer_mixin_args

        teacher_names_or_paths = mixin_args.get("teacher_model_names_or_paths", None)
        teacher_models_cache_dir = mixin_args.get("teacher_models_cache_dir", None)
        kd_ensemble_weights = mixin_args.get("kd_ensemble_weights", None)
        kd_factor_init = mixin_args.get("kd_factor_init", 1.0)
        kd_factor_end = mixin_args.get("kd_factor_end", 1.0)
        kd_temperature_init = mixin_args.get("kd_temperature_init", 1.0)
        kd_temperature_end = mixin_args.get("kd_temperature_end", 1.0)

        # Validate teacher models
        assert (
            isinstance(teacher_names_or_paths, list) and len(teacher_names_or_paths) > 0
        ), "When using KD mixin, teacher_model_names_or_paths must be defined"

        seq_length = get_model_seq_length(self.model)
        teacher_models = []
        for model_name_or_path in teacher_names_or_paths:
            teacher_model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path,
                cache_dir=teacher_models_cache_dir
            )
            if self.args.fp16:
                teacher_model.half()
            teacher_model.resize_token_embeddings(len(self.tokenizer))
            teacher_model = resize_position_embeddings(teacher_model, seq_length)
            teacher_model = teacher_model.eval().to(self.args.device)

            # Use deepspeed inference mode on teacher models
            if self.args.deepspeed:
                ds_engine = deepspeed.init_inference(
                    teacher_model, dtype=torch.half, replace_method="auto")
                teacher_model = ds_engine.module

            teacher_models.append(teacher_model)

        if len(teacher_models) == 1:
            logging.info(f"KD single teacher class: {teacher_models.__class__}")
        else:
            logging.info(f"KD teacher is ensemble of {len(teacher_models)} models")

        self.teacher_models = teacher_models

        # Validate knowledge Distillation factor
        assert 0 <= kd_factor_init <= 1, "kd_factor_init should be >= 0 and <= 1"
        assert 0 <= kd_factor_end <= 1, "kd_factor_end should be >= 0 and <= 1"
        logging.info(f"KD factor: {kd_factor_init} {kd_factor_end}")

        # Validate Knowledge softmax temperature factor
        logging.info(
            f"KD softmax temperature: {kd_temperature_init} {kd_temperature_end}"
        )

        # Validate ensemble weighting
        num_models = len(teacher_models)
        if kd_ensemble_weights is None:
            kd_ensemble_weights = [1.0 / num_models for _ in range(num_models)]
        else:
            assert (
                len(kd_ensemble_weights) == num_models
            ), "Number of ensemble weights should match number of teacher models"
        logging.info(f"Ensemble weights: {kd_ensemble_weights}")

        # Initialize KD as a label smoother
        self.label_smoother = KDLoss(
            num_classes=list(self.model.parameters())[-1].size()[0],
            kd_ensemble_weights=kd_ensemble_weights,
            kd_factor_init=kd_factor_init,
            kd_factor_end=kd_factor_end,
            kd_temperature_init=kd_temperature_init,
            kd_temperature_end=kd_temperature_end,
        )

    def activate_distillation(self):
        self.kd_active = True

    def deactivate_distillation(self):
        self.kd_active = False

    def train(self, *args, **kwargs):
        self.activate_distillation()
        return super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.deactivate_distillation()
        return super().evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self.deactivate_distillation()
        return super().predict(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override original compute_loss.
        KDLoss is used as a label smoother.
        Defaults to standard loss calculation if kd_active is False

        Original: how the loss is computed by Trainer. By default, all models return
        the loss in the first element.
        """
        if self.kd_active and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # main model output
        outputs = model(**inputs)

        # Save past state if it exists (block kept from original function)
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.kd_active and labels is not None:
            # distillation outputs
            with torch.no_grad():
                teacher_outputs = [teacher(**inputs) for teacher in self.teacher_models]
            loss = self.label_smoother(
                model_output=outputs,
                teacher_outputs=teacher_outputs,
                labels=labels,
                trainer_state=self.state
            )
        else:
            # don't use .loss here since model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


class KDLoss:
    """
    Calculates knowledge distillation loss. To be used in place of label smoother

    :param num_classes: Number of units in the output layer.
    :param kd_ensemble_weights: List of weights to apply to each teacher model
                                during distillation.
                                If the total is > 1 the loss will be scaled out
                                of proportion, acting in practice as a scaling factor
                                to the learning rate (the equivalence is true
                                in the composite loss model, and only approximate
                                for the regular distillation model. Scaling the
                                softmax out of proportion creates a target that
                                is impossible to reach, since the output distribution
                                can only sum to 1)
    :param kd_factor_init: Determines the percentage of the target that comes
                           from the teacher model. Value should be float
                           between 0 and 1. Defaults to 1.
    :param kd_factor_end: KD factor at last epoch. Will calculate linear decay
                          based on initial kd_factor_init and kd_factor_end.
                          Value should be float between 0 and 1.
                          If None, no decay is applied. Defaults to None.
    :param kd_temperature_init: Determines the temperature T applied to softmax.
                                If T > 1, it smoothes the softmax distribution.
                                If T < 1, it sharpens the distribution (more mass to
                                few points). If kd_temperature_end is also defined,
                                this variable equals the temperature at the beginning
                                of training. Defaults to 1.0
    :param kd_temperature_end: Determines the temperature applied to softmax.
                               Will calculate linear decay based on
                               kd_temperature_init and kd_temperature_end.
                               If None, no decay is applied. Defaults to None.
    :param verbose: Whether or not to log information on temperature and factor decay
    """

    def __init__(
        self,
        num_classes,
        kd_ensemble_weights,
        kd_factor_init,
        kd_factor_end,
        kd_temperature_init,
        kd_temperature_end,
        verbose=True,
    ):
        self.num_classes = num_classes
        self.kd_ensemble_weights = kd_ensemble_weights
        self.kd_factor_init = kd_factor_init
        self.kd_factor_end = kd_factor_end
        self.kd_temperature_init = kd_temperature_init
        self.kd_temperature_end = kd_temperature_end
        self.verbose = verbose
        self.loss_fn = masked_soft_cross_entropy
        self.ignore_index = -100

    def compute_kd_factor(self, trainer_state):
        """Calculates kd factor based on a linear decay"""
        kd_factor = calculate_linear_decay(
            first_step_value=self.kd_factor_init,
            last_step_value=self.kd_factor_end,
            current_step=trainer_state.global_step,
            total_steps=trainer_state.max_steps,
        )
        if self.verbose and (
            (trainer_state.global_step / trainer_state.max_steps) % 0.1 == 0
        ):
            logging.debug(
                f"KD factor: {kd_factor:.3f} "
                f"@ step {trainer_state.global_step}"
            )
        return kd_factor

    def compute_kd_temperature(self, trainer_state):
        """Calculates softmax temperature based on a linear decay"""
        kd_temperature = calculate_linear_decay(
            first_step_value=self.kd_temperature_init,
            last_step_value=self.kd_temperature_end,
            current_step=trainer_state.global_step,
            total_steps=trainer_state.max_steps,
        )
        if self.verbose and (
            (trainer_state.global_step / trainer_state.max_steps) % 0.1 == 0
        ):
            logging.debug(
                f"KD temperature: {kd_temperature:.3f} "
                f"@ step{trainer_state.global_step}"
            )

        return kd_temperature

    def __call__(self, model_output, teacher_outputs, labels, trainer_state):

        # Compute KD factor and temperature according to current step
        kd_factor = self.compute_kd_factor(trainer_state)
        kd_temperature = self.compute_kd_temperature(trainer_state)

        def get_logits(output):
            return output["logits"] if isinstance(output, dict) else output[0]

        student_logits = get_logits(model_output)

        with torch.no_grad():
            # If ensemble, linearly combine outputs of softmax
            softmax_output_teacher = None
            for w_factor, t_output in zip(self.kd_ensemble_weights, teacher_outputs):
                teacher_logits = get_logits(t_output)
                if softmax_output_teacher is None:
                    softmax_output_teacher = \
                        F.softmax(teacher_logits / kd_temperature, dim=-1) * w_factor
                else:
                    softmax_output_teacher += (
                        F.softmax(teacher_logits / kd_temperature, dim=-1) * w_factor
                    )

            # Replace labels according to kd_factor
            # if conditional avoids unnecessary computation when kd_factor is 1
            # TODO: fix bug in output combination
            if kd_factor < 1:
                # target is linear combination of teacher and target softmaxes
                ohe_target = F.one_hot(labels, num_classes=self.num_classes)
                kd_target = (
                    kd_factor * softmax_output_teacher + (1 - kd_factor) * ohe_target
                )
            else:
                kd_target = softmax_output_teacher

        # calculate padding mask
        if labels.dim() == kd_target.dim() - 1:
            labels = labels.unsqueeze(-1)
        padding_mask = labels.eq(self.ignore_index)

        loss = self.loss_fn(
            output=student_logits,
            target=kd_target,
            padding_mask=padding_mask
        )

        return loss


def soft_cross_entropy(output, target, reduction="mean"):
    """ Cross entropy that accepts soft targets
    Args:
    :param output: predictions for neural network
    :param targets: targets, can be soft
    :param size_average: if false, sum is returned instead of mean

    Examples::

        output = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        output = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(output, target)
        loss.backward()

    see: https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/5
    """
    if reduction == "mean":
        return torch.mean(torch.sum(-target * F.log_softmax(output, dim=1), dim=1))
    elif reduction == "sum":
        return torch.sum(torch.sum(-target * F.log_softmax(output, dim=1), dim=1))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def masked_soft_cross_entropy(output, target, padding_mask):
    """
    Soft cross entropy which accepts a padding mask. Loss is scaled to take into
    account the actual number of elements used to calculate the loss.

    Args:
    :param output: predictions for neural network
    :param targets: targets, can be soft
    :param padding_mask: mask used to pad loss functions for tokens which are not
                         part of the label for that batch
    """
    loss_vec = -target * F.log_softmax(output, dim=-1)
    loss_vec.masked_fill_(padding_mask, 0.0)
    loss = loss_vec.sum()
    num_active_elements = padding_mask.numel() - padding_mask.sum()
    return loss / num_active_elements


def calculate_linear_decay(
    first_step_value, last_step_value, current_step, total_steps
):
    """
    Calculates value for a current step in a linear decay.

    :param first_step_value: Value at first step (before training).
    :param last_step_value: Value at last step (after training).
    :param current_step: Current step. Assumes first step is 0.
    :param total_steps: Total number of steps in training.
    """
    step_size = (first_step_value - last_step_value) / (total_steps - 1)
    return first_step_value - step_size * current_step


def resize_position_embeddings(model, new_seq_length):
    """
    Resizes model's position embeddings matrices if the size of max position embedding
    doesn't match new sequence length.
    (size of position embedding equals size of the attention window)

    :param new_seq_length: Tokenizer sequence length.
    """

    position_embeddings = filter_modules(
        model, include_patterns=[".*position_embeddings.*"]
    )
    for module_name, module in position_embeddings.items():
        original_embed_data = module.weight.data
        max_position_embeddings, embed_hidden_size = original_embed_data.size()
        if max_position_embeddings != new_seq_length:
            new_embed = torch.nn.Embedding(new_seq_length, embed_hidden_size).to(
                device=original_embed_data.device, dtype=original_embed_data.dtype
            )
            new_embed.weight.data[:, :] = original_embed_data[:new_seq_length, :]
            set_module_attr(model, module_name, new_embed)

    return model


def get_model_seq_length(model):
    """
    Returns size of seq length according to first found position embedding in model
    """
    for module_name, module in model.named_modules():
        if "position_embeddings" in module_name:
            return module.weight.data.size()[0]
