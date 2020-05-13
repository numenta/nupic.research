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

import functools
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

class KnowledgeDistillation(object):
    """
    Sets the network to learn from a teacher model
    """
    def __init__(self):
        super().__init__()

        # initialize variables
        self.teacher_model = None
        self.kd_factor_init = None
        self.kd_factor_end = None

    def setup_experiment(self, config):
        """
        Add following variables to config

        :param config: Dictionary containing the configuration parameters

            - teacher_model_class: Class for pretrained model to be used as teacher
                                   in knowledge distillation.
            - kd_factor_init: Determines the percentage of the target that comes
                              from the teacher model. Value should be float
                              between 0 and 1. Defaults to 1.
            - kd_factor_end: KD factor at last epoch. Will calculate linear decay
                             based on initial kd_factor_init and kd_factor_end.
                             Value should be float between 0 and 1.
                             If None, no decay is applied. Defaults to None.
        """
        super().setup_experiment(config)

        # Teacher model and knowledge distillation variables
        teacher_model_class = config.get("teacher_model_class", None)
        if teacher_model_class is not None:
            # load teacher model, set to eval only and transfer to GPU
            self.teacher_model = teacher_model_class()
            self.teacher_model.eval()
            self.teacher_model.to(self.device)
            self.logger.info(f"KD teacher class: {teacher_model_class}")

            # initalize Knowledge Distillation factor
            self.kd_factor_init = config.get("kd_factor_init", 1)
            assert 0 <= self.kd_factor_init <= 1, \
                "KD factor at first epoch should be >= 0 and <= 1"
            self.kd_factor_end = config.get("kd_factor_end", None)
            if self.kd_factor_end is not None:
                assert 0 <= self.kd_factor_end <= 1, \
                    "KD factor at last epoch should be >= 0 and <= 1"
            self.logger.info(f"KD factor: {self.kd_factor_init} {self.kd_factor_end}")

        # Set new training method to use a teacher as default
        self.train_model = config.get("train_model_func", train_model_with_teacher)

    def train_epoch(self, epoch):
        # linear decay knowledge distillation factor if required
        if self.kd_factor_end is not None:
            kd_factor = linear_decay(first_epoch_value=self.kd_factor_init,
                                     last_epoch_value=self.kd_factor_end,
                                     current_epoch=epoch,
                                     total_epochs=self.epochs)
            self.logger.debug(f"KD factor: {kd_factor:.3f} at epoch {epoch}")
        else:
            kd_factor = self.kd_factor_init

        self.train_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.loss_function,
            batches_in_epoch=self.batches_in_epoch,
            pre_batch_callback=functools.partial(self.pre_batch, epoch=epoch),
            post_batch_callback=functools.partial(self.post_batch, epoch=epoch),
            teacher_model=self.teacher_model,
            kd_factor=kd_factor,
        )

def train_model_with_teacher(
    model,
    loader,
    optimizer,
    device,
    freeze_params=None,
    criterion=F.nll_loss,
    batches_in_epoch=sys.maxsize,
    pre_batch_callback=None,
    post_batch_callback=None,
    progress_bar=None,
    teacher_model=None,
    kd_factor=None
):
    """Train the given model by iterating through mini batches. An epoch ends
    after one pass through the training set, or if the number of mini batches
    exceeds the parameter "batches_in_epoch".

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: train dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
           This function will train the model on every batch using this optimizer
           and the :func:`torch.nn.functional.nll_loss` function
    :param batches_in_epoch: Max number of mini batches to train.
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param freeze_params: List of parameters to freeze at specified indices
     For each parameter in the list:
     - parameter[0] -> network module
     - parameter[1] -> weight indices
    :type param: list or tuple
    :param criterion: loss function to use
    :type criterion: function
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :type post_batch_callback: function
    :param pre_batch_callback: Callback function to be called before every batch
                               with the following parameters: model, batch_idx
    :type pre_batch_callback: function
    :param progress_bar: Optional :class:`tqdm` progress bar args.
                         None for no progress bar
    :type progress_bar: dict or None
    :param teacher model: Teacher model used for knowledge distillation
    :type teacher model: torch.nn.Module
    :param kd_factor: Determines the percentage of the target that comes
                      from the teacher model.
    :type kd_factor: float

    :return: mean loss for epoch
    :rtype: float
    """
    model.train()
    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    if progress_bar is not None:
        loader = tqdm(loader, **progress_bar)
        # update progress bar total based on batches_in_epoch
        if batches_in_epoch < len(loader):
            loader.total = batches_in_epoch

    # Check if training with Apex Mixed Precision
    # FIXME: There should be another way to check if 'amp' is enabled
    use_amp = hasattr(optimizer, "_amp_stash")
    try:
        from apex import amp
    except ImportError:
        if use_amp:
            raise ImportError(
                "Mixed precision requires NVIDA APEX."
                "Please install apex from https://www.github.com/nvidia/apex")

    t0 = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        num_images = len(target)
        data = data.to(device, non_blocking=async_gpu)
        target = target.to(device, non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()

        output = model(data)

        # alternative knowledge distillation training
        if teacher_model is not None:
            with torch.no_grad():
                # target is linear combination of teacher and target softmaxes
                softmax_output_teacher = F.softmax(teacher_model(data))
                one_hot_target = F.one_hot(target, num_classes=output.shape[-1])
                combined_target = (kd_factor * softmax_output_teacher
                                   + (1 - kd_factor) * one_hot_target)
            # requires a custom loss function.
            del softmax_output_teacher, one_hot_target
            loss = soft_cross_entropy(output, combined_target)
            del combined_target
        # regular training
        else:
            loss = criterion(output, target)

        del data, target, output

        t2 = time.time()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if freeze_params is not None:
            with torch.no_grad():
                for param in freeze_params:
                    param_module = param[0]
                    param_indices = param[1]
                    param_module.grad[param_indices, :] = 0.0

        t3 = time.time()
        optimizer.step()
        t4 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1, t3 - t2,
                                                              t4 - t3)
            post_batch_callback(model=model, loss=loss.detach(), batch_idx=batch_idx,
                                num_images=num_images, time_string=time_string)
        del loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()

def soft_cross_entropy(output, target, size_average=True):
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
    if size_average:
        return torch.mean(torch.sum(-target * F.log_softmax(output, dim=1), dim=1))
    else:
        return torch.sum(torch.sum(-target * F.log_softmax(output, dim=1), dim=1))

def linear_decay(first_epoch_value, last_epoch_value, current_epoch, total_epochs):
    """
    Calculates value for a current epoch in a linear decay.

    :param first_epoch_value: Value at first epoch (before training).
    :param last_epoch_value: Value at last epoch (before training).
    :param current_epoch: Current epoch. Assumes first epoch is 0.
    :param total_epochs: Total number of epochs in training.
    """
    step_size = (first_epoch_value - last_epoch_value) / (total_epochs - 1)
    return first_epoch_value - step_size * current_epoch

