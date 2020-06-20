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

import time

import torch
from tqdm import tqdm


class CompositeLoss():
    """
    Defines a new training loop that has direct access to Experiment attributes
    and can be customized for more complex loss functions
    """

    def train_epoch(self):
        """Overwrites train model to use private train_model method"""
        self._train_model()

    def _train_model(self):
        """Private train model that has access to Experiment attributes"""
        self.model.train()
        # Use asynchronous GPU copies when the memory is pinned
        # See https://pytorch.org/docs/master/notes/cuda.html
        loader = self.train_loader
        async_gpu = loader.pin_memory
        progress_bar = None
        freeze_params = None
        if progress_bar is not None:
            loader = tqdm(loader, **progress_bar)
            # update progress bar total based on batches_in_epoch
            if self.batches_in_epoch < len(loader):
                loader.total = self.batches_in_epoch

        # Check if training with Apex Mixed Precision
        use_amp = hasattr(self.optimizer, "_amp_stash")
        try:
            from apex import amp
        except ImportError:
            if use_amp:
                raise ImportError(
                    "Mixed precision requires NVIDA APEX."
                    "Please install apex from https://www.github.com/nvidia/apex")

        t0 = time.time()
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= self.batches_in_epoch:
                break

            num_images = len(target)
            if self.transform_data_to_device is None:
                data = data.to(self.device, non_blocking=async_gpu)
                target = target.to(self.device, non_blocking=async_gpu)
            else:
                data, target = self.transform_data_to_device(
                    data, target, self.device, non_blocking=async_gpu)
            t1 = time.time()

            if self.pre_batch is not None:
                self.pre_batch(model=self.model, batch_idx=batch_idx)

            self.optimizer.zero_grad()

            # override loss calculation
            error_loss, complexity_loss = \
                self.calculate_composite_loss(data, target, async_gpu=async_gpu)
            loss = (error_loss + complexity_loss
                    if complexity_loss is not None
                    else error_loss)

            t2 = time.time()
            if use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            del loss

            if freeze_params is not None:
                with torch.no_grad():
                    for param in freeze_params:
                        param_module = param[0]
                        param_indices = param[1]
                        param_module.grad[param_indices, :] = 0.0

            t3 = time.time()
            self.optimizer.step()
            t4 = time.time()

            if self.post_batch_wrapper is not None:
                time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                               + "weight update: {:.3f}s").format(t1 - t0, t2 - t1,
                                                                  t3 - t2, t4 - t3)
                self.post_batch_wrapper(model=self.model,
                                        error_loss=error_loss.detach(),
                                        complexity_loss=(complexity_loss.detach()
                                                         if complexity_loss is not None
                                                         else None),
                                        batch_idx=batch_idx,
                                        num_images=num_images,
                                        time_string=time_string)
            del error_loss, complexity_loss
            t0 = time.time()

        if progress_bar is not None:
            loader.n = loader.total
            loader.close()

    def calculate_composite_loss(self, data, target, async_gpu=True):
        """
        :param data: input to the training function, as specified by dataloader
        :param target: target to be matched by model, as specified by dataloader
        :param async_gpu: define whether or not to use
                          asynchronous GPU copies when the memory is pinned
        """
        # error loss
        output = self.model(data)
        error_loss = self.error_loss(output, target)
        del data, target, output

        # complexity loss
        complexity_loss = (self.complexity_loss(self.model)
                           if self.complexity_loss is not None
                           else None)

        return error_loss, complexity_loss

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["train_epoch"] = ["CompositeLoss.train_epoch"]
        eo["_train_model"] = ["CompositeLoss._train_model"]
        eo["calculate_composite_loss"] = ["CompositeLoss.calculate_composite_loss"]
        return eo
