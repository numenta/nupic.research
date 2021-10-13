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

import pytorch_lightning as pl
import torch
from torch.backends import cudnn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from nupic.research.frameworks.vernon.network_utils import create_model

__all__ = [
    "SupervisedModel",
]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class SupervisedModel(pl.LightningModule):
    """
    General experiment class used to train neural networks in supervised
    learning tasks.
    """
    trainer_requirements = dict(
        automatic_optimization=False,  # Required for complexity_loss
    )

    def __init__(self, config):
        super().__init__()

        self.config = config
        self._loss_function = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )

        self.model = create_model(
            model_class=config["model_class"],
            model_args=config.get("model_args", {}),
            init_batch_norm=config.get("init_batch_norm", False),
            checkpoint_file=config.get("checkpoint_file", None),
            load_checkpoint_args=config.get("load_checkpoint_args", {}),
        )

        self.epochs = config["epochs"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        x, target = batch
        output = self(x)
        loss = self.error_loss(output, target)
        self.log("train_loss", loss
                 # Flag that makes sense but slows things down:
                 # , sync_dist=True
                 # TODO: understand implications.
                 )
        self.manual_backward(loss, optimizer)

        complexity_loss = self.complexity_loss()
        if complexity_loss is not None:
            self.log("complexity_loss", complexity_loss)
            self.manual_backward(complexity_loss, optimizer)

        optimizer.step()
        optimizer.zero_grad()

    def validation_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        loss = self.error_loss(out, target)

        pred = torch.argmax(out, dim=1)
        val_acc = torch.sum(pred == target).float() / len(target)

        # TODO: Logging these every step may be wasteful.
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", val_acc, sync_dist=True)

        return loss

    def configure_optimizers(self):
        group_decay, group_no_decay = [], []
        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if self.should_decay_parameter(module, name, param, self.config):
                    group_decay.append(param)
                else:
                    group_no_decay.append(param)

        optimizer_class = self.config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = self.config.get("optimizer_args", {})
        optimizer = optimizer_class([dict(params=group_decay),
                                     dict(params=group_no_decay,
                                          weight_decay=0.)],
                                    **optimizer_args)

        lr_scheduler_class = self.config.get("lr_scheduler_class", None)
        if lr_scheduler_class is not None:
            lr_scheduler_args = self.config.get("lr_scheduler_args", {})
            lr_scheduler_args = self.expand_lr_scheduler_args(
                lr_scheduler_class, lr_scheduler_args)
            lr_scheduler = lr_scheduler_class(optimizer,
                                              **lr_scheduler_args)

            if (self.config.get("lr_scheduler_step_every_batch", False)
               or isinstance(lr_scheduler, (OneCycleLR, ComposedLRScheduler))):
                lr_scheduler = dict(scheduler=lr_scheduler, interval="step")

            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def setup(self, stage):
        self.train_dataset = self.load_dataset(self.config, train=True)
        self.val_dataset = self.load_dataset(self.config, train=False)

    def train_dataloader(self):
        return self.create_train_loader(self.current_epoch)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.get("val_batch_size",
                                       self.config.get("batch_size", 1)),
            num_workers=self.config.get("workers", 0),
            pin_memory=torch.cuda.is_available(),
        )

    #
    # Utility methods
    #

    def expand_lr_scheduler_args(self, lr_scheduler_class, lr_scheduler_args):
        """
        Return a new lr_scheduler_args with extra args inserted.
        :param lr_scheduler_class: Class of lr-scheduler
        :param lr_scheduler_args: User-specified args
        :return: New lr_scheduler_args
        """
        if issubclass(lr_scheduler_class, OneCycleLR):
            # Update OneCycleLR parameters
            epochs = lr_scheduler_args["epochs"]
            lr_scheduler_args = {
                **lr_scheduler_args,
                "total_steps": sum(self.compute_steps_in_epoch(epoch)
                                   for epoch in range(epochs)),
            }

        return lr_scheduler_args

    def compute_steps_in_epoch(self, epoch):
        """
        Get the number of optimizer steps in a given epoch.
        :param epoch: Epoch number
        :return: Number of optimizer steps
        """
        return len(self.create_train_loader(epoch))

    @classmethod
    def load_dataset(cls, config, train=True):
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        dataset_args = config.get("dataset_args", {})
        dataset_args.update(train=train)
        return dataset_class(**dataset_args)

    def create_train_loader(self, epoch):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.get("batch_size", 1),
            num_workers=self.config.get("workers", 0),
            pin_memory=torch.cuda.is_available(),
            drop_last=self.config.get("train_loader_drop_last", True),
        )

    def should_decay_parameter(self, module, parameter_name, parameter, config):
        if isinstance(module, _BatchNorm):
            return config.get("batch_norm_weight_decay", True)
        elif parameter_name == "bias":
            return config.get("bias_weight_decay", True)
        else:
            return True

    def error_loss(self, output, target, reduction="mean"):
        """
        The error loss component of the loss function.
        """
        return self._loss_function(output, target, reduction=reduction)

    def complexity_loss(self):
        """
        The model complexity component of the loss function.
        """
        pass
