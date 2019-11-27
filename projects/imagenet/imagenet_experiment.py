#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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
import os
from collections.abc import Mapping
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.utils.data
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from nupic.research.frameworks.pytorch.dataset_utils import CachedDatasetFolder
from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model

__all__ = ["ImagenetExperiment"]


class ImagenetExperiment:
    """
    Experiment class used to train different models on Imagenet
    """

    def __init__(self):
        self.config = None
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None
        self.summary = None
        self.rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ids = None
        self.epoch = 0
        self.steps_per_epoch = 0

    def setup(self, config):
        if isinstance(config, Mapping):
            self.config = SimpleNamespace(**config)
        else:
            self.config = config

        if self.config.distributed:
            self.rank = dist.get_rank()
            if torch.cuda.is_available():
                # When using distributed training, assume rank represents GPU
                self.device = self.rank
                self.device_ids = [self.device]

        self.train_loader = self._create_train_dataloader()
        self.steps_per_epoch = len(self.train_loader)
        self.val_loader = self._create_validation_dataloader()
        self.model = self._create_model()
        self.optimizer = self._create_optimizer(self.model)
        self.lr_scheduler = self._create_lr_scheduler(self.optimizer)
        self.loss_function = self.config.loss_function
        if self.rank == 0:
            self.summary = SummaryWriter(log_dir=self.config.logdir)

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        if self.config.progress_bar:
            progress_bar = dict(desc="validating")
        else:
            progress_bar = None

        results = evaluate_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.loss_function,
            batches_in_epoch=self.config.batches_in_epoch,
            progress=progress_bar
        )
        if self.summary is not None:
            for k, v in results.items():
                self.summary.add_scalar(k, v, self.epoch)

        return results

    def train(self, epoch):
        if self.config.progress_bar:
            progress_bar = dict(desc="training : {}".format(epoch))
        else:
            progress_bar = None

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.epoch = epoch
        train_model(model=self.model,
                    loader=self.train_loader,
                    optimizer=self.optimizer,
                    device=self.device,
                    criterion=self.loss_function,
                    batches_in_epoch=self.config.batches_in_epoch,
                    progress_bar=progress_bar,
                    pre_batch_callback=self.pre_batch,
                    post_batch_callback=self.post_batch,
                    )

    def pre_epoch(self, epoch):
        pass

    def pre_batch(self, model, batch_idx):
        if self.summary is not None:
            global_step = self.epoch * self.steps_per_epoch + batch_idx
            self.summary.add_scalar("lr", self.get_lr()[0], global_step)

    def post_batch(self, model, batch_idx):
        if isinstance(self.lr_scheduler, OneCycleLR):
            self.lr_scheduler.step(batch_idx)

    def post_epoch(self, epoch):
        if not isinstance(self.lr_scheduler, OneCycleLR):
            self.lr_scheduler.step(epoch)

    def save(self, checkpoint_path):
        filename = os.path.join(checkpoint_path, "checkpoint.pt")
        torch.save(self.model.state_dict(), filename)
        return filename

    def restore(self, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            filename = os.path.join(checkpoint_path, "checkpoint.pt")
        else:
            filename = checkpoint_path

        self.model.load_state_dict(
            torch.load(filename, map_location=self.device)
        )

    def stop(self):
        if self.summary is not None:
            self.summary.close()

    def get_lr(self):
        """
        Returns the current learning rate
        :return: list of learning rates used by the optimizer
        """
        return [p["lr"] for p in self.optimizer.param_groups]

    def _create_train_dataloader(self):
        dataset = CachedDatasetFolder(
            root=os.path.join(self.config.data, self.config.train_dir),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]))

        if self.config.distributed:
            self.train_sampler = DistributedSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=self.train_sampler is None,
            num_workers=self.config.workers,
            sampler=self.train_sampler,
            pin_memory=torch.cuda.is_available())

    def _create_validation_dataloader(self):
        dataset = CachedDatasetFolder(
            root=os.path.join(self.config.data, self.config.val_dir),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]))
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            pin_memory=False)

    def _create_model(self):
        model_class = self.config.model_class
        model_args = self.config.model_args
        model = model_class(**model_args)
        model.to(self.device)
        if self.config.distributed:
            model = DistributedDataParallel(model, device_ids=self.device_ids)
        else:
            model = DataParallel(model, device_ids=self.device_ids)
        return model

    def _create_lr_scheduler(self, optimizer):
        if self.config.lr_scheduler_class is None:
            return None

        lr_scheduler_class = self.config.lr_scheduler_class
        lr_scheduler_args = self.config.lr_scheduler_args

        if lr_scheduler_class == OneCycleLR:
            lr_scheduler_args["epochs"] = self.config.epochs
            lr_scheduler_args["steps_per_epoch"] = self.steps_per_epoch

        return lr_scheduler_class(optimizer, **lr_scheduler_args)

    def _create_optimizer(self, model):
        optimizer_class = self.config.optimizer_class
        optimizer_args = self.config.optimizer_args
        optimizer = optimizer_class(model.parameters(), **optimizer_args)
        return optimizer
