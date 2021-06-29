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

import io
import sys
import time
from pprint import pformat

import torch
from torch.backends import cudnn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from nupic.research.frameworks.pytorch.model_utils import (
    deserialize_state_dict,
    evaluate_model,
    serialize_state_dict,
    train_model,
)
from nupic.research.frameworks.vernon.experiment_utils import create_lr_scheduler
from nupic.research.frameworks.vernon.experiments.components.experiment_base import (
    ExperimentBase,
)
from nupic.research.frameworks.vernon.network_utils import (
    create_model,
    get_compatible_state_dict,
)

try:
    from apex import amp
except ImportError:
    amp = None


__all__ = [
    "SupervisedExperiment",
]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class SupervisedExperiment(ExperimentBase):
    """
    General experiment class used to train neural networks in supervised learning tasks.
    """
    def __init__(self):
        self.model = None
        self.optimizer = None
        self._loss_function = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batches_in_epoch = sys.maxsize
        self.batches_in_epoch_val = sys.maxsize
        self.epochs = 1
        self.mixed_precision = False
        self.total_batches = 0
        self.epochs_to_validate = []
        self.current_epoch = 0
        self.rank = 0

    def setup_experiment(self, config):
        """
        Configure the experiment for training
        :param config: Dictionary containing the configuration parameters
            - data: Dataset path
            - progress: Show progress during training
            - train_dir: Dataset training data relative path
            - batch_size: Training batch size
            - val_dir: Dataset validation data relative path
            - val_batch_size: Validation batch size
            - workers: how many data loading processes to use
            - train_loader_drop_last: Whether to skip last batch if it is
                                      smaller than the batch size
            - num_classes: Limit the dataset size to the given number of classes
            - model_class: Model class. Must inherit from "torch.nn.Module"
            - model_args: model model class arguments passed to the constructor
            - init_batch_norm: Whether or not to Initialize running batch norm
                               mean to 0.
            - optimizer_class: Optimizer class.
                               Must inherit from "torch.optim.Optimizer"
            - optimizer_args: Optimizer class class arguments passed to the
                              constructor
            - lr_scheduler_class: Learning rate scheduler class.
                                 Must inherit from "_LRScheduler"
            - lr_scheduler_args: Learning rate scheduler class class arguments
                                 passed to the constructor
            - lr_scheduler_step_every_batch: Whether to step the lr-scheduler after
                                             after every batch (e.g. for OneCycleLR)
            - loss_function: Loss function. See "torch.nn.functional"
            - epochs: Number of epochs to train
            - batches_in_epoch: Number of batches per epoch.
                                Useful for debugging
            - batches_in_epoch_val: Number of batches per epoch in validation.
                                   Useful for debugging
            - mixed_precision: Whether or not to enable apex mixed precision
            - mixed_precision_args: apex mixed precision arguments.
                                    See "amp.initialize"
            - sample_transform: Transform acting on the training samples. To be used
                                additively after default transform or auto-augment.
            - target_transform: Transform acting on the training targets.
            - replicas_per_sample: Number of replicas to create per sample in the batch.
                                   (each replica is transformed independently)
                                   Used in maxup.
            - train_model_func: Optional user defined function to train the model,
                                expected to behave similarly to `train_model`
                                in terms of input parameters and return values
            - evaluate_model_func: Optional user defined function to validate the model
                                   expected to behave similarly to `evaluate_model`
                                   in terms of input parameters and return values
            - checkpoint_file: if not None, will start from this model. The model
                               must have the same model_args and model_class as the
                               current experiment.
            - load_checkpoint_args: args to be passed to `load_state_from_checkpoint`
            - epochs_to_validate: list of epochs to run validate(). A -1 asks
                                  to run validate before any training occurs.
                                  Default: last three epochs.
            - launch_time: time the config was created (via time.time). Used to report
                           wall clock time until the first batch is done.
                           Default: time.time() in this setup_experiment().
        """

        self.launch_time = config.get("launch_time", time.time())
        super().setup_experiment(config)
        self.logger.info("Execution order: %s",
                         pformat(self.get_execution_order()))

        # Configure model
        self.device = config.get("device", self.device)
        self.model = self.create_model(config, self.device)
        self.transform_model()

        self.logger.debug(self.model)

        # Configure and create optimizer
        self.optimizer = self.create_optimizer(config, self.model)

        # Validate mixed precision requirements
        self.mixed_precision = config.get("mixed_precision", False)
        if self.mixed_precision and amp is None:
            self.mixed_precision = False
            self.logger.error(
                "Mixed precision requires NVIDA APEX."
                "Please install apex from https://www.github.com/nvidia/apex"
                "Disabling mixed precision training.")

        # Configure mixed precision training
        if self.mixed_precision:
            amp_args = config.get("mixed_precision_args", {})
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, **amp_args)
            self.logger.info("Using mixed precision")

        self._loss_function = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )

        self.num_classes = config.get("num_classes", 1000)
        self.epochs = config.get("epochs", 1)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.batches_in_epoch_val = config.get("batches_in_epoch_val", sys.maxsize)
        self.current_epoch = 0

        # Configure data loaders
        self.create_loaders(config)
        self.total_batches = len(self.train_loader,)

        self.epochs_to_validate = config.get("epochs_to_validate",
                                             range(self.epochs - 3,
                                                   self.epochs + 1))

        # Configure learning rate scheduler
        self.lr_scheduler = self.create_lr_scheduler(
            config, self.optimizer, self.total_batches)
        if self.lr_scheduler is not None:
            lr_scheduler_class = self.lr_scheduler.__class__.__name__
            lr_scheduler_args = config.get("lr_scheduler_args", {})
            self.logger.info("LR Scheduler class: " + lr_scheduler_class)
            self.logger.info("LR Scheduler args:")
            self.logger.info(pformat(lr_scheduler_args))
            self.logger.info("steps_per_epoch=%s", self.total_batches)

        self.step_lr_every_batch = config.get("lr_scheduler_step_every_batch", False)
        if isinstance(self.lr_scheduler, (OneCycleLR, ComposedLRScheduler)):
            self.step_lr_every_batch = True

        # Set train and validate methods.
        self.train_model = config.get("train_model_func", train_model)
        self.evaluate_model = config.get("evaluate_model_func", evaluate_model)

        self.progress = config.get("progress", False)
        if self.logger.disabled:
            self.progress = False

    @classmethod
    def create_model(cls, config, device):
        """
        Create `torch.nn.Module` model from an experiment config
        :param config:
            - model_class: Model class. Must inherit from "torch.nn.Module"
            - model_args: model model class arguments passed to the constructor
            - init_batch_norm: Whether or not to Initialize running batch norm
                               mean to 0.
            - checkpoint_file: if not None, will start from this model. The
                               model must have the same model_args and
                               model_class as the current experiment.
            - load_checkpoint_args: args to be passed to `load_state_from_checkpoint`
        :param device:
            Pytorch device
        :return:
                Model instance
        """
        return create_model(
            model_class=config["model_class"],
            model_args=config.get("model_args", {}),
            init_batch_norm=config.get("init_batch_norm", False),
            device=device,
            checkpoint_file=config.get("checkpoint_file", None),
            load_checkpoint_args=config.get("load_checkpoint_args", {}),
        )

    @classmethod
    def create_optimizer(cls, config, model):
        """
        Create optimizer from an experiment config.

        :param optimizer_class: Callable or class to instantiate optimizer. Must return
                                object inherited from "torch.optim.Optimizer"
        :param optimizer_args: Arguments to pass to the optimizer.
        """
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        return optimizer_class(model.parameters(), **optimizer_args)

    @classmethod
    def create_lr_scheduler(cls, config, optimizer, total_batches):
        """
        Create lr scheduler from the experiment config
        :param config:
            - lr_scheduler_class: (optional) Class of lr-scheduler
            - lr_scheduler_args: (optional) dict of args to pass to lr-class
        :param optimizer: torch optimizer
        :param total_batches: number of batches/steps in an epoch
        """
        lr_scheduler_class = config.get("lr_scheduler_class", None)
        if lr_scheduler_class is not None:
            lr_scheduler_args = config.get("lr_scheduler_args", {})
            return create_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler_class=lr_scheduler_class,
                lr_scheduler_args=lr_scheduler_args,
                steps_per_epoch=total_batches)

    def create_loaders(self, config):
        """Create and assign train and val dataloaders"""

        self.train_loader = self.create_train_dataloader(config)
        self.val_loader = self.create_validation_dataloader(config)

    @classmethod
    def load_dataset(cls, config, train=True):
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        dataset_args = dict(config.get("dataset_args", {}))
        dataset_args.update(train=train)
        return dataset_class(**dataset_args)

    @classmethod
    def create_train_sampler(cls, config, dataset):
        return None

    @classmethod
    def create_train_dataloader(cls, config, dataset=None):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """
        if dataset is None:
            dataset = cls.load_dataset(config, train=True)

        sampler = cls.create_train_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size", 1),
            shuffle=sampler is None,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=config.get("train_loader_drop_last", True),
        )

    @classmethod
    def create_validation_sampler(cls, config, dataset):
        return None

    @classmethod
    def create_validation_dataloader(cls, config, dataset=None):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """
        if dataset is None:
            dataset = cls.load_dataset(config, train=False)

        sampler = cls.create_validation_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("val_batch_size",
                                  config.get("batch_size", 1)),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    def transform_model(self):
        """Placeholder for any model transformation required prior to training"""
        pass

    def run_pre_experiment(self):
        """Run validation before training."""
        if -1 in self.epochs_to_validate:
            self.logger.debug("Validating before any training:")
            return self.validate()

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        return self.evaluate_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def train_epoch(self):
        self.train_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch,
            pre_batch_callback=self.pre_batch,
            post_batch_callback=self.post_batch_wrapper,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def run_epoch(self):
        self.pre_epoch()
        self.train_epoch()
        self.post_epoch()
        t1 = time.time()

        if self.current_epoch in self.epochs_to_validate:
            ret = self.validate()
        else:
            ret = {
                "total_correct": 0,
                "total_tested": 0,
                "mean_loss": 0.0,
                "mean_accuracy": 0.0,
            }

        ret.update(
            learning_rate=self.get_lr()[0],
        )

        self.logger.debug("validate time: %s", time.time() - t1)
        self.logger.debug("---------- End of run epoch ------------")
        self.logger.debug("")

        self.current_epoch += 1
        return ret

    def pre_epoch(self):
        pass

    def pre_batch(self, model, batch_idx):
        pass

    def post_batch(self, model, error_loss, complexity_loss, batch_idx,
                   num_images, time_string):
        # Update 1cycle learning rate after every batch
        if self.step_lr_every_batch:
            self.lr_scheduler.step()

        if self.progress and self.current_epoch == 0 and batch_idx == 0:
            self.logger.info("Launch time to end of first batch: %s",
                             time.time() - self.launch_time)

        if self.progress and (batch_idx % 40) == 0:
            total_batches = self.total_batches
            current_batch = batch_idx
            if hasattr(self.train_loader.sampler, "num_replicas"):
                # Compute actual batch size from distributed sampler
                total_batches *= self.train_loader.sampler.num_replicas
                current_batch *= self.train_loader.sampler.num_replicas
            self.logger.debug("End of batch for rank: %s. Epoch: %s, Batch: %s/%s, "
                              "loss: %s, Learning rate: %s num_images: %s",
                              self.rank, self.current_epoch, current_batch,
                              total_batches, error_loss, self.get_lr(),
                              num_images)
            self.logger.debug("Timing: %s", time_string)

    def post_batch_wrapper(self, **kwargs):
        self.post_optimizer_step(self.model)
        self.post_batch(**kwargs)

    def post_optimizer_step(self, model):
        pass

    def post_epoch(self):
        self.logger.debug("End of epoch %s LR/weight decay before step: %s/%s",
                          self.current_epoch, self.get_lr(), self.get_weight_decay())

        # Update learning rate
        if self.lr_scheduler is not None and not self.step_lr_every_batch:
            self.lr_scheduler.step()

        self.logger.debug("End of epoch %s LR/weight decay after step: %s/%s",
                          self.current_epoch, self.get_lr(), self.get_weight_decay())

    def error_loss(self, output, target, reduction="mean"):
        """
        The error loss component of the loss function.
        """
        return self._loss_function(output, target, reduction=reduction)

    def complexity_loss(self, model):
        """
        The model complexity component of the loss function.
        """
        pass

    def transform_data_to_device(self, data, target, device, non_blocking):
        """
        This provides an extensibility point for performing any final
        transformations on the data or targets.
        """
        data = data.to(self.device, non_blocking=non_blocking)
        target = target.to(self.device, non_blocking=non_blocking)
        return data, target

    @classmethod
    def get_readable_result(cls, result):
        keep_keys = ["total_correct", "total_tested", "complexity_loss",
                     "learning_rate"]
        change_keys = {"mean_loss": "validation_loss",
                       "mean_accuracy": "validation_accuracy"}

        return {
            **{k: result[k]
               for k in keep_keys
               if k in result},
            **{k2: result[k1]
               for k1, k2 in change_keys.items()
               if k1 in result}
        }

    def get_state(self):
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        state = {
            "current_epoch": self.current_epoch,
        }

        # Save state into a byte array to avoid ray's GPU serialization issues
        # See https://github.com/ray-project/ray/issues/5519
        with io.BytesIO() as buffer:
            model = self.model
            if hasattr(model, "module"):
                # DistributedDataParallel
                model = model.module
            serialize_state_dict(buffer, model.state_dict())
            state["model"] = buffer.getvalue()

        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.optimizer.state_dict())
            state["optimizer"] = buffer.getvalue()

        if self.lr_scheduler is not None:
            with io.BytesIO() as buffer:
                state_dict = self.lr_scheduler.state_dict()
                if "anneal_func" in state_dict:
                    # FIXME: This is a workaround for a PyTorch bug.
                    # https://github.com/pytorch/pytorch/issues/42376
                    del state_dict["anneal_func"]
                serialize_state_dict(buffer, state_dict)
                state["lr_scheduler"] = buffer.getvalue()

        if self.mixed_precision:
            with io.BytesIO() as buffer:
                serialize_state_dict(buffer, amp.state_dict())
                state["amp"] = buffer.getvalue()

        return state

    def set_state(self, state):
        """
        Restore the experiment from the state returned by `get_state`
        :param state: dictionary with "model", "optimizer", "lr_scheduler", and "amp"
                      states
        """
        if "model" in state:
            with io.BytesIO(state["model"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            model = self.model
            if hasattr(model, "module"):
                # DistributedDataParallel
                model = model.module
            state_dict = get_compatible_state_dict(state_dict, model)
            model.load_state_dict(state_dict)

        if "optimizer" in state:
            with io.BytesIO(state["optimizer"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            self.optimizer.load_state_dict(state_dict)

        if "lr_scheduler" in state:
            with io.BytesIO(state["lr_scheduler"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            self.lr_scheduler.load_state_dict(state_dict)

        if "amp" in state and amp is not None:
            with io.BytesIO(state["amp"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            amp.load_state_dict(state_dict)

        if "current_epoch" in state:
            self.current_epoch = state["current_epoch"]
        else:
            # Try to recover current epoch from LR Scheduler state
            last_epoch = self.lr_scheduler.last_epoch + 1
            if isinstance(self.lr_scheduler, ComposedLRScheduler):
                self.current_epoch = last_epoch // self.lr_scheduler.steps_per_epoch
            elif isinstance(self.lr_scheduler, OneCycleLR):
                steps_per_epoch = self.lr_scheduler.total_steps // self.epochs
                self.current_epoch = last_epoch // steps_per_epoch
            else:
                self.current_epoch = last_epoch

    def run_iteration(self):
        return self.run_epoch()

    def should_stop(self):
        """
        Whether or not the experiment should stop. Usually determined by the
        number of epochs but customizable to any other stopping criteria
        """
        return self.current_epoch >= self.epochs

    def get_lr(self):
        """
        Returns the current learning rate
        :return: list of learning rates used by the optimizer
        """
        return [p["lr"] for p in self.optimizer.param_groups]

    def get_current_epoch(self):
        """
        Returns the current epoch of the running experiment
        """
        return self.current_epoch

    def get_weight_decay(self):
        """
        Returns the current weight decay
        :return: list of weight decays used by the optimizer
        """
        return [p["weight_decay"] for p in self.optimizer.param_groups]

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "SupervisedExperiment"
        # Extended methods
        eo["setup_experiment"].append(exp + ".setup_experiment")
        eo["get_state"].append(exp + ": Model, optimizer, LR scheduler, epoch")
        eo["set_state"].append(exp + ": Model, optimizer, LR scheduler, epoch")

        eo.update(
            # Overwritten methods
            get_readable_result=[exp + ": Basic keys"],
            run_iteration=[exp + ".run_iteration"],
            should_stop=[exp + ".should_stop"],
            run_pre_experiment=[exp + ".run_pre_experiment"],

            # New methods
            create_model=[exp + ".create_model"],
            create_lr_scheduler=[exp + ".create_lr_scheduler"],
            create_optimizer=[exp + ".create_optimizer"],
            transform_model=[exp + ".transform_model"],
            validate=[exp + ".validate"],
            create_loaders=[exp + ".create_loaders"],
            create_train_dataloader=[exp + ".create_train_dataloader"],
            create_train_sampler=[exp + ".create_train_sampler"],
            create_validation_dataloader=[exp + ".create_validation_dataloader"],
            create_validation_sampler=[exp + ".create_validation_sampler"],
            train_epoch=[exp + ".train_epoch"],
            run_epoch=[exp + ".run_epoch"],
            pre_epoch=[],
            post_epoch=[],
            pre_batch=[],
            post_batch=[exp + ": Logging"],
            post_optimizer_step=[],
            transform_data_to_device=[exp + ".transform_data_to_device"],
            error_loss=[exp + ".error_loss"],
            complexity_loss=[],
            load_dataset=[exp + ".load_dataset"],
        )

        return eo
