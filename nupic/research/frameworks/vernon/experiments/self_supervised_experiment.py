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

from nupic.research.frameworks.pytorch.self_supervised_utils import (
    train_model_supervised,
    train_model_unsupervised,
    evaluate_model,
)
from nupic.research.frameworks.vernon.experiment_utils import create_lr_scheduler
from nupic.research.frameworks.vernon.experiments import SupervisedExperiment
from nupic.research.frameworks.vernon.network_utils import (
    create_model,
    get_compatible_state_dict,
)

try:
    from apex import amp
except ImportError:
    amp = None


__all__ = [
    "SelfSupervisedExperiment",
]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class SelfSupervisedExperiment(SupervisedExperiment):
    """
    General experiment class used to train neural networks in self-supervised learning
    tasks.

    Self-supervised experiments have three important dataset splits: unsupervised, test,
    and supervised. The unsupervised set consists of unlabeled data for representation
    learning, the supervised set consists of a typically smaller amount of labeled
    data for which to train a classifier, and the test set is used to evaluate the
    classifier.

    The validation step trains a new classifier on top of the existing frozen model,
    and then proceeds to test this classifier on the test set. The number of
    supervised training epochs to train for each validation is given by
    supervised_training_epochs_per_validation.
    """

    #add a train supervised classifier method
    #override run_epoch


    def __init__(self):
        super(SelfSupervisedExperiment, self).__init__()
        self.unsupervised_loader = None
        self.supervised_loader = None
        self.supervised_training_epochs_per_validation = 3

        #params for self supervised training
        """
        unsupervised_batch_size
        supervised_batch_size
        val_batch_size
        
        pre_model_transformation_func (patchify)
        post_model_transformation_func (aggregate, like an adaptive average pool)
        
        classifier_class
        classifier_args
        init_batch_norm_classifier
        checkpoint_file_classifier
        load_checkpoint_args_classifier
        reset_classifier_on_validate



        """

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
            - lr_scheduler_step_every_batch: Whether to step the lr-scheduler
                                             after every batch (e.g. for OneCycleLR)
            - unsupervised_loss_function: Loss function for unsupervised training.
            Can be a
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
        self.classifier = self.create_classifier(config, self.device)
        self.transform_model()

        self.logger.debug(self.model)
        self.logger.debug(self.classifier)

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

        self._loss_function_unsupervised = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )
        self._loss_function_args = config.get(
            "loss_function_args", dict()
        )
        if isinstance(self._loss_function_unsupervised, torch.nn.Module):
            self._loss_function_unsupervised_class = self._loss_function_unsupervised
            self._loss_function_unsupervised = \
                self._loss_function_unsupervised_class(**self._loss_function_args)
            self._loss_function_unsupervised.to(self.device)
            self.optimizer = self.create_optimizer(config,
                [self.model.parameters(),self._loss_function_unsupervised.parameters()])
        else:
            # Configure and create optimizer
            self.optimizer = self.create_optimizer(config, self.model.parameters())

        self._loss_function_supervised = config.get(
            "loss_function_supervised", torch.nn.functional.cross_entropy
        )
        self.optimizer_supervised = self.create_optimizer(config,
                                                          self.classifier.parameters())

        self.num_classes = config.get("num_classes", 1000)
        self.epochs = config.get("epochs", 1)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.batches_in_epoch_supervised = config.get("batches_in_epoch_supervised",
                                                      sys.maxsize)
        self.batches_in_epoch_val = config.get("batches_in_epoch_val", sys.maxsize)
        self.current_epoch = 0
        self.reset_classifier_on_validate = config.get(
            "reset_classifier_on_validate", False)

        # Configure data loaders
        self.create_loaders(config)
        self.total_batches = len(self.train_loader, )

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
        self.train_model_unsupervised = config.get("train_model_unsupervised_func",
                                            train_model_unsupervised)
        self.train_model_supervised = config.get("train_model_supervised_func",
                                                 train_model_supervised)
        self.evaluate_model = config.get("evaluate_model_func", evaluate_model)

        self.progress = config.get("progress", False)
        if self.logger.disabled:
            self.progress = False
        self.supervised_training_epochs_per_validation = config.get(
            "supervised_training_epochs_per_validation", 3)

    @classmethod
    def create_optimizer(cls, config, parameter_list):
        """
        Create optimizer from an experiment config. Override here is to provide
        parameters lists (so that loss module parameters can be included)

        :param optimizer_class: Callable or class to instantiate optimizer. Must return
                                object inherited from "torch.optim.Optimizer"
        :param optimizer_args: Arguments to pass to the optimizer.
        """
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        return optimizer_class(parameter_list, **optimizer_args)


    def create_loaders(self, config):
        self.supervised_loader = self.create_supervised_dataloader(config)
        self.unsupervised_loader = self.create_unsupervised_dataloader(config)
        self.val_loader = self.create_validation_dataloader(config)


    @classmethod
    def create_unsupervised_dataloader(cls, config, dataset=None):
        """
        Creates a dataloader for the unlabeled subset of the dataset.

        This method is a class method so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """
        if dataset is None:
            dataset = cls.load_dataset(config, split='unsupervised')

        sampler = cls.create_unsupervised_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("unsupervised_batch_size", 1),
            shuffle=sampler is None,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=config.get("unsupervised_loader_drop_last", True),
        )

    @classmethod
    def create_supervised_dataloader(cls, config, dataset=None):
        """
        Creates a dataloader for the supervised training data.

        This method is a class method so that it can be used directly by
        analysis
        tools, while also being easily overrideable.
        """
        if dataset is None:
            dataset = cls.load_dataset(config, split='supervised')

        sampler = cls.create_unsupervised_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("supervised", 1),
            shuffle=sampler is None,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=config.get("supervised_loader_drop_last", True),
        )

    @classmethod
    def create_validation_loader(cls, config, dataset=None):
        """
        Creates a dataloader for the validation data.

        This method is a class method so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """
        if dataset is None:
            dataset = cls.load_dataset(config, split='validation')

        sampler = cls.create_unsupervised_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("val_batch_size", 1),
            shuffle=sampler is None,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=config.get("validation_loader_drop_last", True),
        )


    @classmethod
    def load_dataset(cls, config, split="unsupervised"):
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        if split == "unsupervised":
            dataset_args = dict(config.get("train_dataset_args", {"split":"unlabeled"}))
        elif split == "supervised":
            dataset_args = dict(config.get("supervised_dataset_args",
                                           {"split":"train"}))
        elif split == "validation":
            dataset_args = dict(config.get("val_dataset_args", {"split":"test"}))
        return dataset_class(**dataset_args)

    @classmethod
    def create_classifier(cls, config, device):
        """
        Create `torch.nn.Module` classifier model from an experiment config
        :param config:
            - model_class: Model class. Must inherit from "torch.nn.Module"
            - model_args: model class arguments passed to the constructor
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
            model_class=config["classifier_class"],
            model_args=config.get("classifier_args", {}),
            init_batch_norm=config.get("init_batch_norm_classifier", False),
            device=device,
            checkpoint_file=config.get("checkpoint_file_classifier", None),
            load_checkpoint_args=config.get("load_checkpoint_args_classifier", {}),
        )

    def train_epoch(self):
        self.train_model_unsupervised

    def validate(self):
        if self.reset_classifier_on_validate:
            for layer in self.classifier.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        for _ in range(self.supervised_training_epochs_per_validation):
            self.train_model_supervised(
                self.model,
                self.classifier,
                self.supervised_loader,
                self.optimizer,
                self.device,
                criterion=self.error_loss,
                batches_in_supervised_epoch=self.batches_in_supervised_epoch,
            )

        return self.train_model_unsupervised(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            criterion=self.error_loss,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )


