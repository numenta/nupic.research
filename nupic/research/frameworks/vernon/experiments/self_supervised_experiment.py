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
import torch.nn.functional as F

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
        self.supervised_loader = None
        self.supervised_training_epochs_per_validation = 3

        #params for self supervised training
        """

        classifier_class
        classifier_args
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
            - supervised_dir: Dataset supervised data relative path
            - supervised_batch_size: Supervised training batch size
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
            - loss_function: Loss function for unsupervised training.
            Can be a
            - epochs: Number of epochs to train
            - batches_in_epoch: Number of batches per epoch.
                                Useful for debugging
            - batches_in_epoch_supervised: Number of batches per epoch in supervised
                                           training
            - batches_in_epoch_val: Number of batches per epoch in validation.
                                   Useful for debugging
            - mixed_precision: Whether or not to enable apex mixed precision
            - mixed_precision_args: apex mixed precision arguments.
                                    See "amp.initialize"
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
            - checkpoint_file
            - epochs_to_validate: list of epochs to run validate(). A -1 asks
                                  to run validate before any training occurs.
                                  Default: last three epochs.
            - launch_time: time the config was created (via time.time). Used to report
                           wall clock time until the first batch is done.
                           Default: time.time() in this setup_experiment().
        """

        super().setup_experiment(config)

        self.classifier = self.create_classifier(config, self.device)
        self.logger.debug(self.classifier)

        self._loss_function_supervised = config.get(
            "loss_function_supervised", torch.nn.functional.cross_entropy
        )
        self.optimizer_supervised = self.create_optimizer(config, self.classifier)

        self.batches_in_epoch_supervised = config.get("batches_in_epoch_supervised",
                                                      sys.maxsize)
        self.reset_classifier_on_validate = config.get(
            "reset_classifier_on_validate", False)

        # Set train and validate methods.
        self.train_model = config.get("train_model_func", train_model_unsupervised)
        self.train_model_supervised = config.get("train_model_supervised_func",
                                                 train_model_supervised)
        self.evaluate_model = config.get("evaluate_model_func", evaluate_model)

        self.supervised_training_epochs_per_validation = config.get(
            "supervised_training_epochs_per_validation", 3)


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

    #TODO: Ask Lucas about how to pass in datasets
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
        self.train_model(self.model,
                                      self.train_loader,
                                      self.optimizer,
                                      self.device,
                                      criterion=F.mse_loss,
                                      complexity_loss_fn=None,
                                      batches_in_epoch=sys.maxsize,
                                      pre_batch_callback=None,
                                      post_batch_callback=None,
                                      transform_to_device_fn=None,
                                      progress_bar=None,)

    def validate(self):
        if self.reset_classifier_on_validate:
            for layer in self.classifier.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        for _ in range(self.supervised_training_epochs_per_validation):
            self.train_model(
                self.model,
                self.classifier,
                self.supervised_loader,
                self.optimizer,
                self.device,
                criterion=self.error_loss,
                batches_in_supervised_epoch=self.batches_in_epoch_supervised,
            )

        return self.train_model_unsupervised(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            criterion=self.error_loss,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )


