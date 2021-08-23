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

import sys

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from nupic.research.frameworks.pytorch.self_supervised_utils import EncoderClassifier
from nupic.research.frameworks.vernon.experiments.supervised_experiment import (
    SupervisedExperiment,
)

__all__ = ["SelfSupervisedExperiment"]


class SelfSupervisedExperiment(SupervisedExperiment):
    """
    General experiment class used to train neural networks in self-supervised learning
    tasks.

    Self-supervised experiments have three important dataset splits: unsupervised,
    supervised, and validation. The unsupervised set consists of unlabeled data for
    representation learning, the supervised set consists of a typically smaller
    amount of labeled data for which to train a classifier, and the test set is used
    to evaluate the classifier.

    The model passed into this experiment under the name 'model' should be some kind
    of encoder which extends torch.nn.Module and implements both forward() and encode():

    class YourEncoderHere(torch.nn.Module):
            # this forward pass is used during unsupervised training
            def forward(self, x):
                x = ...
                return x

            # the encode function is used during supervised training
            def encode(self, x):
                encoded = ...
                return encoded

    The reason for this requirement is that the loss for the unsupervised training
    can (and usually does) operate on a different output than the representation
    that is used during supervised training. For example, an autoencoder's forward()
    should reconstruct its input, while its encode() function should yield the
    intermediate representation of a typically lower dimension.

    The model passed into this experiment under the name 'classifier' should take
    the encoding provided by .encode() as input and output a classification in its
    forward() method.

    The validation step trains a new classifier on top of the existing frozen model,
    and then proceeds to test this classifier on the test set. The number of
    supervised training epochs to train for each validation is given by
    supervised_training_epochs_per_validation.
    """

    def setup_experiment(self, config):
        """
        Configure the experiment for training.
        :param config: Dictionary containing the configuration parameters
            - classifier_config: Dictionary containing parameters for the classifier
                                 model
                - model_class: Classifier model class. Must inherit from
                                          "torch.nn.Module"
                - model_args: model class arguments passed to the
                                         constructor
                - init_batch_norm: Whether or not to Initialize running batch
                                              norm mean to 0.
                - checkpoint_file: if not None, will start from this model. The
                                              model must have the same model_args and
                                              model_class as the current experiment.
                - load_checkpoint_args_classifier: args to be passed to
                                                   `load_state_from_checkpoint`
                - optimizer_class: Optimizer class for the optimizer.
                                              Must inherit from "torch.optim.Optimizer"
                - optimizer_args: Optimizer class arguments passed to the
                                             constructor for the classifier optimizer
                - lr_scheduler_class: Learning rate scheduler class.
                                     Must inherit from "_LRScheduler"
                - lr_scheduler_args: Learning rate scheduler class class
                                                arguments passed to the constructor
                - lr_scheduler_step_every_batch: Whether to step the lr-scheduler
                                                 after every batch (e.g. for OneCycleLR)
                - loss_function: Loss function for supervised training.
                - reset_on_validate: optionally reset the parameters of the
                                     classifier during each validation training loop
            - supervised_batch_size: Supervised training batch size
            - batches_in_epoch_supervised: Number of batches per epoch in supervised
                                           training
            - supervised_loader_drop_last: Whether to skip last batch if it is
                                           smaller than the batch size
            - supervised_training_epochs_per_validation: number of epochs to train
                                                         the classifier for each
                                                         validation loop
            - reuse_unsupervised_dataset: if True, will reuse the unsupervised
                                          dataset during supervised training
        """

        super().setup_experiment(config)
        classifier_config = config.get("classifier_config", None)
        if classifier_config is None:
            raise ValueError("Must provide 'classifier_config' in config")

        self.encoder = self.model
        self.classifier = self.create_model(classifier_config, self.device)
        self.logger.debug(self.classifier)
        self.encoder_classifier = EncoderClassifier(self.encoder, self.classifier)
        self.encoder_classifier.to(self.device)

        self.encoder_optimizer = self.optimizer
        self.classifier_optimizer = self.create_optimizer(
            classifier_config, self.classifier
        )

        self._loss_function_unsupervised = self._loss_function = config.get(
            "loss_function_unsupervised", config.get("loss_function", F.mse_loss)
        )
        self._loss_function_supervised = classifier_config.get(
            "loss_function_supervised",
            classifier_config.get("loss_function", F.cross_entropy),
        )

        self.total_batches_unsupervised = self.total_batches
        self.total_batches_supervised = len(self.supervised_loader)
        self.batches_in_epoch_supervised = config.get(
            "batches_in_epoch_supervised", sys.maxsize
        )

        self.lr_scheduler_classifier = self.create_lr_scheduler(
            classifier_config, self.classifier_optimizer, self.total_batches_supervised
        )
        self.step_lr_every_batch_classifier = classifier_config.get(
            "lr_scheduler_step_every_batch", False
        )
        if isinstance(self.lr_scheduler_classifier, (OneCycleLR, ComposedLRScheduler)):
            self.step_lr_every_batch_classifier = True

        self.reset_classifier_on_validate = classifier_config.get(
            "reset_on_validate", False
        )

        self.supervised_training_epochs_per_validation = config.get(
            "supervised_training_epochs_per_validation", 3
        )

    def create_loaders(self, config):

        unsupervised_data = self.load_dataset(config, dataset_type="unsupervised")
        if config.get("reuse_unsupervised_dataset", False):
            supervised_data = unsupervised_data
        else:
            supervised_data = self.load_dataset(config, dataset_type="supervised")
        val_data = self.load_dataset(config, dataset_type="validation")

        self.unsupervised_loader = (
            self.train_loader
        ) = self.create_unsupervised_dataloader(config, unsupervised_data)

        self.supervised_loader = self.create_supervised_dataloader(
            config, supervised_data
        )
        self.val_loader = self.create_validation_dataloader(config, val_data)

    @classmethod
    def create_unsupervised_dataloader(cls, config, dataset):
        """
        Creates a dataloader for the unlabeled subset of the dataset.

        This method is a class method so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """

        sampler = cls.create_unsupervised_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get(
                "batch_size", config.get("unsupervised_batch_size", 1)
            ),
            shuffle=sampler is None,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=config.get(
                "train_loader_drop_last",
                config.get("unsupervised_loader_drop_last", True),
            ),
        )

    @classmethod
    def create_supervised_dataloader(cls, config, dataset):
        """
        Creates a dataloader for the supervised training data.

        This method is a class method so that it can be used directly by
        analysis
        tools, while also being easily overrideable.
        """

        sampler = cls.create_supervised_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size_supervised", 1),
            shuffle=sampler is None,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=config.get("supervised_loader_drop_last", True),
        )

    @classmethod
    def create_validation_dataloader(cls, config, dataset):
        """
        Creates a dataloader for the validation data.

        This method is a class method so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """
        sampler = cls.create_validation_sampler(config, dataset)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("val_batch_size", config.get("batch_size", 1)),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def create_unsupervised_sampler(cls, config, dataset):
        return None

    @classmethod
    def create_supervised_sampler(cls, config, dataset):
        return None

    @classmethod
    def create_validation_sampler(cls, config, dataset):
        return None

    @classmethod
    def load_dataset(cls, config, train=True, dataset_type="unsupervised"):
        """
        Loads one of three types of datasets: unsupervised, supervised, and validation

        The dataset_class argument in the config file should either be a single
        dataset class, or a dict with the following structure:
        {
         "unsupervised": unsupervised_dataset_class
         "supervised": supervised_dataset_class
         "validation": validation_dataset_class
        }

        Similarly, the dataset_args argument in the config should be a dict of dicts
        with keys corresponding to the different dataset types as described above.
        """
        if not train:
            dataset_type = "validation"

        dataset_class = config.get("dataset_class", None)
        dataset_args = config.get("dataset_args", {})

        if isinstance(dataset_class, dict):
            dataset_class = dataset_class.get(dataset_type, None)

        dataset_args = dict(dataset_args.get(dataset_type, {}))

        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        return dataset_class(**dataset_args)

    def unsupervised_loss(self, output, target, reduction="mean"):
        """
        The error loss component of the loss function.
        """
        return self._loss_function_unsupervised(output, reduction=reduction)

    def supervised_loss(self, output, target, reduction="mean"):
        """
        The error loss component of the loss function.
        """
        return self._loss_function_supervised(output, target, reduction=reduction)

    def encoder_classifier_complexity_loss(self, model):
        """
        The encoder complexity component of the loss function.
        """
        pass

    def pre_batch_supervised(self, model, batch_idx):
        pass

    def post_batch_supervised(
        self, model, error_loss, complexity_loss, batch_idx, num_images, time_string
    ):
        # Update 1cycle learning rate after every batch
        if self.step_lr_every_batch_classifier:
            self.lr_scheduler_classifier.step()

    def post_optimizer_step_supervised(self, model):
        pass

    def post_batch_wrapper_supervised(self, **kwargs):
        self.post_optimizer_step_supervised(self.encoder_classifier)
        self.post_batch_supervised(**kwargs)

    def transform_data_to_device_unsupervised(self, data, target, device, non_blocking):
        """
        This provides an extensibility point for performing any final
        transformations on the data or targets. This will suffice for most
        self-supervised experiments. The first copy of data gets passed through the
        model, and the second copy is optionally used in the loss function.
        """
        data = data.to(self.device, non_blocking=non_blocking)
        return data, data

    def train_epoch(self):
        """
        Trains the encoder in a self-supervised fashion. Train epoch will call the
        .forward() method on the encoder, and the optimizer being used here is also
        specific to the encoder. The classifier will not be affected by this method.

        """
        self.train_model(
            model=self.encoder,
            loader=self.unsupervised_loader,
            optimizer=self.encoder_optimizer,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch,
            pre_batch_callback=self.pre_batch,
            post_batch_callback=self.post_batch_wrapper,
            transform_to_device_fn=self.transform_data_to_device_unsupervised,
        )

    def validate(self):
        """
        This method has three parts.

        First, it optionally resets the classifier if specified in the config. This
        can be used to ensure "fairness" between validation trials. Defaults to
        false, which should not cause problems so long as
        supervised_epochs_per_validation is enough for the classifier to converge.

        Second, this method trains the classifier in a supervised fashion for a
        specified number of epochs. self.encoder_classifier refers to the entire
        EncoderClassifier model, whose .forward() method is as follows:

            def forward(self, x):
                with torch.no_grad():
                    encoded = self.encoder.encode(x)
                out = self.classifier(encoded)
                return out

        The forward pass involves the encoder and classifier, but there are no
        gradients computed for the encode() pass, and the optimizer being used here,
        self.classifier_optimizer, only updates the parameters of the classifier.

        Last, this method validates the whole model using evaluate_model(),
        again calling forward on the EncoderClassifier model and evaludating on the
        validation set.


        """
        if self.reset_classifier_on_validate:
            for layer in self.classifier.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        for _ in range(self.supervised_training_epochs_per_validation):
            self.train_model(
                model=self.encoder_classifier,
                loader=self.supervised_loader,
                optimizer=self.classifier_optimizer,
                device=self.device,
                criterion=self.supervised_loss,
                complexity_loss_fn=self.encoder_classifier_complexity_loss,
                batches_in_epoch=self.batches_in_epoch_supervised,
                pre_batch_callback=self.pre_batch_supervised,
                post_batch_callback=self.post_batch_wrapper_supervised,
                transform_to_device_fn=self.transform_data_to_device,
            )

        return self.evaluate_model(
            model=self.encoder_classifier,
            loader=self.val_loader,
            device=self.device,
            criterion=self.supervised_loss,
            complexity_loss_fn=self.encoder_classifier_complexity_loss,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "SelfSupervisedExperiment"
        # Extended methods
        eo["setup_experiment"].append(exp + ".setup_experiment")

        eo.update(
            # New methods
            create_unsupervised_dataloader=[exp + ".create_unsupervised_dataloader"],
            create_unsupervised_sampler=[exp + ".create_unsupervised_sampler"],
            create_supervised_dataloader=[exp + ".create_supervised_dataloader"],
            create_supervised_sampler=[exp + ".create_supervised_sampler"],
            unsupervised_loss=[exp + ".unsupervised_loss"],
            supervised_loss=[exp + ".supervised_loss"],
            encoder_classifier_complexity_loss=[
                exp + ".encoder_classifier_complexity_loss"
            ],
            pre_batch_supervised=[exp + ".pre_batch_supervised"],
            post_batch_supervised=[exp + ".post_batch_supervised"],
            post_optimizer_step_supervised=[exp + ".post_optimizer_step_supervised"],
            post_batch_wrapper_supervised=[exp + ".post_batch_wrapper_supervised"],
            transform_data_to_device_unsupervised=[
                exp + ".transform_data_to_device_unsupervised"
            ],
            # Overwritten methods
            validate=[exp + ".validate"],
            create_loaders=[exp + ".create_loaders"],
            create_validation_dataloader=[exp + ".create_validation_dataloader"],
            create_validation_sampler=[exp + ".create_validation_sampler"],
            train_epoch=[exp + ".train_epoch"],
            transform_data_to_device=[exp + ".transform_data_to_device"],
            load_dataset=[exp + ".load_dataset"],
        )
        return eo
