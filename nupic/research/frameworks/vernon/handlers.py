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
import copy
import io
import logging
import math
import sys
import time
from collections import defaultdict
from pprint import pformat

import numpy as np
import torch
import torch.distributed as dist
from torch import multiprocessing
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

from nupic.research.frameworks.continual_learning.maml_utils import clone_model
from nupic.research.frameworks.pytorch import datasets
from nupic.research.frameworks.pytorch.dataset_utils.samplers import (
    TaskDistributedSampler,
    TaskRandomSampler,
)
from nupic.research.frameworks.pytorch.distributed_sampler import (
    UnpaddedDistributedSampler,
)
from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from nupic.research.frameworks.pytorch.model_utils import (
    aggregate_eval_results,
    deserialize_state_dict,
    evaluate_model,
    serialize_state_dict,
    set_random_seed,
    train_model,
)
from nupic.research.frameworks.vernon.evaluation_metrics import ContinualLearningMetrics
from nupic.research.frameworks.vernon.experiment_utils import (
    create_lr_scheduler,
    get_free_port,
    get_node_ip_address,
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
    "ImagenetExperiment",
    "ContinualLearningExperiment",
    "MetaContinualLearningExperiment",
]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class SupervisedExperiment:
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
        self.batch_size = 1
        self.epochs = 1
        self.distributed = False
        self.mixed_precision = False
        self.rank = 0
        self.total_batches = 0
        self.progress = False
        self.logger = None
        self.seed = 42
        self.launch_time = 0
        self.epochs_to_validate = []
        self.current_epoch = 0

    def setup_experiment(self, config):
        """
        Configure the experiment for training

        :param config: Dictionary containing the configuration parameters

            - distributed: Whether or not to use Pytorch Distributed training
            - backend: Pytorch Distributed backend ("nccl", "gloo")
                    Default: nccl
            - world_size: Total number of processes participating
            - rank: Rank of the current process
            - data: Dataset path
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
            - batch_norm_weight_decay: Whether or not to apply weight decay to
                                       batch norm modules parameters
                                       See https://arxiv.org/abs/1807.11205
            - bias_weight_decay: Whether or not to apply weight decay to
                                       bias parameters
            - lr_scheduler_class: Learning rate scheduler class.
                                 Must inherit from "_LRScheduler"
            - lr_scheduler_args: Learning rate scheduler class class arguments
                                 passed to the constructor
            - lr_scheduler_step_every_batch: Whether to step the lr-scheduler after
                                             after every batch (e.g. for OneCycleLR)
            - loss_function: Loss function. See "torch.nn.functional"
            - local_dir: Results path
            - logdir: Directory generated by Ray Tune for this Trial
            - epochs: Number of epochs to train
            - batches_in_epoch: Number of batches per epoch.
                                Useful for debugging
            - batches_in_epoch_val: Number of batches per epoch in validation.
                                   Useful for debugging
            - log_timestep_freq: Configures mixins and subclasses that log every
                                 timestep to only log every nth timestep (in
                                 addition to the final timestep of each epoch).
                                 Set to 0 to log only at the end of each epoch.
            - progress: Show progress during training
            - name: Experiment name. Used as logger name
            - log_level: Python Logging level
            - log_format: Python Logging format
            - seed: the seed to be used for pytorch, python, and numpy
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
            - checkpoint_at_init: boolean argument for whether to create a checkpoint
                                  of the initialized model. this differs from
                                  `checkpoint_at_start` for which the checkpoint occurs
                                  after the first epoch of training as opposed to
                                  before it
            - epochs_to_validate: list of epochs to run validate(). A -1 asks
                                  to run validate before any training occurs.
                                  Default: last three epochs.
            - extra_validations_per_epoch: number of additional validations to
                                           perform mid-epoch. Additional
                                           validations are distributed evenly
                                           across training batches.
            - launch_time: time the config was created (via time.time). Used to report
                           wall clock time until the first batch is done.
                           Default: time.time() in this setup_experiment().
        """
        # Configure logging related stuff
        log_format = config.get("log_format", logging.BASIC_FORMAT)
        log_level = getattr(logging, config.get("log_level", "INFO").upper())
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger(config.get("name", type(self).__name__))
        self.logger.setLevel(log_level)
        self.logger.addHandler(console)
        self.progress = config.get("progress", False)
        self.launch_time = config.get("launch_time", time.time())
        self.logdir = config.get("logdir", None)

        # Configure seed
        self.seed = config.get("seed", self.seed)
        set_random_seed(self.seed, False)

        # Configure distribute pytorch
        self.distributed = config.get("distributed", False)
        self.rank = config.get("rank", 0)

        if self.rank == 0:
            self.logger.info(
                f"Execution order: {pformat(self.get_execution_order())}")

        if self.distributed:
            dist_url = config.get("dist_url", "tcp://127.0.0.1:54321")
            backend = config.get("backend", "nccl")
            world_size = config.get("world_size", 1)
            dist.init_process_group(
                backend=backend,
                init_method=dist_url,
                rank=self.rank,
                world_size=world_size,
            )
            # Only enable logs from first process
            self.logger.disabled = self.rank != 0
            self.progress = self.progress and self.rank == 0

        # Configure model
        self.device = config.get("device", self.device)
        self.model = self.create_model(config, self.device)
        self.transform_model()

        if self.rank == 0:
            self.logger.debug(self.model)

        # Configure optimizer
        group_decay, group_no_decay = [], []
        for module in self.model.modules():
            for name, param in module.named_parameters(recurse=False):
                if self.should_decay_parameter(module, name, param, config):
                    group_decay.append(param)
                else:
                    group_no_decay.append(param)

        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        self.optimizer = optimizer_class([dict(params=group_decay),
                                          dict(params=group_no_decay,
                                               weight_decay=0.)],
                                         **optimizer_args)

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

        # Apply DistributedDataParallel after all other model mutations
        if self.distributed:
            self.model = DistributedDataParallel(self.model)
        else:
            self.model = DataParallel(self.model)

        self._loss_function = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )

        self.num_classes = config.get("num_classes", 1000)
        self.epochs = config.get("epochs", 1)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.batches_in_epoch_val = config.get("batches_in_epoch_val", sys.maxsize)
        self.current_epoch = 0

        # Get initial batch size
        self.batch_size = config.get("batch_size", 1)

        # CUDA runtime does not support the fork start method.
        # See https://pytorch.org/docs/stable/notes/multiprocessing.html
        multiprocessing.set_start_method("spawn", force=True)

        # Configure data loaders
        self.train_loader, self.val_loader = self.create_loaders(config)
        self.total_batches = len(self.train_loader,)

        self.epochs_to_validate = config.get("epochs_to_validate",
                                             range(self.epochs - 3,
                                                   self.epochs + 1))

        extra_validations = config.get("extra_validations_per_epoch", 0)
        batches_to_validate = np.linspace(
            min(self.total_batches, self.batches_in_epoch),
            0,
            1 + extra_validations,
            endpoint=False
        )[::-1].round().astype("int").tolist()
        self.additional_batches_to_validate = batches_to_validate[:-1]
        if extra_validations > 0:
            self.logger.info(
                f"Extra validations per epoch: {extra_validations}, "
                f"batch indices: {self.additional_batches_to_validate}")

        # Used for logging. Conceptually, it is a version number for the model's
        # parameters. By default, this is the elapsed number of batches that the
        # model has been trained on. Experiments may also increment this on
        # other events like model prunings. When validation is performed after a
        # training batch, the validation results are assigned to the next
        # timestep after that training batch, since it was performed on the
        # subsequent version of the parameters.
        self.current_timestep = 0
        self.log_timestep_freq = config.get("log_timestep_freq", 1)

        # A list of [(timestep, result), ...] for the current epoch.
        self.extra_val_results = []

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

    @classmethod
    def create_model(cls, config, device):
        """
        Create imagenet model from an ImagenetExperiment config
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
    def create_lr_scheduler(cls, config, optimizer, total_batches):
        """
        Create lr scheduler from an ImagenetExperiment config
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

    @classmethod
    def create_loaders(cls, config):
        """Create train and val dataloaders."""

        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        dataset_args = config.get("dataset_args", {})
        dataset_args.update(train=True)
        train_set = dataset_class(**dataset_args)
        dataset_args.update(train=False)
        val_set = dataset_class(**dataset_args)

        train_loader = cls.create_train_dataloader(train_set, config)
        val_loader = cls.create_validation_dataloader(val_set, config)

        return train_loader, val_loader

    @classmethod
    def create_train_sampler(cls, dataset, config):
        if config.get("distributed", False):
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
        return sampler

    @classmethod
    def create_train_dataloader(cls, dataset, config):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """

        sampler = cls.create_train_sampler(dataset, config)
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
    def create_validation_sampler(cls, dataset, config):
        if config.get("distributed", False):
            sampler = UnpaddedDistributedSampler(dataset, shuffle=False)
        else:
            sampler = None
        return sampler

    @classmethod
    def create_validation_dataloader(cls, dataset, config):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """

        sampler = cls.create_validation_sampler(dataset, config)
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

    def should_decay_parameter(self, module, parameter_name, parameter, config):
        if isinstance(module, _BatchNorm):
            return config.get("batch_norm_weight_decay", True)
        elif parameter_name == "bias":
            return config.get("bias_weight_decay", True)
        else:
            return True

    def pre_experiment(self):
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
        timestep_begin = self.current_timestep
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
            timestep_begin=timestep_begin,
            timestep_end=self.current_timestep,
            learning_rate=self.get_lr()[0],
            extra_val_results=self.extra_val_results,
        )

        if self.rank == 0:
            self.logger.debug("validate time: %s", time.time() - t1)
            self.logger.debug("---------- End of run epoch ------------")
            self.logger.debug("")

        self.extra_val_results = []
        self.current_epoch += 1
        return ret

    def pre_epoch(self):
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)

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
            if self.distributed:
                # Compute actual batch size from distributed sampler
                total_batches *= self.train_loader.sampler.num_replicas
                current_batch *= self.train_loader.sampler.num_replicas
            self.logger.debug("End of batch for rank: %s. Epoch: %s, Batch: %s/%s, "
                              "loss: %s, Learning rate: %s num_images: %s",
                              self.rank, self.current_epoch, current_batch,
                              total_batches, error_loss, self.get_lr(),
                              num_images)
            self.logger.debug("Timing: %s", time_string)

    def post_batch_wrapper(self, model, error_loss, complexity_loss, batch_idx,
                           *args, **kwargs):
        """
        Perform the post_batch updates, then maybe validate.

        This method exists because post_batch is designed to be overridden, and
        validation needs to wait until after all post_batch overrides have run.
        """
        self.post_batch(model, error_loss, complexity_loss, batch_idx,
                        *args, **kwargs)
        self.current_timestep += 1
        validate = (batch_idx in self.additional_batches_to_validate
                    and self.current_epoch in self.epochs_to_validate)
        if validate:
            result = self.validate()
            self.extra_val_results.append(
                (self.current_timestep, result)
            )
            self.model.train()

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
    def aggregate_results(cls, results):
        """
        Aggregate multiple processes' "run_epoch" results into a single result.

        :param results:
            A list of return values from run_epoch from different processes.
        :type results: list

        :return:
            A single result dict with results aggregated.
        :rtype: dict
        """
        ret = cls.aggregate_validation_results(results)

        extra_val_aggregated = []
        for i in range(len(ret["extra_val_results"])):
            timestep = ret["extra_val_results"][i][0]
            val_results = [process_result["extra_val_results"][i][1]
                           for process_result in results]
            extra_val_aggregated.append(
                (timestep, aggregate_eval_results(val_results))
            )
        ret["extra_val_results"] = extra_val_aggregated

        return ret

    @classmethod
    def aggregate_validation_results(cls, results):
        """
        Aggregate multiple processes' "validate" results into a single result.

        This method exists separately from "aggregate_results" to support
        running validation outside of "run_epoch" and aggregating those results
        without causing error. Subclasses / mixins implementing
        "aggregate_results" may expect all results to have the extra data
        appended during run_epoch.

        :param results:
            A list of return values from validate from different processes.
        :type results: list

        :return:
            A single result dict with results aggregated.
        :rtype: dict
        """
        result = copy.deepcopy(results[0])
        result.update(aggregate_eval_results(results))
        return result

    @classmethod
    def aggregate_pre_experiment_results(cls, results):
        if all(results):
            return cls.aggregate_validation_results(results)

    @classmethod
    def get_printable_result(cls, result):
        """
        Return a stripped down version of result that has its large data structures
        removed so that the result can be printed to the console.
        """
        keys = ["total_correct", "total_tested", "mean_loss", "mean_accuracy",
                "learning_rate"]
        return {key: result[key]
                for key in keys
                if key in result}

    @classmethod
    def expand_result_to_time_series(cls, result, config):
        """
        Given the result of a run_epoch call, returns a mapping from timesteps to
        results. The mapping is stored as a dict so that subclasses and mixins
        can easily add data to it.

        Result keys are converted from Ray Tune requirements to better names,
        and the keys are filtered to those that make useful charts.

        :return: defaultdict mapping timesteps to result dicts
        """
        result_by_timestep = defaultdict(dict)

        k_mapping = {
            "mean_loss": "validation_loss",
            "mean_accuracy": "validation_accuracy",
            "learning_rate": "learning_rate",
            "complexity_loss": "complexity_loss",
        }
        val_results = (result["extra_val_results"]
                       + [(result["timestep_end"], result)])
        for timestep, val_result in val_results:
            result_by_timestep[timestep].update({
                k2: val_result[k1]
                for k1, k2 in k_mapping.items()
                if k1 in val_result
            })

        return result_by_timestep

    def get_state(self):
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        state = {
            "current_epoch": self.current_epoch,
            "current_timestep": self.current_timestep,
        }

        # Save state into a byte array to avoid ray's GPU serialization issues
        # See https://github.com/ray-project/ray/issues/5519
        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.model.module.state_dict())
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
            state_dict = get_compatible_state_dict(state_dict, self.model.module)
            self.model.module.load_state_dict(state_dict)

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

        if "current_timestep" in state:
            self.current_timestep = state["current_timestep"]
        else:
            self.current_timestep = self.total_batches * self.current_epoch

    def stop_experiment(self):
        if self.distributed:
            dist.destroy_process_group()

    def should_stop(self):
        """
        Whether or not the experiment should stop. Usually determined by the
        number of epochs but customizable to any other stopping criteria
        """
        return self.current_epoch >= self.epochs

    def should_log_batch(self, train_batch_idx):
        """
        Returns true if the current timestep should be logged, either because it's a
        logged timestep or the final training batch of an epoch.
        """
        return (train_batch_idx == self.total_batches - 1
                or (self.log_timestep_freq > 0
                    and (self.current_timestep % self.log_timestep_freq) == 0))

    @classmethod
    def get_recorded_timesteps(cls, result, config):
        """
        Given an epoch result dict and config, returns a list of timestep numbers
        that are supposed to be logged for that epoch.
        """
        log_timestep_freq = config.get("log_timestep_freq", 1)
        timestep_end = result["timestep_end"]
        if log_timestep_freq == 0:
            ret = [timestep_end - 1]
        else:
            # Find first logged timestep in range
            logged_begin = int(math.ceil(result["timestep_begin"]
                                         / log_timestep_freq)
                               * log_timestep_freq)

            ret = list(range(logged_begin, timestep_end, log_timestep_freq))

            last_batch_timestep = timestep_end - 1
            if last_batch_timestep % log_timestep_freq != 0:
                ret.append(last_batch_timestep)

        return ret

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

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return get_node_ip_address()

    def get_free_port(self):
        """Returns free TCP port in the current node"""
        return get_free_port()

    @classmethod
    def get_execution_order(cls):
        exp = "SupervisedExperiment"
        return dict(
            setup_experiment=[exp + ".setup_experiment"],
            create_model=[exp + ".create_model"],
            transform_model=[exp + ".transform_model"],
            create_loaders=[exp + ".create_loaders"],
            create_train_dataloader=[exp + ".create_train_dataloader"],
            create_train_sampler=[exp + ".create_train_sampler"],
            create_validation_dataloader=[exp + ".create_validation_dataloader"],
            create_validation_sampler=[exp + ".create_validation_sampler"],
            create_lr_scheduler=[exp + ".create_lr_scheduler"],
            pre_experiment=[exp + ".pre_experiment"],
            validate=[exp + ".validate"],
            train_epoch=[exp + ".train_epoch"],
            run_epoch=[exp + ".run_epoch"],
            pre_epoch=[exp + ".pre_epoch"],
            post_epoch=[exp + ".post_epoch"],
            pre_batch=[exp + ".pre_batch"],
            post_batch=[exp + ".post_batch"],
            error_loss=[exp + ".error_loss"],
            complexity_loss=[exp + ".complexity_loss"],
            should_decay_parameter=[exp + ".should_decay_parameter"],
            transform_data_to_device=[exp + ".transform_data_to_device"],
            aggregate_results=[exp + ".aggregate_results"],
            aggregate_validation_results=[exp + ".aggregate_validation_results"],
            aggregate_pre_experiment_results=[
                exp + ".aggregate_pre_experiment_results"
            ],
            get_printable_result=[exp + ".get_printable_result"],
            expand_result_to_time_series=[exp + ": validation results"],
            stop_experiment=[exp + ".stop_experiment"]
        )


class ContinualLearningExperiment(ContinualLearningMetrics, SupervisedExperiment):

    def setup_experiment(self, config):

        super().setup_experiment(config)
        # Override epochs to validate to not validate within the inner loop over epochs
        self.epochs_to_validate = []

        self.current_task = 0

        # Defines how many classes should exist per task
        self.num_tasks = config.get("num_tasks", 1)

        self.num_classes = config.get("num_classes", None)
        assert self.num_classes is not None, "num_classes should be defined"

        self.num_classes_per_task = math.floor(self.num_classes / self.num_tasks)

        # Applying target transform depending on type of CL task
        # Task - we know the task, so the network is multihead
        # Class - we don't know the task, network has as many heads as classes
        self.cl_experiment_type = config.get("cl_experiment_type", "class")
        if self.cl_experiment_type == "task":
            self.logger.info("Overriding target transform")
            self.dataset_args["target_transform"] = (
                transforms.Lambda(lambda y: y % self.num_classes_per_task)
            )

        # Whitelist evaluation metrics
        self.evaluation_metrics = config.get(
            "evaluation_metrics", ["eval_all_visited_tasks"]
        )
        for metric in self.evaluation_metrics:
            if not hasattr(self, metric):
                raise ValueError(f"Metric {metric} not available.")

    def should_stop(self):
        """
        Whether or not the experiment should stop. Usually determined by the
        number of epochs but customizable to any other stopping criteria
        """
        return self.current_task >= self.num_tasks

    def run_task(self):
        """Run outer loop over tasks"""
        # configure the sampler to load only samples from current task
        self.logger.info("Training...")
        self.train_loader.sampler.set_active_tasks(self.current_task)

        # Run epochs, inner loop
        # TODO: return the results from run_epoch
        for _ in range(self.epochs):
            self.run_epoch()

        ret = {}
        for metric in self.evaluation_metrics:
            eval_function = getattr(self, metric)
            temp_ret = eval_function()
            self.logger.debug(temp_ret)
            for k, v in temp_ret.items():
                ret[f"{metric}__{k}"] = v

        # TODO: Fix aggregate results function to not
        # require these parameters, in order to be more flexible
        ret.update(
            timestep_begin=0,
            timestep_end=1,
            learning_rate=self.get_lr()[0],
            extra_val_results=[],
        )

        self.current_task += 1
        return ret

    @classmethod
    def aggregate_results(cls, results):
        """Run validation in single GPU"""
        return results[0]

    @classmethod
    def create_train_sampler(cls, dataset, config):
        return cls.create_task_sampler(dataset, config, train=True)

    @classmethod
    def create_validation_sampler(cls, dataset, config):
        return cls.create_task_sampler(dataset, config, train=False)

    @classmethod
    def create_task_sampler(cls, dataset, config, train):
        # Assume dataloaders are already created
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(dataset):
            class_indices[target].append(idx)

        # Defines how many classes should exist per task
        num_tasks = config.get("num_tasks", 1)
        num_classes = config.get("num_classes", None)
        assert num_classes is not None, "num_classes should be defined"
        num_classes_per_task = math.floor(num_classes / num_tasks)

        task_indices = defaultdict(list)
        for i in range(num_tasks):
            for j in range(num_classes_per_task):
                task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])

        # Change the sampler in the train loader
        distributed = config.get("distributed", False)
        if distributed and train:
            sampler = TaskDistributedSampler(
                dataset,
                task_indices
            )
        else:
            # TODO: implement a TaskDistributedUnpaddedSampler
            # mplement the aggregate results
            # after above are implemented, remove this if else
            sampler = TaskRandomSampler(task_indices)

        return sampler

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "ContinualLearningExperiment"
        eo["create_train_sampler"] = [exp + ".create_train_sampler"]
        eo["create_validation_sampler"] = [exp + ".create_validation_sampler"]
        eo["create_task_sampler"] = [exp + ".create_task_sampler"]
        eo["run_task"] = [exp + ".run_task"]
        eo["should_stop"] = [exp + ".should_stop"]
        eo["aggregate_results"] = [exp + ".aggregate_results"]
        return eo


class MetaContinualLearningExperiment(SupervisedExperiment):
    """
    Experiment class for meta-continual learning (based on OML and ANML meta-continual
    learning setups). There are 2 main phases in a meta-continual learning setup:

        - meta-training: Meta-learning representations for continual learning over all
        tasks
        - meta-testing: Montinual learning of new tasks and model evaluation

    where learning a "task" corresponds to learning a single classification label. More
    specifically, each phase is divided into its own training and testing phase, hence
    we have 4 such phases.

        - meta-training training: Train the inner loop learner (i.e., slow parameters)
        for a specific task
        - meta-training testing: Train the outer loop learner (i.e., fast parameters)
        by minimizing the test loss

        - meta-testing training: Train the inner loop learner continually on a sequence
        of holdout tasks
        - meta-testing testing: Evaluate the inner loop learner on the same tasks

    The parameters for a model used in a meta-continual learning setup are broken down
    into 2 groups:

        - fast parameters: Used to update slow parameters via online learning during
        meta-training training, and updated along with slow parameters during
        meta-training testing
        - slow parameters: Updated during the outer loop (meta-training testing)
    """

    def setup_experiment(self, config):
        """
        Configure the experiment for training

        :param config: Dictionary containing the configuration parameters, most of
        which are defined in SupervisedExperiment, but some of which are specific to
        MetaContinualLearningExperiment

            - experiment_class: Class used to run experiments, specify
            `MetaContinualLearningExperiment` for meta-continual learning
            - adaptation_lr: Learning rate used to update the fast parameters during
            the inner loop of the meta-training phase
            - tasks_per_epoch: Number of different classes used for training during the
            execution of the inner loop
            - slow_batch_size: Number of examples in a single batch used to update the
            slow parameters during the meta-training testing phase, where the examples
            are sampled from tasks_per_epoch difference tasks
            - replay_batch_size: Number of examples in a single batch sampled from all
            data, also used to train the slow parameters (the replay batch is used to
            sample examples to update the slow parameters during meta-training testing
            to prevent the learner from forgetting other tasks)
        """
        if "num_classes" not in config["model_args"]:
            # manually set `num_classes` in `model_args`
            num_classes = config["num_classes"]
            config["model_args"]["num_classes"] = num_classes

        super().setup_experiment(config)
        self.train_slow_loader = self.train_loader
        self.train_fast_loader, self.val_fast_loader, self.train_replay_loader = \
            self.create_fast_slow_loaders(config)

        self.epochs_to_validate = []
        self.tasks_per_epoch = config.get("tasks_per_epoch", 1)
        self.num_classes = min(
            config.get("num_classes", 50),
            self.train_fast_loader.sampler.num_classes
        )

        self.adaptation_lr = config.get("adaptation_lr", 0.03)

        self.batch_size = config.get("batch_size", 5)
        self.val_batch_size = config.get("val_batch_size", 15)
        self.slow_batch_size = config.get("slow_batch_size", 64)
        self.replay_batch_size = config.get("replay_batch_size", 64)

    @classmethod
    def create_loaders(cls, config):
        """Create train and val dataloaders."""

        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        # Create datasets -> same, only initialize two of them
        dataset_args = config.get("dataset_args", {})
        dataset_args.update(train=True)
        main_set = dataset_class(**dataset_args)

        # All loaders share tasks and dataset, but different indices and batch sizes
        train_slow_loader = cls.create_slow_train_dataloader(main_set, config)

        # For pre/post epoch and batch processing slow loader is equiv to train loader

        # note: modify return values? right now, it's returning the torch dataloader
        # that will be stored in self.train_loader, and None (since there's no need
        # for an eval loader)
        return train_slow_loader, None

    @classmethod
    def create_fast_slow_loaders(cls, config):
        """ Creates and returns

        - a torch dataloader for inner loop updates to fast parameters  (i.e.,
        meta-training training)
        - a torch dataloader for evaluating the inner loop learner
        - a torch dataloader for outer loop updates to slow parameters
        """
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        # Create datasets -> same, only initialize two of them
        dataset_args = config.get("dataset_args", {})
        dataset_args.update(train=True)
        main_set = dataset_class(**dataset_args)

        train_fast_loader = cls.create_train_dataloader(main_set, config)
        val_fast_loader = cls.create_validation_dataloader(main_set, config)
        train_replay_loader = cls.create_replay_dataloader(main_set, config)

        return train_fast_loader, val_fast_loader, train_replay_loader

    @classmethod
    def create_train_sampler(cls, dataset, config):
        return cls.create_task_sampler(dataset, config, mode="train")

    @classmethod
    def create_validation_sampler(cls, dataset, config):
        return cls.create_task_sampler(dataset, config, mode="test")

    @classmethod
    def create_replay_sampler(cls, dataset, config):
        return cls.create_task_sampler(dataset, config, mode="replay")

    @classmethod
    def create_task_sampler(cls, dataset, config, mode="replay"):
        """In meta continuous learning paradigm, one task equals one class"""
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(dataset):
            class_indices[target].append(idx)

        if mode == "train":
            fast_sample_size = config.get("fast_sample_size", 5)
            for c in class_indices:
                class_indices[c] = class_indices[c][:fast_sample_size]
        elif mode == "test":
            slow_sample_size = config.get("slow_sample_size", 15)
            for c in class_indices:
                class_indices[c] = class_indices[c][slow_sample_size:]
        elif mode == "replay":
            pass

        distributed = config.get("distributed", False)
        if distributed:
            sampler = TaskDistributedSampler(
                dataset,
                class_indices
            )
        else:
            sampler = TaskRandomSampler(class_indices)

        return sampler

    @classmethod
    def create_slow_train_dataloader(cls, dataset, config):
        sampler = cls.create_validation_sampler(dataset, config)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("slow_batch_size", 64),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def create_replay_dataloader(cls, dataset, config):
        sampler = cls.create_replay_sampler(dataset, config)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("replay_batch_size", 64),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    def run_epoch(self):

        self.pre_epoch()

        timestep_begin = self.current_timestep
        self.optimizer.zero_grad()

        # Clone model - clone fast params and the slow params. The latter will be frozen
        cloned_adaptation_net = self.clone_model()

        tasks_train = np.random.choice(
            self.num_classes, self.tasks_per_epoch, replace=False
        )
        for task in tasks_train:
            self.run_task(task, cloned_adaptation_net)

        # Concatenate slow and replay sets
        self.train_slow_loader.sampler.set_active_tasks(tasks_train)
        slow_data, slow_target = next(iter(self.train_slow_loader))
        replay_data, replay_target = next(iter(self.train_replay_loader))

        slow_data = torch.cat([slow_data, replay_data]).to(self.device)
        slow_target = torch.cat([slow_target, replay_target]).to(self.device)

        # Take step for outer loop. This will backprop through to the original
        # slow and fast params.
        output = cloned_adaptation_net(slow_data)
        output = self.model(slow_data)
        loss = self._loss_function(output, slow_target)
        loss.backward()

        self.optimizer.step()

        # Report statistics for the outer loop
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(slow_target.view_as(pred)).sum().item()
        total = output.shape[0]
        results = {
            "total_correct": correct,
            "total_tested": total,
            "mean_loss": loss.item(),
            "mean_accuracy": correct / total if total > 0 else 0,
        }
        self.logger.debug(results)

        results.update(
            timestep_begin=timestep_begin,
            timestep_end=self.current_timestep,
            learning_rate=self.get_lr()[0],
            extra_val_results=self.extra_val_results,
        )

        self.current_epoch += 1
        self.extra_val_results = []

        self.post_epoch()

        return results

    def run_task(self, task, cloned_adaptation_net):
        self.train_fast_loader.sampler.set_active_tasks(task)
        self.val_fast_loader.sampler.set_active_tasks(task)

        # Train, one batch
        data, target = next(iter(self.train_fast_loader))
        data = data.to(self.device)
        target = target.to(self.device)
        train_loss = self._loss_function(
            cloned_adaptation_net(data), target
        )
        # Update in place
        self.adapt(cloned_adaptation_net, train_loss)

        # Evaluate the adapted model
        with torch.no_grad():
            data, target = next(iter(self.val_fast_loader))
            data = data.to(self.device)
            target = target.to(self.device)

            preds = cloned_adaptation_net(data)
            valid_error = self._loss_function(preds, target)
            valid_error /= len(data)
            self.logger.debug(f"Valid error meta train training: {valid_error}")

            # calculate accuracy
            preds = preds.argmax(dim=1).view(target.shape)
            valid_accuracy = (preds == target).sum().float() / target.size(0)
            self.logger.debug(f"Valid accuracy meta train training: {valid_accuracy}")

    @classmethod
    def update_params(cls, params, loss, lr, distributed=False):
        """
        Takes a gradient step on the loss and updates the cloned parameters in place.
        """
        gradients = torch.autograd.grad(
            loss, params,
            retain_graph=True, create_graph=True
        )

        if distributed:
            size = float(dist.get_world_size())
            for grad in gradients:
                dist.all_reduce(grad.data, op=dist.reduce_op.SUM)
                grad.data /= size

        if gradients is not None:
            params = list(params)
            for p, g in zip(params, gradients):
                if g is not None:
                    p.add_(g, alpha=-lr)

    def adapt(self, cloned_adaptation_net, train_loss):
        fast_params = list(self.get_fast_params(cloned_adaptation_net))
        self.update_params(
            fast_params, train_loss, self.adaptation_lr, distributed=self.distributed
        )

    def clone_model(self, keep_as_reference=None):
        """
        Clones self.model by cloning some of the params and keeping those listed
        specified `keep_as_reference` via reference.
        """
        model = clone_model(self.model.module, keep_as_reference=None)

        if not self.distributed:
            model = DataParallel(model)
        else:
            # Instead of using DistributedDataParallel, the grads will be reduced
            # manually since we won't call loss.backward()
            model

        return model

    def get_slow_params(self):
        if hasattr(self.model, "module"):
            return self.model.module.slow_params
        else:
            return self.model.slow_params

    def get_fast_params(self, clone=None):
        model = clone if clone is not None else self.module
        if hasattr(model, "module"):
            return model.module.fast_params
        else:
            return model.fast_params

    @classmethod
    def aggregate_results(cls, results):
        """Run validation in single GPU"""
        return results[0]

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "MetaContinualLearningExperiment"
        eo["create_loaders"] = [exp + ".create_loaders"]
        eo["create_fast_slow_loaders"] = [exp + ".create_fast_slow_loaders"]
        eo["create_train_sampler"] = [exp + ".create_train_sampler"]
        eo["create_validation_sampler"] = [exp + ".create_validation_sampler"]
        eo["create_replay_sampler"] = [exp + ".create_replay_sampler"]
        eo["create_task_sampler"] = [exp + ".create_task_sampler"]
        eo["create_slow_train_dataloader"] = [exp + ".create_slow_train_dataloader"]
        eo["run_epoch"] = [exp + ".run_epoch"]
        eo["run_task"] = [exp + ".run_task"]
        eo["update_params"] = [exp + ".update_params"]
        eo["adapt"] = [exp + ".adapt"]
        eo["clone_model"] = [exp + ".clone_model"]
        eo["get_slow_params"] = [exp + ".get_slow_params"]
        eo["get_fast_params"] = [exp + ".get_fast_params"]
        eo["aggregate_results"] = [exp + ".aggregate_results"]
        return eo


class ImagenetExperiment(SupervisedExperiment):
    """
    Experiment class used to train Sparse and dense versions of Resnet50 v1.5
    models on Imagenet dataset
    """

    @classmethod
    def create_loaders(cls, config):
        dataset_args = {}
        config.setdefault("dataset_class", datasets.imagenet)
        config.setdefault("dataset_args", dataset_args)

        dataset_args.update(
            data_path=config["data"],
            train_dir=config.get("train_dir", "train"),
            val_dir=config.get("val_dir", "val"),
            num_classes=config.get("num_classes", 1000),
            use_auto_augment=config.get("use_auto_augment", False),
            sample_transform=config.get("sample_transform", None),
            target_transform=config.get("target_transform", None),
            replicas_per_sample=config.get("replicas_per_sample", 1),
        )

        return super().create_loaders(config)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["create_loaders"].insert(0, "ImagenetExperiment.create_loaders")
        return eo
