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

from nupic.research.frameworks.pytorch.datasets import ImagenetDataset
from nupic.research.frameworks.pytorch.distributed_sampler import (
    UnpaddedDistributedSampler,
)
from nupic.research.frameworks.pytorch.imagenet.experiment_utils import (
    create_lr_scheduler,
    get_free_port,
    get_node_ip_address,
)
from nupic.research.frameworks.pytorch.imagenet.network_utils import (
    create_model,
    get_compatible_state_dict,
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

try:
    from apex import amp
except ImportError:
    amp = None

__all__ = [
    "SupervisedExperiment",
    "ImagenetExperiment",
]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class SupervisedExperiment:
    """
    Experiment class used to train Sparse and dense versions of Resnet50 v1.5
    models on Imagenet dataset
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
        self.current_epoch = 0

        # Get initial batch size
        self.batch_size = config.get("batch_size", 1)

        # CUDA runtime does not support the fork start method.
        # See https://pytorch.org/docs/stable/notes/multiprocessing.html
        multiprocessing.set_start_method("spawn", force=True)

        # Configure data loaders
        self.train_loader, self.val_loader, self.test_loader = (
            self.create_loaders(config)
        )
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
        """Create train, val, and test dataloaders."""

        dataset_class = config.get("dataset", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")
        dataset_args = config.get("dataset_args", {})
        dataset = dataset_class(**dataset_args)

        train_set = dataset.get_train_dataset()
        val_set = dataset.get_val_dataset()
        test_set = dataset.get_test_dataset()

        train_loader = cls.create_train_dataloader(train_set, config)
        val_loader = cls.create_validation_dataloader(val_set, config)
        test_loader = cls.create_validation_dataloader(test_set, config)
        return train_loader, val_loader, test_loader

    @classmethod
    def create_train_dataloader(cls, dataset, config):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """

        if config.get("distributed", False):
            sampler = DistributedSampler(dataset)
        else:
            sampler = None
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
    def create_validation_dataloader(cls, dataset, config):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """

        if config.get("distributed", False):
            sampler = UnpaddedDistributedSampler(dataset, shuffle=False)
        else:
            sampler = None
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
            batches_in_epoch=self.batches_in_epoch,
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
        return dict(
            setup_experiment=["SupervisedExperiment.setup_experiment"],
            create_model=["SupervisedExperiment.create_model"],
            transform_model=["SupervisedExperiment.transform_model"],
            create_loaders=["SupervisedExperiment.create_loaders"],
            create_train_dataloader=["SupervisedExperiment.create_train_dataloader"],
            create_validation_dataloader=[
                "SupervisedExperiment.create_validation_dataloader"
            ],
            create_lr_scheduler=["SupervisedExperiment.create_lr_scheduler"],
            pre_experiment=["SupervisedExperiment.pre_experiment"],
            validate=["SupervisedExperiment.validate"],
            train_epoch=["SupervisedExperiment.train_epoch"],
            run_epoch=["SupervisedExperiment.run_epoch"],
            pre_epoch=["SupervisedExperiment.pre_epoch"],
            post_epoch=["SupervisedExperiment.post_epoch"],
            pre_batch=["SupervisedExperiment.pre_batch"],
            post_batch=["SupervisedExperiment.post_batch"],
            error_loss=["SupervisedExperiment.error_loss"],
            complexity_loss=["SupervisedExperiment.complexity_loss"],
            should_decay_parameter=[
                "SupervisedExperiment.should_decay_parameter"
            ],
            transform_data_to_device=[
                "SupervisedExperiment.transform_data_to_device"
            ],
            aggregate_results=["SupervisedExperiment.aggregate_results"],
            aggregate_validation_results=[
                "SupervisedExperiment.aggregate_validation_results"
            ],
            aggregate_pre_experiment_results=[
                "SupervisedExperiment.aggregate_pre_experiment_results"
            ],
            get_printable_result=["SupervisedExperiment.get_printable_result"],
            expand_result_to_time_series=[
                "SupervisedExperiment: validation results"
            ],
            stop_experiment=["SupervisedExperiment.stop_experiment"]
        )


class ImagenetExperiment(SupervisedExperiment):

    @classmethod
    def create_loaders(cls, config):
        dataset = ImagenetDataset
        dataset_args = {}
        config.setdefault("dataset", dataset)
        config.setdefault("dataset_args", dataset_args)

        dataset_args.update(
            data=config["data"],
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
