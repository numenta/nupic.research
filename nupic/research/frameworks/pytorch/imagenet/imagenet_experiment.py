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
import functools
import io
import logging
import multiprocessing
import sys
import time
from pprint import pformat

import ray.services
import torch
import torch.autograd
import torch.distributed as dist
import torch.utils.data
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import OneCycleLR

from nupic.research.frameworks.pytorch.imagenet.experiment_utils import (
    create_lr_scheduler,
    create_model,
    create_optimizer,
    create_train_dataloader,
    create_validation_dataloader,
)
from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler
from nupic.research.frameworks.pytorch.model_utils import (
    count_nonzero_params,
    deserialize_state_dict,
    evaluate_model,
    serialize_state_dict,
    set_random_seed,
    train_model,
)
from nupic.torch.modules import rezero_weights, update_boost_strength

try:
    from apex import amp
except ImportError:
    amp = None

__all__ = ["ImagenetExperiment"]


# Improves performance when using fixed size images (224) and CNN
cudnn.benchmark = True


class ImagenetExperiment:
    """
    Experiment class used to train Sparse and dense versions of Resnet50 v1.5
    models on Imagenet dataset
    """

    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_function = None
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
        self.profile = False
        self.launch_time = 0
        self.epochs_to_validate = []

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
            - lr_scheduler_class: Learning rate scheduler class.
                                 Must inherit from "_LRScheduler"
            - lr_scheduler_args: Learning rate scheduler class class arguments
                                 passed to the constructor
            - loss_function: Loss function. See "torch.nn.functional"
            - local_dir: Results path
            - epochs: Number of epochs to train
            - batches_in_epoch: Number of batches per epoch.
                                Useful for debugging
            - progress: Show progress during training
            - profile: Whether or not to enable torch.autograd.profiler.profile
                       during training
            - name: Experiment name. Used as logger name
            - log_level: Python Logging level
            - log_format: Python Logging format
            - seed: the seed to be used for pytorch, python, and numpy
            - mixed_precision: Whether or not to enable apex mixed precision
            - mixed_precision_args: apex mixed precision arguments.
                                    See "amp.initialize"
            - create_train_dataloader: Optional user defined function to create
                                       the training data loader. See below for
                                       input params.
            - create_validation_dataloader: Optional user defined function to create
                                            the validation data loader. See below for
                                            input params.
            - train_model_func: Optional user defined function to train the model,
                                expected to behave similarly to `train_model`
                                in terms of input parameters and return values
            - evaluate_model_func: Optional user defined function to validate the model
                                   expected to behave similarly to `evaluate_model`
                                   in terms of input parameters and return values
            - init_hooks: list of hooks (functions) to call on the model
                          just following its initialization
            - post_epoch_hooks: list of hooks (functions) to call on the model
                                following each epoch of training
            - checkpoint_file: if not None, will start from this model. The model
                               must have the same model_args and model_class as the
                               current experiment.
            - checkpoint_at_init: boolean argument for whether to create a checkpoint
                                  of the initialized model. this differs from
                                  `checkpoint_at_start` for which the checkpoint occurs
                                  after the first epoch of training as opposed to
                                  before it
            - epochs_to_validate: list of epochs to run validate(). A -1 asks
                                  to run validate before any training occurs.
                                  Default: last three epochs.
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

        # Configure seed
        self.seed = config.get("seed", self.seed)
        set_random_seed(self.seed, False)

        # Configure distribute pytorch
        self.distributed = config.get("distributed", False)
        self.rank = config.get("rank", 0)
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
        model_class = config["model_class"]
        model_args = config.get("model_args", {})
        init_batch_norm = config.get("init_batch_norm", False)
        init_hooks = config.get("init_hooks", None)
        self.model = create_model(
            model_class=model_class,
            model_args=model_args,
            init_batch_norm=init_batch_norm,
            device=self.device,
            init_hooks=init_hooks,
            checkpoint_file=config.get("checkpoint_file", None)
        )
        if self.rank == 0:
            self.logger.debug(self.model)
            params_sparse, nonzero_params_sparse2 = count_nonzero_params(self.model)
            self.logger.debug("Params total/nnz %s / %s = %s ",
                              params_sparse, nonzero_params_sparse2,
                              float(nonzero_params_sparse2) / params_sparse)

        # Configure optimizer
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        batch_norm_weight_decay = config.get("batch_norm_weight_decay", True)
        self.optimizer = create_optimizer(
            model=self.model,
            optimizer_class=optimizer_class,
            optimizer_args=optimizer_args,
            batch_norm_weight_decay=batch_norm_weight_decay,
        )

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

        self.loss_function = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )

        # Configure data loaders
        self.epochs = config.get("epochs", 1)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.epochs_to_validate = config.get("epochs_to_validate",
                                             range(self.epochs - 3, self.epochs + 1))
        workers = config.get("workers", 0)
        data_dir = config["data"]
        train_dir = config.get("train_dir", "train")
        num_classes = config.get("num_classes", 1000)

        # Get initial batch size
        self.batch_size = config.get("batch_size", 1)

        # CUDA runtime does not support the fork start method.
        # See https://pytorch.org/docs/stable/notes/multiprocessing.html
        if torch.cuda.is_available():
            multiprocessing.set_start_method("spawn")

        # Configure Training data loader
        self.create_train_dataloader = config.get(
            "create_train_dataloader", create_train_dataloader)
        self.train_loader = self.create_train_dataloader(
            data_dir=data_dir,
            train_dir=train_dir,
            batch_size=self.batch_size,
            workers=workers,
            distributed=self.distributed,
            num_classes=num_classes,
            use_auto_augment=config.get("use_auto_augment", False),
        )
        self.total_batches = len(self.train_loader)

        # Configure Validation data loader
        val_dir = config.get("val_dir", "val")
        val_batch_size = config.get("val_batch_size", self.batch_size)
        self.create_validation_dataloader = config.get(
            "create_validation_dataloader", create_validation_dataloader)
        self.val_loader = self.create_validation_dataloader(
            data_dir=data_dir,
            val_dir=val_dir,
            batch_size=val_batch_size,
            workers=workers,
            num_classes=num_classes,
        )

        # Configure learning rate scheduler
        lr_scheduler_class = config.get("lr_scheduler_class", None)
        if lr_scheduler_class is not None:
            lr_scheduler_args = config.get("lr_scheduler_args", {})
            self.logger.info("LR Scheduler args:")
            self.logger.info(pformat(lr_scheduler_args))
            self.logger.info("steps_per_epoch=%s", self.total_batches)
            self.lr_scheduler = create_lr_scheduler(
                optimizer=self.optimizer,
                lr_scheduler_class=lr_scheduler_class,
                lr_scheduler_args=lr_scheduler_args,
                steps_per_epoch=self.total_batches)

        # Only profile from rank 0
        self.profile = config.get("profile", False) and self.rank == 0

        # Set train and validate methods.
        self.train_model = config.get("train_model_func", train_model)
        self.evaluate_model = config.get("evaluate_model_func", evaluate_model)

        # Register post-epoch hooks. To be used as `self.model.apply(post_epoch_hook)`
        self.post_epoch_hooks = config.get("post_epoch_hooks", [])

    def validate(self, epoch, loader=None):
        if loader is None:
            loader = self.val_loader

        if epoch in self.epochs_to_validate:
            results = self.evaluate_model(
                model=self.model,
                loader=loader,
                device=self.device,
                criterion=self.loss_function,
                batches_in_epoch=self.batches_in_epoch,
            )
        else:
            results = {
                "total_correct": 0,
                "mean_loss": 0.0,
                "mean_accuracy": 0.0,
            }

        results.update(
            learning_rate=self.get_lr()[0],
        )
        self.logger.info(results)

        return results

    def train_epoch(self, epoch):
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(),
                                             enabled=self.profile) as prof:
            self.train_model(
                model=self.model,
                loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                criterion=self.loss_function,
                batches_in_epoch=self.batches_in_epoch,
                pre_batch_callback=functools.partial(self.pre_batch, epoch=epoch),
                post_batch_callback=functools.partial(self.post_batch, epoch=epoch),
            )
        if self.profile and prof is not None:
            self.logger.info(prof.key_averages().table(sort_by="self_cpu_time_total"))

    def run_epoch(self, epoch):
        if -1 in self.epochs_to_validate and epoch == 0:
            self.logger.debug("Validating before any training:")
            self.validate(epoch=-1)
        self.pre_epoch(epoch)
        self.train_epoch(epoch)
        self.post_epoch(epoch)
        t1 = time.time()
        ret = self.validate(epoch)

        if self.rank == 0:
            self.logger.debug("validate time: %s", time.time() - t1)
            self.logger.debug("---------- End of run epoch ------------")
            self.logger.debug("")

        return ret

    def pre_epoch(self, epoch):
        self.model.apply(update_boost_strength)
        if self.distributed:
            self.train_loader.sampler.set_epoch(epoch)

    def pre_batch(self, model, batch_idx, epoch):
        pass

    def post_batch(self, model, loss, batch_idx, epoch, num_images, time_string):
        # Update 1cycle learning rate after every batch
        if isinstance(self.lr_scheduler, (OneCycleLR, ComposedLRScheduler)):
            self.lr_scheduler.step()

        if self.progress and epoch == 0 and batch_idx == 0:
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
                              self.rank, epoch, current_batch, total_batches, loss,
                              self.get_lr(), num_images)
            self.logger.debug("Timing: %s", time_string)

    def post_epoch(self, epoch):
        count_nnz = self.logger.isEnabledFor(logging.DEBUG) and self.rank == 0
        if count_nnz:
            params_sparse, nonzero_params_sparse1 = count_nonzero_params(self.model)

        self.model.apply(rezero_weights)
        if self.post_epoch_hooks:
            for hook in self.post_epoch_hooks:
                self.model.apply(hook)

        if count_nnz:
            params_sparse, nonzero_params_sparse2 = count_nonzero_params(self.model)
            self.logger.debug("Params total/nnz before/nnz after %s %s / %s = %s",
                              params_sparse, nonzero_params_sparse1,
                              nonzero_params_sparse2,
                              float(nonzero_params_sparse2) / params_sparse)

        self.logger.debug("End of epoch %s LR/weight decay before step: %s/%s", epoch,
                          self.get_lr(), self.get_weight_decay())

        # Update learning rate
        if not isinstance(self.lr_scheduler, (OneCycleLR, ComposedLRScheduler)):
            self.lr_scheduler.step()

        self.logger.debug("End of epoch %s LR/weight decay after step: %s/%s", epoch,
                          self.get_lr(), self.get_weight_decay())

    def get_state(self):
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        # Save state into a byte array to avoid ray's GPU serialization issues
        # See https://github.com/ray-project/ray/issues/5519
        state = {}
        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.model.module.state_dict())
            state["model"] = buffer.getvalue()

        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.optimizer.state_dict())
            state["optimizer"] = buffer.getvalue()

        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.lr_scheduler.state_dict())
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

    def stop_experiment(self):
        if self.distributed:
            dist.destroy_process_group()

    def get_lr(self):
        """
        Returns the current learning rate
        :return: list of learning rates used by the optimizer
        """
        return [p["lr"] for p in self.optimizer.param_groups]

    def get_weight_decay(self):
        """
        Returns the current weight decay
        :return: list of weight decays used by the optimizer
        """
        return [p["weight_decay"] for p in self.optimizer.param_groups]

    def get_node_ip(self):
        """Returns the IP address of the current ray node."""
        return ray.services.get_node_ip_address()
