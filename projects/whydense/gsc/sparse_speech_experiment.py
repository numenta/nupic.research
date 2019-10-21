# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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
import logging
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from nupic.research.frameworks.pytorch.dataset_utils import PreprocessedDataset
from nupic.research.frameworks.pytorch.model_utils import (
    count_nonzero_params,
    evaluate_model,
    set_random_seed,
    train_model,
)
from nupic.research.frameworks.pytorch.models.le_sparse_net import LeSparseNet
from nupic.research.frameworks.pytorch.models.resnet_models import resnet9
from nupic.torch.models.sparse_cnn import GSCSparseCNN, GSCSuperSparseCNN
from nupic.torch.modules import rezero_weights, update_boost_strength


def get_logger(name, verbose):
    """Configure Logger based on verbose level (0: ERROR, 1: INFO, 2: DEBUG)"""
    logger = logging.getLogger(name)
    if verbose == 0:
        logger.setLevel(logging.ERROR)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    return logger


class SparseSpeechExperiment(object):
    """This experiment tests the Google Speech Commands dataset, available
    here:

    http://download.tensorflow.org/data/speech_commands_v0.01.tar
    """

    def __init__(self, config):
        """Called once at the beginning of each experiment."""
        self.start_time = time.time()
        self.logger = get_logger(config["name"], config.get("verbose", 2))
        self.logger.debug("Config: %s", config)

        # Setup random seed
        seed = config["seed"]
        set_random_seed(seed)

        # Get our directories correct
        self.data_dir = config["data_dir"]

        # Configure Model
        self.model_type = config["model_type"]
        self.num_classes = 12
        self.log_interval = config["log_interval"]
        self.batches_in_epoch = config["batches_in_epoch"]
        self.batch_size = config["batch_size"]
        self.background_noise_dir = config["background_noise_dir"]
        self.noise_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.min_epoch_for_checkpoint = config.get("min_epoch_for_checkpoint", 0)
        self.last_training_epoch = -1
        self.best_accuracy = 0.0

        self.load_datasets()

        if self.model_type == "le_sparse":
            model = LeSparseNet(
                input_shape=config.get("input_shape", (1, 32, 32)),
                cnn_out_channels=config["cnn_out_channels"],
                cnn_activity_percent_on=config["cnn_percent_on"],
                cnn_weight_percent_on=config["cnn_weight_sparsity"],
                linear_n=config["linear_n"],
                linear_activity_percent_on=config["linear_percent_on"],
                linear_weight_percent_on=config["weight_sparsity"],
                boost_strength=config["boost_strength"],
                boost_strength_factor=config["boost_strength_factor"],
                use_batch_norm=config["use_batch_norm"],
                dropout=config.get("dropout", 0.0),
                num_classes=self.num_classes,
                k_inference_factor=config["k_inference_factor"],
                activation_fct_before_max_pool=config.get(
                    "activation_fct_before_max_pool", False),
                consolidated_sparse_weights=config.get(
                    "consolidated_sparse_weights", False),
                use_kwinners_local=config.get("use_kwinner_local", False),
            )

        elif self.model_type == "resnet9":
            model = resnet9(
                num_classes=self.num_classes, in_channels=1
            )

        elif self.model_type == "gsc_sparse_cnn":
            model = GSCSparseCNN()

        elif self.model_type == "gsc_super_sparse_cnn":
            model = GSCSuperSparseCNN()

        else:
            raise RuntimeError("Unknown model type: " + self.model_type)

        self.use_cuda = torch.cuda.is_available()
        self.logger.debug("use_cuda %s", self.use_cuda)
        if self.use_cuda:
            self.device = torch.device("cuda")
            model = model.cuda()
        else:
            self.device = torch.device("cpu")

        self.logger.debug("device %s", self.device)
        if torch.cuda.device_count() > 1:
            self.logger.debug("Using %s GPUs", torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

        self.model = model
        self.logger.debug("Model: %s", self.model)
        self.logger.debug("Model non-zero params: %s", count_nonzero_params(self.model))
        self.learning_rate = config["learning_rate"]
        self.optimizer = self.create_optimizer(config, self.model)
        self.lr_scheduler = self.create_learning_rate_scheduler(config, self.optimizer)

    def save(self, checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def restore(self, checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

    def create_learning_rate_scheduler(self, params, optimizer):
        """Creates the learning rate scheduler and attach the optimizer."""
        lr_scheduler = params.get("lr_scheduler", None)
        if lr_scheduler is None:
            return None

        if lr_scheduler == "StepLR":
            lr_scheduler_params = (
                "{'step_size': 1, 'gamma':" + str(params["learning_rate_factor"]) + "}"
            )

        else:
            lr_scheduler_params = params.get("lr_scheduler_params", None)
            if lr_scheduler_params is None:
                raise ValueError(
                    "Missing 'lr_scheduler_params' for {}".format(lr_scheduler)
                )

        # Get lr_scheduler class by name
        clazz = eval("torch.optim.lr_scheduler.{}".format(lr_scheduler))

        # Parse scheduler parameters from config
        lr_scheduler_params = eval(lr_scheduler_params)

        return clazz(optimizer, **lr_scheduler_params)

    def create_optimizer(self, params, model):
        """Create a new instance of the optimizer."""
        lr = params["learning_rate"]
        print("Creating optimizer with learning rate=", lr)
        if params["optimizer"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=params["momentum"],
                weight_decay=params["weight_decay"],
            )
        elif params["optimizer"] == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise LookupError("Incorrect optimizer value")

        return optimizer

    def train(self, epoch):
        """Train one epoch of this model by iterating through mini batches.

        An epoch ends after one pass through the training set, or if the
        number of mini batches exceeds the parameter "batches_in_epoch".
        """
        self.logger.info("epoch: %s", epoch)

        t0 = time.time()

        self.logger.info(
            "Learning rate: %s",
            self.learning_rate
            if self.lr_scheduler is None
            else self.lr_scheduler.get_lr(),
        )

        self.pre_epoch()
        train_model(self.model, self.train_loader, self.optimizer, self.device,
                    batches_in_epoch=self.batches_in_epoch)
        self.post_epoch()

        self.logger.info("training duration: %s", time.time() - t0)
        self.last_training_epoch = epoch

    def post_epoch(self):
        self.model.apply(rezero_weights)
        self.lr_scheduler.step()
        self.train_loader.dataset.load_next()

    def pre_epoch(self):
        self.model.apply(update_boost_strength)

    def test(self, test_loader=None):
        """Test the model using the given loader and return test metrics."""
        if test_loader is None:
            test_loader = self.test_loader

        ret = evaluate_model(self.model, test_loader, self.device)
        ret["mean_accuracy"] = 100.0*ret["mean_accuracy"]

        entropy = self.entropy()
        ret.update({
            "entropy": float(entropy),
            "total_samples": len(test_loader.sampler),
            "non_zero_parameters": count_nonzero_params(self.model)[1],
            "should_checkpoint":
                (self.best_accuracy < ret["mean_accuracy"]
                 and self.last_training_epoch >= self.min_epoch_for_checkpoint)
        })

        # Keep track of best accuracy that we have checkpointed so far
        if self.last_training_epoch >= self.min_epoch_for_checkpoint:
            self.best_accuracy = max(ret["mean_accuracy"], self.best_accuracy)
        return ret

    def entropy(self):
        """Returns the current entropy."""
        entropy = 0
        for module in self.model.modules():
            if module == self.model:
                continue
            if hasattr(module, "entropy"):
                entropy += module.entropy()

        return entropy

    def validate(self):
        """Run validation."""
        if self.validation_loader:
            return self.test(self.validation_loader)
        return None

    def run_noise_tests(self):
        """
        Test the model with different noise values and return test metrics.
        Loads pre-generated noise dataset with noise transforms included
        """
        ret = {}
        for noise in self.noise_values:
            noise_qualifier = "{:02d}".format(int(100 * noise))
            self.test_loader.dataset.load_qualifier(noise_qualifier)
            ret[noise] = self.test(self.test_loader)
        return ret

    def load_datasets(self):
        """
        GSC specifies specific files to be used as training, test, and validation.

        We assume the data has already been processed using the pre-processing scripts
        here: https://github.com/numenta/nupic.torch/tree/master/examples/gsc
        """
        validation_dataset = PreprocessedDataset(
            cachefilepath=self.data_dir,
            basename="gsc_valid",
            qualifiers=[""],
        )

        test_dataset = PreprocessedDataset(
            cachefilepath=self.data_dir,
            basename="gsc_test_noise",
            qualifiers=["{:02d}".format(int(100 * n)) for n in self.noise_values],
        )
        train_dataset = PreprocessedDataset(
            cachefilepath=self.data_dir,
            basename="gsc_train",
            qualifiers=range(30),
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.validation_loader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
