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

import numpy as np
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


class ContinuousSpeechExperiment(object):
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
        self.test_data_dir = config["test_dir"]
        
        # Configure Model
        self.model_type = config["model_type"]
        self.num_classes = 12
        self.log_interval = config["log_interval"]
        self.batches_in_epoch = config["batches_in_epoch"]
        self.batch_size = config["batch_size"]
        self.background_noise_dir = config["background_noise_dir"]
        self.noise_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
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
            print("model on GPU")
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
        

    def train(self, epoch, training_class):
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
        train_model(self.model, self.train_loader[training_class], self.optimizer, self.device,
                    batches_in_epoch=self.batches_in_epoch)
        self.post_epoch(training_class)

        self.logger.info("training duration: %s", time.time() - t0)

    def post_epoch(self, class_):
        self.model.apply(rezero_weights)
        self.lr_scheduler.step()
        self.train_loader[class_].dataset.load_next()

    def pre_epoch(self):
        self.model.apply(update_boost_strength)

    def test_class(self, class_, test_loader=None):
        """Test the model using the given loader and return test metrics."""
        if test_loader is None:
            test_loader = self.test_loader
        try:
            loader = test_loader[class_]
        except:
            loader = self.validation_loader
            
        ret = evaluate_model(self.model, loader, self.device)
        ret["mean_accuracy"] = 100.0 * ret["mean_accuracy"]

        entropy = self.entropy()
        ret.update({
            "entropy": float(entropy),
            "total_samples": len(loader.sampler),
            "non_zero_parameters": count_nonzero_params(self.model)[1],
        })

        return ret
    
    def test(self, test_loader=None):
        if test_loader is None:
            test_loader = self.gen_test_loader
        
        ret = evaluate_model(self.model, test_loader, self.device)
        ret["mean_accuracy"] = 100. * ret["mean_accuracy"]
        entropy = self.entropy()
        ret.update({
            "entropy": float(entropy),
            "total_samples": len(test_loader.sampler),
            "non_zero_parameters": count_nonzero_params(self.model)[1],
        })

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

    def combine_classes(self,class1,class2):

        data1 = torch.load(self.data_dir + "data_train_{}.npz".format(class1))
        data2 = torch.load(self.data_dir + "data_train_{}.npz".format(class2))
        combined_samples = torch.cat((data1[0],data2[0]), dims=0)
        combined_labels = torch.cat((data1[1],data2[1]), dims=0)
        combined_dataset = list((combined_samples, combined_labels))

        self.data_loader = DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        

    def load_datasets(self):
        """
        GSC specifies specific files to be used as training, test, and validation.

        We assume the data has already been processed using the pre-processing scripts
        here: https://github.com/numenta/nupic.torch/tree/master/examples/gsc
        """
        self.test_loader = []
        self.train_loader = []
        
        validation_dataset = ClasswiseDataset(
                        cachefilepath=self.data_dir,
                        basename="data_valid",
                        qualifiers=[""],
            )
        self.validation_loader = DataLoader(
                validation_dataset, batch_size=self.batch_size, shuffle=False
            )
        
        gen_test_dataset = PreprocessedDataset(
                cachefilepath=self.test_data_dir,
                basename="gsc_test_noise",
                qualifiers = ["{:02d}".format(int(100 * n)) for n in self.noise_values],
        )

        self.gen_test_loader = DataLoader(
            gen_test_dataset, batch_size=self.batch_size, shuffle=True
        )

        for class_ in np.arange(1,12):

            test_dataset = ClasswiseDataset(
                cachefilepath=self.data_dir,
                basename="data_test_",
                qualifiers=range(class_,class_+1)
            )
            train_dataset = ClasswiseDataset(
                cachefilepath=self.data_dir,
                basename="data_train_",
                qualifiers=range(class_,class_+1),
            )

            self.train_loader.append(DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            ))


            self.test_loader.append(DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            ))


class ClasswiseDataset(PreprocessedDataset):
    def load_qualifier(self, qualifier):
        """
        Call this to load the a copy of a dataset with the specific qualifier into
        memory.

        :return: Name of the file that was actually loaded.
        """
        file_name = os.path.join(self.path, self.basename + "{}.npz".format(qualifier))
#         self.tensors = list(np.load(file_name, allow_pickle=True))
        self.tensors = list(torch.load(file_name))
        return file_name
