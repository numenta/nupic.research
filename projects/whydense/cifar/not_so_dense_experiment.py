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
import logging
import os
import sys
import time

import click
import torch
import torch.nn as nn  # noqa F401
from torchvision import datasets
from torchvision.transforms import transforms

from nupic.research.frameworks.pytorch.model_utils import (
    evaluate_model,
    set_random_seed,
    train_model,
)
from nupic.research.frameworks.pytorch.models.not_so_densenet import NoSoDenseNetCIFAR
from nupic.research.support import parse_config
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


class NotSoDenseExperiment(object):
    def __init__(self, config):
        super(NotSoDenseExperiment, self).__init__()
        self.logger = get_logger(config["name"], config["verbose"])
        self.logger.debug("Config: %s", config)

        seed = config["seed"]
        set_random_seed(seed)
        self.batches_in_epoch = config["batches_in_epoch"]
        self.epochs = config["iterations"]
        self.batch_size = config["batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.test_batches_in_epoch = config.get("test_batches_in_epoch",
                                                sys.maxsize)
        data_dir = config["data_dir"]

        normalize_tensor = [transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010))]
        data_augmentation = []
        if config.get("data_augmentation", False):
            data_augmentation = [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip()]

        train_dataset = datasets.CIFAR10(root=data_dir,
                                         train=True,
                                         download=True,
                                         transform=transforms.Compose(
                                             data_augmentation + normalize_tensor))
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)
        test_dataset = datasets.CIFAR10(root=data_dir,
                                        train=False,
                                        download=False,
                                        transform=transforms.Compose(normalize_tensor))
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=self.test_batch_size,
                                                       shuffle=True)

        self.model = NoSoDenseNetCIFAR(block_config=config.get("block_config"),
                                       depth=config.get("depth"),
                                       growth_rate=config["growth_rate"],
                                       reduction=config["reduction"],
                                       num_classes=config["num_classes"],
                                       bottleneck_size=config["bottleneck_size"],
                                       avg_pool_size=config["avg_pool_size"],
                                       dense_percent_on=config["dense_percent_on"],
                                       transition_percent_on=config[
                                           "transition_percent_on"],
                                       classifier_percent_on=config[
                                           "classifier_percent_on"],
                                       k_inference_factor=config["k_inference_factor"],
                                       boost_strength=config["boost_strength"],
                                       boost_strength_factor=config[
                                           "boost_strength_factor"],
                                       duty_cycle_period=config["duty_cycle_period"],
                                       )
        self.logger.debug("Model: %s", self.model)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
        else:
            self.device = torch.device("cpu")

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=config["learning_rate"],
                                         momentum=config["momentum"],
                                         nesterov=config["nesterov"],
                                         weight_decay=config["weight_decay"])
        self.loss_function = config["loss_function"]

        if "learning_scheduler_milestones" in config:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                gamma=config["learning_scheduler_gamma"],
                milestones=config["learning_scheduler_milestones"])
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                gamma=config["learning_scheduler_gamma"],
                step_size=config["learning_scheduler_step_size"])

    def train(self, epoch):
        self.logger.info("epoch: %s", epoch)
        t0 = time.time()
        self.pre_epoch()
        train_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            batches_in_epoch=self.batches_in_epoch,
            criterion=self.loss_function,
            post_batch_callback=self.post_batch,
        )
        self.post_epoch()
        self.logger.info("training duration: %s", time.time() - t0)

    def test(self, loader=None):
        if loader is None:
            loader = self.test_loader
        t0 = time.time()
        results = evaluate_model(model=self.model, device=self.device,
                                 loader=loader,
                                 batches_in_epoch=self.test_batches_in_epoch)
        self.logger.info("testing duration: %s", time.time() - t0)
        self.logger.info("mean_accuracy: %s", results["mean_accuracy"])
        self.logger.info("mean_loss: %s", results["mean_loss"])
        return results

    def save(self, checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def restore(self, checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.pt")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

    def pre_epoch(self):
        self.logger.info("learning rate: %s", self.scheduler.get_lr())

    def post_epoch(self):
        self.model.apply(update_boost_strength)
        self.scheduler.step()

    def post_batch(self, *args, **kwargs):
        self.model.apply(rezero_weights)


@click.command()
@click.option(
    "-c",
    "--config",
    type=open,
    default="not_so_dense.cfg",
    show_default=True,
    help="your experiments config file",
)
@click.option(
    "-e",
    "--experiment",
    "experiments",
    multiple=True,
    help="run only selected experiments, by default run all "
         "experiments in config file.",
)
def main(config, experiments):
    options = parse_config(config, experiments)
    for exp in options:
        print("Experiment:", exp)
        params = options[exp]
        params["name"] = exp
        params["loss_function"] = eval(params["loss_function"], globals(), locals())
        exp = NotSoDenseExperiment(params)
        print(exp.model)
        for epoch in range(exp.epochs):
            exp.train(epoch)
            print(exp.test())


if __name__ == "__main__":
    main()
