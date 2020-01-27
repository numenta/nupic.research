# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import os

import torch
import torch.nn as nn
from tqdm import tqdm

import nupic.research.frameworks.backprop_structure.dataset_managers as dm
import nupic.research.frameworks.backprop_structure.networks as networks


class Supervised(object):
    def __init__(self,
                 model_alg, model_params,
                 dataset_name, dataset_params,
                 training_iterations,
                 optim_alg, optim_params,
                 lr_scheduler_alg, lr_scheduler_params,
                 batch_size_train, batch_size_test,
                 use_tqdm=False, tqdm_mininterval=None):
        (self.batch_size_train_first_epoch,
         self.batch_size_train) = batch_size_train
        self.batch_size_test = batch_size_test
        self.use_tqdm = use_tqdm
        self.tqdm_mininterval = tqdm_mininterval

        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")

        model_constructor = getattr(networks, model_alg)
        self.model = model_constructor(**model_params)
        self.model.to(self.device)

        dm_constructor = getattr(dm, dataset_name)
        self.dataset_manager = dm_constructor(**dataset_params)

        self.training_iterations = training_iterations

        optim_constructor = getattr(torch.optim, optim_alg)
        self.optimizer = optim_constructor(self._get_parameters(),
                                           **optim_params)

        sched_constructor = getattr(torch.optim.lr_scheduler, lr_scheduler_alg)
        self.lr_scheduler = sched_constructor(self.optimizer,
                                              **lr_scheduler_params)

        self.loss_func = nn.CrossEntropyLoss()

        # Caching
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset_manager.get_test_dataset(),
            batch_size=self.batch_size_test,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )

    def test(self, loader):
        self.model.eval()
        val_loss = 0
        num_val_batches = 0
        val_correct = 0
        with torch.no_grad():
            if self.use_tqdm:
                batches = tqdm(loader, leave=False, desc="Testing")
            else:
                batches = loader

            for data, target in batches:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_func(output, target).item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                num_val_batches += 1

        return {
            "mean_accuracy": val_correct / len(loader.dataset),
            "mean_loss": val_loss / num_val_batches,
            "total_correct": val_correct,
        }

    def run_epoch(self, iteration):
        self.model.train()
        self._before_train_epoch()

        batch_size = (self.batch_size_train_first_epoch
                      if iteration == 0
                      else self.batch_size_train)

        train_loader = torch.utils.data.DataLoader(
            self.dataset_manager.get_train_dataset(iteration),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )

        if self.use_tqdm:
            batches = tqdm(train_loader, leave=False,
                           desc="Training", mininterval=self.tqdm_mininterval)
        else:
            batches = train_loader

        train_loss = 0.
        train_correct = 0.
        num_train_batches = 0

        for data, target in batches:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_func(output, target) + self._regularization()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._after_optimizer_step()

            with torch.no_grad():
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                num_train_batches += 1

        self.lr_scheduler.step()
        self._after_train_epoch()

        result = {
            "mean_train_accuracy": train_correct / len(train_loader.dataset),
            "mean_training_loss": train_loss / num_train_batches,
            "lr": self.optimizer.state_dict()["param_groups"][0]["lr"],
            "done": iteration + 1 >= self.training_iterations,
        }

        test_result = self.test(self.test_loader)
        result.update(test_result)
        return result

    def save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def restore(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))

    def _get_parameters(self):
        return self.model.parameters()

    def _regularization(self):
        return 0

    def _before_train_epoch(self):
        pass

    def _after_train_epoch(self):
        pass

    def _after_optimizer_step(self):
        pass
