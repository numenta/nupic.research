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


class Supervised(object):
    def __init__(self,
                 network_class, network_args,
                 dataset_class, dataset_args,
                 training_iterations,
                 batch_size_train, batch_size_test,
                 logdir,
                 optim_class=None, optim_args=None,
                 lr_scheduler_class=None, lr_scheduler_args=None,
                 lr_step_every_batch=False,
                 use_tqdm=False, tqdm_mininterval=None,
                 parallel=False):
        self.logdir = logdir

        (self.batch_size_train_first_epoch,
         self.batch_size_train) = batch_size_train
        self.batch_size_test = batch_size_test
        self.use_tqdm = use_tqdm
        self.tqdm_mininterval = tqdm_mininterval

        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")

        self.network = network_class(**network_args)
        self.network.to(self.device)

        if parallel and torch.cuda.device_count() > 1:
            self.network = nn.DataParallel(self.network)

        self.dataset_manager = dataset_class(**dataset_args)

        self.training_iterations = training_iterations

        if optim_class is not None:
            self.optimizer = optim_class(self._get_parameters(), **optim_args)
        else:
            # The caller or overrider is taking responsibility for setting the
            # optimizer before calling run_epoch.
            self.optimizer = None

        if lr_scheduler_class is not None:
            self.lr_scheduler = lr_scheduler_class(self.optimizer,
                                                   **lr_scheduler_args)
        else:
            self.lr_scheduler = None
        self.lr_step_every_batch = lr_step_every_batch

        self.loss_func = nn.CrossEntropyLoss()

        # Caching
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset_manager.get_test_dataset(),
            batch_size=self.batch_size_test,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )

    def test(self, loader):
        self.network.eval()
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
                output = self.network(data)
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
        self.network.train()
        self._before_train_epoch(iteration)

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

        for data, target in batches:
            data, target = data.to(self.device), target.to(self.device)
            output = self.network(data)
            loss = self.loss_func(output, target) + self._regularization()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._after_optimizer_step()

            if self.lr_scheduler is not None and self.lr_step_every_batch:
                self.lr_scheduler.step()

        if self.lr_scheduler is not None and not self.lr_step_every_batch:
            self.lr_scheduler.step()
        self._after_train_epoch(iteration)

        result = {
            "lr": self.optimizer.param_groups[0]["lr"],
            "done": iteration + 1 >= self.training_iterations,
        }

        test_result = self.test(self.test_loader)
        result.update(test_result)
        return result

    def on_finished(self):
        pass

    def save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.network.state_dict(), checkpoint_path)
        return checkpoint_path

    def restore(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        self.network.load_state_dict(torch.load(checkpoint_path))

    def _get_parameters(self):
        return self.network.parameters()

    def _regularization(self):
        return 0

    def _before_train_epoch(self, iteration):
        pass

    def _after_train_epoch(self, iteration):
        pass

    def _after_optimizer_step(self):
        pass
