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

import numpy as np
import torch

from nupic.research.frameworks.vernon import MetaContinualLearningExperiment


class MetaCLDataSplitExperiment(MetaContinualLearningExperiment):
    """
    This explicitly overrides `run_epoch` so that that replay dataset may be
    sampled from classes 0 to 480` and the slow/fast datasets may be sampled
    from classes 481 to 962. Everything else is the same.
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)
        assert config["model_args"]["num_classes"] >= 962
        self.replay_classes = list(range(0, 481))
        self.slowfast_classes = list(range(481, 963))

    def run_epoch(self):

        self.pre_epoch()

        self.optimizer.zero_grad()

        # Sample tasks for inner loop.
        tasks_train = np.random.choice(
            self.slowfast_classes,
            size=self.tasks_per_epoch,
            replace=False
        )

        # Run pre_task; For instance, may reset parameters as needed.
        self.pre_task(tasks_train)

        # Clone model - clone fast params and the slow params. The latter will be frozen
        cloned_adaptation_net = self.clone_model()

        # Inner loop: Train over sampled tasks.
        for task in tasks_train:
            self.run_task(task, cloned_adaptation_net)

        # Sample from the replay set.
        self.train_replay_loader.sampler.set_active_tasks(self.replay_classes)
        replay_data, replay_target = next(iter(self.train_replay_loader))

        # Sample from the slow set.
        slow_data, slow_target = [], []
        for task in tasks_train:
            self.train_slow_loader.sampler.set_active_tasks(task)
            x, y = next(iter(self.train_slow_loader))
            slow_data.append(x)
            slow_target.append(y)

        # Concatenate the slow and replay set.
        slow_data = torch.cat(slow_data + [replay_data]).to(self.device)
        slow_target = torch.cat(slow_target + [replay_target]).to(self.device)

        # Take step for outer loop. This will backprop through to the original
        # slow and fast params.
        output = cloned_adaptation_net(slow_data)
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
            "learning_rate": self.get_lr()[0],
        }
        self.logger.debug(results)

        self.post_epoch()

        self.current_epoch += 1

        return results